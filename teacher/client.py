"""Thin async client for the teacher (any OpenAI-compatible chat endpoint).

Why not the `openai` SDK: we need three things it makes awkward — a hard concurrency
cap shared across many generators, retries that understand vLLM's failure modes
(a 5xx while a worker restarts, a truncated JSON body), and a per-call token log we
can size a 48-hour run from. Sixty lines of httpx cover all three.

Config comes from `.env` (TEACHER_BASE_URL, TEACHER_API_KEY, TEACHER_MODEL), never from
code, so the same generators run against a different endpoint by editing one file.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """`KEY=value` lines → environment. Existing variables win, so a shell export overrides."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return {k: os.environ.get(k, "") for k in ("TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL")}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    failures: int = 0
    seconds: float = 0.0

    def add(self, u: dict, dt: float) -> None:
        self.prompt_tokens += int(u.get("prompt_tokens", 0))
        self.completion_tokens += int(u.get("completion_tokens", 0))
        self.requests += 1
        self.seconds += dt


@dataclass
class Reply:
    text: str
    usage: dict
    seconds: float
    finish_reason: str | None = None
    meta: dict = field(default_factory=dict)

    def json(self):
        """Parse the reply as JSON, tolerating a ```json fence a model adds despite JSON mode."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            t = t.rsplit("```", 1)[0]
        return json.loads(t)


class TeacherError(RuntimeError):
    pass


class Teacher:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        concurrency: int = 8,
        max_retries: int = 5,
        timeout: float = 600.0,
        log_path: str | Path | None = "data/teacher/usage.jsonl",
    ):
        env = load_env()
        self.base_url = (base_url or env["TEACHER_BASE_URL"]).rstrip("/")
        self.api_key = api_key or env["TEACHER_API_KEY"]
        self.model = model or env["TEACHER_MODEL"]
        if not (self.base_url and self.model):
            raise TeacherError("TEACHER_BASE_URL / TEACHER_MODEL missing — copy .env.example to .env")
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.usage = Usage()
        self.log_path = Path(log_path) if log_path else None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def models(self) -> list[str]:
        r = await self._client.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        json_mode: bool = False,
        schema: dict | None = None,
        thinking: bool | None = None,
        tag: str = "",
        **extra,
    ) -> Reply:
        """One chat completion under the concurrency cap, retried on transient failures.

        `schema` asks vLLM for grammar-constrained JSON (`json_schema`); `json_mode` is
        the weaker `json_object`. `thinking=False` turns a hybrid model's scratchpad off
        via chat_template_kwargs — for structured data we want answers, not reasoning
        tokens we pay for and throw away.
        """
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}

        async with self.sem:
            last: Exception | None = None
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    r = await self._client.post("/chat/completions", json=body)
                    if r.status_code in _RETRYABLE:
                        raise TeacherError(f"HTTP {r.status_code}: {r.text[:200]}")
                    r.raise_for_status()
                    data = r.json()
                    dt = time.perf_counter() - t0
                    choice = data["choices"][0]
                    usage = data.get("usage") or {}
                    self.usage.add(usage, dt)
                    self._log(tag, usage, dt, choice.get("finish_reason"))
                    return Reply(
                        text=choice["message"].get("content") or "",
                        usage=usage,
                        seconds=dt,
                        finish_reason=choice.get("finish_reason"),
                        meta={"reasoning": choice["message"].get("reasoning_content")},
                    )
                except (httpx.HTTPError, TeacherError, KeyError, json.JSONDecodeError) as e:
                    last = e
                    self.usage.failures += 1
                    if attempt == self.max_retries:
                        break
                    await asyncio.sleep(min(60, 2**attempt) + random.random())
            raise TeacherError(f"teacher failed after {self.max_retries + 1} attempts: {last}")

    def _log(self, tag: str, usage: dict, dt: float, finish: str | None) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "model": self.model, "tag": tag, "s": round(dt, 3),
                                "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                                "finish": finish}) + "\n")


async def gather_limited(coros, desc: str = "", every: int = 50):
    """asyncio.gather that prints progress; exceptions are returned, not raised, so one
    bad call never kills a long generation run."""
    done = 0
    out = [None] * len(coros)

    async def run(i, c):
        nonlocal done
        try:
            out[i] = await c
        except Exception as e:  # noqa: BLE001
            out[i] = e
        done += 1
        if every and done % every == 0:
            print(f"  {desc} {done}/{len(coros)}", flush=True)

    await asyncio.gather(*(run(i, c) for i, c in enumerate(coros)))
    return out
