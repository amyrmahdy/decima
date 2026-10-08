"""Router: TypeSafe-style questions in, `/v1/systemone`-shaped answers out, plus ready-made presets.

    from decima import Router
    r = Router()                                      # the preset picks the model; decima-base for your own questions
    r.decide_text("I was charged twice and nobody answers", preset="triage")
    r.predict("My card was stolen", {"block": {"type": "noul", "instructions": "The customer wants the card blocked."}})
    → {"model": "decima-base", "answers": {"block": {"type": "noul", "noul": 0.97}}, "usage": {...}}

Presets are JSON files in decima/presets/ (or any path): {"description", "model", "state"?, "questions"}.
`state` turns plain text into the state the questions were written for (e.g. {"tool": "Bash", "command": "{text}"});
a question with `requires: [keys]` is asked only when the state object has those keys, so one preset can cover
several state shapes (kg-judge: sentences, mention pairs, record pairs).

Models: a Hub repo id, a short name (decima-base, decima-agent, decima-small) or a local export dir.
DECIMA_MODELS="decima-base=/models/base,decima-agent=/models/agent" maps names to local dirs (air-gapped hosts, Docker).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .systemone import _confidence, _r, to_question
from .types import Question

DEFAULT_MODEL = "amyrmahdy/decima-base"
PRESETS_DIR = Path(__file__).parent / "presets"
_LOADED: dict[tuple, Any] = {}


def resolve_model(name: str) -> str:
    """Short name / repo id / path → what DecimaOnnx.from_pretrained takes, after DECIMA_MODELS overrides."""
    local = dict(kv.split("=", 1) for kv in os.environ.get("DECIMA_MODELS", "").split(",") if "=" in kv)
    for k in (name, name.split("/")[-1]):
        if k in local:
            return local[k]
    if "/" not in name and not Path(name).exists():
        return f"amyrmahdy/{name}"
    return name


def short_name(name: str) -> str:
    return Path(str(name).rstrip("/")).name


def list_presets() -> dict[str, str]:
    return {p.stem: json.loads(p.read_text())["description"] for p in sorted(PRESETS_DIR.glob("*.json"))}


def load_preset(name_or_path: str | dict) -> dict:
    """A preset by name (decima/presets/<name>.json), a path to a preset JSON file, or a preset dict."""
    if isinstance(name_or_path, dict):
        return name_or_path
    p = Path(name_or_path)
    if p.suffix != ".json" or not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise ValueError(f"unknown preset {name_or_path!r}; available: {', '.join(list_presets())}")
    return json.loads(p.read_text())


def fill(template: Any, text: str) -> Any:
    """Put `text` wherever the template has the string "{text}"."""
    if template == "{text}":
        return text
    if isinstance(template, dict):
        return {k: fill(v, text) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, text) for v in template]
    return template


def applicable(questions: dict, state: Any) -> dict:
    """Drop questions whose `requires` keys the state lacks; strip `requires` from the rest."""
    keys = set(state) if isinstance(state, dict) else None
    out = {}
    for qid, spec in questions.items():
        req = spec.get("requires") if isinstance(spec, dict) else None
        if req and keys is not None and not set(req) <= keys:
            continue
        out[qid] = {k: v for k, v in spec.items() if k != "requires"} if req else spec
    return out


def answer(t: str, keys: list[str], p: list[float]) -> dict:
    """One `/v1/systemone` answer from probabilities over the question's options (same fields as decima.systemone)."""
    if t == "choice":
        i = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": keys[i], "confidence": _r(_confidence(p)), "probabilities": {k: _r(v) for k, v in zip(keys, p)}}
    if t == "score":
        return {"type": "score", "score": _r(sum(i * v for i, v in enumerate(p))), "confidence": _r(_confidence(p)),
                "legend": {str(i): lv for i, lv in enumerate(keys)}, "probabilities": {str(i): _r(v) for i, v in enumerate(p)}}
    return {"type": "noul", "noul": _r(p[0])}


class Router:
    """Decisions over TypeSafe-style question dicts, on CPU. Models load lazily and stay cached.

    model: fixes the model for every call; None lets each preset use its recommended model (decima-base otherwise).
    """

    def __init__(self, model: str | None = None, precision: str = "int8", threads: int = 1):
        self.model, self.precision, self.threads = model, precision, threads

    def load(self, name: str):
        """The model behind a name, loaded once per process (Routers with the same settings share it)."""
        key = (resolve_model(name), self.precision, self.threads)
        if key not in _LOADED:
            from .runtime import DecimaOnnx

            _LOADED[key] = DecimaOnnx.from_pretrained(key[0], precision=self.precision, threads=self.threads)
        return _LOADED[key]

    def model_for(self, preset: dict | None = None) -> str:
        return self.model or (preset or {}).get("model") or DEFAULT_MODEL

    def _decide(self, model, items: list[tuple[str, Question]]) -> list[list[float]]:
        """Probabilities for every (state, question) pair: all questions about one state in one batched call."""
        out: list[list[float] | None] = [None] * len(items)
        groups: dict[str, list[int]] = {}
        for i, (s, _) in enumerate(items):
            groups.setdefault(s, []).append(i)
        for s, idx in groups.items():
            for i, d in zip(idx, model.decide_many(s, [items[i][1] for i in idx])):
                out[i] = d.probs
        return out

    def _plan(self, state: Any, questions: dict | None, preset: str | dict | None, lang: str):
        if isinstance(state, (bool, int, float)) or state is None:
            raise ValueError("state must be a string, object or array")
        ps = load_preset(preset) if preset is not None else None
        qs = applicable({**(ps or {}).get("questions", {}), **(questions or {})}, state)
        if not qs:
            raise ValueError(f"no question applies to this state (keys: {sorted(state) if isinstance(state, dict) else 'text'})")
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return ps, s, [(qid, *to_question(qid, spec, lang)) for qid, spec in qs.items()]

    def predict(self, state: Any, questions: dict | None = None, *, preset: str | dict | None = None, lang: str = "en") -> dict:
        """Answer TypeSafe-style `questions` (and/or a preset's) about one state, in the `/v1/systemone` response shape."""
        return self.predict_many([state], questions, preset=preset, lang=lang)[0]

    def predict_many(self, states: list[Any], questions: dict | None = None, *, preset: str | dict | None = None, lang: str = "en") -> list[dict]:
        preset = load_preset(preset) if preset is not None else None
        plans = [self._plan(s, questions, preset, lang) for s in states]
        name = self.model_for(plans[0][0] if plans else None)
        model = self.load(name)
        items = [(s, q) for _, s, qs in plans for _, _, q, _ in qs]
        t0 = time.perf_counter()
        probs = iter(self._decide(model, items))
        ms = (time.perf_counter() - t0) * 1e3 / max(len(plans), 1)
        out = []
        for _, s, qs in plans:
            n = len(model.tok(s)["input_ids"])
            nq = max(len(model.tok(q.text)["input_ids"]) for _, _, q, _ in qs) if model.cfg.get("question_in_state") else 0
            r = {"model": short_name(name), "answers": {qid: answer(t, keys, next(probs)) for qid, t, _, keys in qs},
                 "usage": {"input_tokens": n, "output_tokens": 0}, "latency_ms": round(ms, 1)}
            limit = model.cfg.get("max_state_tokens", 512)
            if n + nq > limit:
                r["warnings"] = [f"state is {n} tokens; {short_name(name)} reads the first {limit}"]
            out.append(r)
        return out

    def decide_text(self, text: str, preset: str | dict = "triage", lang: str = "en") -> dict:
        """Run a preset on plain text, shaped into the state its questions expect. JSON object text is used as is."""
        ps = load_preset(preset)
        return self.predict(as_state(text, ps), preset=ps, lang=lang)


def as_state(text: str, preset: str | dict | None = None) -> Any:
    """Text from a user or a file → state: a JSON object or array stays structured; plain text goes through the
    preset's `state` template."""
    if text.lstrip().startswith(("{", "[")):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return fill(load_preset(preset).get("state", "{text}"), text) if preset is not None else text
