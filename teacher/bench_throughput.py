"""Aggregate teacher throughput, and a few multilingual samples to pick the teacher by.

    uv run python -m teacher.bench_throughput                       # tok/s at concurrency 1/8/32/64, both models
    uv run python -m teacher.bench_throughput --samples             # 20 decisions per FA/AR/RU per model

vLLM batches requests, so the single-stream tok/s a chat user sees says nothing about
how fast a data run finishes. We fire the *real* generation prompt at several
concurrency levels and count completion tokens per wall-second. Requests per second
is what sizes the dataset: every request yields ~10 examples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from tabulate import tabulate

from teacher.client import Teacher, gather_limited
from teacher.prompts import DOMAINS, Cell, generate_prompt, loads_lenient, parse_generate

MODELS = ["rande-fast-local", "rande-strong-local"]


def _cells(n: int, seed: int, langs=("en", "fa", "ar", "ru"), kinds=("choose", "verify", "score", "rank")) -> list[Cell]:
    rng = random.Random(seed)
    doms = list(DOMAINS)
    out = []
    for i in range(n):
        kind = rng.choice(kinds)
        nc = 2 if kind == "verify" else rng.choice([3, 5, 8, 12])
        lang = rng.choice(langs)
        out.append(Cell(domain=rng.choice(doms), kind=kind, n_choices=nc, state_lang=lang, choice_lang=lang,
                        none=kind == "choose" and rng.random() < 0.25, n_examples=10, seed=seed * 1000 + i))
    return out


async def measure(model: str, levels: list[int], max_tokens: int, thinking: bool | None) -> list[dict]:
    rows = []
    for c in levels:
        n_req = min(64, max(4, 2 * c))   # enough to fill the level; the workers cap in-flight sequences anyway
        async with Teacher(model=model, concurrency=c, log_path=None) as t:
            cells = _cells(n_req, seed=c)
            t0 = time.perf_counter()
            res = await gather_limited(
                [t.chat(generate_prompt(cell), max_tokens=max_tokens, thinking=thinking, tag="bench") for cell in cells],
                desc=f"{model} c={c}", every=0)
            wall = time.perf_counter() - t0
            ok = [r for r in res if not isinstance(r, Exception)]
            comp = sum(r.usage.get("completion_tokens", 0) for r in ok)
            prompt = sum(r.usage.get("prompt_tokens", 0) for r in ok)
            valid = 0
            for r, cell in zip(res, cells):
                if not isinstance(r, Exception):
                    try:
                        valid += len(parse_generate(loads_lenient(r.text), cell)[0])
                    except Exception:  # noqa: BLE001
                        pass
            row = {"model": model, "concurrency": c, "requests": n_req, "ok": len(ok), "wall_s": round(wall, 1),
                   "completion_tok": comp, "prompt_tok": prompt, "tok_s": round(comp / wall, 1),
                   "req_s": round(len(ok) / wall, 3), "examples_s": round(valid / wall, 2),
                   "valid_examples": valid, "mean_completion_tok": round(comp / max(1, len(ok)))}
            rows.append(row)
            print(f"  {model:20s} c={c:3d}  {row['tok_s']:7.1f} tok/s  {row['req_s']:.3f} req/s  "
                  f"{row['examples_s']:.2f} ex/s  ({len(ok)}/{n_req} ok, {wall:.0f}s)", flush=True)
    return rows


async def samples(models: list[str], per_lang: int, thinking: bool | None, out_dir: Path) -> None:
    """20 decisions per language per model: 2 calls × 10 examples, mixed kinds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        recs = []
        async with Teacher(model=model, concurrency=8, log_path=None) as t:
            cells = []
            for lang in ("fa", "ar", "ru"):
                for j in range(max(1, per_lang // 10)):
                    kind = ["choose", "verify", "score", "rank"][j % 4]
                    nc = 2 if kind == "verify" else [5, 8, 3, 5][j % 4]
                    cells.append(Cell(domain=random.Random(j * 7 + len(lang)).choice(list(DOMAINS)), kind=kind, n_choices=nc,
                                      state_lang=lang, choice_lang=lang, none=kind == "choose", n_examples=10, seed=100 + j))
            res = await gather_limited([t.chat(generate_prompt(c), max_tokens=4096, thinking=thinking, tag="sample") for c in cells], every=0)
            for r, c in zip(res, cells):
                if isinstance(r, Exception):
                    print(f"  {model} {c.state_lang} {c.kind}: FAILED {r}")
                    continue
                try:
                    ok, rej = parse_generate(loads_lenient(r.text), c)
                except Exception as e:  # noqa: BLE001
                    print(f"  {model} {c.state_lang} {c.kind}: unparseable ({e}); finish={r.finish_reason} tokens={r.usage.get('completion_tokens')}")
                    continue
                print(f"  {model} {c.state_lang} {c.kind} n={c.n_choices}: {len(ok)} valid, {len(rej)} rejected {rej[:3]}; "
                      f"{r.usage.get('completion_tokens')} tok in {r.seconds:.0f}s")
                for e in ok:
                    e["model"] = model
                recs += ok
        p = out_dir / f"samples-{model}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        print(f"  saved {len(recs)} → {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--levels", nargs="*", type=int, default=[1, 8, 32, 64])
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--per-lang", type=int, default=20)
    args = ap.parse_args()
    thinking = {"on": True, "off": False, "default": None}[args.thinking]

    if args.samples:
        asyncio.run(samples(args.models, args.per_lang, thinking, Path("data/teacher")))
        return

    rows = []
    for m in args.models:
        rows += asyncio.run(measure(m, args.levels, args.max_tokens, thinking))
    cols = ["model", "concurrency", "requests", "ok", "wall_s", "tok_s", "req_s", "examples_s", "mean_completion_tok"]
    print()
    print(tabulate([[r[c] for c in cols] for r in rows], headers=cols))
    out = Path(f"runs/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-teacher-throughput.json")
    out.write_text(json.dumps({"thinking": args.thinking, "max_tokens": args.max_tokens, "rows": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
