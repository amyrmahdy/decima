"""Generator 1: the teacher invents whole decisions.

    uv run python -m teacher.generate --calls 200                # pilot (~2k examples)
    uv run python -m teacher.generate --calls 30000 --concurrency 16   # full run

The grid is seeded and deterministic: call i always asks for the same
(domain × languages × kind × n_choices × none-option) cell, so a run can be stopped
and resumed by cell id, and two runs never produce the same call twice. Every valid
example is appended to data/teacher/generate-<model>.jsonl as it arrives; rejects go to
a sibling log so a bad prompt shows up as a number, not a silent gap.

Mix (by count): languages EN 30 / FA 25 / AR 25 / RU 20; ~15 % cross-lingual (state in
one language, question+choices in another, mostly English choices — the MASSIVE shape);
kinds choose 40 / verify 22 / score 20 / rank 18 — verify, score and out-of-scope are
where the baseline is weakest, so they get more than their share.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

from teacher.client import Teacher, TeacherError
from teacher.prompts import DOMAINS, Cell, generate_prompt, loads_lenient, parse_generate

LANG_W = {"en": 0.30, "fa": 0.25, "ar": 0.25, "ru": 0.20}
KIND_W = {"choose": 0.40, "verify": 0.22, "score": 0.20, "rank": 0.18}
CHOOSE_N = ([2, 3, 5, 8, 12, 20, 50, 100], [0.06, 0.12, 0.16, 0.16, 0.14, 0.14, 0.12, 0.10])
SCORE_N = ([3, 5, 7], [0.4, 0.45, 0.15])
RANK_N = ([3, 5, 8, 12, 20], [0.2, 0.3, 0.25, 0.15, 0.10])
CROSS_LINGUAL = 0.15
NONE_RATE = 0.25
PER_CALL = 10


def _pick(rng: random.Random, w: dict):
    return rng.choices(list(w), weights=list(w.values()))[0]


def cell(i: int, seed: int = 0) -> Cell:
    rng = random.Random(f"{seed}:{i}")
    kind = _pick(rng, KIND_W)
    if kind == "verify":
        n = 2
    elif kind == "score":
        n = rng.choices(*SCORE_N)[0]
    elif kind == "rank":
        n = rng.choices(*RANK_N)[0]
    else:
        n = rng.choices(*CHOOSE_N)[0]
    state_lang = _pick(rng, LANG_W)
    choice_lang = state_lang
    if rng.random() < CROSS_LINGUAL:
        others = [l for l in LANG_W if l != state_lang]
        choice_lang = "en" if state_lang != "en" and rng.random() < 0.7 else rng.choice(others)
    none = kind == "choose" and rng.random() < NONE_RATE
    return Cell(domain=rng.choice(list(DOMAINS)), kind=kind, n_choices=n, state_lang=state_lang,
                choice_lang=choice_lang, none=none, n_examples=PER_CALL, seed=i)


def example_id(rec: dict) -> str:
    key = rec["state"] + "\x1f" + rec["question"] + "\x1f" + "\x1f".join(rec["choices"])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_done(path: Path) -> tuple[set[int], set[str]]:
    cells, ids = set(), set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells.add(r["cell"]); ids.add(r["id"])
    return cells, ids


async def run(args) -> None:
    out = Path(args.out or f"data/teacher/generate-{args.model or Teacher().model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_log = out.with_suffix(".rejects.jsonl")
    done_cells, seen = load_done(out)
    todo = [i for i in range(args.offset, args.offset + args.calls) if i not in done_cells]
    print(f"{len(done_cells)} cells done, {len(seen)} examples on disk; {len(todo)} calls to go → {out}", flush=True)
    if not todo:
        return

    t0 = time.time()
    stats = {"calls": 0, "failed": 0, "examples": 0, "rejects": 0, "dupes": 0}
    fo, fr = out.open("a"), rej_log.open("a")

    async with Teacher(model=args.model, concurrency=args.concurrency) as t:
        async def one(i: int):
            c = cell(i, args.seed)
            try:
                budget = args.max_tokens * 2 if c.n_choices >= 50 else args.max_tokens   # 100-entry catalogues overflow 4k
                r = await t.chat(generate_prompt(c), max_tokens=budget, temperature=args.temperature,
                                 json_mode=True, thinking=args.thinking, tag=f"gen:{c.kind}:{c.n_choices}")
                recs, rejects = parse_generate(loads_lenient(r.text), c)
            except (TeacherError, ValueError) as e:
                stats["failed"] += 1
                fr.write(json.dumps({"cell": i, "error": str(e)[:300]}) + "\n"); fr.flush()
                return
            for why in rejects:
                fr.write(json.dumps({"cell": i, "reject": why}) + "\n")
            stats["rejects"] += len(rejects)
            for rec in recs:
                rec["id"] = example_id(rec)
                if rec["id"] in seen:
                    stats["dupes"] += 1
                    continue
                seen.add(rec["id"])
                rec.update(source="generate", model=t.model, cell=i)
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["examples"] += 1
            if not recs:   # keep the cell marked done even if everything was rejected
                fo.write(json.dumps({"id": f"empty-{i}", "cell": i, "source": "empty"}) + "\n")
            fo.flush(); fr.flush()
            stats["calls"] += 1
            if stats["calls"] % args.log_every == 0:
                el = time.time() - t0
                rate = stats["examples"] / el
                eta = (len(todo) - stats["calls"] - stats["failed"]) * PER_CALL / max(rate, 1e-9) / 3600
                print(f"  {stats['calls']}/{len(todo)} calls  {stats['examples']} ex  {rate*3600:.0f} ex/h  "
                      f"rej={stats['rejects']} dup={stats['dupes']} fail={stats['failed']}  "
                      f"tok={t.usage.completion_tokens}  eta={eta:.1f}h", flush=True)

        await asyncio.gather(*(one(i) for i in todo))
    fo.close(); fr.close()
    print(f"done: {stats}  {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    args.thinking = {"on": True, "off": False, "default": None}[args.thinking]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
