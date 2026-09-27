"""Generator 2: the teacher labels EXISTING states under NEW questions.

    uv run python -m teacher.label --per-source 200                # pilot
    uv run python -m teacher.label --per-source 6000 --concurrency 16

Why a second generator: invented states come with invented choice sets that fit them
too neatly. Real utterances (bench TRAIN splits, never test/validation) and our own
generated states re-asked under a different question give the student decisions where
state and choices were not written together.

Stage 1 — per (source, choice language), the teacher invents ~8 question+choice sets
(cached in data/teacher/label-questions.json); one extra question per source is a random
10–20-entry subset of the source's real catalogue plus "none of the above".
Stage 2 — states are labeled in batches of 20 under one question per call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from decima.normalize import normalize
from teacher.client import Teacher, TeacherError
from teacher.generate import example_id
from teacher.prompts import DOMAINS, NONE_TEXT, label_prompt, loads_lenient, parse_label, questions_prompt
from teacher.sources import TRAIN_SPLITS, generated_states, train_states

BATCH = 20
CROSS_LINGUAL = 0.15
Q_PER_SOURCE = 8


def _choice_lang(state_lang: str, rng: random.Random) -> str:
    if rng.random() >= CROSS_LINGUAL:
        return state_lang
    return "en" if state_lang != "en" and rng.random() < 0.7 else rng.choice([l for l in ("en", "fa", "ar", "ru") if l != state_lang])


def _valid_question(q: dict) -> bool:
    k, ch = q.get("kind"), q.get("choices")
    if k not in ("choose", "score", "verify", "rank") or not isinstance(ch, list) or not q.get("question"):
        return False
    if any(not isinstance(c, str) or not c.strip() for c in ch) or len(set(ch)) != len(ch):
        return False
    return (len(ch) == 2) if k == "verify" else (2 <= len(ch) <= 20)


async def invent_questions(t: Teacher, cache: Path, sources: dict, seed: int, thinking) -> dict:
    """{source_key: [ {kind, question, choices, choice_lang, none_idx} ]}, cached."""
    have = json.loads(cache.read_text()) if cache.exists() else {}
    rng = random.Random(seed)

    async def one(key, info):
        states, catalogue, lang, desc = info
        cl = _choice_lang(lang, rng)
        try:
            r = await t.chat(questions_prompt(desc, states, lang, cl, Q_PER_SOURCE, catalogue), max_tokens=3000,
                             temperature=0.8, json_mode=True, thinking=thinking, tag="label:q")
            qs = [q for q in loads_lenient(r.text).get("questions", []) if _valid_question(q)]
        except (TeacherError, ValueError, AttributeError) as e:
            print(f"  questions for {key}: FAILED {e}")
            qs = []
        out = []
        for q in qs:
            ch = [normalize(c, cl) for c in q["choices"]]
            none_idx = len(ch) - 1 if (q["kind"] == "choose" and ch[-1].lower() == NONE_TEXT[cl].lower()) else None
            out.append({"kind": q["kind"], "question": normalize(q["question"], cl), "choices": ch, "choice_lang": cl, "none_idx": none_idx})
        if len(catalogue) >= 3:   # a slice of the real label space, always with an out-of-scope exit
            for _ in range(2):
                k = rng.randint(min(10, len(catalogue)), min(20, len(catalogue)))
                sub = rng.sample(catalogue, k) + [NONE_TEXT["en"]]
                out.append({"kind": "choose", "question": "Which of these best describes what the user wants?",
                            "choices": sub, "choice_lang": "en", "none_idx": len(sub) - 1})
        have[key] = out
        print(f"  {key}: {len(out)} questions ({cl})", flush=True)

    await asyncio.gather(*(one(k, v) for k, v in sources.items() if k not in have))
    cache.write_text(json.dumps(have, ensure_ascii=False, indent=1))
    return have


async def run(args) -> None:
    out = Path(args.out or f"data/teacher/label-{args.model or Teacher().model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_log = out.with_suffix(".rejects.jsonl")
    seen = set()
    done_batches = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            seen.add(r["id"]); done_batches.add(r.get("batch"))
    rng = random.Random(args.seed)

    # ---- sources: bench train splits + our generated states
    sources: dict[str, tuple[list[str], list[str], str, str]] = {}
    for key in args.sources:
        st, names, lang, desc = train_states(key, args.per_source, args.seed)
        sources[key] = ([normalize(s, lang) for s in st], names, lang, desc)
    for (dom, lang), st in generated_states(args.generated, args.per_source // 4, args.seed).items():
        if len(st) >= BATCH:
            sources[f"gen:{dom}/{lang}"] = (st, [], lang, DOMAINS[dom])
    print(f"{len(sources)} sources, {sum(len(v[0]) for v in sources.values())} states", flush=True)

    async with Teacher(model=args.model, concurrency=args.concurrency) as t:
        questions = await invent_questions(t, Path("data/teacher/label-questions.json"), sources, args.seed, args.thinking)

        # ---- batches: each state gets `--asks` different questions
        batches = []
        for key, (states, _, lang, _) in sources.items():
            qs = questions.get(key, [])
            if not qs:
                continue
            for a in range(args.asks):
                order = states[:]
                rng.shuffle(order)
                for b in range(0, len(order) - BATCH + 1, BATCH):
                    q = rng.choice(qs)
                    bid = f"{key}|{a}|{b}" if args.seed == 0 else f"{key}|s{args.seed}|{a}|{b}"   # seed 0 keeps the first run's ids
                    if bid in done_batches:
                        continue
                    batches.append((bid, key, lang, q, order[b : b + BATCH]))
        print(f"{len(batches)} batches of {BATCH} to label", flush=True)

        t0 = time.time()
        stats = {"calls": 0, "failed": 0, "examples": 0, "rejects": 0, "dupes": 0}
        fo, fr = out.open("a"), rej_log.open("a")

        async def one(bid, key, lang, q, states):
            items = [{"state": s, "question": q["question"], "choices": q["choices"]} for s in states]
            try:
                r = await t.chat(label_prompt(items, q["kind"], q["choice_lang"]), max_tokens=args.max_tokens,
                                 temperature=0.3, json_mode=True, thinking=args.thinking, tag=f"label:{q['kind']}")
                recs, rejects = parse_label(loads_lenient(r.text), items, q["kind"])
            except (TeacherError, ValueError) as e:
                stats["failed"] += 1
                fr.write(json.dumps({"batch": bid, "error": str(e)[:300]}) + "\n"); fr.flush()
                return
            stats["rejects"] += len(rejects)
            for why in rejects:
                fr.write(json.dumps({"batch": bid, "reject": why}) + "\n")
            for rec in recs:
                rec.update(kind=q["kind"], domain=key.split("/")[0].replace("gen:", ""), state_lang=lang,
                           choice_lang=q["choice_lang"], none_idx=q["none_idx"], source="label", model=t.model, batch=bid)
                rec["id"] = example_id(rec)
                if rec["id"] in seen:
                    stats["dupes"] += 1
                    continue
                seen.add(rec["id"])
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["examples"] += 1
            if not recs:
                fo.write(json.dumps({"id": f"empty-{bid}", "batch": bid, "source": "empty"}) + "\n")
            fo.flush(); fr.flush()
            stats["calls"] += 1
            if stats["calls"] % args.log_every == 0:
                el = time.time() - t0
                rate = stats["examples"] / el
                eta = (len(batches) - stats["calls"] - stats["failed"]) * BATCH / max(rate, 1e-9) / 3600
                print(f"  {stats['calls']}/{len(batches)} calls  {stats['examples']} ex  {rate*3600:.0f} ex/h  "
                      f"rej={stats['rejects']} dup={stats['dupes']} fail={stats['failed']}  eta={eta:.1f}h", flush=True)

        await asyncio.gather(*(one(*b) for b in batches))
        fo.close(); fr.close()
        print(f"done: {stats}  {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=list(TRAIN_SPLITS))
    ap.add_argument("--generated", default=None, help="generate-*.jsonl whose states get re-asked")
    ap.add_argument("--per-source", type=int, default=200, help="states sampled per source")
    ap.add_argument("--asks", type=int, default=1, help="how many different questions each state is labeled under")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    args.thinking = {"on": True, "off": False, "default": None}[args.thinking]
    if args.generated is None:
        cands = sorted(Path("data/teacher").glob("generate-*.jsonl")) if Path("data/teacher").exists() else []
        args.generated = str(cands[0]) if cands else "/nonexistent"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
