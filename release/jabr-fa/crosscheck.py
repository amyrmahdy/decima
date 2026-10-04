"""Second-model check of the Persian edition: which translations may have changed the answer?

    uv run python release/jabr-fa/crosscheck.py --cases <cb>/cases/v2.toml --dir release/jabr-fa

A different model family from the translator (the local teacher, free) answers every case twice, on the English original and on the
Persian translation (question and state both Persian). A case it answers correctly in English but not in
Persian is a candidate translation error and goes to the native-speaker review, together with the cases whose
numbers changed and a seeded random sample. Writes crosscheck.jsonl (resumable) and review.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import tomllib
from pathlib import Path

from teacher.client import Teacher, TeacherError


def ask(q: dict, ttype: str, state: str) -> str:
    if ttype == "choice":
        opts = "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())
        tail = f"Options:\n{opts}\nAnswer with the option key only."
    elif ttype == "noul":
        c = q.get("criteria") or {}
        tail = "".join(f"\n{k}: {v}" for k, v in c.items()) + "\nAnswer true or false only."
    else:
        lv = "\n".join(f"{i}: {v}" for i, v in enumerate(q["criteria"]))
        tail = f"Levels:\n{lv}\nAnswer with the level number only."
    return f"Text:\n{state}\n\nQuestion: {q['instructions']}\n{tail}"


def parse(ans: str, ttype: str, q: dict):
    a = ans.strip().strip(".`'\"").lower()
    if ttype == "noul":
        return True if a.startswith("true") else False if a.startswith("false") else None
    if ttype == "score":
        return int(a.split()[0]) if a[:1].isdigit() else None
    return a.split()[0] if a and a.split()[0] in q["criteria"] else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--dir", default="release/jabr-fa")
    ap.add_argument("--model", default="rande-fast-local")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--sample", type=int, default=40)
    a = ap.parse_args()
    d = Path(a.dir)
    en = {t["id"]: t for t in tomllib.loads(Path(a.cases).read_text())["task"]}
    fa = {t["id"][: -len("_fa")]: t for t in tomllib.loads((d / "v2-fa.toml").read_text())["task"]}
    flags = {json.loads(l)["id"]: json.loads(l)["fa"].get("flags", []) for l in (d / "translations.jsonl").open()}
    out = d / "crosscheck.jsonl"
    done = {(r["id"], r["i"], r["lang"]): r for r in map(json.loads, out.open())} if out.exists() else {}
    done = {k: r for k, r in done.items() if (k[0], k[1], "en") in done and (k[0], k[1], "fa") in done}  # same checker for both
    jobs = [(tid, i, lang) for tid in fa for i in range(len(en[tid]["cases"])) for lang in ("en", "fa")
            if (tid, i, lang) not in done]
    async with Teacher(model=a.model, concurrency=a.streams) as client:
        async def one(tid, i, lang):
            t = en[tid] if lang == "en" else fa[tid]
            try:
                r = await client.chat([{"role": "user", "content": ask(t["question"], t["type"], t["cases"][i]["state"])}],
                                      max_tokens=20, temperature=0, thinking=False, tag="fa-check")
                pred = parse(r.text, t["type"], t["question"])
            except TeacherError:
                pred = None
            row = {"id": tid, "i": i, "lang": lang, "pred": pred, "ok": pred == t["cases"][i]["expected"], "model": a.model}
            done[(tid, i, lang)] = row
            with out.open("a") as fo:
                fo.write(json.dumps(row) + "\n")
        await asyncio.gather(*(one(*j) for j in jobs))
    n = len([k for k in done if k[2] == "en"])
    acc = {lang: sum(r["ok"] for k, r in done.items() if k[2] == lang) / n for lang in ("en", "fa")}
    suspect = sorted({(tid, i) for (tid, i, lang) in done if lang == "en" and done[(tid, i, "en")]["ok"]
                      and not done[(tid, i, "fa")]["ok"]})
    numflag = sorted({(tid, i) for tid in fa for i in flags.get(tid, [])} - set(suspect))
    rng = random.Random(0)
    pool = sorted({(tid, i) for tid in fa for i in range(len(en[tid]["cases"]))} - set(suspect) - set(numflag))
    sample = sorted(rng.sample(pool, min(a.sample, len(pool))))
    items = []
    for why, keys in (("answer changed", suspect), ("numbers changed", numflag), ("random", sample)):
        for tid, i in keys:
            items.append({"id": tid, "i": i, "why": why, "type": en[tid]["type"], "expected": en[tid]["cases"][i]["expected"],
                          "en": en[tid]["cases"][i]["state"], "fa": fa[tid]["cases"][i]["state"],
                          "question_en": en[tid]["question"]["instructions"], "question_fa": fa[tid]["question"]["instructions"]})
    (d / "review.json").write_text(json.dumps({"model": a.model, "acc": acc, "items": items}, ensure_ascii=False, indent=1))
    print(f"{a.model}: en {acc['en']:.3f} fa {acc['fa']:.3f}; review {len(suspect)} changed + {len(numflag)} numbers + {len(sample)} random")


if __name__ == "__main__":
    asyncio.run(main())
