"""Step 0 of PLAN-v2: how good is the teacher itself on jabr/classifier-benchmark? (the student's ceiling)

    uv run python -m teacher.eval_teacher_jabr --cases <classifier-benchmark>/cases --out runs/jabr-teacher.json

One call per case, the question in TypeSafe's schema, a plain JSON answer. Nothing from the benchmark is
written anywhere except this score file; it is a measurement, never training data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import tomllib
from pathlib import Path

from teacher.client import Teacher, TeacherError
from teacher.prompts import loads_lenient

SYSTEM = "You are a careful decision engine. Read the input and answer the typed question with calibrated probabilities. Output JSON only."


def prompt(task: dict, state: str) -> list[dict]:
    q = task["question"]
    t = task["type"]
    if t == "choice":
        ask = 'Return {"probs": {<every option key>: probability}} (sum 1).'
    elif t == "noul":
        ask = 'The question is a statement to judge for this input. Return {"p_true": probability that it is true}.'
    else:
        ask = 'The criteria are ordered levels, level 0 first. Return {"probs": [one probability per level]} (sum 1).'
    body = f"INPUT:\n{state}\n\nQUESTION ({t}):\n{json.dumps(q, ensure_ascii=False)}\n\n{ask}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def predicted(task: dict, ans: dict):
    t = task["type"]
    if t == "choice":
        p = ans["probs"]
        return max(p, key=lambda k: float(p[k]))
    if t == "noul":
        return float(ans["p_true"]) >= 0.5
    p = [float(v) for v in ans["probs"]]
    return max(range(len(p)), key=p.__getitem__)


async def run(a) -> None:
    tasks = []
    for suite in ("v1", "v2"):
        for t in tomllib.load(open(Path(a.cases) / f"{suite}.toml", "rb"))["task"]:
            tasks.append((suite, t))
    jobs = [(s, t, c) for s, t in tasks for c in t["cases"]]
    out = []
    async with Teacher(model=a.model, concurrency=a.streams) as tc:
        async def one(s, t, c):
            try:
                r = await tc.chat(prompt(t, c["state"]), max_tokens=400, temperature=0.0, thinking=a.thinking, json_mode=True, tag="jabr")
                pred = predicted(t, loads_lenient(r.text))
            except (TeacherError, ValueError, KeyError, TypeError, AttributeError):
                pred = None
            out.append({"suite": s, "task": t["id"], "type": t["type"], "correct": pred == c["expected"]})
        await asyncio.gather(*(one(*j) for j in jobs))
    res = {}
    for suite in ("v1", "v2"):
        by = {}
        for r in out:
            if r["suite"] == suite:
                by.setdefault(r["task"], []).append(r["correct"])
        types = {}
        for r in out:
            if r["suite"] == suite:
                types.setdefault(r["type"], {}).setdefault(r["task"], []).append(r["correct"])
        res[suite] = {"macro_acc": st.mean(st.mean(v) for v in by.values()),
                      "micro_acc": st.mean(r["correct"] for r in out if r["suite"] == suite),
                      "by_type": {k: st.mean(st.mean(v) for v in d.values()) for k, d in types.items()}}
        print(suite, json.dumps(res[suite]))
    Path(a.out).write_text(json.dumps({"model": a.model or "teacher", "thinking": a.thinking, "results": res}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
