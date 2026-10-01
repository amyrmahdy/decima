"""Re-label s2 rows with the main teacher (Gemma-4-26B-A4B), task by task.

    uv run python -m teacher.relabel_s2 --src data/teacher/s2q-rande-strong-local.jsonl \
        --out data/teacher/s2q-relabelled.jsonl --streams 4

Why: a second teacher (Qwen3-Coder-Next, 0.874 on jabr v2 vs Gemma's 0.942) writes extra English tasks and
states to double throughput, but its labels are weaker. Writing states is the expensive part; labelling a
task's states is cheap (a few numbers per state). So the main teacher labels every row again, and its
distribution becomes the target. The writer's label is kept as `writer_probs`/`writer_p_true` and
`agree` records whether both teachers picked the same answer (useful for analysis and weighting).
Resumable: groups already in --out are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from teacher.client import Teacher, TeacherError
from teacher.generate_s1 import Reject, _norm_probs
from teacher.prompts import loads_lenient

SYSTEM = ("You label data for a decision model. For each input, give the probability a careful senior practitioner "
          "would assign, with honest uncertainty. Output JSON only.")


def prompt(q: dict, states: list[str]) -> list[dict]:
    t = q["type"]
    shape = {"choice": '{"key": probability, ...} over every criteria key (sum 1)',
             "noul": "the probability that the statement is true for this input",
             "score": "[one probability per level, level 0 first] (sum 1; mass on adjacent levels)"}[t]
    question = {"type": t, "instructions": q["instructions"], "criteria": q["criteria"]}
    lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(states))
    body = (f"QUESTION ({t}):\n{json.dumps(question, ensure_ascii=False)}\n\nINPUTS:\n{lines}\n\n"
            f'Return {{"labels": [ ... one entry per input, in order ... ]}} where each entry is {shape}.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def parse(q: dict, v):
    if q["type"] == "choice":
        if not isinstance(v, dict) or set(v) != set(q["criteria"]):
            raise Reject("keys")
        return {"probs": dict(zip(q["criteria"], _norm_probs([float(v[k]) for k in q["criteria"]])))}
    if q["type"] == "score":
        if not isinstance(v, list) or len(v) != len(q["criteria"]):
            raise Reject("levels")
        return {"probs": _norm_probs([float(x) for x in v])}
    p = float(v["p_true"] if isinstance(v, dict) else v)
    if not 0 <= p <= 1:
        raise Reject("p")
    return {"p_true": round(min(max(p, 1e-3), 1 - 1e-3), 4)}


def top(q: dict, lab: dict):
    if q["type"] == "noul":
        return lab["p_true"] >= 0.5
    p = lab["probs"]
    return max(p, key=p.get) if isinstance(p, dict) else max(range(len(p)), key=p.__getitem__)


async def run(a) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in open(a.src):
        r = json.loads(line)
        if r.get("source") == "s2gen":
            groups[r["group"]].append(r)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["group"] for l in out.open()}
    todo = [g for g in groups if g not in done]
    print(f"{len(groups)} tasks, {len(done)} done, {len(todo)} to relabel", flush=True)
    fo = out.open("a")
    st = {"rows": 0, "agree": 0, "failed": 0}
    async with Teacher(model=a.model, concurrency=a.streams) as t:
        async def one(g: str):
            rows = groups[g]
            q = rows[0]
            try:
                r = await t.chat(prompt(q, [x["state"] for x in rows]), max_tokens=2500, temperature=0.0,
                                 thinking=False, json_mode=True, tag="relabel")
                labs = loads_lenient(r.text).get("labels")
                if not isinstance(labs, list) or len(labs) != len(rows):
                    raise Reject("label count")
            except (TeacherError, Reject, ValueError, AttributeError) as e:
                st["failed"] += 1
                return
            for row, v in zip(rows, labs):
                try:
                    lab = parse(q, v)
                except (Reject, TypeError, ValueError, KeyError):
                    continue
                writer = {f"writer_{k}": row[k] for k in ("probs", "p_true") if k in row}
                agree = top(q, lab) == top(q, row)
                new = {k: v for k, v in row.items() if k not in ("probs", "p_true")}
                fo.write(json.dumps({**new, **lab, **writer, "agree": agree, "labeller": t.model}, ensure_ascii=False) + "\n")
                st["rows"] += 1; st["agree"] += agree
            fo.flush()
        await asyncio.gather(*(one(g) for g in todo))
    fo.close()
    print(f"done: {st}  agreement {st['agree'] / max(st['rows'], 1):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="labelling teacher (default: TEACHER_MODEL)")
    ap.add_argument("--streams", type=int, default=4)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
