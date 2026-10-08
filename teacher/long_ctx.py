"""Long-context rows: an existing decision, its state buried inside 2k–8k tokens of unrelated material (added 2026-10-08).

    uv run python -m teacher.long_ctx --n 40000 --out data/long/long-train.jsonl
    uv run python -m teacher.long_ctx --n 1500 --split test --out data/long/long-test.jsonl

Laya and Kev read 8k–65k-token documents; Decima read 512 (base) or 2,048 (agent) and refused anything longer. mmBERT, the
backbone, supports 8,192 positions natively. To teach the model to find the decisive part of a long input, take a row
whose label is already exact or agreed (agent gates, knowledge-graph judgments, typed decisions) and surround its state
with other states from the same pool as filler — shuffled so the decisive part sits anywhere. The question and label do
not change; filler states are other rows' inputs, so none of them answers this row's question. Rows whose question
refers to "the document" or a named field are wrapped as one JSON object with the original state under its own key, so
the reference stays unambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SOURCES = {   # pool → (files, weight). Test uses the *-test files of the same generators.
    "agent": (["data/agent/proc-{split}.jsonl", "data/agent/tools-{split}.jsonl", "data/agent/proc-shell-{split}.jsonl"], 0.35),
    "kg": (["data/kg/proc-{split}.jsonl", "data/kg/llm-docs-{split}.jsonl", "data/kg/llm-threads-{split}.jsonl"], 0.4),
    "typed": (["data/mix-base2/s2-train.jsonl"], 0.25),           # train only (no test split of the same form)
}
FILLER_KEYS = ["notes", "context", "history", "attachments", "related", "appendix", "log", "background"]


def sid(*p):
    return hashlib.sha1("\x1f".join(map(str, p)).encode()).hexdigest()[:16]


def load(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += [json.loads(l) for l in open(p) if '"empty"' not in l]
    return rows


def wrap(rng, r, filler, target_chars):
    """Bury r's state among filler states. JSON states keep their keys; the filler goes under other keys."""
    try:
        core = json.loads(r["state"])
    except (ValueError, TypeError):
        core = None
    parts, size = [], len(r["state"])
    while size < target_chars:
        f = rng.choice(filler)["state"]
        parts.append(f); size += len(f)
    if isinstance(core, dict):
        out, keys = dict(core), rng.sample(FILLER_KEYS, k=min(len(FILLER_KEYS), 3))
        for k in keys:
            out[k] = []
        for i, f in enumerate(parts):
            out[keys[i % len(keys)]].append(f)
        items = list(out.items())
        rng.shuffle(items)                      # the decisive keys can come anywhere
        return json.dumps(dict(items), ensure_ascii=False)
    pos = rng.randint(0, len(parts))
    return "\n\n---\n\n".join(parts[:pos] + [r["state"]] + parts[pos:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=8000)     # ≈ 2k tokens
    ap.add_argument("--max-chars", type=int, default=30000)    # ≈ 7.5k tokens
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(f"long:{a.split}:{a.seed}")
    pools = {}
    for name, (files, w) in SOURCES.items():
        rows = load([f.format(split=a.split) for f in files])
        if a.split == "test" and name == "typed":
            continue
        if rows:
            pools[name] = (rows, w)
    from teacher.decontam_agent import Decontam
    dc = Decontam()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    names, weights = list(pools), [pools[n][1] for n in pools]
    n = 0
    with out.open("w") as fo:
        while n < a.n:
            name = rng.choices(names, weights)[0]
            rows = pools[name][0]
            r = rng.choice(rows)
            if dc.hit(r):
                continue
            target = rng.randint(a.min_chars, a.max_chars)
            st = wrap(rng, r, rows, target)
            row = {k: r[k] for k in ("kind", "question", "choices", "probs", "gold", "state_lang", "choice_lang")}
            row.update({"state": st, "id": sid("long", a.split, n, r.get("id")), "source": "long-ctx", "task": f"long-{name}"})
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows → {out}")


if __name__ == "__main__":
    main()
