"""95 % cluster-bootstrap confidence intervals for the small benchmarks (JevBench public, typed-decisions).

    uv run python scripts/audit/bootstrap.py [system ...]      # default: v1i kev-0.5b kev-0.8b laya laya-multilingual v0

Resamples *cases* (items sharing a state — typed-decisions asks 5 questions per case) with replacement,
1,000 draws, seed 0; reports accuracy and the 2.5 / 97.5 percentiles. Also paired differences vs the
first system on the same resamples (diff = other − first, so v1i − V0 is the negated V0 entry).
→ runs/audit/bootstrap.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ITEMS = "runs/items/jevtyped.jsonl"
SUITES = ("jevbench/public", "typed/test")


def main() -> None:
    systems = sys.argv[1:] or ["v1i", "kev-0.5b", "kev-0.8b", "laya", "laya-multilingual", "v0"]
    items = [json.loads(l) for l in open(ITEMS)]
    items = [i for i in items if i["split"] == "eval" and "group" not in i]
    preds = {n: {r["id"]: r["probs"] for r in map(json.loads, open(f"runs/preds/jevtyped--{n}.jsonl"))} for n in systems}
    top = lambda p: max(range(len(p)), key=p.__getitem__)
    out = {}
    for suite in SUITES:
        cases = defaultdict(list)
        for i in items:
            if i["suite"] == suite:
                cases[i["state"]].append(i)
        C = list(cases.values())
        ok = {n: {i["id"]: top(preds[n][i["id"]]) == i["gold"] for c in C for i in c} for n in systems}
        rng = random.Random(0)
        draws = [[rng.randrange(len(C)) for _ in C] for _ in range(1000)]
        res = {}
        for n in systems:
            acc = lambda idx: sum(ok[n][i["id"]] for j in idx for i in C[j]) / sum(len(C[j]) for j in idx)
            point = acc(range(len(C)))
            bs = sorted(acc(d) for d in draws)
            diff = sorted(acc(d) - sum(ok[systems[0]][i["id"]] for j in d for i in C[j]) / sum(len(C[j]) for j in d) for d in draws)
            res[n] = {"acc": point, "ci95": [bs[25], bs[974]], f"diff_vs_{systems[0]}_ci95": [diff[25], diff[974]]}
            print(f"{suite:16s} {n:18s} {point:.3f} [{bs[25]:.3f}, {bs[974]:.3f}]  Δ vs {systems[0]} [{diff[25]:+.3f}, {diff[974]:+.3f}]")
        out[suite] = {"cases": len(C), "items": sum(map(len, C)), "systems": res}
    Path("runs/audit").mkdir(parents=True, exist_ok=True)
    Path("runs/audit/bootstrap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
