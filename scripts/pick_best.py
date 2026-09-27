"""Pick the best Decima run from its scoreboard files (runs/<name>-<set>.json, written by train_v1.sh).

    uv run python scripts/pick_best.py v1a v1b          # prints a table, last line = best name

Score = mean over item sets (kev, laya, decima, jevtyped, btzsc) of the mean metric over that set's
suites — every set counts equally, so no single protocol dominates. btzsc is scored by macro-F1 over
its 18 clean datasets only (the true zero-shot check: rottentomatoes/banking77/massive/agnews overlap
our training data), so a run that overfits our label spaces cannot win. Suites missing for any
candidate are left out for all of them; a set missing for any candidate is left out for all.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SETS = ("kev", "laya", "decima", "jevtyped", "btzsc")
BTZSC_CONTAMINATED = {"btzsc/rottentomatoes", "btzsc/banking77", "btzsc/massive", "btzsc/agnews"}


def set_scores(name: str) -> dict[str, dict[str, float]]:
    out = {}
    for s in SETS:
        p = Path(f"runs/{name}-{s}.json")
        if p.exists():
            res = json.loads(p.read_text())["results"].get(name, {})
            if s == "btzsc":
                out[s] = {k: v["macro_f1"] for k, v in res.items() if v.get("n") and k not in BTZSC_CONTAMINATED}
            else:
                out[s] = {k: v["acc"] for k, v in res.items() if v.get("n")}
    return out


def main() -> None:
    names = sys.argv[1:]
    per = {n: set_scores(n) for n in names}
    rows, total = [], {n: [] for n in names}
    for s in SETS:
        if not names or any(s not in per[n] for n in names):
            continue
        common = set.intersection(*(set(per[n][s]) for n in names))
        if not common:
            continue
        means = {n: statistics.mean(per[n][s][k] for k in common) for n in names}
        rows.append((s, len(common), means))
        for n in names:
            total[n].append(means[n])
    print(f"{'set':10s} {'suites':>6s} " + " ".join(f"{n:>10s}" for n in names))
    for s, k, m in rows:
        print(f"{s:10s} {k:6d} " + " ".join(f"{m[n]:10.3f}" for n in names))
    score = {n: statistics.mean(v) if v else float("-inf") for n, v in total.items()}
    print(f"{'SCORE':10s} {'':6s} " + " ".join(f"{score[n]:10.3f}" for n in names))
    print(max(names, key=score.get))


if __name__ == "__main__":
    main()
