"""The one headline calibration metric, computed once for every document and figure.

    uv run python scripts/audit/calibration.py          # → runs/audit/calibration.json + a table

Definition (the only "calibration error as shipped" number any release document may quote):
  ECE = 15-bin expected calibration error on top-1 confidence, per suite, from `bench/score.py`
  (the `ece` field of runs/v1i-{set}.json), on the probabilities each system ships
  (Decima: its shipped temperature; Kev: raw, T = 1, as Kev publishes; Laya: its shipped temperatures).
  Pairwise: for each competitor, the mean of per-suite ECE over the suites BOTH systems were run on,
  across all five item sets (kev, laya, decima, jevtyped, btzsc; no competitor was run on BTZSC, so it
  contributes no pairs). FarsTail is excluded (its calibration items overlap its test items, EVAL §3).

Also reported, never as a headline: the same pairing on `ece_cal` (one temperature per suite fitted on
that suite's calib items; suites without calib items — JevBench, BTZSC — drop out), and plain set means.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

SETS = ("kev", "laya", "decima", "jevtyped", "btzsc")
DECIMA = os.environ.get("DECIMA", "v1i")      # the released model; DECIMA=v1k for Decima 1.1
COMPETITORS = ("kev-0.5b", "kev-0.8b", "laya", "laya-multilingual")


def ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def pairwise(res: dict, comp: str, metric: str) -> dict:
    a, b, suites = [], [], []
    for r in res.values():
        if comp not in r:
            continue
        for suite, v in r[DECIMA].items():
            w = r[comp].get(suite)
            if not w or "farstail" in suite or not v.get("n") or not w.get("n"):
                continue
            if ok(v.get(metric)) and ok(w.get(metric)):
                a.append(v[metric]); b.append(w[metric]); suites.append(suite)
    return {"suites": len(a), "decima": sum(a) / len(a), comp: sum(b) / len(b), "suite_ids": suites}


def main() -> None:
    res = {s: json.load(open(f"runs/{DECIMA}-{s}.json"))["results"] for s in SETS}
    out = {"definition": __doc__.split("Definition")[1].split("Also reported")[0].strip(),
           "headline_as_shipped": {}, "refitted_per_suite_T": {}, "set_means_as_shipped": {}}
    for c in COMPETITORS:
        out["headline_as_shipped"][c] = pairwise(res, c, "ece")
        out["refitted_per_suite_T"][c] = pairwise(res, c, "ece_cal")
    for s, r in res.items():
        out["set_means_as_shipped"][s] = {}
        for sysname, suites in r.items():
            vals = [v["ece"] for v in suites.values() if v.get("n") and ok(v.get("ece"))]
            if vals:
                out["set_means_as_shipped"][s][sysname] = sum(vals) / len(vals)
    Path("runs/audit").mkdir(parents=True, exist_ok=True)
    Path(f"runs/audit/calibration{'' if DECIMA == 'v1i' else '-' + DECIMA}.json").write_text(json.dumps(out, indent=2))
    print("pairwise ECE (15-bin, per-suite mean, FarsTail excluded)   as shipped            refitted per suite")
    for c in COMPETITORS:
        h, f = out["headline_as_shipped"][c], out["refitted_per_suite_T"][c]
        print(f"  vs {c:18s} {h['suites']:3d} suites  {h['decima']:.3f} vs {h[c]:.3f}    "
              f"{f['suites']:3d} suites  {f['decima']:.3f} vs {f[c]:.3f}")
    print("→ runs/audit/calibration.json")


if __name__ == "__main__":
    main()
