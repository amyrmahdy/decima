"""x86 CPU (ONNX fp32 / int8) vs the GX10 reference (PyTorch fp32 on CUDA) for Decima-small v1i.

    uv run python runs/x86/compare_x86.py        # after runs/x86/run_acc.sh → runs/x86/compare-v1i.json

Every metric comes from bench.score.score_suite, so it is computed exactly like docs/EVAL.md.
Set means are means of per-suite values (BTZSC: macro-F1, also over the clean 18).
Agreement = share of eval items where both systems pick the same option; max|Δp| = largest
absolute probability difference on any option of any item."""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.items import read
from bench.score import score_suite

SETS = ["decima", "kev", "laya", "jevtyped", "btzsc"]
BTZSC_DIRTY = ("rottentomatoes", "banking77", "massive", "agnews")


def load(p):
    return {r["id"]: r["probs"] for r in map(json.loads, open(p))} if Path(p).exists() else None


out = {}
for s in SETS:
    items = [i for i in read(f"runs/items/{s}.jsonl") if i["split"] == "eval" and "group" not in i]
    systems = {"gx10-gpu-fp32": load(f"runs/preds/{s}--v1i.jsonl"),
               "x86-cpu-fp32": load(f"runs/x86/preds/{s}--v1i-x86-fp32.jsonl"),
               "x86-cpu-int8": load(f"runs/x86/preds/{s}--v1i-x86-int8.jsonl")}
    systems = {k: v for k, v in systems.items() if v}
    by_suite = defaultdict(list)
    for i in items:
        by_suite[i["suite"]].append(i)
    metric = "macro_f1" if s == "btzsc" else "acc"
    res = {k: {su: score_suite(its, P) for su, its in by_suite.items()} for k, P in systems.items()}
    summ = {}
    for k, r in res.items():
        vals = [v[metric] for v in r.values() if v.get("n")]
        e = {"mean_" + metric: float(np.mean(vals)), "mean_ece": float(np.mean([v["ece"] for v in r.values() if v.get("n")])),
             "n_items": sum(v["n"] for v in r.values() if v.get("n")), "coverage": min(v["coverage"] for v in r.values())}
        if s == "btzsc":
            e["clean18_macro_f1"] = float(np.mean([v[metric] for su, v in r.items() if v.get("n") and not any(d in su for d in BTZSC_DIRTY)]))
        summ[k] = e
    agree = {}
    ref = systems.get("gx10-gpu-fp32")
    for a, b in (("x86-cpu-fp32", "gx10-gpu-fp32"), ("x86-cpu-int8", "gx10-gpu-fp32"), ("x86-cpu-int8", "x86-cpu-fp32")):
        A, B = systems.get(a), systems.get(b)
        if not (A and B):
            continue
        ids = [i["id"] for i in items if i["id"] in A and i["id"] in B]
        top = [int(np.argmax(A[k]) == np.argmax(B[k])) for k in ids]
        dp = max(float(np.max(np.abs(np.asarray(A[k]) - np.asarray(B[k])))) for k in ids)
        agree[f"{a} vs {b}"] = {"items": len(ids), "top1_agreement": float(np.mean(top)), "max_abs_dp": dp}
    out[s] = {"metric": metric, "summary": summ, "agreement": agree,
              "per_suite": {k: {su: {m: v.get(m) for m in ("n", "acc", "macro_f1", "ece")} for su, v in r.items()} for k, r in res.items()}}
    print(s, json.dumps(summ), json.dumps(agree), flush=True)

Path("runs/x86/compare-v1i.json").write_text(json.dumps(out, indent=2))
print("→ runs/x86/compare-v1i.json")
