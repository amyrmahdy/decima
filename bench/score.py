"""Score any number of systems on the same scoreboard items, side by side.

    uv run python -m bench.score --items runs/items/phase0.jsonl \
        --preds decima-v0=runs/preds/phase0-decima-v0.jsonl laya=runs/preds/phase0-laya.jsonl \
        --out runs/phase0-scoreboard.json

Per suite and system: accuracy · macro-F1 (over choice text) · Brier · NLL · ECE as shipped ·
ECE after temperature fitted on that suite's calib items (never on eval items; blank when the
suite has no non-test split) · flip rate (answer changes when the same choices are reordered).
Choice sets may differ per item (Laya's random distractors), so everything is computed per item.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

from bench.items import read
from decima.calibration import fit_temperature

BINS = 15


def _probs(p: list[float], kind: str) -> np.ndarray:
    """Renormalised; for rank (independent sigmoids) this only serves the argmax-style metrics."""
    a = np.asarray(p, dtype=np.float64)
    return a / max(a.sum(), 1e-12)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def macro_f1(pred: list[str], gold: list[str]) -> float:
    f1s = []
    for c in set(gold):
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _temper(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def score_suite(items: list[dict], preds: dict[str, list[float]]) -> dict:
    ev = [i for i in items if i["split"] == "eval" and "group" not in i and i["id"] in preds]
    ca = [i for i in items if i["split"] == "calib" and i["id"] in preds]
    fl = [i for i in items if "group" in i and i["id"] in preds and i["group"] in preds]
    n_expected = sum(i["split"] == "eval" and "group" not in i for i in items)
    if not ev:
        return {"n": 0, "coverage": 0.0}

    P = [_probs(preds[i["id"]], i["kind"]) for i in ev]
    gold = np.array([i["gold"] for i in ev])
    pred = np.array([int(p.argmax()) for p in P])
    conf = np.array([float(p.max()) for p in P])
    correct = (pred == gold).astype(float)
    brier = float(np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(P, gold)]))
    nll = float(np.mean([-math.log(max(p[g], 1e-12)) for p, g in zip(P, gold)]))

    T, ece_cal = float("nan"), float("nan")
    ranked = any(i["kind"] == "rank" for i in ev)
    if ca and not ranked:
        # fit one T on calib logits (log-probs of the shipped system; ragged → pad with -inf)
        n = max(len(preds[i["id"]]) for i in ca)
        Z = np.full((len(ca), n), -1e9)
        for r, i in enumerate(ca):
            p = np.clip(np.asarray(preds[i["id"]], dtype=np.float64), 1e-12, 1)
            Z[r, : len(p)] = np.log(p)
        T = fit_temperature(Z, np.array([i["gold"] for i in ca]))
        Pc = [_temper(p, T) for p in P]
        ece_cal = ece(np.array([p.max() for p in Pc]), correct)

    flip = float("nan")
    if fl:
        orig_pred = {i["id"]: int(np.argmax(preds[i["id"]])) for i in ev}
        flips = [i["perm"][int(np.argmax(preds[i["id"]]))] != orig_pred[i["group"]] for i in fl if i["group"] in orig_pred]
        flip = float(np.mean(flips)) if flips else float("nan")

    return {
        "n": len(ev), "coverage": len(ev) / max(n_expected, 1), "choices": int(np.median([len(p) for p in P])),
        "kind": ev[0]["kind"], "lang": ev[0]["lang"],
        "acc": float(correct.mean()),
        "macro_f1": macro_f1([i["choices"][k] for i, k in zip(ev, pred)], [i["choices"][g] for i, g in zip(ev, gold)]),
        "brier": brier, "nll": nll, "ece": ece(conf, correct), "ece_cal": ece_cal, "T": T, "flip": flip,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--metric", default="acc", help="metric shown in the side-by-side table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = read(args.items)
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_suite[i["suite"]].append(i)

    results: dict[str, dict[str, dict]] = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        preds = {r["id"]: r["probs"] for r in read(path)}
        results[name] = {s: score_suite(its, preds) for s, its in by_suite.items()}

    names = list(results)
    for name in names:
        rows = [[s, r.get("n"), r.get("choices"), r.get("acc"), r.get("macro_f1"), r.get("brier"), r.get("ece"), r.get("ece_cal"), r.get("flip")]
                for s, r in results[name].items() if r.get("n")]
        print(f"\n== {name}")
        print(tabulate(rows, headers=["suite", "n", "choices", "acc", "macroF1", "brier", "ece", "ece_cal", "flip"], floatfmt=".3f"))

    m = args.metric
    print(f"\n== side by side: {m}")
    table = []
    for s in by_suite:
        table.append([s] + [results[n][s].get(m, float("nan")) if results[n][s].get("n") else None for n in names])
    fams = sorted({s.split("/")[0] for s in by_suite})
    for f in fams:
        row = [f"{f}/* mean"]
        for n in names:
            v = [results[n][s][m] for s in by_suite if s.startswith(f + "/") and results[n][s].get("n")]
            ok = [x for x in v if not math.isnan(x)]   # e.g. flip is undefined for ordered-level suites
            row.append(float(np.mean(ok)) if ok and len(v) == sum(s.startswith(f + "/") for s in by_suite) else None)
        table.append(row)
    print(tabulate(table, headers=["suite"] + names, floatfmt=".3f", missingval="—"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"items": args.items, "preds": dict(s.split("=", 1) for s in args.preds), "results": results}, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
