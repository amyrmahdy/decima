"""Decima Bench runner.

    uv run python -m bench.run                      # everything, 1000 examples per task
    uv run python -m bench.run --tasks massive/fa banking77/en --limit 200
    uv run python -m bench.run --system decima --checkpoint checkpoints/v0/best --no-latency

Reports, per task: accuracy · Brier · NLL · ECE (raw and after temperature scaling on a
held-out calibration slice) · CPU latency p50/p95 per decision with choices cached.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tabulate import tabulate

from decima.baseline import DEFAULT_MODEL, ZeroShotBiEncoder
from decima.calibration import fit_temperature, softmax
from decima.model import Decima
from bench.datasets import REGISTRY
from bench.metrics import summarize


def latency(model, task, n: int = 100) -> tuple[float, float]:
    """Per-decision wall time, choices already embedded (the steady-state case)."""
    q = task.question
    model.logits(task.states[:1], q)   # warm the choice cache
    times = []
    for s in task.states[:n]:
        t0 = time.perf_counter()
        model.logits([s], q)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def run_task(model, key: str, limit: int, calib: int, do_latency: bool) -> dict:
    task = REGISTRY[key](limit, calib)
    q = task.question
    z = model.logits(task.states, q)
    y = np.array(task.gold)
    raw = summarize(softmax(z), y)

    T = 1.0
    cal = raw
    if task.calib_states:
        zc = model.logits(task.calib_states, q)
        T = fit_temperature(zc, np.array(task.calib_gold))
        cal = summarize(softmax(z / T), y)

    p50, p95 = latency(model, task) if do_latency else (float("nan"), float("nan"))
    return {
        "task": key, "lang": task.lang, "kind": q.kind, "n": len(y), "choices": task.n_choices,
        "acc": raw["acc"], "brier": raw["brier"], "ece_raw": raw["ece"], "ece_cal": cal["ece"],
        "nll_cal": cal["nll"], "T": T, "p50_ms": p50, "p95_ms": p95,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(REGISTRY))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--calib", type=int, default=500)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--system", choices=["baseline", "decima"], default="baseline")
    ap.add_argument("--checkpoint", default=None, help="checkpoints/<run>/best for --system decima")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.system == "decima":
        if not args.checkpoint:
            ap.error("--system decima needs --checkpoint")
        model = Decima(args.checkpoint, device=args.device)
        system, model_name = "decima late-interaction scorer", args.checkpoint
    else:
        model = ZeroShotBiEncoder(args.model, device=args.device)
        system, model_name = "zero-shot bi-encoder", args.model
    rows = []
    for key in args.tasks:
        t0 = time.time()
        r = run_task(model, key, args.limit, args.calib, not args.no_latency)
        rows.append(r)
        print(f"  {key:14s} acc={r['acc']:.3f} ece={r['ece_raw']:.3f}→{r['ece_cal']:.3f} p50={r['p50_ms']:.1f}ms ({time.time()-t0:.0f}s)", flush=True)

    cols = ["task", "kind", "choices", "n", "acc", "brier", "ece_raw", "ece_cal", "nll_cal", "T", "p50_ms", "p95_ms"]
    print()
    print(tabulate([[r[c] for c in cols] for r in rows], headers=cols, floatfmt=".3f"))

    out = Path(args.out or f"runs/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{args.system}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model_name, "system": system, "limit": args.limit,
        "machine": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version()},
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
