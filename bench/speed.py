"""Release speed table: CPU (ONNX fp32 / int8) and GPU (PyTorch), single request and batched.

    uv run python -m bench.speed --name v1i --export export/v1i --checkpoint checkpoints/v1i/best \
        --out runs/speed-v1i.json            # add --no-gpu / --no-cpu to skip a side

Inputs are real: short states (MASSIVE utterances, ~10–20 tokens) and long states (typed-decisions
cases, ~300–500 tokens), with real option sets of 4 (AG News), 20 (Laya's MASSIVE protocol) and 77
(banking77). Choice encodings are cached (the steady state of an app asking the same question), and
every number says what it is: latency of one call (p50/p95) for a batch of B states, and throughput in
decisions/s. Never compare single-request latency with someone else's batched throughput.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from bench.items import read
from decima.types import Question


def inputs() -> tuple[dict[str, list[str]], dict[int, Question]]:
    laya = [i for i in read("runs/items/laya.jsonl") if i["suite"] == "laya/massive-en" and i["split"] == "eval" and "group" not in i]
    dec = {i["suite"]: i for i in read("runs/items/decima.jsonl") if i["split"] == "eval" and "group" not in i}
    typed = [i for i in read("runs/items/jevtyped.jsonl") if i["suite"] == "typed/test" and i["split"] == "eval" and "group" not in i]
    states = {"short": [i["state"] for i in laya[:60]], "long": list(dict.fromkeys(i["state"] for i in typed))[:60]}
    qs = {}
    for n, src in ((4, dec["decima/agnews/en"]), (20, laya[0]), (77, dec["decima/banking77/en"])):
        qs[n] = Question(text=src["question"], choices=src["choices"], kind="choose", lang="en")
    return states, qs


def timed(fn, reps: int) -> list[float]:
    fn()                                              # warm-up (also fills the choice cache)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); out.append((time.perf_counter() - t0) * 1e3)
    return out


def row(side, prec, threads, batch, length, n, ms) -> dict:
    p50, p95 = float(np.percentile(ms, 50)), float(np.percentile(ms, 95))
    return {"side": side, "precision": prec, "threads": threads, "batch": batch, "state": length, "choices": n,
            "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "decisions_per_s": round(batch * 1e3 / p50, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--export", help="ONNX export dir (fp32); <dir>-int8 is used for int8")
    ap.add_argument("--checkpoint", help="PyTorch checkpoint dir for the GPU side")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 10, 50])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    states, qs = inputs()
    rows = []
    if not args.no_cpu and args.export:
        from decima.runtime import DecimaOnnx

        for prec, d in (("fp32", Path(args.export)), ("int8", Path(args.export + "-int8"))):
            for th in args.threads:
                rt = DecimaOnnx(d, threads=th)
                for length, S in states.items():
                    for n, q in qs.items():
                        it = iter(S * (args.reps + 2))
                        ms = timed(lambda: rt.decide_logits(next(it), q), args.reps)
                        rows.append(row("cpu", prec, th, 1, length, n, ms)); print(rows[-1], flush=True)
    if not args.no_gpu and args.checkpoint:
        import torch

        from decima.model import Decima

        m = Decima(args.checkpoint, device="cuda")
        sync = torch.cuda.synchronize
        for B in args.batches:
            for length, S in states.items():
                for n, q in qs.items():
                    batch = (S * (B // len(S) + 1))[:B]
                    def call():
                        m.logits(batch, q, batch_size=B, max_pairs=max(256, B * n)); sync()
                    ms = timed(call, max(5, args.reps // 3))
                    rows.append(row("gpu", "fp32", None, B, length, n, ms)); print(rows[-1], flush=True)

    meta = {"name": args.name, "machine": platform.machine(), "cpu": platform.processor() or platform.machine(),
            "x86": platform.machine() in ("x86_64", "AMD64"), "python": platform.python_version()}
    if not args.no_gpu and args.checkpoint:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
