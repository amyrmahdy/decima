"""Run one of our systems over scoreboard items → preds JSONL {"id", "probs"}.

    uv run python -m bench.predict --items runs/items/phase0.jsonl --system decima \
        --checkpoint checkpoints/v0/best --device cuda --out runs/preds/phase0-decima-v0.jsonl
    uv run python -m bench.predict --items runs/items/phase0.jsonl --system baseline --out runs/preds/phase0-e5.jsonl

`--system onnx --checkpoint export/v0-int8` scores the exported graphs (fp32 or int8) on CPU.
Competitors have their own predictors under bench/external/ (own venvs, same output format).
Items sharing (question, choices, lang, kind) are batched so choice encodings are computed once.
Probabilities are the system as shipped: Decima with its trained temperature, the baseline at T=0.05.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.items import read
from decima.calibration import softmax
from decima.types import Question


def load_system(system: str, checkpoint: str | None, model: str | None, device: str, threads: int = 4):
    if system == "decima":
        from decima.model import Decima

        return Decima(checkpoint, device=device), checkpoint
    if system == "onnx":
        from decima.runtime import DecimaOnnx

        return DecimaOnnx(checkpoint, threads=threads), checkpoint
    from decima.baseline import DEFAULT_MODEL, ZeroShotBiEncoder

    return ZeroShotBiEncoder(model or DEFAULT_MODEL, device=device), model or DEFAULT_MODEL


def predict(sys, items: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        groups[(it["question"], tuple(it["choices"]), it["lang"], it["kind"])].append(it)
    out = []
    done, t0 = 0, time.time()
    for (qt, choices, lang, kind), its in groups.items():
        q = Question(text=qt, choices=list(choices), kind=kind, lang=lang)
        z = sys.logits([i["state"] for i in its], q)
        p = 1 / (1 + np.exp(-z)) if kind == "rank" else softmax(z)
        out.extend({"id": i["id"], "probs": [float(x) for x in row]} for i, row in zip(its, p))
        done += len(its)
        if len(groups) > 50 and done % 2000 < len(its):
            print(f"  {done}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--system", choices=["decima", "onnx", "baseline"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None, help="encoder id for --system baseline")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for --system onnx (the GX10 overheats on all 20)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.system in ("decima", "onnx") and not args.checkpoint:
        ap.error(f"--system {args.system} needs --checkpoint (onnx: an export dir)")

    # model first, then items: on the GB10's unified memory a CUDA context created after a large item
    # file is in RAM has failed to allocate even the weights
    sys, name = load_system(args.system, args.checkpoint, args.model, args.device, args.threads)
    items = read(args.items)
    t0 = time.time()
    preds = predict(sys, items)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    print(f"{len(preds)} preds from {name} in {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
