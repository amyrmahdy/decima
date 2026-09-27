"""int8 vs fp32 on real eval items (ONNX exports, CPU) — the release check for quantisation damage.

    uv run python scripts/int8_check.py v1i          # uses export/v1i and export/v1i-int8

Items: every eval item (no option-order copies) of the Kev, Laya and Decima-bench sets. Prints top-1
agreement and accuracy for both precisions; the synthetic near-duplicate options in bench/latency.py
are not a quantisation measure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    n = sys.argv[1]
    out = Path("runs/final"); (out / "preds").mkdir(parents=True, exist_ok=True)
    items_path = out / f"int8-items-{n}.jsonl"
    items = {}
    with items_path.open("w") as f:
        for s in ("kev", "laya", "decima"):
            for line in open(f"runs/items/{s}.jsonl"):
                r = json.loads(line)
                if r["split"] == "eval" and "group" not in r:
                    items[r["id"]] = r
                    f.write(line)
    preds = {}
    for tag, d in (("fp32", f"export/{n}"), ("int8", f"export/{n}-int8")):
        p = out / "preds" / f"onnx-{tag}--{n}.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            subprocess.run(["uv", "run", "python", "-m", "bench.predict", "--items", str(items_path), "--system", "onnx",
                            "--checkpoint", d, "--threads", "4", "--out", str(p)], check=True)
        preds[tag] = {r["id"]: r["probs"] for r in map(json.loads, p.open())}
    top = lambda p: max(range(len(p)), key=p.__getitem__)
    ids = [k for k in preds["fp32"] if k in preds["int8"]]
    agree = sum(top(preds["fp32"][k]) == top(preds["int8"][k]) for k in ids) / len(ids)
    acc = {t: sum(top(preds[t][k]) == items[k]["gold"] for k in ids) / len(ids) for t in preds}
    print(f"{n} int8 vs fp32 (ONNX, {len(ids)} real eval items): top-1 agreement {agree:.4f} · "
          f"accuracy fp32 {acc['fp32']:.4f} int8 {acc['int8']:.4f}")


if __name__ == "__main__":
    main()
