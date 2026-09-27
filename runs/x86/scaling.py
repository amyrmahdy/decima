"""Latency vs number of options (int8/fp32 ONNX, x86, pinned), cached and cold option sets.

Option texts are real intent labels (banking77 + CLINC150 + MASSIVE), extended with numbered
variants past ~290 so every option is distinct; cost depends on token count, not content.

    OMP_NUM_THREADS=1 taskset -c 2 uv run python runs/x86/scaling.py int8 1     # or fp32
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from bench.items import read
from decima.runtime import DecimaOnnx
from decima.types import Question

prec, threads = sys.argv[1], int(sys.argv[2])
labels = []
for i in read("runs/items/decima.jsonl"):
    if i["suite"] in ("decima/banking77/en", "decima/clinc150/en", "decima/massive/en") and i["split"] == "eval" and "group" not in i:
        for c in i["choices"]:
            if c not in labels: labels.append(c)
base = list(labels)
k = 2
while len(labels) < 1000:
    labels += [f"{c} (variant {k})" for c in base]; k += 1
states = [i["state"] for i in read("runs/items/laya.jsonl") if i["suite"] == "laya/massive-en" and i["split"] == "eval" and "group" not in i][:40]
rt = DecimaOnnx(f"export/v1i{'-int8' if prec == 'int8' else ''}", threads=threads)
rows = []
for n in (4, 20, 77, 150, 300, 500, 1000):
    q = Question(text="What does the user want?", choices=labels[:n], kind="choose", lang="en")
    t0 = time.perf_counter(); rt._choices(q); cold_enc = (time.perf_counter() - t0) * 1e3
    rt.decide_logits(states[0], q)
    reps = 30 if n <= 150 else 12
    ms = []
    for j in range(reps):
        t0 = time.perf_counter(); rt.decide_logits(states[j % len(states)], q); ms.append((time.perf_counter() - t0) * 1e3)
    r = {"precision": prec, "threads": threads, "choices": n, "cached_p50_ms": round(float(np.percentile(ms, 50)), 1),
         "cached_p95_ms": round(float(np.percentile(ms, 95)), 1), "cold_option_encode_ms": round(cold_enc, 1)}
    rows.append(r); print(r, flush=True)
    rt._cache.clear()
json.dump(rows, open(f"runs/x86/scaling-v1i-{prec}-{threads}t.json", "w"), indent=2)
