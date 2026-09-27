# Decima-small on x86 CPU — release benchmark

Model: **Decima-small = run `v1i`** (frozen 2026-09-26). Measured 2026-09-26/27 on an x86 laptop, the
deployment-class machine the model card's "CPU x86" row was waiting for. Companion to
[EVAL.md](EVAL.md), whose numbers are fp32 PyTorch on the GX10's GPU.

## Short version

- **x86 reproduces the GX10 exactly.** fp32 ONNX on x86 CPU picks the same answer as fp32 PyTorch on the
  GB10 GPU on **100 %** of 22,050 eval items (largest probability difference 5 × 10⁻⁶).
- **int8 costs nothing measurable in accuracy.** Top-1 agreement with fp32 is **98.6–99.8 %** per set on
  45,850 real eval items (not the 0.93 EVAL.md measured on synthetic latency-probe items), and every
  set-level score moves by ≤ 0.004. EVAL.md's "int8 accuracy equal to these numbers" can move from
  *must not claim* to *can claim*, citing this file.
- **Fast for small option sets:** int8, 1 thread: **20 ms** for 4 options, **42 ms** for 20, **83 ms** for 77
  (short state, option encodings cached).
- **Latency grows linearly with the number of options, about 1 ms per option.** 1,000 options take
  ~1.06 s. The original target "<50 ms at 1,000 choices" is **not** met; <50 ms holds up to roughly
  20–40 options depending on option length. Large option sets need the retrieval tier (README roadmap
  step 5).
- **Release blocker found:** `export/v1i-int8/tokenizer` is an **absolute symlink** into the GX10's home
  directory, so the int8 export fails to load on any other machine. Fix below.

## Setup

| | |
|---|---|
| Model files | `checkpoints/v1i/best`, `export/v1i` (fp32 ONNX), `export/v1i-int8` — rsynced from the GX10; **all 15 files match the sha256 values in `release/manifest-v1i.json`** |
| Code | commit `31436ab` (fetched from the GX10) |
| CPU | Intel Core Ultra 7 155H (Meteor Lake, hybrid 6P + 8E + 2LP-E, 22 threads), AVX2 + AVX-VNNI, no AVX-512 |
| Memory / power | 30 GB, on AC, `balanced` platform profile, `powersave` governor |
| Runtime | onnxruntime 1.30.0 (CPUExecutionProvider), Python 3.11 |
| Pinning | latency runs are pinned with `taskset`: 1 thread → cpu 2 (a 4.8 GHz P-core); 4 threads → cpus 2,4,7,9 (one hyperthread on each of four different P-cores) |
| Items | the frozen item files `runs/items/{decima,kev,laya,jevtyped,btzsc}.jsonl`, eval split, no option-order copies |

## 1. Accuracy on x86

Set scores are means of per-suite values, computed with `bench.score.score_suite` (the scorer behind
EVAL.md). "GX10 GPU fp32" is the development predictions `runs/preds/<set>--v1i.jsonl`.

| set (suites, items) | metric | GX10 GPU fp32 | x86 CPU fp32 | **x86 CPU int8** | int8 − GPU |
|---|---|---:|---:|---:|---:|
| Decima bench (12, 12,000) | acc | 0.785 | 0.785 | **0.784** | −0.0003 |
| Kev protocol (8, 1,350) | acc | 0.768 | 0.768 | **0.767** | −0.0017 |
| Laya protocol (29, 8,700) | acc | 0.764 | 0.764 | **0.763** | −0.0003 |
| JevBench public + typed (2, 2,231) | acc | 0.485 | — | **0.489** | +0.0036 |
| BTZSC, all 22 (21,569) | macro-F1 | 0.621 | — | **0.622** | +0.0004 |
| BTZSC, clean 18 (zero-shot) | macro-F1 | 0.570 | — | **0.571** | +0.0004 |
| Decima bench | ECE (as shipped) | 0.031 | 0.031 | 0.032 | |
| Laya protocol | ECE (as shipped) | 0.056 | 0.056 | 0.053 | |

fp32 was run only on the three sets `scripts/int8_check.py` defines, to measure quantization damage.
JevBench and BTZSC ran int8 only (the shipping artifact).

### Agreement (same answer on the same item)

| set | x86 fp32 vs GPU fp32 | x86 int8 vs GPU fp32 | max \|Δp\| int8 vs GPU |
|---|---:|---:|---:|
| Decima bench | 100.00 % (max \|Δp\| 2.5e-6) | 99.41 % | 0.066 |
| Kev protocol | 100.00 % (5.2e-6) | 99.78 % | 0.028 |
| Laya protocol | 100.00 % (2.9e-6) | 99.28 % | 0.077 |
| JevBench public + typed | — | 98.57 % | 0.031 |
| BTZSC | — | 98.69 % | 0.050 |

### Decima bench per suite (EN / FA / AR / RU)

| suite | GPU fp32 | x86 int8 | Δ |
|---|---:|---:|---:|
| sst5/en | 0.514 | 0.512 | −0.002 |
| agnews/en | 0.904 | 0.904 | 0.000 |
| xnli/en | 0.749 | 0.746 | −0.003 |
| xnli/ar | 0.659 | 0.659 | 0.000 |
| xnli/ru | 0.689 | 0.690 | +0.001 |
| farstail/fa | 0.819 | 0.816 | −0.003 |
| massive/en | 0.876 | 0.876 | 0.000 |
| massive/fa | 0.846 | 0.847 | +0.001 |
| massive/ar | 0.760 | 0.763 | +0.003 |
| massive/ru | 0.837 | 0.838 | +0.001 |
| banking77/en | 0.887 | 0.886 | −0.001 |
| clinc150/en | 0.875 | 0.875 | 0.000 |

Laya protocol groups (int8 / GPU): MASSIVE en 0.917 / 0.913 · MASSIVE 13 other languages 0.851 / 0.852 ·
XNLI en 0.773 / 0.767 · XNLI 14 other languages 0.670 / 0.671.

## 2. Speed on x86

One decision = one state scored against one option set, batch 1, option encodings cached (the steady
state of an app asking the same question repeatedly). Real inputs from `bench/speed.py`: short states
are MASSIVE utterances (~10–20 tokens), long states are typed-decisions cases (~300–500 tokens); option
sets are AG News (4), Laya's MASSIVE protocol (20) and banking77 (77). 40 timed calls per cell after
warm-up.

### Model-card table — p50 / p95 ms

| precision | threads | state | 4 options | 20 options | 77 options |
|---|---:|---|---:|---:|---:|
| **int8** | **1** | **short** | **20 / 23** | **42 / 46** | **83 / 92** |
| int8 | 1 | long | 102 / 113 | 126 / 142 | 204 / 221 |
| int8 | 4 | short | 31 / 43 | 62 / 83 | 108 / 138 |
| int8 | 4 | long | 106 / 176 | 158 / 216 | 246 / 292 |
| fp32 | 1 | short | 24 / 27 | 75 / 80 | 181 / 186 |
| fp32 | 1 | long | 96 / 104 | 179 / 193 | 407 / 441 |
| fp32 | 4 | short | 17 / 35 | 84 / 115 | 178 / 226 |
| fp32 | 4 | long | 120 / 184 | 192 / 234 | 344 / 432 |

- **Use 1 thread.** At batch 1, 4 threads is *slower* than 1 thread for int8 in every cell and has much
  worse p95; the graphs are too small to amortize thread synchronization. For throughput, run several
  single-threaded workers instead.
- **int8 is 2.2× faster than fp32 once options dominate** (77 options: 83 vs 181 ms); for 4 options and
  long states the encoder dominates and the gain is small.
- For reference, the GX10's own indicative Grace-CPU probe (`runs/latency-v1i-gx10-indicative.json` on the GX10,
  int8, 1 thread) gave 16 ms at 4 options and 76 ms at 77: the laptop x86 core is in the same range.

### Scaling with the number of options (int8 / fp32, 1 thread, short state)

The option *encodings* are cached, but every call still runs the scorer over every option: each
option's tokens cross-attend to the state (`decima/runtime.py`, `decide_logits`). That is what makes
answers independent of option order. It also makes the cost linear in the number of options. Option
texts here are short intent labels (banking77 + CLINC150 + MASSIVE, with numbered variants past ~290).
Option length matters too, which is why 20 short labels (26 ms) are cheaper than Laya's 20 options above
(42 ms).

| options | int8 p50 | int8 p95 | fp32 p50 | fp32 p95 | new option set: one-time encode (int8) |
|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 20 | 20 | 22 | 27 |
| 20 | 26 | 29 | 44 | 46 | 133 |
| 77 | 75 | 80 | 159 | 167 | 745 |
| 150 | 130 | 140 | 313 | 342 | 1,405 |
| 300 | 286 | 293 | 690 | 722 | 3,118 |
| 500 | 534 | 574 | 1,367 | 1,493 | 5,916 |
| 1,000 | 1,058 | 1,108 | 2,820 | 2,996 | 12,431 |

All values are ms. In words (int8, 1 thread, short state): 16 ms at 4 options, ~1 ms per extra option,
1.06 s at 1,000 — the release documents quote these measured points, not a fitted formula. Encoding a
*new* option set costs ≈ 12 ms per option, once. The cache holds up to 256 option sets and is cleared
entirely when it overflows (`decima/runtime.py`; not an LRU).

### Footprint

| | int8 | fp32 |
|---|---:|---:|
| ONNX files on disk (encoder + scorer, from the manifest's byte counts) | 127 MB + 17 MB tokenizer | 489 MB + 17 MB tokenizer |
| Peak RSS of the Python process (load + first decision) | ~1.0 GB | ~1.4 GB |
| Load time (import + sessions + tokenizer) | 5.1 s | 5.2 s |
| First decision after load (cold option set of 4) | 54 ms | 57 ms |

Most of the RSS and load time is the Python stack (`transformers` is imported for the tokenizer), not the
model. Not measured: loading the tokenizer with the `tokenizers` library alone.

## 3. Findings to act on before release

1. **Blocker: the int8 export is not portable.** `decima/quantize.py:183` creates
   `tokenizer -> (src / "tokenizer").resolve()`, an absolute path
   (`export/v1i/tokenizer`). On any other machine `DecimaOnnx("…-int8")`
   fails with `OSError: Repo id must be in the form 'repo_name'…`. Every `export/*-int8` has the same link.
   For these measurements the local copy was repointed to `../v1i/tokenizer` (the target files are
   sha256-identical per the manifest). Fix: copy the tokenizer files into the int8 folder (safest for an
   HF upload, where symlinks do not survive), or at least make the link relative. Then re-export `v1i-int8`
   and update the manifest; `encoder.onnx` / `scorer.onnx` do not change.
2. **Restate the speed claim.** The honest headline is "~20 ms per decision on one laptop CPU core for
   small option sets (int8, 4 options); ~1 ms per additional option". "1,000 options" is supported for
   *correctness* (no context-window limit, order-independent) but costs ~1 s per decision. The model card's
   "the number of options is not bounded by a prompt" is true; add the latency slope next to it.
3. **Default to `threads=1`** in the quickstart and runtime docs (see the speed table).
4. **int8 is now a measured claim.** Done: EVAL.md §5 lists "int8 / ONNX accuracy equal to these numbers"
   under *claims we can make*, and the card quotes the numbers in §1 (the old "98.8 %" sentence is gone).

### Model-card fill-ins (Speed section)

| | precision | threads / batch | 4 options | 20 options | 77 options |
|---|---|---|---:|---:|---:|
| CPU x86 (Core Ultra 7 155H, 1 P-core) — short state | int8 | 1 thread | 20 ms | 42 ms | 83 ms |
| CPU x86 (Core Ultra 7 155H, 1 P-core) — long state (~400 tokens) | int8 | 1 thread | 102 ms | 126 ms | 204 ms |

p50 values; p95 in §2. The GX10 Grace-CPU and GB10-GPU rows still need `bench/speed.py` on a quiet GX10.

## 4. Reproduce

```bash
# model: rsync checkpoints/v1i/best, export/v1i, export/v1i-int8 and the five runs/items/*.jsonl, then
#   repoint export/v1i-int8/tokenizer → ../v1i/tokenizer (finding 1) and check release/manifest-v1i.json

# speed (model-card table)
OMP_NUM_THREADS=1 taskset -c 2 uv run python -m bench.speed --name v1i --export export/v1i \
    --threads 1 --reps 40 --no-gpu --out runs/x86/speed-v1i-x86-1t.json
OMP_NUM_THREADS=4 taskset -c 2,4,7,9 uv run python -m bench.speed --name v1i --export export/v1i \
    --threads 4 --reps 40 --no-gpu --out runs/x86/speed-v1i-x86-4t.json

# latency vs number of options (4 … 1,000) and one-time option-encode cost
OMP_NUM_THREADS=1 taskset -c 2 uv run python runs/x86/scaling.py int8 1
OMP_NUM_THREADS=1 taskset -c 2 uv run python runs/x86/scaling.py fp32 1

# accuracy: int8 on the 5 release sets + fp32 on kev/laya/decima, then the comparison
runs/x86/run_acc.sh
uv run python runs/x86/compare_x86.py        # → runs/x86/compare-v1i.json
```

Pick the `taskset` CPUs from `/sys/devices/cpu_core/cpus` and `…/topology/thread_siblings_list` on
other hybrid Intel machines.

## Files

| file | contents |
|---|---|
| `runs/x86/speed-v1i-x86-{1,4}t.json` | `bench/speed.py` output (fp32 + int8, short/long, 4/20/77 options) |
| `runs/x86/scaling-v1i-{int8,fp32}-1t.json` | latency vs 4…1,000 options; one-time option-encode cost |
| `runs/x86/compare-v1i.json` | per-suite and per-set scores for all three systems, agreement, max \|Δp\| |
| `runs/x86/preds/*.jsonl` | x86 predictions (`{"id", "probs"}`), same format as `runs/preds/` |
| `runs/x86/run_acc.sh`, `runs/x86/compare_x86.py`, `runs/x86/scaling.py` | the accuracy pass, the comparison, the option-count scaling probe |
