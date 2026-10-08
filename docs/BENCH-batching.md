# Many questions, long documents, abstention — runtime notes and benchmark

Measured 2026-10-08 on the GX10 (Grace, 20 Arm cores: Cortex-X925 + A725), onnxruntime 1.30.0,
`export/agent3-int8` (mmBERT-base body, 2,048-token states) and `export/base2-int8` (512), with the code
and default batch settings of the commit that adds this file. The machine was shared with a GPU training
run and another CPU job, so treat single numbers as ±10 %; time the process spent stopped by the thermal
guard is excluded.
Raw rows: `runs/bench-batching.json`. Reproduce with `bench/batching.py` (commands in its docstring).

## Short version

- **Same numbers, bitwise.** `decide_many` / `decide_batch` give exactly the probabilities of `decide`
  one by one: max |Δp| = **0.0** on both models, 1 and 4 threads, 5 states (30 → 2,350 tokens, the last one
  cut) × 8 questions of every kind (`python -m bench.batching --check …`, also in `tests/`). The scorer
  sees the same tensors as before, so the Rust runtime's parity is untouched.
- **Batching helps short states only: 1.15–1.3×.** Six questions on a short message: 182 → 143 ms at
  4 threads, 457 → 390 ms at 1 thread; a first request (option sets not cached yet) 1.13–1.19×.
- **It does nothing for long states, and cannot.** 48 questions on a 1,950-token document take
  **~175 s at 4 threads and ~420 s at 1 thread, before and after** (3.7 s / 8.7 s per question). The
  work is the same and it is compute-bound; see "Why long states are slow" for what would help.
- **`long="chunk"`** opens documents of any length (tested to 8k tokens) without the 422, at a cost
  linear in the document: 5 windows of 512 tokens for a 2k document (base2: 1.5 s per question at
  4 threads), 20 for 8k (6.5 s); with agent3, 5 windows of 2,048 for 8k (15 s per question).

## 1. Latency before / after (agent3-int8, choice encodings cached unless "cold")

"before" = `decide` once per question (what `serve.py` did); "after" = `decide_many` (what it does now).
Median of 20 / 10 runs for the short cases, one run for the long ones.

| case | state tokens | threads | before | after | per question | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| 1 question, short state | 30 | 1 | 70 ms | 74 ms | 74 ms | 0.94 (noise: same calls) |
| 1 question, short state | 30 | 4 | 27 ms | 28 ms | 28 ms | 0.99 |
| 6 questions, short state | 30 | 1 | 457 ms | 390 ms | 65 ms | **1.17** |
| 6 questions, short state | 30 | 4 | 182 ms | 143 ms | 24 ms | **1.27** |
| 6 questions, short state, cold | 30 | 1 | 1,025 ms | 911 ms | 152 ms | **1.13** |
| 6 questions, short state, cold | 30 | 4 | 369 ms | 310 ms | 52 ms | **1.19** |
| 24 questions, long state | 1,950 | 1 | 209 s | 207 s | 8.6 s | 1.01 |
| 24 questions, long state | 1,950 | 4 | 86 s | 87 s | 3.6 s | 0.99 |
| 48 questions, long state | 1,950 | 1 | 418 s | 419 s | 8.7 s | 1.00 |
| 48 questions, long state | 1,950 | 4 | 174 s | 177 s | 3.7 s | 0.99 |

A single question goes through the same encoder and scorer calls either way (a profile puts > 99 % of
the time in ONNX Runtime for both), so the 1-question rows differ by noise only.

**Batch settings.** `batch_size=16`, `max_batch_tokens=512` (padded tokens per encoder call). A sweep with
12 questions (agent3, 1 and 4 threads) found sharing a call worth 10–25 % for rows up to ~150 tokens,
neutral at ~300 and 0–15 % *slower* beyond: on CPU the encoder is compute-bound, and a padded batch of
long rows only adds memory traffic for its L² attention tensors. The 512-token budget therefore batches
short rows and runs long ones alone. Choice sets that are not cached yet are encoded together across
the questions of a request (the "cold" rows).

## 2. Why long states are slow

`question_in_state` puts the question in front of the state, so every question re-encodes the whole
document — 48 questions are 48 full encoder passes. And the cost of one pass grows much faster than the
length (agent3-int8, 1 question, 4 threads):

| state tokens | 155 | 469 | 981 | 1,985 |
|---|---:|---:|---:|---:|
| latency | 78 ms | 292 ms | 1,032 ms | 3,564 ms |

A fit `a·L + b·L²` puts **~85 % of the 2k-token time in the L² term**. mmBERT attends globally in only 8
of its 22 layers; the other 14 use a 128-token sliding window, but the exported graph computes them as
full L×L attention with a mask. Two follow-ups would change this, neither done here because both change
the ONNX files that the Ollaya runtime and the published model share:

1. **Re-export with banded attention in the sliding-window layers** (same weights, same maths up to float
   rounding): the L² term drops to 8/22 of today's, an estimated **~2× at 2k tokens** for every question.
2. **A model that encodes the document once** (state cached, question only on the option side, as V0
   does): each further question then costs one scorer pass over the cached document instead of an
   encoder pass. V0 showed this costs accuracy on questions whose content must meet the text inside the
   encoder (NLI, yes/no reading), so it is a training-side decision, not a runtime one.

Until then, the practical levers are threads (4 threads: 2.4× over 1 at 2k tokens) and, for documents
that do not need to be read as a whole, chunking with a short-window model (base2 chunked: 1.5 s per
question on a 2k document vs 3.6 s for agent3 reading it whole).

## 3. Long documents: `long="chunk"`

Opt-in in the Python API (`decide(…, long="chunk")`, `decide_many(…, long="chunk", aggregate=…, overlap=64)`)
and per request in `serve.py` (`"long": "chunk"`, `"aggregate"`; documents up to `--max-doc-tokens`, default
16,384, else 422 `STATE_TOO_LONG`). Without it nothing changes: the Python API cuts an over-long state at
its end as before and the server answers 422 `STATE_TRUNCATED`.

**Windows.** The state is split into character spans whose rendering (question + window) fits the model's
`max_state_tokens`. A window ends at a sentence boundary when one falls in its second half; the next one
starts `overlap` tokens (default 64) earlier, moved to a sentence start when one is near. A state that fits
is decided exactly as without the option. Each window is an ordinary decision.

**Aggregation** (per question type; the default is `max`):

| `aggregate` | noul / verify | choice / score | rank |
|---|---|---|---|
| `max` (default) | the window with the highest P(yes): "yes if any window says so" | the single most confident window, its distribution unchanged | per option, its highest P |
| `mean` | average over windows | average distribution | average |
| `sum` | summed log-probabilities, renormalised (product of experts) | same | not allowed |

Why `max` by default: in a long document the evidence for an answer usually sits in one place, and the
windows that do not contain it are not evidence against it. `mean` dilutes one window's evidence n-fold
(n windows); `sum` treats windows as independent witnesses and becomes over-confident when they overlap or
agree for the same reason. `max` returns one real forward pass — the window that decided — whose
probabilities carry the model's calibration on a window-sized input. Its cost: for noul the false-positive
rate grows with the number of windows (n chances for a spurious "yes"), and it is wrong for statements
about the document as a whole ("the contract is one-sided") and for negated ones ("never mentions a
penalty"): use `mean` for those, or don't chunk.

**What it returns.** `Decision.meta["window"]` (index of the deciding window), `meta["windows"]` (character
spans of the state) and `meta["window_probs"]`. The server adds `"window": {"index", "start", "end", "count"}`
to answers that needed more than one window.

**What it cannot do.** Anything that needs the whole document at once: comparing a clause on page 1 with
one on page 9, counting or summing across windows, resolving a reference whose antecedent is in another
window, or judging the document as a whole. Each window is read alone. And chunking is only as good as
the model on one window: in the test with a fact placed at 75 % of a 1.5k-token document (base2, 4
windows), "Does the text mention a vault access code?" found it (window P(yes) 0.72 vs ≤ 0.28 elsewhere),
while the phrasing "The document gives a vault access code." stayed under 0.5 in every window.

| model | document | windows / question | questions | total | per question (4 threads) |
|---|---:|---:|---:|---:|---:|
| base2-int8 (512) | 1,948 tokens | 5 | 6 | 9.3 s | 1.5 s |
| base2-int8 (512) | 1,948 tokens | 5 | 24 | 37 s | 1.5 s |
| base2-int8 (512) | 7,990 tokens | 20 | 6 | 39 s | 6.5 s |
| agent3-int8 (2,048) | 7,990 tokens | 5 | 6 | 90 s | 15 s |

## 4. Abstention (confidence gating)

`decision.abstain(threshold)` sets and returns `decision.abstained` (top probability below `threshold`);
`decision.answer` is the top choice, or `None` when abstained; `decide` / `decide_many` / `decide_batch`
take `min_confidence=` to do it in one call. In the server, `"min_confidence": 0.6` adds `"abstain": true |
false` to every answer, compared with the answer's TypeSafe `confidence` (`(K·p_max − 1)/(K − 1)`; for noul
`|2·noul − 1|`) — the number the client sees, comparable across option counts. Without the option the
answers are unchanged.
