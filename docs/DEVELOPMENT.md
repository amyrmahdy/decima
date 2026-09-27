# Decima

A tiny, calibrated **decision model**: `state + question + arbitrary choices → probabilities`.
Not a chatbot, not a classifier with a fixed label set. The choices are supplied at inference
time as natural language, so one model answers any decision your system needs — routing,
intent, triage, verification, ranking — on CPU, in milliseconds.

```
LLM              reasons about the problem
Decima           makes the decision           ← this repo
your system      takes the action
```

**Target claim** (published only once measured on this bench):

> ~30M body · <50 ms CPU · 1,000 arbitrary choices · calibrated · EN + FA — measured across EN / FA / AR / RU

## Primitives

| kind     | question                                | output                                |
|----------|-----------------------------------------|---------------------------------------|
| `choose` | pick one of N options                   | softmax over choices                  |
| `score`  | pick one of N *ordered* levels          | softmax, ordinal-aware                |
| `verify` | is this proposition true?               | P(yes), P(no)                         |
| `rank`   | score each option independently         | one probability per choice            |

## Decima Bench

There is no standard benchmark for decision models, so this one is part of the product.
Every dataset is reshaped into the inference format above; label names become readable
choice text; nothing is trained on the test sets.

| task        | lang | kind   | choices | what it measures                              |
|-------------|------|--------|--------:|-----------------------------------------------|
| sst5        | en   | score  | 5       | ordinal decisions (Laya's weakest type)       |
| agnews      | en   | choose | 4       | easy sanity floor                             |
| xnli        | en/ar/ru | choose | 3   | verification / entailment                     |
| farstail    | fa   | verify | 2       | Persian entailment                            |
| massive     | en/fa/ar/ru | choose | 60 | intents; fa/ar/ru state vs **English** choices (cross-lingual) |
| banking77   | en   | choose | 77      | high cardinality (Laya: 0.425, Jev: 0.870)    |
| clinc150    | en   | choose | 151     | very high cardinality + out-of-scope option   |

Metrics per task: accuracy · Brier · NLL · ECE (raw, and after temperature scaling on a
held-out slice) · CPU latency p50 / p95 per decision with choice embeddings cached.

```bash
uv sync
uv run python -m bench.run                           # full matrix, 1000 examples/task
uv run python -m bench.run --tasks massive/fa banking77/en --limit 200
```

Results land in `runs/*.json`.

## Roadmap

1. **Zero-shot baseline** — `decima/baseline.py`: multilingual-e5-small bi-encoder, no training.
   The honest floor, and the backbone we intend to keep. *(this commit)*
2. **Teacher distillation** — synthetic decisions in EN/FA/AR/RU (+ cross-lingual mixes) with
   full probability distributions from a teacher model served locally on the GX10 (vLLM).
3. **Student** — e5-small body + late-interaction option scorer (state encoded once, options
   cross-attend to state tokens; options never compete for context window) + ordinal head +
   first-class "none of the above". KL-to-teacher + NLL, then temperature scaling.
4. **Runtime** — ONNX int8; Python + `onnxruntime-web` packages; `choose / score / verify / rank` API.
5. **Retrieval tier** — ANN over choice embeddings → top-k → scorer, for 1k–1M choice spaces.

## Teacher distillation data

The teacher is Gemma-4-26B-A4B (NVFP4, vLLM behind LiteLLM, served locally with vLLM on an NVIDIA GB10 box;
endpoint in `.env`, see `.env.example`). Qwen3-Coder-Next was measured too and rejected: 3 of 6 sample
calls unparseable, no valid Persian. Two generators write JSONL to `data/teacher/` (git-ignored),
seeded, resumable, deduplicated by (state, question, choices):

- `teacher.generate` — the teacher invents state + question + choices + gold + full distribution over a
  (domain × language × kind × choice-count × none-option) grid; ≥ 20 choices use one shared catalogue
  per call with sparse top-k distributions.
- `teacher.label` — the teacher labels existing states (bench **train** splits only, never test or
  validation; plus our own generated states) under freshly invented questions and choice sets.

Mix by count: EN 30 / FA 25 / AR 25 / RU 20, ~15 % cross-lingual, kinds choose 40 / verify 22 /
score 20 / rank 18, "none of the above" in ~25 % of choose examples and correct in ~a third of those.
Every distribution is validated as a proper probability vector; Persian/Arabic pass through
`decima.normalize` before storage. `teacher.bench_throughput` measures aggregate tok/s; `teacher.stats`
prints the histograms and calibration statistics of a dataset.

## Student

`decima/model.py`: the multilingual-e5-small transformer body (fine-tuned fully) + a late-interaction
option scorer: the state is encoded once to token states; each choice (question prepended) is encoded
separately and runs through 2 decoder-style layers (self-attention among choice tokens, cross-attention
over state tokens); scalar score per choice, plus the centred bi-encoder similarity as a skip so training
starts at the zero-shot baseline. Choices never share a context window, so the choice count is unbounded
and both encodings are cacheable. Heads: softmax (choose/verify), cumulative-link ordinal with thresholds
derived from the level texts (score), per-choice sigmoid (rank). Loss: soft cross-entropy to the teacher
distribution (= KL up to the teacher's entropy) + 0.3 · NLL on gold; temperature fitted afterwards on a
5 % hold-out of the teacher data.

Parameter counts (exact): transformer body **21,639,552** + scorer/heads **4,735,109** = **26.4M**
trainable outside the vocabulary table; the 96,014,208-parameter vocab table is a lookup, not compute,
and is not counted in the "~30M" of the target claim.

```bash
uv run python -m train.train --data "data/teacher/*.jsonl" --out checkpoints/v0 --epochs 3
uv run python -m bench.run --system decima --checkpoint checkpoints/v0/best --no-latency
uv run python -m bench.compare runs/baseline-*.json -- runs/<decima-run>.json
uv run python -m decima.export checkpoints/v0/best --out export/v0      # encoder.onnx + scorer.onnx
```

## Results — V0 (measured 2026-09-23, 1000 examples/task, no bench split used for weights or temperature)

Teacher data: 303,114 examples (87k invented, 216k labeled) from Gemma-4-26B in ~24 h wall time on
the GX10 (workers at 32 streams). Student: 3 epochs, 2 h 43 min, bf16; hold-out (5 % of teacher
data, 14,960 examples) accuracy 0.710 (choose 0.728 · score 0.587 · verify 0.853 · rank 0.657);
temperature 0.95 fitted on that hold-out.

| task          | kind   | choices | baseline acc | **V0 acc** | V0 ECE (cal) |
|---------------|--------|--------:|-------------:|-----------:|-------------:|
| sst5/en       | score  |       5 | 0.270 | **0.484** | 0.045 |
| agnews/en     | choose |       4 | 0.826 | **0.872** | 0.037 |
| xnli/en       | choose |       3 | 0.394 | **0.639** | 0.070 |
| xnli/ar       | choose |       3 | 0.374 | **0.574** | 0.062 |
| xnli/ru       | choose |       3 | 0.396 | **0.584** | 0.077 |
| farstail/fa   | verify |       2 | 0.486 | **0.695** | 0.021 |
| massive/en    | choose |      60 | 0.488 | **0.672** | 0.042 |
| massive/fa    | choose |      60 | 0.348 | **0.635** | 0.053 |
| massive/ar    | choose |      60 | 0.236 | **0.451** | 0.075 |
| massive/ru    | choose |      60 | 0.402 | **0.591** | 0.041 |
| banking77/en  | choose |      77 | 0.526 | **0.622** | 0.041 |
| clinc150/en   | choose |     151 | 0.506 | **0.573** | 0.059 |
| **mean**      |        |         | 0.438 | **0.616** | 0.052 |

Baseline = zero-shot bi-encoder (`runs/baseline-e5-small-500-part*.json`); V0 = `runs/v0-decima-1000.json`.
banking77 sits between Laya (0.425) and Jev (0.870); per-task ECE after temperature scaling is 0.021–0.077
(Laya: 0.081). CPU latency for V0 is not yet measured — `export/v0/{encoder,scorer}.onnx` are the fp32
graphs to quantise and time on x86.

## Layout

```
decima/         the model side: types · normalize (fa/ar script) · calibration · baseline · model · export
teacher/        the teacher side: client · prompts · generate · label · sources · stats · bench_throughput
train/          data packing (by choice budget) · training loop → checkpoints/ (git-ignored)
bench/          the bench: datasets (unified loaders) · metrics · run · compare
runs/           benchmark results (json) and train logs, git-ignored except *.json
```

## Training box (GX10 / DGX Spark class, ARM64 + Blackwell)

The bench and baseline run on any CPU. Teacher serving and student training run on the GX10.

```bash
# on the GX10
git clone https://github.com/amyrmahdy/decima.git && cd decima
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is missing
uv sync                                                   # on aarch64 this resolves the CUDA torch build
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -m bench.run --limit 200                    # sanity: same numbers as the laptop, modulo latency
```

The torch source is selected by machine in `pyproject.toml`: x86_64 gets CPU wheels, aarch64 gets
CUDA 12.8 wheels. CPU latency numbers for the model card are always taken on x86, never on the Grace CPU.
