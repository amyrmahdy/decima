---
pretty_name: Decima Bench Predictions
license: cc-by-4.0
language:
- en
- fa
- ar
- ru
- multilingual
task_categories:
- text-classification
- zero-shot-classification
size_categories:
- 1M<n<10M
tags:
- evaluation
- reproducibility
- predictions
- calibration
- benchmark
---

# Decima Bench Predictions

This is the reproducibility package for every number in the
[Decima-small](https://huggingface.co/amyrmahdy/decima-small) model card and technical report.
It contains **each system's predicted probability vector for each evaluation item**, the scores computed
from those predictions, and pointers to the code that rebuilds the items and scores them.

**No evaluation text is redistributed here.** An item is identified by a stable `id`, and its state,
question, choices and gold label are **rebuilt locally from the original public datasets** by the
loaders in `bench/`. Several source datasets are non-commercial (XNLI, ANLI), share-alike (BoolQ), or have no
licence (AG News, SST-5, Yelp) — see `release/LICENSING.md`. Shipping only ids and numbers means we
never re-license anyone's text, and anyone can still re-score, compare or plot every system without
running a model.

## Contents

```
preds/<set>--<system>.parquet      id: str, probs: list<float64>  — aligned with the item's choices
preds-x86/<set>--v1i-x86-{fp32,int8}.parquet   ONNX on x86 CPU (int8 = shipped runtime)
items/<set>.manifest.parquet       id, suite, split, lang, kind, n_choices, group, perm, sha256
                                   — NO text and NO gold; sha256 = sha256(json [state, question, choices,
                                   gold]) so a rebuilt item can be checked against ours
results/<system>-<set>.json        bench.score / bench.jdi_index output (per-suite acc, macro-F1, Brier,
                                   NLL, ECE, ECE after temperature, flip rate, T) + the exact inputs used
results/phase1*-summary.md, results/latency-*-gx10-indicative.json, results/x86/*.json
audit/*.json                       overlap, bootstrap CIs, calibration, parameter counts
determinism.txt                    Decima-small's release re-run vs its development run (max |Δp| per set)
code-ref.json                      repository, tag, commit, item-builder commands, JevBench / JDI kit
                                   commits, and the resolved Hub revision of every source dataset
files.json                         sha256 and size of every file
```

The package is built by `release/hf/datasets/build_predictions.py` in the code repository. It never
copies item text: manifests are derived from the local item files by keeping only identifiers and shape
and adding the hash. Decima-small's predictions on the kev, laya, decima, jevtyped and btzsc sets are a
clean re-run in fresh processes on the frozen items; `determinism.txt` records the largest probability
difference against the development run that selected the checkpoint. Everything else (ablations,
competitors, JDI) is the development run, unchanged. `code-ref.json` says which files the re-run replaced.

### Item sets

| set | suites | items (incl. flip copies + calib) | source (eval split) | protocol |
|---|---|---:|---|---|
| `kev` | 8 | 4,800 | banking77, BoolQ, AG News, MNLI, SST-5, Yelp full (+ yes/no variants) | Kev's evaluation reproduced item for item (`bench/suites_kev.py`) |
| `laya` | 29 | 26,100 | MASSIVE test (14 langs), XNLI test (15 langs) | Laya's protocol: first 300 test rows, 19 seeded distractors (`bench/suites_laya.py`) |
| `decima` | 12 | 29,000 | SST-5, AG News, XNLI en/ar/ru, FarsTail, MASSIVE en/fa/ar/ru, banking77, CLINC150 | full label sets, 1,000 eval + 500 calib per suite (`bench/datasets.py`) |
| `jevtyped` | 2 | 4,644 | JevBench v1.4 public (repo @2fa63fa), LocalLLaMA/typed-decisions test | `bench/suites_public.py` |
| `btzsc` | 22 | 42,448 | btzsc/btzsc test | the harness's own sample, ≤ 1,000 per dataset |
| `jdi-all` | 25 (19 panel benchmarks) | 96,054 | Jev Decision Index 0.1, rebuilt from pinned sources (kit apolinario/decision-index @52a6989) | `bench/jdi_index.py` |
| `jdi` / `jdi-rest` | panel split in two (V0 only) | 75,854 / 20,200 | as above | as above |

Every non-`score` eval item (except in JDI) also has a `#flip` copy with its choices permuted by a seed. The copy is
used to measure order sensitivity (flip rate). The `calib` items are used only to fit a temperature
for `ece_cal`.

### Systems × sets (prediction files that exist)

| system | what | kev | laya | decima | jevtyped | btzsc | jdi-all |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **v1i** | **Decima-small (released)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| v0 | Decima V0 (teacher data only) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (+ `jdi`, `jdi-rest`) |
| v1a, v1a2, v1b, v1c, v1e, v1f, v1g, v1h, v1j | Phase-1 runs (data/curriculum ablations) | ✓ | ✓ | ✓ | ✓ | all but v1a, v1e | v1a2, v1g |
| v2b, v2c | Decima-base candidates (multilingual-e5-base; not released, see the technical report) | ✓ | ✓ | ✓ | ✓ | ✓ | v2b |
| s-a2g, s-bc, s-bh | weight soups (v1a2+v1g, v1b+v1c, v1b+v1h) | ✓ | ✓ | ✓ | ✓ | ✓ | |
| e5 | zero-shot `intfloat/multilingual-e5-small` bi-encoder (T = 0.05) | ✓ | ✓ | ✓ | ✓ | ✓ | |
| kev-0.5b / kev-0.8b | `jaredpalmer/kev-*` (Apache-2.0), run with its own code (`bench/external/kev_predict.py`) | ✓ / ✓ | ✓ / – | – / ✓ | ✓ / ✓ | | |
| laya / laya-multilingual | `convaiinnovations/laya*` (Apache-2.0), run with its own code (`bench/external/laya_predict.py`) | ✓ | ✓ | ✓ | ✓ | | |
| v1i-x86-fp32 / -int8 | ONNX export on x86 CPU | ✓ | ✓ | ✓ | int8 | int8 | |

That is 104 GPU prediction files plus 8 x86 files: 2.58 M prediction rows. Every ablation ships: the
negative results are part of the evidence. As zstd-compressed Parquet the whole package is about 275 MB
(the same predictions are about 1.2 GB as JSONL).

Competitor probabilities are those the competitor's own code produces *as shipped*: Laya with its
shipped temperatures, Kev raw. Before any comparison, both protocols were verified by reproducing the
competitors' published numbers (Laya 0.783 / 0.860, Kev 0.799).

## How to reproduce every table

```bash
git clone https://github.com/amyrmahdy/decima && cd decima && git checkout v1.0.0
uv sync

# 1. rebuild the items from the original datasets (downloads them; nothing comes from this repo)
#   calibration items per suite: kev/laya 300, decima 500 (the default), typed 1000, others none
uv run python -m bench.items --suites 'kev/*'    --calib 300      --out runs/items/kev.jsonl
uv run python -m bench.items --suites 'laya/*'   --calib 300      --out runs/items/laya.jsonl
uv run python -m bench.items --suites 'decima/*' --calib 500      --out runs/items/decima.jsonl
uv run python -m bench.items --suites 'jevbench/public' 'typed/test' --calib 1000 --out runs/items/jevtyped.jsonl
uv run python -m bench.items --suites 'btzsc/*'                   --out runs/items/btzsc.jsonl
uv run python -m bench.items --suites 'jdi/*' --no-flips --calib 0 --limit 1000000000 --out runs/items/jdi.jsonl
#   (jdi-all = the panel subset: see scripts/eval_winner.sh)

# 2. check the rebuilt items against ours (same ids, same sha256 of [state, question, choices, gold])
uv run python -m bench.verify_items --items runs/items/kev.jsonl --manifest items/kev.manifest.parquet
#    exit 0 = identical; otherwise it lists missing / extra / changed ids

# 3. score with the downloaded predictions — e.g. the Kev table
uv run python -m bench.score --items runs/items/kev.jsonl --metric acc \
    --preds v0=preds/kev--v0.parquet v1i=preds/kev--v1i.parquet kev-0.5b=preds/kev--kev-0.5b.parquet \
            kev-0.8b=preds/kev--kev-0.8b.parquet laya=preds/kev--laya.parquet laya-multilingual=preds/kev--laya-multilingual.parquet \
    --out kev.json
#   BTZSC uses --metric macro_f1; JDI uses bench.jdi_index (official = no truncation):
uv run python -m bench.jdi_index --items runs/items/jdi.jsonl --preds v1i-official=preds/jdi-all--v1i.parquet \
    --decima-checkpoint <Decima-small> --budget-systems v1i-official --out jdi.json

# 4. (optional) regenerate our predictions from the released weights
uv run python -m bench.predict --items runs/items/kev.jsonl --system decima --checkpoint <Decima-small> --out my-kev.jsonl
uv run python -m bench.predict --items runs/items/kev.jsonl --system onnx --checkpoint <Decima-small>/onnx/int8 --out my-kev-int8.jsonl
```

Each `results/<system>-<set>.json` records the `items` file and the `preds` map it was computed from
(paths as they were in our checkout: `runs/preds/X.jsonl` is `preds/X.parquet` here; `bench.score` and
`bench.jdi_index` read either format). Rerunning step 3 with the same map reproduces it to the last
floating-point digit (differences ≤ 1e-15 come from summation order in macro-F1, which follows Python's
hash seed; set `PYTHONHASHSEED=0` for bit-identical reruns). The
model-card tables map to results files as follows:

| card / report table | results files |
|---|---|
| Kev protocol | `results/v1i-kev.json` (its `preds` map includes V0, Kev and Laya) |
| Laya protocol | `results/v1i-laya.json` |
| Decima bench | `results/v1i-decima.json` |
| JevBench public + typed-decisions | `results/v1i-jevtyped.json`; 95 % cluster-bootstrap CIs: `audit/bootstrap.json` (`scripts/audit/bootstrap.py`) |
| Overlap audit, headline calibration | `audit/overlap.json`, `audit/calibration.json` (`scripts/audit/overlap.py`, `calibration.py`) |
| Parameter counts (all systems) | `audit/params.json` (`scripts/audit/params.py`) |
| BTZSC (macro-F1, 22 and clean 18) | `results/v1i-btzsc.json` |
| Jev Decision Index 0.1 | `results/v1i-jdi.json`, `results/phase0-jdi.json` |
| Ablations and soups | `results/<run>-<set>.json`, `results/phase1*-summary.md` |
| int8 vs fp32, x86 | `results/x86/compare-v1i.json` |

Latency and speed numbers are hardware measurements, not predictions. They live in
`results/latency-*-gx10-indicative.json` and `results/x86/speed-*.json` and are reproduced with `bench/latency.py` /
`bench/speed.py` on your own hardware.

### Caveats for reproduction

- **Dataset revisions.** The loaders call `load_dataset` without `revision=`, so a dataset that
  changes on the Hub changes the items. `code-ref.json` records the Hub revision of every source
  dataset as resolved when our items were built (plus the JevBench and JDI-kit commits, which the
  loaders already pin). `bench.verify_items` detects drift; if it reports differences, load the
  recorded revision (`load_dataset(..., revision=...)`) for that source.
- Some suites use splits that overlap with Decima's training data (AG News and banking77 calib items
  come from train; the Kev calib items come from train). The model card's zero-shot vs
  in-distribution table says which ones.
- GPU predictions are from an NVIDIA GB10 (PyTorch). Other hardware may differ in the last digits. Top-1 ties are rare but possible.

## Licence

- Predictions, scores and manifests: **CC BY 4.0** (the same licence as the synthetic dataset; owner
  decision 2026-09-26). They are our measurements.
- Competitor predictions are outputs of Apache-2.0 models run with their authors' code.
- The evaluation datasets keep their own licences. They are **not** included, and you download them
  from their original sources when you rebuild the items. Item ids from JDI/JevBench reuse upstream
  case identifiers (for example, BFCL function names). These are identifiers, not content.

## Citation

```bibtex
@misc{decima2026bench,
  title  = {Decima Bench Predictions},
  author = {Madani, A. M.},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/amyrmahdy/decima-bench-predictions}}
}
```
