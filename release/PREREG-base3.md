# base3: licence-clean and BTZSC-zero-shot (protocol fixed before training, 2026-10-10)

Written before the run starts, so the claim can be checked against what was planned.

**Why.** Decima-small's BTZSC submission was not zero-shot in the strict sense: four BTZSC source datasets (AG News,
banking77, MASSIVE, SST/Rotten Tomatoes) were in its training data, and BTZSC scores were looked at when choosing
between training runs. The benchmark's author asked for a version where BTZSC is not used for training, development
or model selection. base3 is that version, and it is also free of non-commercial training data.

**Training data** (`data/clean/` + `data/teacher/generate-*.jsonl`):

- removed for licence: ANLI, XNLI, multilingual-NLI-26lang, AG News, SST-5, and teacher labels on their text;
- removed for BTZSC: banking77 and MASSIVE (gold rows and teacher labels on their text), and 6 CLINC150 utterances whose
  text is identical to a BTZSC MASSIVE item (exact match on the first 200 normalised characters);
- check: no training state matches any of the 20,776 BTZSC item texts (same rule) after removal.

**Model selection.** From jhu-clsp/mmBERT-base, one run, fixed hyperparameters (`scripts/phase_base3.sh`). The released
checkpoint is the one with the best accuracy on the run's own random training holdout. No BTZSC number is computed
before that choice is final.

**BTZSC evaluation.** Run once on the chosen checkpoint (int8 ONNX, the benchmark's 22 datasets), reported as it comes,
together with this file. No re-runs, threshold tuning or prompt changes after seeing the score.
