# Decima-small evaluation (checkpoint `checkpoints/v1i/best`)

Frozen 2026-09-26. Every number here is taken from files in `runs/`, or measured for this document
with the committed scripts named in the text (`scripts/audit/{overlap,bootstrap,calibration}.py` →
`runs/audit/*.json`). Decima numbers come from the fp32 PyTorch checkpoint on CUDA
(`bench.predict --device cuda`). They carry over to the shipped int8 ONNX export: on 45,850 real eval
items int8 matches fp32's top answer on 98.6–99.8 % per set, and every set-level score moves by ≤ 0.004
(docs/BENCH-x86.md §1).

Short version:
- On Kev's protocol, Decima is in-distribution except on Yelp (zero-shot); Kev-0.5B is in-distribution; Kev-0.8B's training data is not auditable.
- On Laya's MASSIVE/XNLI protocol, Decima is in-distribution. Laya says it is not; we have not verified that.
- On BTZSC, 18 datasets are zero-shot. rottentomatoes is contaminated (33% measured). banking77, massive and agnews are in-distribution.
- JevBench (public items), typed-decisions and the JDI panel are zero-shot for Decima by data. The Jev-style synthetic data targeted JevBench's format, and the public JevBench items were used for checkpoint selection.

All eval sets were also used to **select** v1i among about 15 candidates (`scripts/pick_best.py`, see §5).

---

## 1. Training data provenance of v1i

### 1a. Lineage (from the `start` events in `runs/train-*.jsonl` and `runs/logs/auto.log`)

Backbone: `intfloat/multilingual-e5-small` (rev 614241f). Its own pre-training and fine-tuning data were
not audited. The mE5 report lists NLI among its fine-tuning sources.

| run | init | `--data` (teacher = `data/teacher/generate-*.jsonl` + `data/teacher/label-*.jsonl`) | epochs | n_train | started |
|---|---|---|---:|---:|---|
| V0 | e5-small | teacher | 3.0 | 288,154 | 09-23 12:13 |
| v1a | V0 | teacher + `data/gold` | 1.0 | 667,483 | 09-23 20:51 |
| v1a2 | v1a | teacher + `data/gold` | 0.3 | 667,483 | 09-23 22:37 |
| v1e | v1a2 | teacher + `data/gold-lite` (25 % subset of gold) | 0.4 | 382,960 | 09-24 03:18 |
| v1f | v1e | teacher + `data/mix-f` = gold-lite + `data/gold-cls` | 0.5 | 666,515 | 09-24 04:25 |
| v1g | v1f | teacher + `data/mix-g` = gold + gold-cls | 0.5 | 951,038 | 09-24 11:24 |
| **v1i** | v1g | teacher + `data/mix-i` = gold + `data/gold-cls-lite` (1/3 subset of gold-cls) + `data/jevgen-snap` ×3 | 0.5 | 779,123 | 09-25 05:04 |

- Every data file is older than the run that read it, so the files on disk are the files that were trained on.
- gold-lite ⊂ gold (99,814/99,814 ids) and gold-cls-lite ⊂ gold-cls (99,467/99,467 ids).
- n_train is after the trainer's 5 % holdout.
- Taken together, v1i's weights have seen: teacher generate + teacher label + gold + gold-cls + jevgen-snap.

### 1b. Sources

**How each row type is labelled:**
- **Teacher-labelled** rows: Gemma-4-26B-A4B invents a question and choice set and gives a probability distribution. The dataset's own label is not used.
- **Gold** rows: the dataset's human (or machine-translated human) label, smoothed to 0.92.

**Decontamination:**
- gold and gold-cls were decontaminated against every `runs/items/*.jsonl` when they were built. The method is `teacher.gold.Contam`: exact segment match or a shared 200-character prefix. The "dropped" counts below come from `data/gold*/stats.json`.
- **The teacher label file was not decontaminated.** It was written at 09-23 12:04, before any item file existed (the first appeared at 17:17). It uses train splits only, and it is the source of every measured overlap in §2 (BTZSC Rotten Tomatoes 33 %, MASSIVE up to 11.7 % per language).

| source dataset (HF id) | split | how used | languages | rows (distinct texts) | dropped as eval-overlapping |
|---|---|---|---|---:|---:|
| mteb/banking77 | train | teacher-labelled | en | 11,339 (5,996) | not checked |
| mteb/banking77 | train | gold (gold-cls) | en | 42,000 (9,195) | 797 |
| clinc/clinc_oos `plus` | train | teacher-labelled | en | 11,275 (5,999) | not checked |
| clinc/clinc_oos `plus` | train | gold (gold-cls) | en | 42,000 (15,131) | 112 |
| mteb/amazon_massive_intent | train | teacher-labelled | en fa ar ru | 44,786 (23,435) | not checked |
| mteb/amazon_massive_intent | train | gold (gold-cls) | all 51 locales (en 20k; fa/ar/ru 15k each; 11 Laya locales 2k; 36 others 1.4k rows) | 137,400 (116,479) | 2,079 |
| fancyzhx/ag_news | train | teacher-labelled | en | 11,371 (6,000) | not checked |
| fancyzhx/ag_news (+ sh0416 titles) | train | gold (gold-cls) | en | 45,000 (45,000) | 868 |
| SetFit/sst5 | train | teacher-labelled | en | 11,374 (5,993) | **not checked** |
| SetFit/sst5 | train | gold (gold-cls) | en | 32,000 (8,038) | 498 |
| facebook/xnli | train | teacher-labelled | en ar ru | 33,642 (17,995) | not checked |
| facebook/xnli | train (MT of MNLI train) | gold | ar ru (26k each); de fr es tr zh hi vi bg el sw ur th (3k each) | 88,000 | 243 |
| nyu-mll/multi_nli | train | gold | en | 66,000 | 1,047 |
| stanfordnlp/snli | train | gold | en | 20,000 | 137 |
| facebook/anli | train_r1–r3 | gold | en | 30,000 | 38,910 |
| alisawuffles/WANLI | train | gold | en | 25,000 | 217 |
| google/boolq | train | gold | en | 17,752 (8,891 passages) | 536 |
| azarijafari/FarsTail | Train-word.csv | gold | fa | 14,504 (7,252 pairs) | 1 |
| MoritzLaurer/multilingual-NLI-26lang-2mil7 | MT of mnli/anli/fever/wanli/ling **train** | gold | fa 42k; ar, ru 30k; de fr es tr zh hi uk pl it pt ja ko id he vi 2.4k | 138,000 | ≈ 6.1k per language |
| mteb/FarsTail | (listed in `teacher/sources.py`, but that repo only ships test) | none (0 rows in the label file) | — | 0 | — |
| teacher.generate (synthetic states) | — | teacher-generated | en fa ar ru (30/25/25/20) | 86,900 | not checked |
| own generated states, re-asked | — | teacher-labelled | en fa ar ru | 92,427 (75,804) | not checked |
| teacher.generate_jev (`data/jevgen-snap`, only in v1i, ×3) | — | teacher-generated, long "Jev-style" states | en 70 / fa ar ru 10 each | 6,004 (1,570 states) | not checked |

---

## 2. Status of each benchmark (measured)

### 2a. Method

Script: `scripts/audit/overlap.py` → `runs/audit/overlap.json` (the only overlap measure release
documents quote). The comparison has two sides:
- **Eval side:** every eval `state` (split = eval, option-order copies excluded), 73 suites.
- **Training side:** every training `state` in all the files in §1 (teacher invented, teacher relabelled,
  Jev-style, gold NLI/BoolQ, gold-cls): 1.02 M rows, 722,128 distinct states.

**Match rule:** the whole state is identical after unwrapping JSON states (Laya's protocol), lower-casing
and collapsing whitespace; states shorter than 8 characters are ignored. The script does not fold
Arabic/Persian letter forms or match segments, so near-duplicates (a shared line, a different diacritic)
are not counted. For every suite with any overlap it also recomputes each system's accuracy on the
non-overlapping items.

JDI items are not in the committed audit.

### 2b. Classification and measured overlap

Labels used:
- **ZS** = zero-shot (no training data from that dataset).
- **ID** = in-distribution (trained on that dataset's train split; test rows differ).
- **CONT** = contaminated (eval texts are in training).

| item set / suite | Decima-small (v1i) | exact whole-state overlap (`overlap.json`) | Kev-0.5B / 0.8B | Laya / Laya-ML |
|---|---|---|---|---|
| **kev/** banking77, agnews, agnews-yn, sst5 | ID | 0 % | Kev-0.5B ID (trained on the train splits of its 6 sources); Kev-0.8B not auditable | agnews: ID ("in training mix"); banking77: ID status unknown; sst5: ZS ("held out") |
| kev/ boolq, mnli | ID (BoolQ train; MNLI train + XNLI/nli26 translations of it) | 0 % | Kev-0.5B ID; Kev-0.8B not auditable | boolq: ID ("in training mix"); mnli: not stated |
| kev/ yelp, yelp-yn | **ZS** (no Yelp data) | 0 % | Kev-0.5B ID; Kev-0.8B not auditable | not stated |
| **laya/** massive-* (14 langs) | ID (MASSIVE train, all 51 locales) | **ru 11.7 %, ar 6.7 %, en 1.3 %**, other 11: 0 % | not stated | Laya states it did not train on MASSIVE (unverified) |
| laya/ xnli-* (15 langs) | ID (MNLI train for en, XNLI train for the other 14) | 0 % | mnli ID; XNLI not stated | Laya states it did not train on XNLI (unverified) |
| **decima/** all 12 suites | ID (the train split of every one of these datasets is in §1) | massive ru 5.0 %, fa 4.7 %, ar 4.7 %, en 0.6 %; clinc150 0.3 %; banking77 0.2 %; sst5 0.1 %; others 0 % | agnews/sst5/banking77 ID; rest not stated | agnews ID; rest not stated |
| **jevbench/public** (231) | **ZS** by data, but used for model selection, and the jevgen data was built to target this style | 0 % | not stated | not stated |
| **typed/test** (2,000) | **ZS** (train split used only as calibration items for `ece_cal`) | 0 % | not stated | ZS (base checkpoints) |
| **btzsc/** 18 clean datasets | **ZS** | 0 % | not run | not run |
| btzsc/ rottentomatoes | **CONT** | **33.0 %** | not run | not run |
| btzsc/ banking77, massive, agnews | ID (the same test splits as decima/*) | 0.1 / 1.1 / 0 % | not run | agnews ID |
| **jdi/** 19 panel benchmarks | **ZS** | not in the committed audit | board reference | board reference |
| jdi/ banking77, clinc150 (display-only) | ID | not in the committed audit | — | — |
| jdi/ anli (display-only) | **ID** (ANLI train r1–r3 is in gold) | not in the committed audit | — | — |
| other jdi display-only | ZS | not in the committed audit | — | — |

Summary: exact whole-state overlap is 0 % on 60 of 73 suites and ≤ 0.3 % on every suite except MASSIVE
(0.6–11.7 % per language) and BTZSC Rotten Tomatoes (33.0 %).

**Findings:**

- **rottentomatoes is still contaminated.** All 330 whole-state matches come from the **teacher-label file** (the SST-5 train states it relabelled; Rotten Tomatoes shares the SST-5 corpus). gold-cls SST-5 contributes 0 matches, because its decontamination (498 texts dropped) worked. The teacher label file was never decontaminated, so the BTZSC clean-18 exclusion is required.
- **MASSIVE overlap comes from upstream duplicates.** MASSIVE's train and test splits share identical utterances, for example «какая сегодня дата» ("what is today's date") and «ما هو الطقس هذا الاسبوع» ("what is the weather this week"). The undecontaminated teacher label file carries them into training. Every match in `overlap.json` is attributed to the teacher-relabelled file.
- **Removing overlapping items barely changes the results.** The table below compares accuracy on all items with accuracy on the non-overlapping items only (`overlap.json`, `accuracy_all_vs_clean`). v1i's largest change on any suite is −0.014 (laya/massive-ru); competitors move by similar amounts.

| suite | overlapping items | v1i all → clean | largest competitor change |
|---|---:|---|---|
| laya/massive-ru | 35/300 | 0.893 → 0.879 | kev-0.5b 0.620 → 0.600 |
| laya/massive-ar | 20/300 | 0.867 → 0.864 | laya 0.127 → 0.118 |
| decima/massive/ru | 50/1000 | 0.837 → 0.832 | kev-0.8b 0.565 → 0.548 |
| decima/massive/fa | 47/1000 | 0.846 → 0.845 | kev-0.8b 0.567 → 0.560 |
| decima/massive/ar | 47/1000 | 0.760 → 0.757 | laya-ML 0.282 → 0.284 |
| btzsc/rottentomatoes (accuracy; macro-F1 not recomputed) | 330/1000 | 0.857 → 0.851 | V0 0.846 → 0.827 |

**Competitors' training data:**
- Kev-0.5B trained on the train splits of its six sources: 1,500 rows each, with the exact templates of its eval (`bench/suites_kev.py`). Its published eval is therefore in-distribution.
- Kev-0.8B's training suite (`decision-v7`, `public_frac 1.0`, per its `training_config.json`) is not auditable from the release.
- Laya's GitHub README marks AG News and BoolQ as "in training mix", and SST-5 and DAIR Emotion as "held out". Laya states it did not train on MASSIVE or XNLI; we have not verified this.

---

## 3. Protocol per benchmark

**Calibration (`bench/score.py`), same procedure for every system:**
- ECE uses 15 equal-width bins on top-1 confidence.
- **ece** is computed on the probabilities as shipped:
  - Decima: its shipped temperature, 0.973.
  - Laya: its shipped temperatures. Laya-multilingual ships 1.0.
  - Kev: raw T = 1. Kev's published numbers are raw.
- **ece_cal**: one temperature per suite, fitted on that suite's *calib* items, then applied to the eval items. Suites with no calib split (BTZSC, JevBench, JDI) get no ece_cal.
- **flip** = the share of items whose answer changes when the same choices are presented in a different order. Ordered (score) suites have no flip copy.

| set | dataset (HF id), eval split | eval items | choices | langs | format | sampling | metric | calib items (for ece_cal) |
|---|---|---:|---:|---|---|---|---|---|
| **kev/** (8 suites) | legacy-datasets/banking77 test; google/boolq validation; fancyzhx/ag_news test; nyu-mll/multi_nli validation_matched; SetFit/sst5 test; Yelp/yelp_review_full test | 1,350 (150/source; agnews-yn 300) | 77 / 2 / 4 / 3 / 5 / 5 / 2 / 2 | en | Kev's exact rendered strings (state wrappers, "key: description" options, yes/no noul) | **Kev's protocol reproduced item for item** (`kev.evaluate --n_per_source 150 --seed 1`; every RNG call replayed; `bench/suites_kev.py`) | accuracy; Kev's "all" = pooled over 1,350 | 300/suite. SST-5 validation, MNLI validation_mismatched. banking77/boolq/agnews/yelp come from **train minus Kev's 1,500 training rows**, which can include rows Decima trained on. |
| **laya/** (29 suites) | mteb/amazon_massive_intent test (14 langs); facebook/xnli test (15 langs) | 8,700 (300/suite) | 20 (MASSIVE), 3 (XNLI) | en de fr es pt ru tr ar hi ta zh ja ko sw (+ ur vi th el bg for XNLI) | state `{"utterance": …}` / `{"premise","hypothesis"}` JSON; options "key: description" | **Laya's protocol reproduced item for item** (published numbers matched to within 0.001): first 300 test rows in file order; MASSIVE options = gold + 19 distractors from `random.Random(13)` over that language's test label set, shuffled (`bench/suites_laya.py`) | accuracy; "other languages" = unweighted mean of non-en suites | 300/suite from the validation split, built the same way. For MASSIVE, 1–13 calib states per suite also occur verbatim in test (upstream duplicates; exact match after lower-casing). |
| **decima/** (12 suites) | SetFit/sst5 test; ag_news test; facebook/xnli test en/ar/ru; mteb/FarsTail test; MASSIVE test en/fa/ar/ru (English intent names for every language); mteb/banking77 test; clinc_oos plus test | 12,000 (1,000/suite, seeded shuffle) | 5 / 4 / 3 / 2 / 60 / 77 / 151 | en fa ar ru | our own: full label set, natural-language label names | seed 0 shuffle, first 1,000 (`bench/datasets.py`) | accuracy | 500/suite. SST-5, XNLI, MASSIVE, CLINC: validation. AG News and banking77: **train** (the same split Decima trained on). **FarsTail: test, and 490/500 calib states are also eval states**, so FarsTail ece_cal is in-sample. |
| **jevbench/public** | JevBench v1.4 public files, repo pinned at 2fa63fa | 231 (easy 48, standard 72, hard 111) | 2–6 | en | noul → verify "no/yes: criterion"; choice / score "label: criterion" | all 231 public items. **This is not the official score**, which needs the 308 sealed and 146 judge items and speed/cost components. | accuracy (+ tier accuracy) | none |
| **typed/test** | LocalLLaMA/typed-decisions test | 2,000 decisions (400 cases × 5 questions) | 2–5 | en | state = compact JSON; question = instructions | whole test split | accuracy vs the discrete teacher label (not correctness) | 1,000 items from the **train** split (a light adaptation to the 4 workflows) |
| **btzsc/** (22) | btzsc/btzsc test | 21,569 (≤ 1,000/dataset) | 2–72 | en | text + the dataset's hypotheses verbatim as choices; one neutral question for Decima | the harness's own `max_samples` draw (groups 0, 1 + `random.Random(0).sample`), 1,000 per dataset; 200 banking77 and 169 massive unlabelled groups skipped (`bench/suites_public.py`) | macro-F1 (primary), mean over 22 and over the clean 18 | none |
| **jdi/** (36 benchmarks, 19 panel) | rebuilt from pinned public sources with the kit's normalizers, byte-identical to the kit manifest; kit 52a6989 | 481,586 items; panel 96,054 predicted | 2–255 | en (+ ar, es, ja tracks) | JDI request → items; empty-state requests move the instruction into the state (`JDI_MOVE_INPUT`) | every selected case (`bench/jdi_index.py`) | Decision Index 0.1 `balanced_raw` (5 area means × 100). **Official = no truncation**: a request over Decima's 512 state / 64 choice token budget is "unsupported" and scores as wrong. "Truncated" = Decima truncates silently (not JDI-legal). | none |

---

## 4. Results

### 4a. Per-set summary (means of per-suite values)

In the table below, a dash (—) means that system was not run on that set. The laya and decima rows are
in-distribution for Decima (Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya reports it did
not). Only the as-shipped ece is bolded; ece_cal (refitted) is reported neutrally.

| set | metric | V0 | **v1i** | kev-0.5b | kev-0.8b | laya | laya-ML |
|---|---|---:|---:|---:|---:|---:|---:|
| kev (8) | acc | 0.629 | **0.768** | 0.779 | 0.794 | 0.681 | 0.580 |
| | acc, Kev "all" (pooled 1,350) | 0.644 | 0.790 | 0.799 | 0.812 | 0.710 | 0.610 |
| | ece / ece_cal | 0.115 / 0.069 | **0.072** / 0.058 | 0.087 / 0.063 | 0.135 / 0.059 | 0.170 / 0.088 | 0.280 / 0.077 |
| | flip | 0.007 | 0.000 | 0.014 | 0.016 | 0.076 | 0.080 |
| laya (29) | acc | 0.593 | **0.764** | 0.527 | — | 0.445 | 0.607 |
| | ece / ece_cal | 0.087 / 0.066 | **0.056** / 0.056 | 0.121 / 0.080 | — | 0.422 / 0.067 | 0.259 / 0.077 |
| | flip | 0.000 | 0.000 | 0.261 | — | 0.230 | 0.184 |
| decima (12) | acc | 0.616 | **0.785** | — | 0.661 | 0.439 | 0.554 |
| | ece / ece_cal | 0.095 / 0.052 | **0.031** / 0.032 | — | 0.162 / 0.050 | 0.398 / 0.047 | 0.196 / 0.065 |
| | flip | 0.000 | 0.000 | — | 0.151 | 0.489 | 0.365 |
| jevtyped (2) | acc | 0.448 | 0.485 | 0.464 | **0.548** | 0.458 | 0.426 |
| | ece | 0.158 | **0.130** | 0.174 | 0.208 | 0.172 | 0.286 |
| | flip | 0.000 | 0.000 | 0.128 | 0.077 | 0.119 | 0.116 |

### 4b. Kev protocol (`runs/v1i-kev.json`)

| suite | V0 | v1i | kev-0.5b | kev-0.8b | laya | laya-ML | v1i ece | v1i ece_cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 0.627 | **0.940** | 0.860 | 0.813 | 0.473 | 0.380 | 0.094 | 0.060 |
| agnews | 0.920 | 0.940 | 0.940 | 0.900 | 0.933 | **0.947** | 0.041 | 0.018 |
| boolq | 0.553 | 0.720 | 0.753 | **0.807** | 0.740 | 0.660 | 0.090 | 0.089 |
| mnli | 0.347 | 0.767 | 0.747 | **0.800** | 0.613 | 0.613 | 0.056 | 0.056 |
| sst5 | 0.420 | 0.487 | **0.533** | 0.507 | 0.320 | 0.300 | 0.104 | 0.094 |
| yelp (ZS for Decima) | 0.520 | 0.487 | 0.553 | **0.640** | 0.580 | 0.267 | 0.056 | 0.061 |
| agnews-yn | 0.770 | **0.960** | **0.960** | 0.953 | 0.937 | 0.843 | 0.052 | 0.026 |
| yelp-yn (ZS for Decima) | 0.873 | 0.847 | 0.887 | **0.933** | 0.853 | 0.633 | 0.080 | 0.058 |
| six-source mean | 0.564 | 0.723 | 0.731 | **0.744** | 0.610 | 0.528 | | |

kev-0.5b reproduces Kev's published numbers exactly: 0.860 / 0.940 / 0.753 / 0.747 / 0.533 / 0.553 / 0.960 / 0.887, "all" 0.799.

Competitor ECE on this set is in `runs/v1i-kev.json`. kev-0.5b raw ECE is 0.013–0.136 per suite; v1i has the lowest raw-ECE mean.

### 4c. Laya protocol (`runs/v1i-laya.json`)

In-distribution for Decima: Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya reports it did not.

| group | V0 | v1i | kev-0.5b | laya | laya-ML |
|---|---:|---:|---:|---:|---:|
| MASSIVE en, acc | 0.820 | **0.913** | 0.753 | 0.783 | 0.657 |
| MASSIVE 13 other langs, mean acc | 0.686 | **0.852** | 0.426 | 0.306 | 0.451 |
| XNLI en, acc | 0.583 | 0.767 | 0.747 | **0.860** | 0.843 |
| XNLI 14 other langs, mean acc | 0.490 | 0.671 | 0.589 | 0.520 | **0.731** |
| MASSIVE non-en, ece / ece_cal | 0.113 / 0.072 | 0.048 / 0.046 | 0.165 / 0.061 | 0.659 / 0.070 | 0.351 / 0.094 |
| XNLI non-en, ece / ece_cal | 0.060 / 0.063 | 0.065 / 0.068 | 0.084 / 0.095 | 0.242 / 0.063 | 0.188 / 0.062 |
| MASSIVE non-en, flip | 0 | 0 | 0.414 | 0.436 | 0.369 |

Our Laya runs reproduce Laya's published numbers: 0.783 / 0.306 / 0.860 / 0.521 (ours 0.520), and for laya-ML 0.657 / 0.451 / 0.843 / 0.731.

Weakest v1i MASSIVE language: sw 0.720. Weakest v1i XNLI language: hi 0.600.

### 4d. Decima bench (`runs/v1i-decima.json`; every suite is ID for Decima)

| suite | V0 | v1i | kev-0.8b | laya | laya-ML | v1i ece | v1i ece_cal |
|---|---:|---:|---:|---:|---:|---:|---:|
| sst5/en | 0.484 | 0.514 | **0.539** | 0.305 | 0.268 | 0.052 | 0.035 |
| agnews/en | 0.872 | 0.904 | 0.880 | **0.922** | **0.922** | 0.022 | 0.043 |
| xnli/en | 0.639 | 0.749 | 0.797 | **0.870** | 0.828 | 0.051 | 0.049 |
| xnli/ar | 0.574 | 0.659 | 0.656 | 0.453 | **0.687** | 0.010 | 0.012 |
| xnli/ru | 0.584 | 0.689 | 0.677 | 0.593 | **0.700** | 0.028 | 0.040 |
| farstail/fa | 0.695 | 0.819 | 0.780 | 0.554 | **0.836** | 0.018 | 0.015* |
| massive/en | 0.672 | **0.876** | 0.609 | 0.458 | 0.473 | 0.029 | 0.028 |
| massive/fa | 0.635 | **0.846** | 0.567 | 0.038 | 0.313 | 0.014 | 0.032 |
| massive/ar | 0.451 | **0.760** | 0.425 | 0.078 | 0.282 | 0.029 | 0.040 |
| massive/ru | 0.591 | **0.837** | 0.565 | 0.185 | 0.389 | 0.028 | 0.032 |
| banking77/en | 0.622 | **0.887** | 0.819 | 0.378 | 0.424 | 0.043 | 0.018 |
| clinc150/en | 0.573 | **0.875** | 0.621 | 0.434 | 0.530 | 0.044 | 0.045 |
| mean | 0.616 | **0.785** | 0.661 | 0.439 | 0.554 | 0.031 | 0.032 |

\* FarsTail ece_cal is fitted on calib items, 490 of 500 of which are eval items (§3). Laya's and Laya-ML's flip on FarsTail is 1.000: they always pick the same position regardless of order.

### 4e. JevBench public and typed-decisions, with 95 % cluster-bootstrap CIs

Script: `scripts/audit/bootstrap.py` → `runs/audit/bootstrap.json`. Clusters are distinct states; 1,000 resamples, seed 0.
- JevBench public has 231 distinct states, so each cluster is one item.
- typed/test has 400 clusters (the cases).

| system | JevBench public acc [95 % CI] | easy / standard / hard | typed/test acc [95 % CI] | typed ece (raw / cal) |
|---|---|---|---|---|
| V0 | 0.506 [0.446, 0.567] | 0.896 / 0.444 / 0.378 | 0.391 [0.365, 0.417] | 0.177 / 0.065 |
| **v1i** | **0.571 [0.511, 0.636]** | 0.979 / 0.458 / **0.468** | 0.399 [0.371, 0.423] | 0.147 / 0.075 |
| kev-0.5b | 0.506 [0.442, 0.571] | 0.958 / 0.514 / 0.306 | 0.421 [0.397, 0.445] | 0.133 / 0.031 |
| kev-0.8b | **0.645 [0.584, 0.710]** | 1.000 / 0.750 / 0.423 | **0.450 [0.421, 0.478]** | 0.204 / 0.064 |
| laya | 0.567 [0.502, 0.632] | 1.000 / 0.667 / 0.315 | 0.348 [0.326, 0.371] | 0.222 / 0.053 |
| laya-ML | 0.511 [0.446, 0.576] | 0.917 / 0.431 / 0.387 | 0.342 [0.321, 0.363] | 0.316 / 0.033 |

Paired differences between v1i and each other system (same resamples):

| comparison | JevBench public | typed/test |
|---|---|---|
| v1i − kev-0.8b | −0.074 [−0.156, +0.013] | −0.052 [−0.082, −0.020] |
| v1i − laya | +0.004 [−0.074, +0.087] | +0.050 [+0.011, +0.087] |
| v1i − kev-0.5b | +0.065 [−0.009, +0.152] | −0.023 [−0.053, +0.009] |
| v1i − V0 | +0.065 [+0.009, +0.121] | +0.008 [−0.016, +0.031] |

(`bootstrap.json` stores other − v1i; the signs above are flipped to read as v1i − other.)

**typed/test baselines:**

| baseline | acc | source |
|---|---:|---|
| uniform | 0.308 | card |
| majority class | 0.461 | card |
| per-question majority from our 1,000 train-split items | 0.475 | measured |
| test-oracle majority | 0.523 | measured |

v1i's whole interval lies below 0.461. kev-0.8b's point estimate (0.450) is below it, but its interval reaches 0.478.
The typed ece columns come from `runs/v1i-jevtyped.json`; v1i's refitted ECE (0.075) is the worst of the six systems.

### 4f. BTZSC (`runs/v1i-btzsc.json`; competitors were not run on BTZSC)

| dataset | V0 | e5-small (0-shot bi-encoder) | **v1i** |
|---|---:|---:|---:|
| amazonpolarity | 0.901 | 0.917 | 0.903 |
| imdb | 0.872 | 0.870 | 0.867 |
| appreviews | 0.899 | 0.901 | 0.872 |
| yelpreviews | 0.931 | 0.944 | 0.933 |
| rottentomatoes (CONT) | 0.846 | 0.753 | 0.857 |
| financialphrasebank | 0.485 | 0.502 | 0.459 |
| emotiondair | 0.360 | 0.398 | 0.330 |
| empathetic | 0.193 | 0.315 | 0.125 |
| banking77 (ID) | 0.647 | 0.532 | 0.853 |
| biasframes_intent | 0.562 | 0.560 | 0.536 |
| massive (ID) | 0.575 | 0.501 | 0.790 |
| agnews (ID) | 0.870 | 0.734 | 0.905 |
| yahootopics | 0.506 | 0.518 | 0.407 |
| trueteacher | 0.422 | 0.438 | 0.518 |
| manifesto | 0.166 | 0.152 | 0.077 |
| capsotu | 0.472 | 0.465 | 0.370 |
| biasframes_offensive | 0.554 | 0.505 | 0.495 |
| biasframes_sex | 0.491 | 0.200 | 0.654 |
| wikitoxic_toxicaggregated | 0.684 | 0.533 | 0.707 |
| wikitoxic_obscene | 0.722 | 0.597 | 0.735 |
| wikitoxic_threat | 0.568 | 0.206 | 0.520 |
| wikitoxic_insult | 0.767 | 0.478 | 0.754 |
| **mean, all 22 (macro-F1)** | 0.613 | 0.546 | **0.621** |
| **mean, clean 18 (macro-F1)** | **0.586** | 0.528 | 0.570 |
| accuracy, all 22 / clean 18 | 0.651 / 0.631 | 0.571 / 0.557 | 0.663 / 0.621 |
| ECE raw, all 22 / clean 18 | 0.079 / 0.078 | 0.245 / 0.218 | 0.102 / 0.100 |

- These are 1,000-per-dataset subsamples, and Decima gets a neutral question. They are **not comparable** to the leaderboard, which uses full test sets (e.g. Qwen3-Embedding-0.6B 0.580, e5-large-v2 0.597, both all-22).
- **v1i is below V0 on the zero-shot clean-18 mean** (0.570 vs 0.586).

### 4g. Jev Decision Index 0.1 (`runs/v1i-jdi.json`, `runs/logs/jdi-score-v1i.txt`, `runs/phase0-jdi.json`)

| system | Index, official (no truncation) | Index, truncated (not JDI-legal) | panel items within budget |
|---|---:|---:|---:|
| uniform-random chance | 27.11 | — | — |
| V0 | 10.16 | 25.73 | — |
| **v1i** | **12.78** | 26.66 | 47,746 / 96,054 (49.7 %) |

For reference, the board's own runs (not reproduced here): Jev 1.13.0 59.51 · Kev 0.5B 30.34 · Kev 0.6B 31.30 · GLiNER2.5-small 23.93 · Laya (421M) 16.39.

Official coverage for v1i is 0 on the tools, contractnli, bright, bpomp, pop909 and habermas benchmarks, because every request there exceeds the 512/64 token budget.

Why the official index is so low: `balanced_raw` weights five areas equally (knowledge, language,
retrieval, tools, arts). v1i's tools area is 0.000 and retrieval 0.033, because those requests exceed the
budget and count as wrong under the no-truncation rule; language (0.150) and arts (0.198) lose
benchmarks the same way. Its knowledge area (0.259) is near chance (0.271). The index is not "dominated
by knowledge": knowledge is one area of five.

The column labelled "v0-official" in `runs/v1i-jdi.json` and `jdi-score-v1i.txt` (25.73) is **mislabelled**: it is V0 *without* the budget rule. The official V0 score is 10.16 (`runs/phase0-jdi.json`).

---

## 5. Claims

### Claims we can make

| claim | supporting row |
|---|---|
| On the Laya MASSIVE protocol (reproduced to within 0.001), Decima-small scores 0.913 in English and 0.852 averaged over 13 other languages, against 0.783 / 0.451 for the better Laya checkpoint in each case. **This must carry the caveat, adjacent to the number: "Decima trained on the MASSIVE train split (all 51 locales; test rows differ); Laya reports it did not train on MASSIVE."** (For XNLI: "Decima trained on the XNLI/MNLI train splits; Laya reports it did not.") Measured exact whole-state overlap is 0–11.7 % per language; excluding those items changes v1i by ≤ 0.014 on any suite. | §4c; §2b table and clean-subset table |
| In every pairwise comparison, v1i has the lower *as-shipped* ECE (15-bin, mean over the suites both systems ran, all five item sets, FarsTail excluded; `scripts/audit/calibration.py` → `runs/audit/calibration.json`): 0.063 vs kev-0.5b 0.117 (39 suites), 0.056 vs kev-0.8b 0.158 (21), 0.056 vs laya 0.370 (50), 0.056 vs laya-ML 0.251 (50). Kev is measured raw (T = 1), as it publishes; Laya with its shipped temperatures. This is the only calibration headline; the set means in §4a (0.072 / 0.056 / 0.031 / 0.130) are supporting detail. | `runs/audit/calibration.json`; §4a |
| The choice order never changes Decima's answer: flip rate 0.000 on every suite, versus 10–27 % for the others as a mean over the Kev, Laya and Decima-bench suites each ran (kev-0.8b 10.3 %, laya-ML 21.4 %, kev-0.5b 21.9 %, laya 27.2 %; `release/figures/make_figures.py`), or 0.014–0.489 as a range of per-set means (§4a). This holds by construction, since each choice is scored independently, not because it was learned. | §4a flip rows |
| On the in-distribution Decima bench, the mean is 0.785 versus kev-0.8b 0.661, laya 0.439 and laya-ML 0.554. **This must be stated as in-distribution for Decima:** every one of these datasets' train splits was trained on. | §4d |
| JevBench *public subset* accuracy is 0.571 [0.511, 0.636]. It is the highest on the hard tier (0.468) but not overall. **This must say:** "231 public items; not the official JevBench score; the public items were used for checkpoint selection." | §4e |
| Relative to V0: large gains on every in-distribution set (Kev +0.139, Laya +0.171, Decima +0.169) and on JevBench public (+0.065 [+0.009, +0.121]: the CI just excludes zero, by about 2 items, and these items were used for selection). | §4a, §4e |
| Size is 26.4 M body+head parameters, or 122.4 M including the 96.0 M multilingual embedding table. Total parameters are 0.29× Laya (421 M), 0.38× Laya-ML (322 M), 0.25× Kev-0.5B (494 M) and 0.16× Kev-0.8B (753 M). Body+head is 0.07× Laya's 369.7 M and 0.21× Laya-ML's 125.3 M. Always state which count is meant. | §6 |
| On typed/test v1i scores higher than both Laya base checkpoints (+0.050 [+0.011, +0.087] vs laya) — only as a relative statement: every system is below the 0.461 majority-class baseline, so this is not a usefulness claim (see the typed row under *must not make*). | §4e |
| int8 / ONNX accuracy equals these numbers: int8 matches fp32's top answer on 98.6–99.8 % of 45,850 real eval items per set, every set-level score within 0.004; x86 fp32 ONNX reproduces the GPU predictions on 100 % of 22,050 items. | docs/BENCH-x86.md §1 |

### Claims we must not make

| do not say | why (row) |
|---|---|
| "the only model that runs on CPU", or "the only CPU decision model" | Laya runs on CPU; its card reports 193–464 ms per call on CPU with the router. |
| any MASSIVE or XNLI comparison with Laya *without* the training-data caveat | Decima trained on the MASSIVE train split (51 locales) and XNLI/MNLI train; Laya says it did not (unverified). §2b. |
| "beats Kev" / "state of the art vs Kev" | Kev set mean 0.768 vs 0.779 (kev-0.5b) and 0.794 (kev-0.8b); pooled "all" 0.790 vs 0.799 / 0.812; kev-0.8b leads on JevBench (0.645) and typed (0.450). §4b, §4e. |
| "better at NLI than Laya" | XNLI en 0.767 vs 0.860; XNLI non-en 0.671 vs 0.731 (laya-ML). §4c. |
| a JevBench score, rank or leaderboard position | only 231 public items; the official composite needs sealed and judge items plus speed and cost. §3. |
| that the gap to kev-0.8b on JevBench is real, or that v1i beats Laya on JevBench | differences of −0.074 [−0.156, +0.013] and +0.004 [−0.074, +0.087] are inside noise. §4e. |
| any "typed-decisions" win or usefulness claim (the relative statement vs Laya above is the only allowed typed comparison) | every tested small model is below the 0.461 majority-class baseline; v1i is at 0.399 [0.371, 0.423]. §4e. |
| a BTZSC number as zero-shot without saying so, or comparing it with the leaderboard | the all-22 mean includes one contaminated and three in-distribution datasets. Clean-18 is 0.570, below V0 (0.586). Our subsample and question are not the leaderboard setup. §4f. |
| any JDI result as competitive, or "the index is dominated by knowledge" | official 12.78 is below uniform chance (27.11) and below Laya (16.39) and Kev 0.5B (30.34) on the board; even truncated (26.66) is below chance. About half of panel requests exceed the token budget and count as wrong; tools = 0.000, retrieval = 0.033; knowledge is one of five equal areas. §4g. |
| "calibrated better than everyone" after temperature scaling | with per-suite temperature the systems are close (pairwise refitted: 0.057 vs 0.075, 0.046 vs 0.052, 0.052 vs 0.066, 0.052 vs 0.074; `runs/audit/calibration.json`); typed/test v1i 0.075 is the worst. Report refitted ECE neutrally, never bolded. §4a, §4e. |
| "no PyTorch" for the installed package | the runtime (`decima.Decima`) imports no PyTorch, but the package as it stands still depends on torch, sentence-transformers and datasets (`pyproject.toml`); a runtime-only package is planned. |
| that the eval sets are untouched held-out data | all five sets (Kev, Laya, Decima, jevtyped, BTZSC clean-18) define `scripts/pick_best.py`'s score, which chose v1i from ~15 candidates (v1i and v1j tie at 0.674). Expect mild optimistic bias. |

---

**Update 2026-09-26 (docs/BENCH-x86.md):** "int8 / ONNX accuracy equal to these numbers" moved to *claims
we can make* (above). The 0.93 top-1 agreement measured earlier came from bench/latency.py's synthetic
near-duplicate options and is not a quantisation measure.

---

## 6. Parameter counts (measured the same way for every system)

Definitions:
- **vocab table** = the token-embedding matrix.
- **body** = the backbone minus the token embeddings (layers, position embeddings, norms).
- **head** = everything outside the backbone.

Sources:
- Decima: `DecimaModel.load("checkpoints/v1i/best", "cpu").param_counts()`.
- Competitors: tensor shapes read from the safetensors headers of the pinned HF revisions (no weights loaded). Kev's head comes from `head.pt`.
- Both are produced by `uv run python scripts/audit/params.py` → `runs/audit/params.json` (the table below is that file; it also lists each system's non-embedding count = body + head).

| system | total (as run) | vocab table | body | head | source |
|---|---:|---:|---:|---:|---|
| **Decima-small (v1i)** | **122,388,869** | 96,014,208 | 21,639,552 | 4,735,109 | `param_counts()`; e5-small backbone, 12 layers × 384 |
| Kev-0.5B | 494,492,032 | 136,134,656 | 357,898,112 | 459,264 | Qwen/Qwen2.5-0.5B @060db64 (tied embeddings) + jaredpalmer/kev-0.5b @9ce2fd3 `head.pt`. The LoRA (8,798,208) is merged into the base weights at load. |
| Kev-0.8B | 752,917,824 | 254,279,680 | 498,113,344 | 524,800 | Qwen/Qwen3.5-0.8B-Base @dc7cdfe, text model (the checkpoint's vision tower and MTP, 121,045,760, are not used) + jaredpalmer/kev-0.8b @54f4f87 `head.pt`. LoRA 10,822,656, merged. |
| Laya | 421,293,830 | 51,576,832 | 343,204,864 | 26,512,134 | convaiinnovations/laya @aa8c91c `model.safetensors`; the card says "421M total" |
| Laya-multilingual | 321,908,998 | 196,608,000 | 110,331,648 | 14,969,350 | convaiinnovations/laya-multilingual @82d57fc; the card says "322M total" |

Decima-small's body+head is 26,374,661. Always say which count is meant; the release documents quote the 122 M total.

---

Reproduce:
- Overlap and clean-subset accuracy: `uv run python scripts/audit/overlap.py` → `runs/audit/overlap.json`.
- Confidence intervals: `uv run python scripts/audit/bootstrap.py` → `runs/audit/bootstrap.json`.
- Headline calibration (pairwise ECE as shipped + refitted): `uv run python scripts/audit/calibration.py` → `runs/audit/calibration.json`.
- Flip means and figure numbers: `uv run --with matplotlib python release/figures/make_figures.py`.
- Scores: `uv run python -m bench.score --items runs/items/<set>.jsonl --preds …` as recorded in each `runs/v1i-<set>.json`.
- Parameter counts (§6), Decima and competitors: `uv run python scripts/audit/params.py` → `runs/audit/params.json` (reads safetensors headers of the pinned revisions; no weights downloaded).
