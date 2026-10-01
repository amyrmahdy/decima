---
license: apache-2.0
language: [en, fa, ar, ru, de, fr, es, pt, tr, hi, ta, zh, ja, ko, sw, ur, vi, th, el, bg]
library_name: onnx
pipeline_tag: zero-shot-classification
tags: [decision-model, system-one, jev, jev-alternative, typed-decisions, calibration, multilingual, onnx, int8, cpu, in-browser]
base_model: intfloat/multilingual-e5-small
---

# Decima-small — an open, CPU-sized Jev-style decision model

**Give it a situation, a question and your options. Get back probabilities that are well calibrated as shipped.**
Like TypeSafe's Jev, Decima is a *System One* model: it returns typed decisions (choose one, yes/no, an ordered
score, rank) with probabilities instead of generating text. Unlike Jev, it is open (Apache-2.0), has 122M
parameters, and runs on one CPU core or entirely in your browser.
A 122M-parameter decision model that runs in ~20 ms on one laptop CPU core (4 options), works across
languages (evaluated in 20), and does not change its answer when you reorder the options.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy) · [amyrmahdy.github.io](https://amyrmahdy.github.io)).
The name: **DECI**sion **MA**king — and Decima is also the Roman Fate who decides.

> **New in 1.1 (2026-10-01).** Trained further on 45,729 teacher-labelled decisions in Jev's own question
> format (choice with option descriptions, true/false statements, described levels). On the community
> [classifier-benchmark](https://github.com/jabr/classifier-benchmark) (49 tasks) it rises from 0.571 to
> **0.616** — ahead of Laya (421M, 0.583) at under a third of the size; Von (0.720) and GLiNER2 (0.684)
> remain ahead. Decima now also speaks **TypeSafe's System One API**: `python -m decima.serve`, and the
> official `typesafe-sdk` works against it unchanged. All numbers below are 1.1 unless marked 1.0;
> 1.0 stays available at revision `v1.0`.

![Shuffle the options: Decima's answer never changes](figures/option_order_flips.png)

- **Small and fast** — 122M parameters, int8 ONNX, **~20 ms** per decision on one x86 core (4 options, short input).
- **Multilingual** — **0.857** on MASSIVE intent across 13 non-English languages under Laya's published
  protocol, re-run by us (Laya-multilingual: 0.451). Decima trained on the MASSIVE and XNLI/MNLI train
  splits; Laya reports it did not.
- **Stable, and well calibrated as shipped** — **0 %** answer changes when options are shuffled (the four
  other open decision models we compared: 10–27 %), and the lowest calibration error as shipped in every
  pairwise comparison we ran.

| accuracy¹ | Decima-small | Kev-0.5B | Kev-0.8B | Laya | Laya-multilingual |
|---|---:|---:|---:|---:|---:|
| Parameters (total) | **122M** | 494M | 753M | 421M | 322M |
| Laya's MASSIVE + XNLI protocol, re-run by us, 29 suites / 19 languages (in-distribution for Decima²) | **0.761** | 0.527 | — | 0.445 | 0.607 |
| Kev's published protocol, re-run by us, 8 suites | 0.762 | 0.779 | **0.794** | 0.681 | 0.580 |
| Decima bench, 12 suites, EN/FA/AR/RU (in-distribution for Decima) | **0.786** | — | 0.661 | 0.439 | 0.554 |
| jabr/classifier-benchmark v2, 49 tasks (zero-shot by data; macro accuracy)⁴ | 0.616 | — | — | 0.583 | — |
| Answer changes when options are shuffled³ | **0.0 %** | 21.9 % | 10.3 % | 27.2 % | 21.4 % |

Calibration error as shipped (15-bin ECE, mean over the suites both models ran, FarsTail excluded;
lower is better): Decima **0.064** vs Kev-0.5B 0.117 (39 suites) · **0.060** vs Kev-0.8B 0.158 (21) ·
**0.058** vs Laya 0.370 (50) · **0.058** vs Laya-multilingual 0.251 (50). Details under *Stability and calibration*.

¹ All numbers are accuracy unless stated, measured by us on one harness with identical inputs for every model (competitor checkpoints run locally; their published numbers were reproduced to within 0.001 first — see [Evaluation](#evaluation)). Decima was trained on the *train* splits of several of these datasets; see *What is zero-shot and what is not* under Results before quoting any row. Protocols, data provenance and measured overlap: [docs/EVAL.md](https://github.com/amyrmahdy/decima/blob/main/docs/EVAL.md).
² Decima trained on the MASSIVE train split (all 51 locales; test rows differ) and on the XNLI/MNLI train splits; Laya reports it did not train on MASSIVE or XNLI.
³ Mean over the Kev, Laya and Decima-bench suites each model was run on.
⁴ Decima int8 on CPU through the benchmark's own harness (we added a backend); Laya's number is the benchmark's published run.

## Quickstart

```bash
pip install "git+https://github.com/amyrmahdy/decima"      # from source: github.com/amyrmahdy/decima
```

The runtime (`decima.Decima`) imports no PyTorch — ONNX Runtime, numpy and a tokenizer — but the package
as it stands still installs the training stack as well (torch, sentence-transformers, datasets). A
runtime-only package without the training dependencies is planned.

```python
from decima import Decima, Question

decima = Decima.from_pretrained("amyrmahdy/decima-small")   # int8 ONNX, one CPU thread

# outputs below are real (Decima-small int8)
q = Question("Which team should handle this request?",
             ["billing", "technical support", "sales", "account security"])

decima.decide("Someone logged into my account from another country.", q)
# → {'billing': 0.004, 'technical support': 0.031, 'sales': 0.001, 'account security': 0.964}

decima.decide("یک نفر از کشور دیگری وارد حسابم شده است.", q)   # the same message in Persian
# → {'billing': 0.012, 'technical support': 0.024, 'sales': 0.002, 'account security': 0.962}
```

`decide` returns a dict of probabilities. The lower-level `decide_logits` returns log-probabilities
(softmax them for `choose`/`verify`/`score`; `exp` them for `rank`). `Question(..., lang="fa")` turns on
Persian/Arabic text normalisation; the example above gives the same output with or without it.

- **Options are free text** chosen at call time — no fixed label set, no retraining.
- **Option encodings are cached** per option set, so repeated questions only pay for the state.
- **Four question kinds**: `choose` (one of N), `score` (ordered levels, ordinal head), `verify`
  (yes/no), `rank` (independent probability per option).
- **Languages**: state, question and options can be in different languages (e.g. a Persian message
  scored against English options).

**Jev / TypeSafe-compatible API.** Decima speaks TypeSafe's System One wire format (`choice`, `score`,
`noul`), so code written for Jev runs on a local Decima:

```bash
python -m decima.serve                     # http://127.0.0.1:11436 · POST /v1/systemone
export TYPESAFE_BASE_URL=http://127.0.0.1:11436 TYPESAFE_API_KEY=local TYPESAFE_DEFAULT_MODEL=decima-small
```

```python
from decima.systemone import system_one   # or in-process, without a server
system_one(decima, "My card was charged twice", {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "tech": "bugs"}},
    "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
})
```

**Good for** routing, triage, intent, topic and sentiment classification, verification and ranking — any
bounded decision where software needs a choice *and* a confidence it can threshold.
**Not recommended** for tool-call or shell-command safety: in our spot checks it called
`rm -rf /var/lib/postgresql/data` non-destructive (0.722).
**Not for** writing text, open-ended answers, arithmetic, world-knowledge questions, or long multi-fact
reasoning (see [Limitations](#limitations) and [Bias, risks and out-of-scope use](#bias-risks-and-out-of-scope-use)).

## Under the hood

The short version of the [technical report](https://github.com/amyrmahdy/decima/blob/main/docs/TECHNICAL-REPORT.md) — the ideas that make a 122M model
behave like this.

**1 · Late interaction ⇒ permutation invariance by construction.** The state (with the question) is
encoded once into token states $H_s$. Each option $k$ is encoded on its own and reads the state through
two cross-attention layers, producing a score $s_k = f(H_s, H_k)$ that depends only on the state and that
option — never on its position or on the other options. The distribution is
$p = \mathrm{softmax}(s/T)$, so permuting the options permutes $p$ identically. The 0 % flip rate holds by
construction for `choose`, `verify` and `rank` (up to floating-point ties between near-equal options), and
we measure it anyway. `score` levels are ordered by definition, so their order is part of the question.

**2 · Option sets of any size, cached.** Because options never share a context window, there is no prompt
to overflow: cost is $\approx t_{\text{enc}}(\text{state}) + n \cdot t_{\text{scorer}}$. Measured on one
x86 core (int8, short state): 16 ms at 4 options, ~1 ms per extra option, 1.06 s at 1,000. Option
encodings are cached per option set, so a repeated question only pays for the new state.

**3 · An ordinal head for ordered choices.** For `score` questions ("low / medium / high", 1–5 stars) a
cumulative-link head predicts $P(y > k) = \sigma(g - \theta_k)$ with a shared latent $g$ and thresholds
$\theta_0 < \theta_1 < \dots$ that are increasing by construction (cumulative softplus gaps) and computed
from each level's vector after it has attended to the state — so they depend on the level texts and the
state, and a 3-level and a 7-level scale each get their own spacing.

**4 · Distilled from distributions, not just labels.** A 26B teacher labelled ~300k decisions with a
probability distribution over the options — full distributions for short option lists; top-k (3–6) with
the leftover mass spread uniformly for catalogues of 20+ options and for the relabelled rows — across a
domain × language × question-kind × option-count grid (plus "none of the above" and cross-lingual cases).
Training minimises soft cross-entropy to the target distribution (teacher, or label-smoothed gold at 0.92
for public train-split rows) + 0.3 × NLL on its argmax label; `rank` uses per-option binary cross-entropy.
One temperature, fitted on a 5 % hold-out of the training mix, ships in the config.

**5 · Model selection that guards zero-shot ability.** We found that data which lifts in-distribution
benchmarks can quietly cost generalisation (BTZSC clean-18 fell from 0.586 to 0.560 across our e5-small
runs). From then on every candidate was selected on a score that includes a strictly zero-shot
benchmark (weight 1/5), so a model that overfits our label spaces is penalised. The released model
1.0 recovered to 0.570; 1.1, trained further on Jev-format data, is at 0.558 — both below V0's 0.586.

**6 · Block-wise 8-bit that preserves the model.** Plain dynamic int8 agreed with fp32 on 88 % of real
items (45 % on a synthetic 77-option stress probe): a few activation channels in the encoder are tens of
times larger than the rest, so one scale per tensor loses precision everywhere else. Block-wise 8-bit in
the encoder (blocks of 32 for weights and activations; the scorer keeps per-channel dynamic int8) confines
each outlier to its own block — int8 now matches fp32 on 99.4 % of 22,050 real test items (1.1; accuracy
0.776 fp32 vs 0.777 int8), at 3.8× smaller size.

**7 · Comparisons we can defend.** Every competitor was run on our hardware from its own checkpoint, and
their published numbers were **reproduced to within 0.001** (Laya 0.783 / 0.860 / 0.520 vs 0.521, Kev
0.799) with their own evaluation protocols before we compared anything.

## Results

<details open>
<summary><b>System One tasks — jabr/classifier-benchmark (the benchmark the "Jev alternatives" comparisons cite)</b></summary>

[jabr/classifier-benchmark](https://github.com/jabr/classifier-benchmark): synthetic tasks in TypeSafe's
question format — `choice`, `noul` (judge a statement true/false), `score` (ordered levels). Macro accuracy
over tasks. Decima: int8 ONNX on CPU through a backend we added to the harness (choice options rendered as
"key: description", noul as a yes/no verify on the statement, score through the ordinal head); the other
rows are the benchmark's own published runs.

| | params | v2 (49 tasks, 866 cases) | v1 (8 tasks, 78 cases) |
|---|---:|---:|---:|
| Jev (hosted) | — | **0.966** | **0.972** |
| Von 1.1 | 395M | 0.720 | 0.927 |
| GLiNER2 | ~0.5B | 0.684 | 0.785 |
| **Decima-small 1.1** | 122M | 0.616 | 0.755 |
| Laya | 421M | 0.583 | 0.619 |
| Decima-small 1.0 | 122M | 0.571 | 0.680 |

By question type (v2): choice 0.702, noul 0.623, score 0.447 (1.0: 0.697, 0.532, 0.403). The cases were
never trained on (the 1.1 data was decontaminated against them), and the benchmark was run once per
released model. Its gold labels and tasks are synthetic, written by a committee of LLMs; our teacher
(Gemma-4-26B-A4B) scores 0.942 on v2 with a plain prompt, so the remaining gap is the student's, not
the labels'.

</details>

<details>
<summary><b>Laya's published protocol (re-run by us) — MASSIVE and XNLI in 19 languages; in-distribution for Decima</b></summary>

First 300 test rows per language; MASSIVE intent = gold + 19 random distractors (20 options), XNLI = 3
described relations. Our re-run of Laya reproduced its published numbers to within 0.001 (0.783 / 0.306
/ 0.860 / 0.520 vs 0.521 published).

⚠ Decima trained on the MASSIVE and XNLI/MNLI train splits; Laya reports it did not.

![On Laya's MASSIVE + XNLI protocol, Decima-small scores 76.4 % at 122M parameters](figures/size_vs_accuracy.png)

| | Decima-small | Laya | Laya-multilingual | Kev-0.5B |
|---|---:|---:|---:|---:|
| MASSIVE intent, English | **0.930** | 0.783 | 0.657 | 0.753 |
| MASSIVE intent, 13 other languages (mean) | **0.857** | 0.306 | 0.451 | 0.426 |
| XNLI, English | 0.760 | **0.860** | 0.843 | 0.747 |
| XNLI, 14 other languages (mean) | 0.660 | 0.520 | **0.731** | 0.589 |

Decima was trained on MASSIVE's train split (all 51 locales) and on XNLI/MNLI train data; Laya reports
it did not train on MASSIVE or XNLI. Test rows differ from training rows (exact overlap ≤ 11.7 % per
MASSIVE language, see below), but for Decima these are in-distribution tasks and for Laya they are
zero-shot.

</details>

<details>
<summary><b>Kev's published protocol (re-run by us) — banking77, AG News, BoolQ, MNLI, SST-5, Yelp</b></summary>

150 records per dataset, Kev's prompts, rendering and option shuffles. Decima and Kev-0.5B trained on the
train splits of these sources (Decima: all but Yelp); Laya's README lists AG News and BoolQ in its mix
(SST-5 held out); Kev-0.8B's training data is not auditable.

| | banking77 | AG News | BoolQ | MNLI | SST-5 | Yelp | AG News y/n | Yelp y/n | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decima-small | **0.913** | **0.947** | 0.713 | 0.740 | 0.513 | 0.473 | 0.953 | 0.847 | 0.762 |
| Kev-0.5B | 0.860 | 0.940 | 0.753 | 0.747 | **0.533** | 0.553 | **0.960** | 0.887 | 0.779 |
| Kev-0.8B | 0.813 | 0.900 | **0.807** | **0.800** | 0.507 | **0.640** | 0.953 | **0.933** | **0.794** |
| Laya | 0.473 | 0.933 | 0.740 | 0.613 | 0.320 | 0.580 | 0.937 | 0.853 | 0.681 |
| Laya-multilingual | 0.380 | **0.947** | 0.660 | 0.613 | 0.300 | 0.267 | 0.843 | 0.633 | 0.580 |

Decima did not train on Yelp (kept out so BTZSC stays zero-shot for it).

</details>

<details>
<summary><b>Decima bench — full label sets in English, Persian, Arabic, Russian (in-distribution for Decima)</b></summary>

Decima trained on the train split of every dataset in this table (MASSIVE and XNLI included; Laya
reports it did not train on MASSIVE or XNLI), so this bench is in-distribution for Decima.

| | SST-5 | AG News | XNLI en | XNLI ar | XNLI ru | FarsTail fa | MASSIVE en | fa | ar | ru | banking77 (77) | CLINC150 (151) | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decima-small | 0.507 | 0.908 | 0.750 | 0.656 | 0.684 | 0.811 | **0.879** | **0.848** | **0.769** | **0.843** | **0.893** | **0.881** | **0.786** |
| Kev-0.8B | **0.539** | 0.880 | 0.797 | 0.656 | 0.677 | 0.780 | 0.609 | 0.567 | 0.425 | 0.565 | 0.819 | 0.621 | 0.661 |
| Laya | 0.305 | **0.922** | **0.870** | 0.453 | 0.593 | 0.554 | 0.458 | 0.038 | 0.078 | 0.185 | 0.378 | 0.434 | 0.439 |
| Laya-multilingual | 0.268 | **0.922** | 0.828 | **0.687** | **0.700** | **0.836** | 0.473 | 0.313 | 0.282 | 0.389 | 0.424 | 0.530 | 0.554 |

MASSIVE here uses all 60 intents as options (Persian/Arabic/Russian utterances, English options).

</details>

<details>
<summary><b>Zero-shot — BTZSC (clean 18 = datasets Decima never trained on)</b></summary>

[BTZSC](https://huggingface.co/datasets/btzsc/btzsc), macro-F1. The **clean 18** excludes the four
datasets that overlap our training data (Rotten Tomatoes ≈ SST-5 corpus, banking77, MASSIVE, AG News).

| | Parameters | clean 18 | all 22 |
|---|---:|---:|---:|
| **Decima-small 1.1** | 122M | 0.558 | 0.611* |
| Decima-small 1.0 | 122M | 0.570 | 0.621* |
| Decima V0 (teacher data only) | 122M | 0.586 | 0.613 |
| multilingual-e5-small, zero-shot (our base encoder, no training) | 118M | 0.528 | 0.546 |

All rows run by us on the harness's own 1,000-per-dataset draw. The benchmark's published leaderboard
uses full test sets, so we do not place Decima on it. Running two or three of the leaderboard's
reference models (e.g. Qwen3-Reranker-0.6B, gte-large) on this same draw, for a like-for-like
comparison, is future work.
*The all-22 figure includes the four in-distribution datasets — quote clean 18. Note that V0 is higher on
clean 18: the gold classification data that lifted our in-distribution benchmarks cost some zero-shot
ability (see the technical report).

</details>

<details>
<summary><b>Long business decisions and the Jev Decision Index (limitations)</b></summary>

| | JevBench public (231) | typed-decisions test (2,000) |
|---|---:|---:|
| Decima-small 1.1 | 0.571 [0.511–0.632] | 0.427 [0.399–0.455] |
| Kev-0.8B | 0.645 [0.584–0.710] | 0.450 [0.421–0.478] |
| Laya | 0.567 [0.502–0.632] | 0.348 [0.326–0.371] |
| Kev-0.5B | 0.506 [0.442–0.571] | 0.421 [0.397–0.445] |
| Laya-multilingual | 0.511 [0.446–0.576] | 0.342 [0.321–0.363] |
| majority-class baseline | — | 0.461 |

95 % cluster-bootstrap intervals (`scripts/audit/bootstrap.py` → `runs/audit/bootstrap.json`).
Decima is zero-shot on both by data, but its Jev-style synthetic data targeted this format and the public
JevBench items were used for checkpoint selection. JevBench here is the **231 public items of JevBench v1.4**, not the
official leaderboard score (which also uses sealed items). On typed-decisions every small model we
tested, Decima included, is below the majority-class baseline: long, multi-fact business cases remain
hard at this size.

**Jev Decision Index 0.1** (measured on 1.0): 12.8 under the official no-truncation rule — **below uniform chance
(27.1)**. The index weights five areas equally (knowledge, language, retrieval, tools, arts). Decima's
tools area is 0.000 and retrieval 0.033 because those requests exceed its 512-token state / 64-token
option budget and count as wrong under the no-truncation rule (only 49.7 % of panel items fit the budget);
its knowledge area is near chance (0.259 vs 0.271). With truncation it scores 26.7, still below chance.
Board reference runs: Jev 59.5, Kev-0.5B 30.3, Laya 16.4.

</details>

<details>
<summary><b>Stability and calibration — how they were measured</b></summary>

- **Option order**: every multi-option question is re-asked with its options shuffled. Decima's answer
  changed on **0.0 %** of questions (mean over the Kev, Laya and Decima-bench suites each model ran:
  Kev-0.5B 21.9 %, Kev-0.8B 10.3 %, Laya 27.2 %, Laya-multilingual 21.4 %; worst single suite: Laya
  100 %, Kev-0.5B 66.7 %). Decima scores each option independently of
  its position, so this holds by construction; we report the measurement.
- **Calibration**: expected calibration error (ECE, 15 bins, top-1 confidence), mean of per-suite ECE
  over the suites both models were run on (Kev, Laya, Decima-bench, JevBench and typed-decisions suites;
  FarsTail excluded — its calibration items overlap its test items). Source: `scripts/audit/calibration.py`
  → `runs/audit/calibration.json`. *As shipped* = each model's own probabilities (Decima: one global
  temperature fitted on a 5 % hold-out of its training mix; Kev: raw, as it publishes; Laya: its shipped
  temperatures). *Refitted* = one temperature per suite fitted on that suite's calibration items, same
  procedure for every model (suites without calibration items drop out).

  | vs | shared suites | Decima as shipped | theirs as shipped | refitted: shared suites | Decima refitted | theirs refitted |
  |---|---:|---:|---:|---:|---:|---:|
  | Kev-0.5B | 39 | **0.064** | 0.117 | 38 | 0.054 | 0.075 |
  | Kev-0.8B | 21 | **0.060** | 0.158 | 20 | 0.043 | 0.052 |
  | Laya | 50 | **0.058** | 0.370 | 49 | 0.049 | 0.066 |
  | Laya-multilingual | 50 | **0.058** | 0.251 | 49 | 0.049 | 0.074 |

  After refitting a temperature per suite the systems are close; on typed-decisions Decima's refitted ECE
  (0.033) ties for best with Laya-multilingual, behind Kev-0.5B (0.031). Our claim is about probabilities as shipped. Laya's
  MASSIVE/XNLI suites are in-distribution for Decima (Decima trained on the MASSIVE and XNLI/MNLI train
  splits; Laya reports it did not).

</details>

<details>
<summary><b>Speed — x86 CPU, int8 and fp32, 4 to 1,000 options</b></summary>

Measured on 1.0; 1.1 has the same architecture and graph shapes, so the same costs apply. One decision = one state scored against one option set, batch 1, option encodings cached. Measured on an
x86 laptop, **Intel Core Ultra 7 155H, one P-core, ONNX Runtime** (p50 / p95, ms). Details and
reproduction: docs/BENCH-x86.md.

| precision | threads | state | 4 options | 20 options | 77 options |
|---|---:|---|---:|---:|---:|
| **int8** | **1** | short (~10–20 tokens) | **20 / 23** | **42 / 46** | **83 / 92** |
| int8 | 1 | long (~400 tokens) | 102 / 113 | 126 / 142 | 204 / 221 |
| fp32 | 1 | short | 24 / 27 | 75 / 80 | 181 / 186 |
| fp32 | 1 | long | 96 / 104 | 179 / 193 | 407 / 441 |

- **Use one thread per process.** At batch 1, four threads were slower than one in every int8 cell; for
  throughput, run several single-threaded workers.
- **Cost grows with the number of options**: 16 ms at 4 options, ~1 ms per extra option, 1.06 s at
  1,000 (int8, 1 x86 core, short state, short intent-label options; the table above uses the benchmarks'
  own option sets). Under 50 ms holds up to roughly 20–40 options depending on option length. Any number
  of options *works* — there is no prompt to fill and order never matters — but large option sets are
  slower; a retrieval pre-filter for 1k+ options is on the roadmap.
  ![Latency vs number of options](figures/speed_vs_options.png)
- **A new option set** costs a one-time encode (~12 ms per option). The cache holds up to 256 option sets
  and is cleared entirely when it overflows.
- **int8 = fp32 in practice**: same top answer on 99.4 % of 22,050 real eval items for 1.1 (accuracy 0.776
  fp32 vs 0.777 int8); for 1.0, 98.6–99.8 % per set over 45,850 items, every set-level score within ±0.004.
  x86 fp32 reproduces our GPU scores exactly (1.0: 100 % agreement on 22,050 items).
- Files: int8 ONNX 127 MB, fp32 ONNX 489 MB (+ 17 MB tokenizer). Peak process memory ≈ 1.0 GB (int8),
  mostly the Python stack.

**GPU** (NVIDIA GB10, PyTorch fp32, option encodings cached; `runs/final/speed-v1i.json`). Latency of one call
for a batch of B states (p50, ms) and throughput (decisions/s):

| batch | state | 4 options | 20 options | 77 options |
|---:|---|---:|---:|---:|
| 1 | short | 4.3 ms · 232/s | 4.6 ms · 217/s | 7.9 ms · 127/s |
| 1 | long | 4.4 ms · 227/s | 6.3 ms · 159/s | 12.4 ms · 81/s |
| 10 | short | 8.9 ms · 1,125/s | 23.5 ms · 425/s | 62.5 ms · 160/s |
| 10 | long | 26.5 ms · 377/s | 54.7 ms · 183/s | 133.7 ms · 75/s |
| 50 | short | 33.8 ms · 1,481/s | 129.1 ms · 387/s | 335.0 ms · 149/s |
| 50 | long | 137.8 ms · 363/s | 297.3 ms · 168/s | 537.7 ms · 93/s |

</details>

<details>
<summary><b>What is zero-shot and what is not — read before quoting a number</b></summary>

| Test | Decima | Kev | Laya |
|---|---|---|---|
| Kev's protocol (banking77, AG News, BoolQ, MNLI, SST-5) | in-distribution (train splits used) | Kev-0.5B in-distribution; Kev-0.8B not auditable | partly (AG News, BoolQ in its mix; SST-5 held out) |
| Kev's protocol, Yelp | zero-shot | Kev-0.5B in-distribution; Kev-0.8B not auditable | unknown |
| Laya's protocol, MASSIVE / XNLI | in-distribution | unknown | zero-shot (as reported, unverified) |
| Decima bench (MASSIVE, banking77, CLINC, AG News, SST-5, XNLI, FarsTail) | in-distribution | partly | partly |
| BTZSC clean 18 | **zero-shot** | not run | not run |
| JevBench public, typed-decisions | zero-shot by data, but the Jev-style synthetic data targeted this format and the public JevBench items were used for model selection (no typed train split used) | not stated | Laya's typed-decisions checkpoint is trained on it — not the one compared here |
| Jev Decision Index | zero-shot | board reference | board reference |
| jabr/classifier-benchmark | zero-shot by data (1.1's System One data decontaminated against it); run once per released model | not run | benchmark's own run |

"In-distribution" = the model trained on that dataset's train split. Measured exact whole-state overlap
between Decima's training states and test states (`scripts/audit/overlap.py` → `runs/audit/overlap.json`)
is 0 % on 60 of 73 suites and ≤ 0.3 % on every suite except MASSIVE (0.6–11.7 % per language; MASSIVE
repeats short commands between its own train and test splits) and BTZSC Rotten Tomatoes (33.0 %, shares
the SST-5 corpus — excluded from the clean-18 score). All of it comes from the teacher-relabelled file,
which was not decontaminated. Removing the overlapping items moves Decima's accuracy by at most 0.014 on
any suite (laya/massive-ru 0.893 → 0.879; competitors move by similar amounts).

**Selection note**: Decima-small 1.0 was chosen among ~15 training runs using these same evaluation sets; 1.1 is a single continuation of it
(there is no separate held-out benchmark), so its scores carry some selection optimism.
The full per-suite overlap table is in [docs/EVAL.md §2](https://github.com/amyrmahdy/decima/blob/main/docs/EVAL.md#2-status-of-each-benchmark-measured).

</details>

## Model details

- **Architecture**: `multilingual-e5-small` encoder (12 layers × 384, 21.6M non-embedding parameters,
  96M word-embedding table) + a 4.7M late-interaction scorer. The state and question are encoded
  together; every option is encoded separately (question + option) and attends to the state tokens
  through two small decoder-style layers, then gets a score. Heads: softmax (choose/verify),
  cumulative-link ordinal (score), independent sigmoids (rank). Inputs: 512 state tokens, 64 per option.
- **Why late interaction**: options never share a context window, so the number of options is not
  bounded by a prompt and option encodings are cacheable; it is also why reordering options cannot
  change the answer.
- **Training** (distillation + gold labels): teacher decisions from Gemma-4-26B-A4B (≈303k invented
  and re-labelled decisions over EN/FA/AR/RU, ~8 % cross-lingual; ≈6k long multi-question rows (1,570
  cases), repeated 3×), plus gold-labelled train splits: NLI (MNLI, SNLI, ANLI, WANLI, XNLI, FarsTail,
  multilingual-NLI-26lang), BoolQ, banking77, CLINC150, MASSIVE (51 languages), AG News, SST-5. Loss:
  soft cross-entropy to the target distribution (teacher — full distributions for short option lists,
  top-k with uniform remainder for large catalogues and relabelled rows — or label-smoothed gold) + 0.3 ×
  NLL on its argmax. The gold-labelled rows were decontaminated against every evaluation item (exact-text
  overlaps removed). **The teacher-relabelled rows were not decontaminated.** This contaminates BTZSC
  Rotten Tomatoes (33 %, excluded from clean-18) and part of MASSIVE (≤ 11.7 % per language; ≤ 0.014
  effect on any suite). Overlap audit: `scripts/audit/overlap.py` → `runs/audit/overlap.json`, [docs/EVAL.md
  §2](https://github.com/amyrmahdy/decima/blob/main/docs/EVAL.md#2-status-of-each-benchmark-measured).
- **Decima 1.1** continues 1.0 for 0.5 epoch on the same mix plus 45,729 teacher-labelled System One
  decisions (Gemma-4-26B-A4B; 4,593 tasks over 62 business settings, EN/FA/AR/RU; `choice` with key →
  description criteria, `noul` statements including policy thresholds and near-misses, `score` with
  described levels), each rendered four ways (option and statement wordings vary), decontaminated
  against every evaluation item and against jabr/classifier-benchmark (11 rows dropped); 5 % of the
  tasks were held out as a dev set (choice 0.70 → 0.79, noul 0.59 → 0.75, score 0.47 → 0.63 vs 1.0).
- **Calibration**: one temperature (0.936; 1.0: 0.973) fitted on a 5 % hold-out of the training mix (teacher + gold
  rows), shipped in the config.

Full protocol, data provenance and overlap audit: [docs/EVAL.md](https://github.com/amyrmahdy/decima/blob/main/docs/EVAL.md).

## Limitations

- **Long, multi-fact business decisions** (JevBench, typed-decisions): Kev-0.8B is ahead; all small
  models are weak on typed-decisions.
- **Knowledge-heavy questions** (MMLU-style facts, math, chess): not what a 122M decision model knows.
- **Logic in English**: XNLI English 0.760 vs Laya 0.860.
- **Applying numeric rules**: still weak. Asked whether a refund is due under a 30-day rule, 1.1 says yes
  for a 12-day-old purchase (0.748) but also, wrongly, for a 45-day-old one (0.713). This is the main target
  of Decima 2.
- **Fine-grained sentiment** (5 levels): ~0.49–0.51.
- **Input length**: 512 tokens of state; longer inputs are truncated.
- **Colloquial loanwords and punctuation** (Persian spot checks): «یکی از یه کشور دیگه وارد اکانتم شده!»
  ("someone from another country got into my account!", with the common loanword «اکانت») was routed to
  technical support (0.525); the same sentence with «حسابم» went to account security (0.919). Adding one
  optional comma moved a Persian sales question from 0.936 to 0.916. Test with text written the way your
  users write.
- **"None of the above" options hurt**: on short option lists, adding "none of the above" pulled real
  tickets into it (18/20 → 11/20 in our demo set). Prefer a confidence threshold (on 20 realistic
  tickets, the 16 answers above 0.8 were all correct and the 4 below it, including both errors, would
  have been escalated) — but a threshold is not an off-topic filter: 1 of 6 off-topic messages still
  reached 0.80. ("None of the above" rows were in the teacher data; the model still over-selects the
  option when the list is short.)
- It picks among the options you give it; it can still pick the wrong one. Use the probabilities:
  act automatically only above a threshold, and escalate the rest.
- **Tool calls and shell commands**: not recommended — in our spot checks it called
  `rm -rf /var/lib/postgresql/data` non-destructive (0.722).

## Bias, risks and out-of-scope use

- **Out of scope**: medical, legal or financial *advice*; any decision about a real person (hiring,
  credit, moderation sanctions, triage of patients) without human review; tool-call or shell-command
  safety gates; open-ended generation.
- **One teacher.** About 300k training labels come from a single 26B teacher (Gemma-4-26B-A4B) and were
  not audited against human judgement beyond gold-label agreement checks. Its biases transfer to the
  student, including in the healthcare-intake and moderation domains of the training grid. The healthcare
  domain covers routing and urgency only, never diagnosis, but the model has no mechanism that enforces
  this.
- **Languages.** Evaluated in 20 languages; the weakest are Swahili (MASSIVE 0.730) and Hindi (XNLI 0.597).
  Performance in unevaluated languages is unknown, and a message can be routed correctly in one language
  and wrongly in another.
- **Selection optimism and in-distribution headlines.** See *What is zero-shot and what is not*.
- **Reporting problems**: open an issue at [github.com/amyrmahdy/decima/issues](https://github.com/amyrmahdy/decima/issues).

## Evaluation

Every number above comes from `bench/` in the repository: one item format for all systems, competitor
predictors run in their own environments, and one scorer (`bench/score.py`). Laya's and Kev's protocols
are reimplemented from their own evaluation code and verified by reproducing their published numbers
(to within 0.001) before comparing. Audit numbers (overlap, confidence intervals, calibration) come from
`scripts/audit/*.py` → `runs/audit/*.json`; parameter counts for every system from
`scripts/audit/params.py`. Code: [github.com/amyrmahdy/decima](https://github.com/amyrmahdy/decima)
(tag v1.0.0); the README's *Reproduce Decima's numbers* section lists the commands. Every system's
predictions, the scores and a text-free item manifest are in
[decima-bench-predictions](https://huggingface.co/datasets/amyrmahdy/decima-bench-predictions), so any
table can be re-scored without running a model.

## License

**Apache-2.0** (model weights, ONNX exports and code). The base encoder, multilingual-e5-small, is MIT;
the teacher, Gemma-4-26B-A4B, is Apache-2.0.

**Training-data disclosure.** Decima was trained on teacher-generated data and on gold labels derived
from public datasets under a range of licences — attribution (CC BY: banking77, CLINC150, MASSIVE,
WANLI, FarsTail), share-alike (SNLI, BoolQ, parts of MultiNLI), **non-commercial (CC BY-NC 4.0: ANLI,
XNLI, multilingual-NLI-26lang)**, and datasets published without a licence for research use (AG News,
SST-5). The model does not contain or reproduce these datasets, but if your use is commercial, review
their terms. Full table: `release/LICENSING.md`.

## Citation

```bibtex
@misc{madani2026decima,
  title  = {Decima: A Small, Calibrated, Permutation-Invariant Decision Model via Late-Interaction Distillation},
  author = {Madani, Amir Mahdi},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/amyrmahdy/decima-small}},
  note   = {Code: \url{https://github.com/amyrmahdy/decima}}
}
```
