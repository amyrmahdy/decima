---
pretty_name: Decima Synthetic Decisions
license: cc-by-4.0
language:
- en
- fa
- ar
- ru
multilinguality:
- multilingual
task_categories:
- text-classification
- zero-shot-classification
task_ids:
- multi-class-classification
- multi-label-classification
- natural-language-inference
size_categories:
- 100K<n<1M
source_datasets:
- original
annotations_creators:
- machine-generated
language_creators:
- machine-generated
tags:
- synthetic
- distillation
- soft-labels
- calibration
- decision-making
- routing
- llm-generated
- gemma
configs:
- config_name: short
  default: true
  data_files:
  - split: train
    path: short/train.parquet
  - split: test
    path: short/test.parquet
- config_name: long
  data_files:
  - split: train
    path: long/train.parquet
  - split: test
    path: long/test.parquet
- config_name: relabel
  data_files:
  - split: train
    path: relabel/train.parquet
  - split: test
    path: relabel/test.parquet
---

# Decima Synthetic Decisions

About 190k **fully synthetic decision problems** in English, Persian, Arabic and Russian, each with a
**calibrated soft label**. The teacher, **Gemma-4-26B-A4B-it**, wrote every part of every row: a *state*
(the text a system has in front of it), a *question*, a list of natural-language *choices*, and the
probability a careful expert would give each choice. This is the synthetic part of the data used to
distil [Decima-small](https://huggingface.co/amyrmahdy/decima-small). Decima-small is a 122M
decision model on a `multilingual-e5-small` body. It reads (state, question, choices) and returns a
calibrated distribution.

The dataset contains **no text from other datasets** (see *Personal data and third-party text*). Gold-labelled public datasets that
Decima-small also trained on are *not* redistributed here. They are rebuilt by scripts, see
[`../README.md`](../README.md).

| config | rows | distinct states | what |
|---|---:|---:|---|
| `short` (default) | 86,900 | 86,893 | teacher invents short states (1–6 sentences) + a decision over a (domain × language × kind × #choices) grid; 2–125 choices |
| `long` | 10,244 | 2,692 | "Jev-style" business cases: one long artefact (log, email thread, invoice, agent trace, contract redline… median 443 words) with 3–5 independent questions |
| `relabel` | 92,427 | 75,804 | `short`-style generated states **re-asked under new questions** the teacher invented, so state and choices were not written together |

Four decision **kinds**, which are all Decima's API has:

- `choose`: exactly one choice is right (intent, route, category, action). The probabilities sum to 1.
- `verify`: a yes/no proposition. There are 2 choices, affirm first. The probabilities sum to 1.
- `score`: ordered levels, lowest to highest (urgency, risk, sentiment…). The probabilities sum to 1,
  and the mass sits on adjacent levels.
- `rank`: each choice is judged **independently** (which tags apply). Each probability is in [0, 1], and
  they **do not** sum to 1.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("amyrmahdy/decima-synthetic-decisions", "short")   # or "long", "relabel"
row = ds["train"][0]
print(row["state"], row["question"], row["choices"], row["probs"])
```

For distillation, train on `probs` (KL / soft cross-entropy). `gold` is the teacher's single best
choice. For `rank`, use a per-choice binary loss.

## Schema and examples

| column | type | notes |
|---|---|---|
| `id` | string | 16-hex sha1 of state ␟ question ␟ choices |
| `config` | string | `short` / `long` / `relabel` |
| `group` | string | rows that share a generation call (`short`: one catalogue/call of 10 examples) or a state (`long`, `relabel`); splits never cut a group |
| `kind` | string | `choose` / `verify` / `score` / `rank` |
| `domain` | string | grid domain (20 in `short`/`relabel`, 30 in `long`) |
| `state_lang`, `choice_lang` | string | `en` `fa` `ar` `ru`. The question is in `choice_lang` for short/relabel and in `state_lang` for long |
| `cross_lingual` | bool | `state_lang != choice_lang` |
| `state`, `question` | string | NFC; for fa/ar also: letters unified to the language's canonical form (ی/ي, ک/ك…), tashkeel and tatweel stripped, Arabic-Indic digits → ASCII (`decima.normalize`) |
| `choices` | list[string] | 2–125 entries |
| `n_choices` | int32 | |
| `gold` | int32 | teacher's best choice (0-based) |
| `probs` | list[float64] | teacher distribution, aligned with `choices`; every entry ≥ 1e-4 except `rank` |
| `label_format` | string | `full`: the teacher wrote every probability. `topk`: the teacher wrote the 3–6 most plausible `[index, prob]` pairs and the leftover mass was spread uniformly (floor 1e-4). Used for ≥ 16-choice catalogues and all `relabel` rows |
| `none_idx` | int32 or null | index of the "none of the above" / "insufficient information to decide" choice, if present |
| `scenario`, `artefact_format`, `q_index` | string/string/int32 or null | `long` only: requested sub-scenario, artefact form, question position |
| `uncertain_requested` | bool or null | `long` only: the prompt asked for a genuinely uncertain label (top ≤ 0.65) |
| `gold_tie` | bool or null | `long` only: `gold` is within 0.05 of the argmax but not the argmax (78 rows) |
| `in_decima_small_training` | bool | the row was in Decima-small (v1i) training data. All `short`/`relabel` rows were; 6,004 of the `long` rows were (a snapshot taken mid-generation) |

`short` example (test split):

```json
{"id": "03cf8ad1efe2d511", "config": "short", "group": "short-6259", "kind": "choose", "domain": "support",
 "state_lang": "en", "choice_lang": "en", "cross_lingual": false,
 "state": "Transcript Excerpt:\nUser: 'How much is the extra storage?'\nAgent: 'It depends on which tier you are currently on.'\nUser: 'I'm on the Basic. Is it a monthly or yearly cost?'\nAgent: 'We offer both, but most people prefer the yearly discount.'",
 "question": "What is the user attempting to do?",
 "choices": ["Purchase an add-on", "Cancel a service", "Troubleshoot an error", "Update a profile", "Compare two different products"],
 "n_choices": 5, "gold": 0, "probs": [0.65, 0.05, 0.05, 0.05, 0.2], "label_format": "full", "none_idx": null,
 "scenario": null, "artefact_format": null, "q_index": null, "uncertain_requested": null, "gold_tie": null,
 "in_decima_small_training": true}
```

`long` example (state truncated here):

```json
{"id": "07e3a1434ecf1bc0", "config": "long", "group": "jev-0-2579", "kind": "verify", "domain": "it_helpdesk",
 "state_lang": "en", "choice_lang": "en", "cross_lingual": false,
 "state": "[2022-03-10 08:14:22] [INFO] [Bot-Core-v2.4.1] Session started for user: j_mendoza_99\n[2022-03-10 08:14:25] [INFO] [Bot-Core-v2.4.1] User matched to: Julian Mendoza (ID: 88291) - Dept: Logistics - Location: Chicago Hub\n …",
 "question": "Was Julian Mendoza's account status found to be inactive during the initial bot check?",
 "choices": ["Yes, it was inactive", "No, it was active"], "n_choices": 2, "gold": 1, "probs": [0.02, 0.98],
 "label_format": "full", "none_idx": null, "scenario": "new-hire access bundle",
 "artefact_format": "chat with the IT bot handing off to a human", "q_index": 1,
 "uncertain_requested": false, "gold_tie": false, "in_decima_small_training": false}
```

## Statistics

Splits: `test` holds about 2 % of **groups**, chosen by `sha1(group) mod 100 < 2`.

| config | train | test |
|---|---:|---:|
| short | 85,035 | 1,865 (188 calls) |
| long | 10,049 | 195 (52 states) |
| relabel | 90,603 | 1,824 (1,477 states) |

**Kind × state language (`short`)**

| kind | en | fa | ar | ru | total |
|---|---:|---:|---:|---:|---:|
| choose | 10,089 | 8,082 | 8,420 | 6,492 | 33,083 |
| verify | 5,923 | 5,111 | 4,866 | 3,949 | 19,849 |
| score | 5,405 | 4,783 | 4,420 | 3,413 | 18,021 |
| rank | 4,879 | 3,827 | 4,137 | 3,104 | 15,947 |
| **total** | 26,296 | 21,803 | 21,843 | 16,958 | 86,900 |

**Kind × state language (`long`)**

| kind | en | fa | ar | ru | total |
|---|---:|---:|---:|---:|---:|
| choose | 3,191 | 431 | 429 | 446 | 4,497 |
| verify | 2,212 | 315 | 325 | 312 | 3,164 |
| score | 1,765 | 280 | 267 | 271 | 2,583 |
| **total** | 7,168 | 1,026 | 1,021 | 1,029 | 10,244 |

`relabel`: choose 24,676 · verify 24,580 · score 24,699 · rank 18,472; en 27,325 · fa 24,161 · ar 23,791 · ru 17,150.

**Choices per row**

| config | 2 | 3–5 | 6–12 | 13–50 | 51–125 |
|---|---:|---:|---:|---:|---:|
| short | 21,902 | 32,656 | 19,515 | 8,984 | 3,843 |
| long | 3,164 | 4,347 | 2,733 | — | — |
| relabel | 24,580 | 61,718 | 6,129 | — | — |

**Other properties**

| | short | long | relabel |
|---|---:|---:|---:|
| cross-lingual rows (state ≠ choice language) | 13,176 | 813 | 4,706 |
| rows with a none / insufficient-info option | 7,800 | 642 | 24,288 |
| …of which it is the gold | 1,819 | 77 | 9,912 |
| `label_format = topk` | 12,827 | 0 | 92,427 |
| state words p10 / median / p90 / max | 19 / 31 / 47 / 97 | 280 / 443 / 712 / 1,463 | 19 / 31 / 47 / 97 |
| mean top probability (non-rank) | 0.72 | 0.73 | 0.84 |
| share with top prob ≥ 0.9 / ≤ 0.6 (non-rank) | 0.24 / 0.36 | 0.29 / 0.36 | 0.41 / 0.13 |
| `gold` = argmax(`probs`) | 94.4 % | 99.2 % | 99.3 % |

Domains: `short`/`relabel` cover support, ops, security, legal, finance, hr, healthcare, ecommerce,
agent_tools, email, docqa, travel, it_helpdesk, government, education, logistics, insurance, social,
realestate and dev, at 4.0–4.9k rows each. `long` covers 30 domains, among them support triage, model
routing, RAG relevance, citation support, tool-call risk, agent verification, security incidents,
invoice/AP, fraud, moderation and code review, at 259–433 rows each.

## How it was made

Teacher: `google/gemma-4-26B-A4B-it`, run as the NVFP4 build `nvidia/Gemma-4-26B-A4B-NVFP4` on vLLM
(one NVIDIA GB10), with thinking off and temperature 0.9. JSON mode was used for `short`/`relabel`.
Generation took about 24 h for `short` + `relabel` together and about 25 h for `long`. The code is in the Decima
repository ([github.com/amyrmahdy/decima](https://github.com/amyrmahdy/decima)):

- `teacher/generate.py` + `teacher/prompts.py` → `short`. It uses a seeded grid in which call *i*
  always asks for the same cell, so runs are resumable and no call is repeated. Mix: languages EN 30 /
  FA 25 / AR 25 / RU 20; about 15 % cross-lingual (mostly non-English state with English choices);
  kinds choose 40 / verify 22 / score 20 / rank 18; choose sizes {2, 3, 5, 8, 12, 20, 50, 100}; 25 % of
  choose cells append "none of the above" and make it the gold in a stated third of the examples. There
  are 10 examples per call. For ≥ 20 choices the teacher first writes one shared *catalogue* (with near
  neighbours) and then labels 10 states against it with sparse top-k probabilities. The prompt asks for
  realistic, concrete states in varied forms (ticket, chat, logs, email, document paragraph, tool
  context). It asks for at least a third of the examples to be genuinely ambiguous (top ≤ 0.6), for the
  answer never to be restated in the state, and for **calibrated** probabilities, with no exact zeros
  for plausible choices.
- `teacher/generate_jev.py` → `long`. One call writes one long state (asked for 300–1,500 words;
  200–2,400 accepted) plus 3–5 questions. The grid covers 30 domains, each with sub-scenarios and
  typical artefact forms. The domain list was written from public descriptions of the use cases, not
  from any evaluation set. Languages EN 70 / FA, AR, RU 10 each (30 % of non-English states have
  English options); kinds choose 45 / verify 30 / score 25; about 15 % of choose questions end in
  "none of the above" or "insufficient information"; about 30 % of questions are asked to be uncertain.
  The prompt asks for invented organisation and person names, realistic noise (signatures, irrelevant
  log lines) and "at least one detail that matters for a question but is easy to miss". The state is
  returned between text markers instead of inside JSON, which keeps quoting and escaping intact.
- `teacher/label.py` → `relabel`. For each (domain, language) group of `short` states, the teacher
  first invents about 8 new question + choice sets covering all four kinds, then labels the states in
  batches of 20 under one question with top-k probabilities.

**Validation.** Every reply is parsed and checked. A malformed example is dropped; the other examples
from the same call are kept. `probs` must have one value per choice, each in [0, 1]. For non-rank
rows, a full vector's mass must be within 0.9–1.1 and a top-k list's mass within 0.5–1.02. The result
is renormalised with a 1e-4 floor per choice. `gold` must be a valid index. `verify` must have exactly 2 choices. Choices must be distinct,
and catalogue sizes must be within 0.8–1.25× of the size asked for. For `long`, `gold` must be the
argmax or within 0.05 of it (then `gold_tie` is set), questions within a state must be distinct, and
the state must be 200–2,400 words. The reject logs hold 3,175 entries for `short` and 589 for `long` (bad examples plus failed calls).
Text is normalised per language (`decima.normalize`).

**Dedup.** Exact duplicates by `id` (sha1 of state + question + choices) are dropped at generation
time, and for `long` also by state hash. `build_synthetic.py` dedups by `id` across configs again and
found 0 duplicates.

**Release build.** `release/hf/datasets/build_synthetic.py` turns the raw generator files into these
splits. It is deterministic: rows are sorted by id and the split comes from a hash, so the gzip JSONL
output is byte-identical across runs. It drops internal fields (the local gateway alias `model`, the
grid index `cell`, `source`, the label-job `batch` key, and empty resume markers) and validates every
row again. Rows the teacher labelled on *public* datasets' text are never exported.

## Intended use

- Distilling or fine-tuning small models that pick among natural-language options with **calibrated**
  confidence: routers, triage, intent/tool selection, verification gates, graded scoring.
- Research on soft-label distillation, calibration, "none of the above" handling, cross-lingual
  decisions, and long-context decisions (`long`).
- The `test` splits are for sanity checks on in-distribution synthetic data. **They are not a
  benchmark of real-world ability.** Decima's public numbers come from real datasets, see
  `decima-bench-predictions`.

Out of scope: medical, legal or financial *advice*. The healthcare domain covers routing and urgency
only. The data is also not meant for any decision about a real person without human review.

## Limitations and biases

- **One teacher, synthetic distribution.** Every state, question and label comes from one model at one
  temperature. Its blind spots, style and errors are the dataset's. Nothing here is a human judgement.
- **Teacher errors.** Labels were not human-verified. `gold` disagrees with argmax(`probs`) in 5.6 % of
  `short` rows (the check was stricter for `long`). Some probability vectors are internally
  inconsistent with the stated gold.
- **Calibration is asked for, not guaranteed.** The probabilities are the teacher's *stated*
  confidence. The prompts push for ambiguity (≈ 36 % of `short`/`long` rows have top ≤ 0.6), which
  probably spreads the mass more than the teacher's real accuracy would justify. Verbalised
  probabilities cluster on round values: 57 % of all values in `full` rows are multiples of 0.05. In `topk` rows the leftover mass is
  spread **uniformly** over unlisted choices, which is an artefact of the format, not a judgement. Treat
  `probs` as soft targets for distillation, not as ground-truth likelihoods. Decima-small applies a
  temperature fitted afterwards.
- **Mode collapse in names and details.** Invented names repeat heavily. For example, "Marcus Thorne"
  appears in 814 of 2,692 `long` states and "Sarah Jenkins" in about 1.2k `short` states, and IP
  `192.168.1.45` appears in about 1.6k states. Organisations such as "Vanguard…", "Lumina…" and
  "Aetheris…" recur. Models may learn spurious name cues.
- **Language imbalance and translation-ese.** `long` is 70 % English. Persian, Arabic and Russian text
  is model-written and can read as translated. Persian dates mix Solar Hijri and Gregorian calendars.
- **Western/corporate framing.** Domains, currencies and institutions lean toward US/EU enterprise
  settings even in non-English rows.
- **Choice lists are written together with the state** in `short`/`long`, so they often fit too
  neatly. `relabel` exists to reduce this.

## Personal data and third-party text

- The prompts ask for invented names and details. States contain synthetic emails, phone numbers,
  IPs, IBAN-like strings and card numbers. Most are obvious test values (`4111 1111 1111 1111`,
  `DE89 3704 0044 0532 …`, `192.168.x.x`). Some invented addresses used real providers or real-looking company domains (`@gmail.com`, `@mail.com`, `@mail.ru`…). This release was built with `--scrub-free-mail`, which gives **every e-mail address in every field** (state, question, options) a reserved RFC 2606 domain — `@gmail.com` → `@gmail.example`, `@vertex-corp.com` → `@vertex-corp.example` — so no address points at a real domain or mailbox (verified: all 25,007 addresses end in `.example` or `.test`). This is the only change from the text Decima-small trained on. A celebrity's name appears as a fictional employee
  name in 2 rows. No real person's data is known to be included. Please report anything that looks
  real and it will be removed.
- **No third-party text.** Each distinct `short` (86,893) and `long` (2,692) state was checked against
  751,601 public texts used elsewhere in Decima's training (exact match and any shared 10-word
  sequence). There were 0 exact matches, and the only overlap was one generic phrase in one `short`
  state. This cannot rule out phrases memorised from the teacher's own pre-training.

## Licence and terms

- Dataset licence: **CC BY 4.0** (owner decision 2026-09-26; see `release/LICENSING.md`).
- Teacher: Gemma 4 is released under **Apache-2.0**
  ([Gemma 4 licence](https://ai.google.dev/gemma/docs/gemma_4_license)). Apache-2.0 places no
  restrictions on model outputs. The Gemma Terms of Use, which cover Gemma 1–3n and have
  "Model Derivatives"/distillation clauses, do not apply to Gemma 4. We ask users, voluntarily, to follow
  Google's [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- Machine-generated text may not be protected by copyright in every jurisdiction. The licence mainly
  states attribution expectations and disclaims warranty.

## Citation

```bibtex
@misc{decima2026synthetic,
  title  = {Decima Synthetic Decisions},
  author = {Madani, A. M.},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/amyrmahdy/decima-synthetic-decisions}}
}
```
