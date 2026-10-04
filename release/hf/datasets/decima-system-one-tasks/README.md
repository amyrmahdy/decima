---
pretty_name: Decima System One Tasks
license: cc-by-4.0
language:
- en
- fa
- ar
- ru
- es
- de
- fr
- tr
- zh
multilinguality:
- multilingual
task_categories:
- text-classification
- zero-shot-classification
size_categories:
- 100K<n<1M
annotations_creators:
- machine-generated
language_creators:
- machine-generated
tags:
- system-one
- jev
- typesafe
- typed-decisions
- noul
- soft-labels
- calibration
- synthetic
- distillation
configs:
- config_name: s2
  default: true
  data_files:
  - {split: train, path: s2/train.parquet}
  - {split: test, path: s2/test.parquet}
- config_name: s1
  data_files:
  - {split: train, path: s1/train.parquet}
  - {split: test, path: s1/test.parquet}
---

# Decima System One Tasks

**About 138k typed decisions in the Jev / TypeSafe schema**, across 15k tasks and nine languages, each with a
calibrated soft label. A *task* is what a developer writes once: a `noul` statement to judge true or false, a `choice`
with named criteria, or a `score` with ordered levels. Each task comes with many *states* it is applied to, the way
System One models are used in production.

This is the System One part of the training data of [decima-base](https://huggingface.co/amyrmahdy/decima-base) and
[decima-agent](https://huggingface.co/amyrmahdy/decima-agent). It is in the raw schema rather than a rendered prompt,
so it fits any model that speaks it: Kev, Laya, Ollaya-style models, or your own.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).
Code: [github.com/amyrmahdy/decima](https://github.com/amyrmahdy/decima).

## Configs

| config | train rows / tasks | test rows / tasks | languages |
|---|---:|---:|---|
| `s2` | 87,410 / 9,970 | 4,586 / 523 | EN 72 %; FA, AR, RU, ES, DE, ZH, TR, FR |
| `s1` | 43,303 / 4,350 | 2,415 / 243 | EN 77 %; FA, AR, RU |

- **`s2`, the second generation, is the one to use.**
  - It is shaped like real use: ops and enterprise, security/DevOps/compliance, safety and moderation, triage and
    services, content, spatial scenes, everyday knowledge.
  - A **reasoning** cluster (about 28 %) holds statements whose answer needs a rule applied to facts in the state:
    numeric thresholds, dates relative to a date in the state, two conditions, "unless" exceptions, negation, a policy
    excerpt plus a statement. It includes near-miss false cases.
  - 1–3 questions per task share the same states, and about 18 % of the states are long (250–900 words: logs, threads,
    contracts, tickets).
- **`s1`** is the first generation: one question per task, short states.

**The test splits hold out whole tasks** (5 % by task group), exactly as in training, so they measure new questions,
not new states of seen ones. Rows matching any evaluation item we report on (including jabr/classifier-benchmark) were
removed.

## Columns

| column | |
|---|---|
| `type` | `noul`, `choice` or `score` |
| `question_id` | snake_case name of the question (`s2` only), as TypeSafe users name them |
| `instructions` | the statement (`noul`) or the question (`choice`, `score`) |
| `criteria` | JSON. `choice`: `{key: when it applies}`. `noul`: optional `{"true": …, "false": …}`. `score`: the ordered levels, lowest first |
| `state` | the input text |
| `p_true` | `noul`: the teacher's probability that the statement is true |
| `probs` | JSON. `choice`: `{key: probability}`. `score`: one probability per level |
| `task_group`, `state_group` | rows of the same task / the same state |
| `cluster`, `domain` | where the task comes from |
| `long_state`, `state_lang`, `question_lang` | |
| `writer`, `labeler` | the model that wrote the row and the one whose distribution is the label |

```python
from datasets import load_dataset
import json
ds = load_dataset("amyrmahdy/decima-system-one-tasks", "s2", split="train")
r = ds[0]
print(r["type"], r["instructions"], json.loads(r["criteria"] or "null"), r["state"], r["p_true"] or json.loads(r["probs"]))
```

## How it was made

- **Writer and teacher: Gemma 4 26B-A4B-it** (Apache-2.0), served locally. One call writes one task and 10 states
  (3 long ones in long-state calls), then gives a probability for every state. 1.4 % of `s2` was written and labelled
  by Qwen3-Coder-Next (Apache-2.0), marked in `writer`.
- **A seeded grid** of type, cluster, domain and language, so the mix is controlled, not whatever the model likes
  to write. Domains were written for the generator; nothing was taken from an evaluation set.
- **Labels are a distribution, not a vote.** The teacher scored 0.942 on jabr/classifier-benchmark v2, so expect a
  few percent of wrong labels, more in the reasoning cluster.
- Invented e-mail addresses at real providers were rewritten to reserved `.example` domains.

## Limitations

- **Synthetic and model-labelled.** It is a distillation set, not human-verified gold.
- **The teacher's style shows.** Names repeat, and the states are cleaner than real user text.
- **Smaller languages are thin:** a few hundred to a few thousand rows each.

## License

CC BY 4.0. Please cite:

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
