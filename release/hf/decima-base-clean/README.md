---
license: apache-2.0
language: [en, fa, ar, ru, de, fr, es, pt, tr, hi, zh, ja, ko, ur, vi, th]
library_name: onnx
pipeline_tag: zero-shot-classification
tags: [decision-model, system-one, jev-alternative, zero-shot, btzsc, calibration, multilingual, onnx, int8, cpu, commercial-use]
base_model: jhu-clsp/mmBERT-base
---

# decima-base-clean

**The licence-clean, truly zero-shot version of [decima-base](https://huggingface.co/amyrmahdy/decima-base).** Same
architecture, same API, trained without any non-commercial data and without any data from the BTZSC benchmark.

Use it when:
- you need a model whose training data has no non-commercial licences anywhere. It's also the base of
  [decima-guard](https://huggingface.co/amyrmahdy/decima-guard).
- you want an honest zero-shot number. Nothing from BTZSC was used for training, development or picking the checkpoint.

Use [decima-base](https://huggingface.co/amyrmahdy/decima-base) when you want the best scores on routing, intent and
NLI-style questions. It trained on MASSIVE, XNLI and similar train splits, so it's stronger on those, and part of its
data is non-commercial.

## Results

| | decima-base 2.0 | **decima-base-clean** | multilingual-e5-small, untrained (zero-shot baseline) |
|---|---:|---:|---:|
| BTZSC, 22 datasets (macro-F1), zero-shot | 0.639 (not zero-shot) | **0.556** (int8: 0.555) | 0.546 |
| Kev's protocol, 8 suites | 0.790 | 0.648 | — |
| Laya's MASSIVE + XNLI protocol | 0.793 | 0.616 | — |
| Decima bench, 12 suites, EN/FA/AR/RU | 0.794 | 0.589 | — |
| Typed decisions (TypeSafe) | 0.484 | 0.477 | — |
| Answer changes when options are shuffled | 0 % | **0 %** | — |

decima-base 2.0's higher numbers on the middle rows come from training on those suites' own train splits. Its BTZSC
number isn't zero-shot either. decima-base-clean saw none of them.

**How the BTZSC number was made** (the protocol was written before training:
[PREREG-base3.md](https://github.com/amyrmahdy/decima/blob/main/release/PREREG-base3.md)):
- **Removed from training:** banking77 and MASSIVE, plus six CLINC utterances identical to a BTZSC item. No training
  text matches any of the 20,776 BTZSC item texts.
- **Model selection:** one run with fixed settings. The checkpoint was picked by the run's own random holdout.
- **The BTZSC run:** once, after the checkpoint was final. We later trained the same model 0.4 epoch longer. It scored
  0.551, and we kept the original.

It also plays: 9 of 10 platformer levels and 22.8 apples per game of Snake, from text sensors.

## Quickstart

```bash
pip install decima-ai
```

```python
from decima import Decima, Question

m = Decima.from_pretrained("amyrmahdy/decima-base-clean")
q = Question("Which team should handle this?", ["billing", "technical support", "sales", "account access"])
print(m.decide("I was charged twice for the same subscription this month.", q).top)
```

## Training data

- **Licence-clean public sets:** CLINC150 (CC-BY), SNLI, MNLI, WANLI, BoolQ and FarsTail.
- **Our own teacher-labelled System One decisions:** Gemma 4 run locally, labels only on texts we can redistribute.
- **Procedural game and grid decisions** with exact labels.

The full list and licences are in
[release/LICENSING.md](https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md). The exact filter is in
`data/clean`.

## Limitations

The same as decima-base, and weaker on intent and routing questions it has no training data for. States are limited to
512 tokens; longer ones are rejected, never cut.

Apache-2.0. Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).
