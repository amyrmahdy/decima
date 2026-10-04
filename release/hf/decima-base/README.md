---
license: apache-2.0
language: [en, fa, ar, ru, de, fr, es, pt, tr, hi, zh, ja, ko, ur, vi, th]
library_name: onnx
pipeline_tag: zero-shot-classification
tags: [decision-model, system-one, jev, jev-alternative, typed-decisions, calibration, multilingual, onnx, int8, cpu]
base_model: jhu-clsp/mmBERT-base
---

# Decima-base: an open, calibrated Jev-style decision model

**Give it a situation, a question and your options. Get back calibrated probabilities.**

Decima is a *System One* model: it returns typed decisions instead of generating text, and speaks TypeSafe's
`/v1/systemone` wire format. The decision types are choose one, yes/no, ordered score and rank.

Decima-base is the second generation and the larger sibling of [decima-small](https://huggingface.co/amyrmahdy/decima-small):

- 321M parameters on mmBERT-base;
- int8 ONNX on a CPU;
- the same answer whatever order the options come in.

For coding-agent hooks (secret gate, bash gate, tool and command choice), use
[decima-agent](https://huggingface.co/amyrmahdy/decima-agent), fine-tuned from this model.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)). The name: **DECI**sion **MA**king, and Decima
is also the Roman Fate who decides.

| | decima-small 1.1 | **decima-base** | Laya-multilingual (same backbone, same size) |
|---|---:|---:|---:|
| Parameters | 122M | **321M** | 322M |
| [jabr/classifier-benchmark](https://github.com/jabr/classifier-benchmark) v2, 49 tasks (macro accuracy)¹ | 0.616 | **0.673** | — |
| The same 49 tasks in Persian (our [Persian edition](https://github.com/amyrmahdy/decima/tree/main/release/jabr-fa))¹ | — | **0.616** | — |
| Laya's MASSIVE + XNLI protocol, re-run by us, 29 suites / 19 languages² | 0.761 | **0.793** | 0.607 |
| Kev's published protocol, re-run by us, 8 suites | 0.762 | **0.790** | 0.580 |
| Decima bench, 12 suites, EN/FA/AR/RU² | 0.786 | **0.794** | 0.554 |
| BTZSC, 18 zero-shot suites (macro-F1)³ | 0.558 | **0.594** | — |
| Calibration error as shipped (ECE, mean over 48 suites) | 0.055 | **0.051** | 0.250 |
| Answer changes when options are shuffled | 0 % | **0 %** | 21 % |

¹ int8 on CPU through the benchmark's own harness. On jabr v2, the benchmark's published runs: Jev 0.966, Von 0.720,
GLiNER2 0.684, Laya 0.583.
² In-distribution for Decima: it trained on the MASSIVE and XNLI/MNLI train splits (test rows differ). Laya reports it
did not.
³ BTZSC suites are zero-shot for Decima: none of their training data was used.

**It also plays games.** These are sensor text in, one decision per move out, with the games' wording never in training:

- it wins **9 of 10** levels of a side-scrolling platformer;
- it averages **22.6 apples** per game of Nokia-style Snake.

## Quickstart

```bash
pip install "git+https://github.com/amyrmahdy/decima"
```

```python
from decima import Decima, Question

m = Decima.from_pretrained("amyrmahdy/decima-base")            # int8 ONNX, CPU
q = Question("Which team should handle this?", ["billing", "technical support", "sales", "account access"])
d = m.decide("I was charged twice for the same subscription this month.", q)
print(d.top, d.probs)
```

As a local TypeSafe-compatible server:

```bash
python -m decima.serve --model amyrmahdy/decima-base --port 11436          # binds to 127.0.0.1
```

The official `typesafe-sdk` works against it unchanged.

## How it works

The architecture is a late-interaction scorer:

- the state and question are encoded once;
- each option is encoded separately and cross-attends to the state;
- there is one logit per option.

Options never see each other, so permuting them permutes the probabilities exactly (0 % flips by construction).
`score` uses an ordinal head, and one fitted temperature calibrates every output.

The state budget is 512 tokens, and a longer state is rejected (HTTP 422), never silently truncated. decima-agent reads
2,048.

## Training

decima-base starts from [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT) and was trained in
two stages, 1.27M examples in the second:

- **Teacher-labelled decisions in the System One format.** The teachers are Gemma and Qwen3-Coder, run locally.
- **Public NLI and classification train splits.**
- **Procedural decisions with exact labels**, including grid and side-scroller scenes, reworded by an LLM under a
  "keep every number" check.

Every evaluation set was decontaminated against the training data. Licensing of the data:
[release/LICENSING.md](https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md).

## Limitations

- **Far from Jev on hard, unfamiliar decisions** (jabr v2: 0.673 vs 0.966). Use it for narrow, well-defined questions
  with clear options, and send the fuzzy ones to a bigger model.
- **Numbers and rules inside the text** (thresholds, dates, counting) remain its weakest area, as for decima-small.
- **Persian costs about 5 points** against English on the same tasks (0.616 vs 0.673). Other languages are less tested.
- **Slower than decima-small**, which is the choice for browsers and the smallest CPUs; a short decision with four options takes about 47 ms on one CPU thread and 21 ms on four (ARM cores of an
  NVIDIA GB10), vs 14 / 7 ms for decima-small.

## License and citation

Apache-2.0.

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
