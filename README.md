# Decima — open, CPU-sized Jev-style decision models

**Situation + question + your options → calibrated probabilities.**
Like TypeSafe's Jev, Decima is a *System One* model: typed decisions with probabilities instead of generated
text. Unlike Jev, it is open (Apache-2.0), runs on a CPU, and speaks the same `/v1/systemone` API.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy) · [amyrmahdy.github.io](https://amyrmahdy.github.io)).
The name: **DECI**sion **MA**king — and Decima is also the Roman Fate who decides.

| Model | Size | For | jabr v2 (49 tasks) |
|---|---|---|---:|
| [decima-agent](https://huggingface.co/amyrmahdy/decima-agent) | 321M | coding-agent hooks: secret gate, bash gate, tool and command choice, model routing | 0.675 |
| [decima-base](https://huggingface.co/amyrmahdy/decima-base) | 321M | general decisions, multilingual | 0.673 |
| [decima-small](https://huggingface.co/amyrmahdy/decima-small) | 122M | the smallest CPUs and the browser | 0.616 |

**2.1 (2026-10-06):**
- **decima-agent 2.1** ([v2.1](https://huggingface.co/amyrmahdy/decima-agent/tree/v2.1)) is trained on field logs: real-shape
  shell commands, local services, recursive deletes. Hand-written agent decisions: 0.90 → 0.93.
- **Long questions no longer tie.** When "question + option" ran past the 64-token choice window, the options were cut off,
  could become identical, and the answer was exactly 0.5. Models that predate the fix keep the rendering they were trained
  with, except in that case; every other decision is unchanged (checked on 4,644 typed-decision items). Models trained from
  now on keep every option whole ([`decima/render.py`](decima/render.py)).
- Decima also runs in [Ollaya](https://ollaya.dev/library/decima) (0.11.0), a Rust runtime that matches this code
  decision for decision.

**2.0 (2026-10-04):**
- **decima-base** (mmBERT-base) is a new generation.
- **decima-agent**, fine-tuned from it, is the decision layer for agent loops: 0.90 on 130 hand-written agent decisions
  (decima-base without agent training: 0.58).
- [**Claude Code hooks**](integrations/claude-code) run locally, put rules first and fail closed.
- A [**Persian edition**](release/jabr-fa) of the jabr benchmark.

**Open data:**

| Dataset | Rows | What |
|---|---:|---|
| [decima-agent-decisions](https://huggingface.co/datasets/amyrmahdy/decima-agent-decisions) | 412k | secret gate, bash gate, read-only, tool and command choice, next step, model tier; held-out test splits and 130 hand-written cases |
| [decima-system-one-tasks](https://huggingface.co/datasets/amyrmahdy/decima-system-one-tasks) | 138k | typed decisions (`noul`, `choice`, `score`) in the Jev/TypeSafe schema, 15k tasks, nine languages |
| [decima-game-decisions](https://huggingface.co/datasets/amyrmahdy/decima-game-decisions) | 333k | grid, side-scroller and Snake decisions with exact labels |
| [jabr-v2-persian](https://huggingface.co/datasets/amyrmahdy/jabr-v2-persian) | 845 | the jabr classifier benchmark v2 in Persian, reviewed case by case |
| [decima-synthetic-decisions](https://huggingface.co/datasets/amyrmahdy/decima-synthetic-decisions) | 190k | decima-small's synthetic decisions |

Decima does not write text. It decides: routing, triage, intent, classification, verification,
ranking — any bounded choice your software needs to make, with a confidence it can threshold.

- **decima-small: 122M parameters**, ONNX int8, **~20 ms** per decision on one laptop CPU core (4 options, short input)
- **Options in plain text**, given at call time; evaluated in 20 languages (weakest: Swahili, Hindi)
- **Order-proof** — shuffling the options never changes the answer (0 %; the four other open decision models we compared: 10–27 %)
- **Well calibrated as shipped** — lowest calibration error as shipped in every pairwise comparison we ran
  (ECE 0.058–0.064 vs 0.117–0.370)

[Model on Hugging Face](https://huggingface.co/amyrmahdy/decima-small) · [Technical report](docs/TECHNICAL-REPORT.md) · [Evaluation & claims audit](docs/EVAL.md) · [x86 benchmark](docs/BENCH-x86.md) · [Playground](https://huggingface.co/spaces/amyrmahdy/decima-playground) · [Predictions dataset](https://huggingface.co/datasets/amyrmahdy/decima-bench-predictions) · [Synthetic training data](https://huggingface.co/datasets/amyrmahdy/decima-synthetic-decisions)

![Shuffle the options: Decima's answer never changes](release/figures/option_order_flips.png)

## Install

```bash
pip install "git+https://github.com/amyrmahdy/decima"      # from source (tag v2.0.0)
```

The runtime (`decima.Decima`) imports no PyTorch and needs no GPU — ONNX Runtime, numpy and a
tokenizer. The package as it stands still installs the training stack as well (torch,
sentence-transformers, datasets), because runtime and training share one `pyproject.toml`. A
runtime-only package without the training dependencies is planned.

## Use

```python
from decima import Decima, Question

decima = Decima.from_pretrained("amyrmahdy/decima-small")   # int8 ONNX, ~140 MB, one CPU thread

# outputs below are real (Decima-small int8, export/v1i-int8)
q = Question("Which team should handle this request?",
             ["billing", "technical support", "sales", "account security"])

decima.decide("Someone logged into my account from another country.", q)
# → {'billing': 0.004, 'technical support': 0.031, 'sales': 0.001, 'account security': 0.964}

decima.decide("یک نفر از کشور دیگری وارد حسابم شده است.", q)   # the same message in Persian
# → {'billing': 0.012, 'technical support': 0.024, 'sales': 0.002, 'account security': 0.962}
```

Four question kinds: `choose` (one of N) · `score` (ordered levels) · `verify` (yes/no) · `rank`
(independent probability per option). State, question and options may be in different languages.
`Question(..., lang="fa")` (or `"ar"`) turns on Persian/Arabic text normalisation; for the Persian
example above the output is the same with or without it.

### Jev / TypeSafe-compatible API

Decima speaks TypeSafe's System One wire format (`choice`, `score`, `noul`), so code written for Jev runs on
a local Decima:

```bash
python -m decima.serve                     # http://127.0.0.1:11436 · POST /v1/systemone, GET /v1/models
export TYPESAFE_BASE_URL=http://127.0.0.1:11436 TYPESAFE_API_KEY=local TYPESAFE_DEFAULT_MODEL=decima-small
```

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:11436")
r = client.system_one("Someone logged into my account from another country and changed my password.", {
    "team": Choice(instructions="Which team should handle this?",
                   criteria={"billing": "charges and refunds", "tech": "bugs and errors", "security": "account takeover"}),
    "urgent": Noul(instructions="The customer needs help within the hour."),
    "anger": Score(instructions="How upset is the customer?", criteria=["calm", "annoyed", "furious"]),
}, model="decima-small")
```

Or in-process, without a server: `from decima.systemone import system_one`.

## How it compares

| | Decima-small | Kev-0.5B | Kev-0.8B | Laya | Laya-multilingual |
|---|---:|---:|---:|---:|---:|
| Parameters | **122M** | 494M | 753M | 421M | 322M |
| Laya's MASSIVE + XNLI protocol, re-run by us (29 suites, 19 languages; in-distribution for Decima¹) | **0.761** | 0.527 | — | 0.445 | 0.607 |
| Kev's published protocol, re-run by us (8 suites) | 0.762 | 0.779 | **0.794** | 0.681 | 0.580 |
| jabr/classifier-benchmark v2, 49 tasks (zero-shot by data) | 0.616 | — | — | 0.583 | — |
| Answer changes when options are shuffled² | **0 %** | 22 % | 10 % | 27 % | 21 % |

¹ Decima trained on the MASSIVE train split (all 51 locales; test rows differ) and on the XNLI/MNLI train
splits; Laya reports it did not train on MASSIVE or XNLI.
² Mean over the Kev, Laya and Decima-bench suites each model was run on.

All models run by us on one harness; competitors' published numbers were reproduced (to within 0.001)
first. Decima trained on the train splits of several of these datasets — see [docs/EVAL.md](docs/EVAL.md)
for what is zero-shot, what is in-distribution, and the limitations (long multi-fact business
decisions, knowledge-heavy questions). Not recommended for tool-call or shell-command safety.

## Reproduce Decima's numbers

Every system's predictions, including Decima's and the competitors', are downloadable in
[decima-bench-predictions](https://huggingface.co/datasets/amyrmahdy/decima-bench-predictions), with the
scores and a text-free manifest of every item; its card has the commands to re-score every table without
running a model. Decima's predictions can also be regenerated from the released weights and the frozen
item builders:

```bash
uv sync
# item sets (calibration items per suite: kev/laya 300, decima 500, typed 1000, others none)
uv run python -m bench.items --suites 'kev/*'    --calib 300  --out runs/items/kev.jsonl
uv run python -m bench.items --suites 'laya/*'   --calib 300  --out runs/items/laya.jsonl
uv run python -m bench.items --suites 'decima/*' --calib 500  --out runs/items/decima.jsonl
uv run python -m bench.items --suites 'jevbench/*' 'typed/*' --calib 1000 --out runs/items/jevtyped.jsonl
uv run python -m bench.items --suites 'btzsc/*'               --out runs/items/btzsc.jsonl
# predictions (int8 ONNX, 1 thread) and scores, one set at a time
for s in kev laya decima jevtyped btzsc; do
  uv run python -m bench.predict --items runs/items/$s.jsonl --system onnx --checkpoint export/v1i-int8 \
      --threads 1 --out runs/preds/$s--decima.jsonl
  M=acc; [ $s = btzsc ] && M=macro_f1
  uv run python -m bench.score --items runs/items/$s.jsonl --metric $M --preds decima=runs/preds/$s--decima.jsonl
done
# audit numbers: overlap, confidence intervals, headline calibration
uv run python scripts/audit/overlap.py && uv run python scripts/audit/bootstrap.py && uv run python scripts/audit/calibration.py
```

`export/v1i-int8` is the int8 ONNX export; the same files are `onnx/int8/` in the
[model repository](https://huggingface.co/amyrmahdy/decima-small)
(`hf download amyrmahdy/decima-small --include 'onnx/int8/*' --local-dir decima-small`, then pass
`--checkpoint decima-small/onnx/int8`). Parameter counts for every system:
`uv run python scripts/audit/params.py`. Competitor predictors (Laya, Kev) run in their own environments: `bench/external/`. Training recipe,
data builders and every experiment — including the ones that failed — are in the
[technical report](docs/TECHNICAL-REPORT.md).

## Repository

| | |
|---|---|
| `decima/` | model, ONNX runtime, export, int8 quantisation |
| `train/` | training (distillation + gold labels, crash-safe checkpoints) |
| `teacher/` | teacher data generation and gold-label builders |
| `bench/` | evaluation harness: one item format for every system, scorer, speed |
| `docs/` | technical report, evaluation audit, benchmarks |

## License & citation

Code and model: [Apache-2.0](LICENSE). Synthetic dataset: CC BY 4.0. Training-data licences
(some non-commercial) are disclosed in the model card and `release/LICENSING.md`. Predictions dataset:
CC BY 4.0.

If you use Decima, please cite it (GitHub's "Cite this repository" reads [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{madani2026decima,
  title  = {Decima: A Small, Calibrated, Permutation-Invariant Decision Model via Late-Interaction Distillation},
  author = {Madani, A. M.},
  year   = {2026},
  howpublished = {\url{https://github.com/amyrmahdy/decima}},
  note   = {Model: \url{https://huggingface.co/amyrmahdy/decima-small}}
}
```

Questions and problems: [GitHub issues](https://github.com/amyrmahdy/decima/issues).
