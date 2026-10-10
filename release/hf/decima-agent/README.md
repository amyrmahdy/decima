---
license: apache-2.0
language: [en]
library_name: onnx
pipeline_tag: zero-shot-classification
tags: [decision-model, system-one, jev, jev-alternative, claude-code, hooks, agents, guardrails, secret-detection, knowledge-graph, entity-resolution, tool-routing, onnx, int8, cpu, local-first]
base_model: amyrmahdy/decima-base
---

# Decima-agent

**A 321M local judge for agents and knowledge graphs. It decides in milliseconds on a CPU, and nothing leaves your machine.**

Give it a situation, a question and your options; it returns calibrated probabilities. It speaks TypeSafe's
`/v1/systemone` format, so code written for Jev works after a one-line change. It is the judge, not the writer: it sits
next to your agent or your extraction pipeline and decides what is safe, what is true, and when the expensive model is
needed.

- **Agent gates:** secret in this edit? Should this command run, ask or be denied? Which tool, which exact command, what next?
- **Knowledge-graph judgments:** is this extracted fact asserted, hypothetical, denied or planned? Are these two records
  the same entity? Are these two things alternatives? Which relation holds?
- **It also plays Snake** from text sensors, one decision per move (the median game below: 45 apples, alive after 600 moves).

![decima-agent 2.2 playing Snake on CPU](snake-agent-2.2.gif)

![decima on the command line, on CPU](decima-cli.gif)

## Quickstart

```bash
pip install decima-ai          # the `decima` package, CLI and server (PyPI; source: github.com/amyrmahdy/decima)
```

```python
from decima import Decima, Question

m = Decima.from_pretrained("amyrmahdy/decima-agent")          # int8 ONNX, CPU
q = Question("Should the agent's proposed command run?",
             ["allow: read-only, builds or tests", "ask: changes something recoverable", "deny: destroys data or history"])
print(m.decide('{"tool": "Bash", "command": "rm -rf .loop/ data/"}', q).top)   # → deny: destroys data or history
```

From the command line, with ready-made presets:

```bash
decima "rm -rf ~/projects/app" --preset agent-gate          # gate: deny 0.98
decima "Acme denied it had acquired Beta Labs." --preset kg-judge   # status: negated 0.99
```

As a local server for any TypeSafe client (binds to 127.0.0.1):

```bash
python -m decima.serve --model amyrmahdy/decima-agent --port 11436
```

For Claude Code, copy the hooks in [`integrations/claude-code`](https://github.com/amyrmahdy/decima/tree/main/integrations/claude-code):
rules first, then Decima, fail closed, shadow mode to start.

## Results

Measured on data the model never trained on.

**Agent decisions: 40 cases written after training, with no exact or near copy in any training data**

| | decima-base | agent 2.0 | agent 2.1 | **agent 2.2** |
|---|---:|---:|---:|---:|
| Read-only or not (12) | 7 | 10 | 9 | **9** |
| Bash gate: allow / ask / deny (18) | 11 | 15 | 15 | **16** |
| Secret gate (10) | 6 | 8 | 9 | **9** |
| **All 40** | 0.60 | 0.83 | 0.83 | **0.85** |

The same on the shipped int8 runtime and in full precision: 34 of 40.

**Knowledge-graph judgments: William Lyon's public notebooks**
([extraction-knowledge-graph-experiments](https://github.com/johnymontana/extraction-knowledge-graph-experiments)): the
same pipeline and questions, with Jev swapped for a local Decima. None of the notebooks' documents or Beer pairs were in
training.

| | Jev (published) | decima-agent 2.1 | **decima-agent 2.2** |
|---|---:|---:|---:|
| Planted "maybe / denied / planned" traps caught | 8/11 | every answer 0.5 | **11/11** |
| Beer entity matching, AUC (450 pairs) | 0.992 | 0.228 | **0.972** |
| Assertion gate over GLiNER edges, F1 (ungated 0.330) | 0.404 | 0.294 | **0.368** |
| GLiNER ∪ relation selection, both gated, F1 | 0.343 | — | **0.364** |
| Alternatives vs not, separated by one threshold | yes | — | **yes** |
| Support threads: intent / priority / resolved (8 each) | 5 / 3 / 8 | — | **7 / 3 / 3** |

Jev is better at deciding whether a support thread is resolved and at filtering wrong facts. decima-agent is better at
catching hedged facts and at relation selection, and it runs on your CPU for free.

**Games** (text sensors, one decision per move, the games' own wording never in training): Snake, **42 apples** a game
on average with 6 of 10 games still alive at the 600-move limit (2.1: 32 apples); a side-scrolling platformer, **9 of 10**
levels.

**Speed:** about 50 ms per decision on one CPU thread and 20 ms on four (ARM cores of an NVIDIA GB10) for a short state
with its options cached. Long states cost more: about 0.6 s at 2,000 tokens.

## What changed in 2.2

- Knowledge-graph judgments: record and mention matching, fact modality, "does the document discuss this?", relation
  selection with direction and `none`, typing a mention by what it refers to, alternatives vs complementary, long
  support threads (about 190k rows; labels exact from code or agreed by two models).
- Agent gates trained on what field logs showed: real-shape shell commands, credentials passed inline (client calls,
  Bearer headers, curl), more key formats, destructive cloud and cluster commands, database commands.
- Snake states where the apple's direction is a closed pocket smaller than the body.
- Options are always kept whole in the choice window, so long questions no longer tie at 0.5.

## Limitations

- **Never the only line of defence.** Keep hard rules first (the hooks do), plus your sandbox and secret scanning in CI.
  In the fresh test it still calls one unseen key format (Mailgun `key-…`) "not a secret", and sends two destructive
  commands to "ask" instead of "deny".
- **States up to 2,048 tokens.** Longer input is refused, never silently cut. Split long documents first.
- **Many questions about one long document are slow:** each question re-reads the document (about 0.6 s per question
  at 2,000 tokens on CPU).
- **English agent data only.** The backbone is multilingual; the agent and knowledge-graph skills were trained in English.
- **Small test sets.** 40 fresh agent cases and 8 support threads mean wide error bars.
- **A correction:** an earlier card reported gains on a 130-case hand-written set that turned out to have near-copies in
  the training data. The 40-case fresh set replaces it, and every training mix now drops rows that copy a test case
  ([`teacher/decontam_agent.py`](https://github.com/amyrmahdy/decima/blob/main/teacher/decontam_agent.py)).

## Training

decima-base (mmBERT-base, 321M) → agent 2.0 → 2.1 → knowledge-graph rounds → 2.2. Each round replays general data so the
general skills mostly stay (decima-base's five benchmark sets: 0.703; this model: 0.672). The training data is public:
[decima-agent-decisions](https://huggingface.co/datasets/amyrmahdy/decima-agent-decisions),
[decima-kg-judgments](https://huggingface.co/datasets/amyrmahdy/decima-kg-judgments),
[decima-game-decisions](https://huggingface.co/datasets/amyrmahdy/decima-game-decisions) and
[decima-system-one-tasks](https://huggingface.co/datasets/amyrmahdy/decima-system-one-tasks).

## License and citation

Apache-2.0. Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
