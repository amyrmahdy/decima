---
license: apache-2.0
language: [en]
library_name: onnx
pipeline_tag: zero-shot-classification
tags: [decision-model, system-one, jev, jev-alternative, claude-code, hooks, agents, guardrails, secret-detection, tool-routing, onnx, int8, cpu, local-first]
base_model: amyrmahdy/decima-base
---

# Decima-agent: a local decision layer for coding agents

**Claude builds. Decima decides, on your machine. The hook enforces it.**

Decima-agent is a 321M-parameter System One model for the small, frequent decisions inside an agent loop:

- Does this file edit contain a real secret?
- Should this command run, or should it ask or be denied?
- Which tool fits this step, and which command does exactly this?
- What should happen next?
- Which model tier does this task need?

It returns calibrated probabilities instead of text, runs as int8 ONNX on a CPU, and nothing it judges ever leaves
the machine. It speaks TypeSafe's `/v1/systemone` wire format, so hooks written for Jev work after a one-line change.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)). The name: **DECI**sion **MA**king, and Decima
is also the Roman Fate who decides.

| On 130 hand-written agent decisions, held out from training | decima-base (no agent training) | **decima-agent** |
|---|---:|---:|
| Secret gate: real credential vs placeholder, env lookup, hash, docs example | 0.50 | **0.95** |
| Bash gate: allow / ask / deny under a stated policy | 0.77 | **0.95** |
| Command choice: the exact command among near misses (`reset --soft` vs `--hard`) | 0.33 | **0.93** |
| Next action: run / look first / test first / small model / escalate | 0.14 | **0.93** |
| Tool choice: Read, Grep, Glob, Edit, Write, Bash, WebFetch, WebSearch, ask the user | 0.50 | **0.83** |
| Model tier router: haiku / sonnet / opus / other | 0.93 | **1.00** |
| **All 130** | 0.58 | **0.90** |

- **Gates that catch what matters.** At the recommended thresholds (block at p ≥ 0.8, ask at p ≥ 0.3), **0**
  of the 9 real secrets in the hand-written set get through.
- **Honest confidence.** Calibration error 0.035. 86 % of the decisions come with confidence ≥ 0.75, and those are
  right 95 % of the time. Send the rest to a bigger model or to a person.
- **Small and local.** int8 ONNX, 355 MB, about 50 ms on one CPU thread and about 20 ms on four (ARM cores of an NVIDIA GB10; short command, the question's options cached) per decision. States up to
  2,048 tokens, so a command plus recent turns and a diff fit.

Numbers are measured on the shipped int8 runtime. On this set fp32 gives the same accuracy (0.90), with an ECE of 0.041.

## Quickstart: hooks for Claude Code

```bash
pip install "git+https://github.com/amyrmahdy/decima"
python -m decima.serve --model amyrmahdy/decima-agent --port 11436     # binds to 127.0.0.1
```

Then add the hooks from [`integrations/claude-code`](https://github.com/amyrmahdy/decima/tree/main/integrations/claude-code)
to `.claude/settings.json`:

| Hook | Rules first | Then Decima | If the server is down |
|---|---|---|---|
| `secret_gate.py` (Write, Edit) | known key formats → deny | deny ≥ 0.8, ask ≥ 0.3, else no opinion | ask |
| `bash_gate.py` (Bash) | destructive patterns → deny, plain read-only → allow | deny ≥ 0.6; ask if p(deny) ≥ 0.2 or unsure; else allow | ask |
| `route.py` (before a run) | — | cheapest tier that can do the task; one tier up when unsure | strongest tier |

**Run a week in shadow mode first (`DECIMA_SHADOW=1`).** The hooks decide and log to `~/.decima/decisions.jsonl` but
never enforce. Thresholds tuned on one model or project do not carry over to another. The log is also the training
set for a gate tuned to your own team's decisions.

## Use it directly

```python
from decima import Decima, Question

m = Decima.from_pretrained("amyrmahdy/decima-agent")
q = Question("Which command does exactly this?",
             ["git reset --soft HEAD~1", "git reset --hard HEAD~1", "git revert HEAD", "git commit --amend"])
d = m.decide('{"goal": "undo the last commit but keep my changes"}', q)
print(d.top, max(d.probs))
```

The same model answers through `/v1/systemone` (choice, noul and score), so the official `typesafe-sdk` works against
`http://127.0.0.1:11436` unchanged.

## What it was trained on

decima-agent is [decima-base](https://huggingface.co/amyrmahdy/decima-base) (mmBERT-base, 321M) fine-tuned on about
160k agent decisions, with about 55k rows of decima-base's own training mix replayed so its general skills stay.
Every agent label is either exact or agreed by two models:

- **Exact, from code:**
  - real-format credentials vs placeholders, env lookups, docs example keys, hashes, UUIDs and public keys;
  - a policy-labelled command catalogue (read-only, build/test, recoverable change, destructive, exfiltration, root,
    pipe-to-shell);
  - tool and command choices with near-miss distractors;
  - next actions from rules over the situation;
  - test-run and ops-alert outcomes.
- **Written by Qwen3-Coder and checked by Gemma:**
  - realistic files with a credential slot that code fills, so the label is exact;
  - realistic commands, tasks and steps, where a row is kept only when the blind second model agrees.
- About a third of the states are wrapped in a long agent context (recent turns, file list, a diff), so the decisive
  line has to be found in 0.5–2k tokens.

The agent data is public: [amyrmahdy/decima-agent-decisions](https://huggingface.co/datasets/amyrmahdy/decima-agent-decisions)
(about 400k rows, with the held-out test splits and the 130 hand-written cases).

The hand-written evaluation set was written separately from all generators. The procedural test splits hold out key
formats, file types, command families, tools and phrasings, and are reported separately in the agentbench files.

## Limitations

- **Never the only line of defence.** Keep hard rules, your sandbox, permission settings and secret scanning in CI.
  Decima removes routine prompts; it does not replace those.
- **Unseen formats and commands are weaker.**
  - On the held-out procedural secret set, where half the credential formats never appear in training, accuracy is 0.75.
  - In realistic files from project types never seen in training, written by an LLM, 25 of 145 real credentials score
    below the ask threshold, so the model alone would let them through. The secret gate's regex rules cover the common
    vendor formats; keep secret scanning in CI.
  - On the held-out command test, where half the goals come from catalogue entries never seen in training, accuracy
    is 0.65.
- **"Is this read-only?" is the weakest single question** (0.63 on 8 hand-written cases).
- **English agent data only.** The underlying model is multilingual, but the agent skills were trained and tested in
  English.
- **General classification is better on decima-base.** Fine-tuning for agent decisions costs a little general accuracy
  (0.703 → 0.681 on decima-base's five benchmark sets), and its general calibration is looser (ECE 0.10 vs 0.05).
- Small hand-written subsets (6–22 cases per decision) mean wide error bars; the procedural and LLM-written test sets
  are larger but are not real logs.

## License and citation

Apache-2.0. The base encoder is [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT). Training data
licensing is described in the [repository](https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md).

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
