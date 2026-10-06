---
pretty_name: Decima Agent Decisions
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
- zero-shot-classification
size_categories:
- 100K<n<1M
annotations_creators:
- machine-generated
- expert-generated
language_creators:
- machine-generated
tags:
- agents
- coding-agents
- claude-code
- hooks
- guardrails
- secret-detection
- command-safety
- tool-use
- model-routing
- soft-labels
- synthetic
- system-one
configs:
- config_name: secret
  default: true
  data_files:
  - {split: train, path: secret/train.parquet}
  - {split: test, path: secret/test.parquet}
- config_name: bash
  data_files:
  - {split: train, path: bash/train.parquet}
  - {split: test, path: bash/test.parquet}
- config_name: readonly
  data_files:
  - {split: train, path: readonly/train.parquet}
  - {split: test, path: readonly/test.parquet}
- config_name: tool
  data_files:
  - {split: train, path: tool/train.parquet}
  - {split: test, path: tool/test.parquet}
- config_name: command
  data_files:
  - {split: train, path: command/train.parquet}
  - {split: test, path: command/test.parquet}
- config_name: next_step
  data_files:
  - {split: train, path: next_step/train.parquet}
  - {split: test, path: next_step/test.parquet}
- config_name: router
  data_files:
  - {split: train, path: router/train.parquet}
  - {split: test, path: router/test.parquet}
- config_name: outcome
  data_files:
  - {split: train, path: outcome/train.parquet}
  - {split: test, path: outcome/test.parquet}
- config_name: agentbench
  data_files:
  - {split: test, path: agentbench/test.parquet}
- config_name: agentbench_fresh
  data_files:
  - {split: test, path: agentbench_fresh/test.parquet}
---

# Decima Agent Decisions

**About 400k labelled decisions that a coding agent's hooks make on every step**, plus a hand-written test set of 130.
The questions are the small, frequent ones:

- Is a real secret about to be written to disk?
- Should this shell command run, ask the user, or be denied?
- Is this command read-only?
- Which tool, and which exact command, fits this step?
- What should happen next: run, look first, test first, hand to a small model, or escalate?
- Which model tier does this task need?

This is the training data of [decima-agent](https://huggingface.co/amyrmahdy/decima-agent), a 321M local decision
model for Claude Code-style hooks. It is released so others can train, audit or beat it.

Built by **A. M. Madani** ([@amyrmahdy](https://github.com/amyrmahdy)).
Code: [github.com/amyrmahdy/decima](https://github.com/amyrmahdy/decima).

## Configs

| config | train | test | what the state is → what is decided |
|---|---:|---:|---|
| `secret` | 126,111 | 2,111 | a Write/Edit (file, new text, often a diff in a long agent context) → is a live credential written? |
| `bash` | 77,379 | 2,513 | a proposed shell command, with a policy → allow / ask / deny, and "does it exfiltrate / delete / need root?" |
| `readonly` | 13,823 | 600 | a command → does it leave every file, setting and remote unchanged? |
| `tool` | 26,246 | 1,503 | a goal → Read, Glob, Grep, Edit, Write, Bash, WebFetch, WebSearch, Task, TodoWrite, AskUserQuestion or an MCP tool |
| `command` | 33,810 | 1,637 | a goal → the exact command among near misses (`git reset --soft` vs `--hard`) |
| `next_step` | 74,246 | 2,258 | a trajectory or a proposed step → the next action |
| `router` | 11,442 | 768 | a coding task → haiku / sonnet / opus-class tier, or a non-coding model |
| `outcome` | 39,035 | 1,205 | test output or an ops alert → did the tests pass? does a human need to step in? |
| `agentbench` | — | 130 | hand-written (see the correction below) |
| `agentbench_fresh` | — | 40 | hand-written after the fact; no exact or near copy in any training data |

About a third of the states are wrapped in a long agent context (recent turns, a file list, the last diff), so the
decisive line has to be found in 0.5–2k tokens.

## Columns

| column | |
|---|---|
| `id` | stable row id |
| `task` | finer task name inside the config (`secret`, `secret-llm`, `bash-llm`, `tool-para`, …) |
| `origin` | `procedural` (exact label from code), `llm-file+code-slot`, `llm-paraphrase`, `llm+blind-check` (see below) |
| `kind` | `choose` (one of N), `verify` (yes/no) or `score` (ordered levels) |
| `state` | the situation, usually the JSON a PreToolUse hook receives |
| `question`, `choices` | the decision, as asked |
| `gold` | index of the right choice |
| `probs` | soft target (most mass on `gold`), the distribution the model was trained on |

`agentbench` uses the [jabr/classifier-benchmark](https://github.com/jabr/classifier-benchmark) / TypeSafe shape
instead: `type` (`noul` or `choice`), `instructions`, `criteria` (JSON), `state`, `choices`, `expected`, `gold`.

```python
from datasets import load_dataset
ds = load_dataset("amyrmahdy/decima-agent-decisions", "bash")
print(ds["test"][0])
```

## How the labels were made

Every agent label is exact or agreed by two different models:

- **`procedural`: exact, from code.**
  - Real-format credentials (AWS, GitHub, Stripe, Slack, GitLab, npm, Hugging Face, JWT, connection strings, …) vs
    placeholders, env lookups, docs example keys, hashes, UUIDs, public keys, test fixtures and base64 images.
  - A policy-labelled command catalogue: read-only, build/test, recoverable change, destructive, exfiltration,
    root, pipe-to-shell.
  - Tool and command choices with near-miss distractors; next actions from rules over the situation; test-run and
    ops-alert outcomes.
- **`llm-file+code-slot`.** Qwen3-Coder-Next writes a realistic file with a `<<VALUE>>` slot; code fills it with a real
  or fake credential, so the label stays exact.
- **`llm+blind-check`.** Qwen3-Coder-Next writes realistic commands, tasks and steps with an intended label; Gemma 4
  (26B-A4B) labels each one blind, and only rows where both agree are kept.
- **`llm-paraphrase`.** Gemma 4 rewrites a procedural goal in natural words; rows that lose a required entity (a
  file name, a flag) are dropped.

All fake credentials are random strings in real formats; none is a working key. Invented e-mail addresses at real
providers were rewritten to reserved `.example` domains.

### Test splits are harder than train on purpose

The test splits hold out whole families rather than random rows: credential formats (Google, SendGrid, Twilio, basic-auth
URLs), fake-value types (certificates), file types (Terraform, INI), command families, catalogue entries, tools and
phrasings, and LLM domains (terraform, helm, gcloud, rsync, openssl). Train rows whose state also appears in test were
removed. Test accuracy therefore measures generalisation to things never seen in training.

### agentbench: 130 hand-written cases, and a correction

Written in the states a real hook sends: secret gate 20, bash gate 22, read-only 8, next action 14, tier router 15, tests
passed 8, ops intervention 10, tool choice 12, command choice 15, next step 6. Credential-shaped values in it are fake.

**Correction (2026-10-06).** This set is not fully held out from the training data of decima-agent 2.0 and 2.1: our
read-only templates were near-copies of its 8 read-only cases, and LLM-written rows contained some of its bash-gate
commands verbatim. This release of the dataset drops every training row that copies a test case exactly or nearly
(1,693 rows; [`teacher/decontam_agent.py`](https://github.com/amyrmahdy/decima/blob/main/teacher/decontam_agent.py)), but
the published models were trained before that.

**`agentbench_fresh`** is the honest measurement: 40 cases (read-only 12, bash gate 18, secret gate 10) written after the
fact and checked against all training data. decima-base 0.60, decima-agent 2.0 0.83, decima-agent 2.1 0.83.

## Limitations

- **English only.** The states are English code, commands and prose.
- **Synthetic.** These are not real agent logs. Real commands are messier (`cd <dir> && python -c "…"`); logs
  from your own hooks are the best data for tuning a gate to your project.
- **The policy is ours.** "Ask" covers network calls, installs and anything outside the workspace; your project may
  want local services allowed. Labels follow the policy text in each question.
- Labels from `llm+blind-check` rows are only as good as two models agreeing; the procedural rows are exact.

## License

CC BY 4.0. The writing models are Apache-2.0 (Qwen3-Coder-Next; Gemma 4 26B-A4B). Please cite:

```bibtex
@software{madani2026decima,
  author = {Madani, Amir Mahdi},
  title  = {Decima: open, calibrated decision models},
  year   = {2026},
  url    = {https://github.com/amyrmahdy/decima}
}
```
