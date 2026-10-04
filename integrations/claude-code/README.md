# Decima hooks for Claude Code

A local decision layer for agent loops: Claude builds, Decima decides on your machine, and the hook enforces it.
No code, command or diff leaves the machine to be judged.

| Hook | Decides | Rules first | Model thresholds | If the server is down |
|---|---|---|---|---|
| `secret_gate.py` (Write, Edit) | does the new text hold a real credential? | known key formats → deny | deny ≥ 0.8, ask ≥ 0.3, else no opinion | ask |
| `bash_gate.py` (Bash) | allow / ask / deny under a stated policy | destructive patterns → deny, plain read-only → allow | deny ≥ 0.6, ask ≥ 0.2 or unsure, else allow | ask |
| `route.py` (before a run) | cheapest model tier that can do the task | — | unsure (< 0.6) → one tier up | strongest tier |

## Setup

```bash
pip install "git+https://github.com/amyrmahdy/decima"
python -m decima.serve --model amyrmahdy/decima-agent --port 11436      # binds to 127.0.0.1; int8 ONNX on CPU
```

Then merge `settings.example.json` into `.claude/settings.json`, with the paths pointing at this folder.

## Shadow first

Thresholds that work on one model or project do not transfer to another. Run a week with `DECIMA_SHADOW=1`: the hooks
decide and log to `~/.decima/decisions.jsonl` but never enforce. Compare the log with what you approved, then turn
enforcement on. The same log is the training set for a gate tuned to your own team's decisions.

## Never the only line of defence

Hard rules run before the model, and a model miss can never allow a command the rules deny. Keep your sandbox,
permission settings and secret scanning in CI. Decima removes the routine prompts; it does not replace those.

Environment: `DECIMA_URL`, `DECIMA_MODEL`, `DECIMA_SHADOW`, `DECIMA_LOG`, `DECIMA_TIMEOUT`, `DECIMA_BASH_ALLOW_AT`,
`DECIMA_BASH_AUTO_ALLOW`, `DECIMA_TIERS`, `DECIMA_ROUTE_MIN`.
