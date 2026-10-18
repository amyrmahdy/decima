#!/usr/bin/env bash
# decima-guard web/API data from public labelled datasets (teacher/guard_web.py). Offline, no teacher, idempotent:
# rebuilds data/guard/test-web.jsonl then data/guard/train-web.jsonl (fixed seed, atomic writes). FORCE=1 to rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1
mkdir -p runs/logs
LOG=runs/logs/guard-data-web.log
if [ -z "${FORCE:-}" ] && [ -s data/guard/test-web.jsonl ] && [ -s data/guard/train-web.jsonl ] \
   && [ data/guard/train-web.jsonl -nt teacher/guard_web.py ]; then
  echo "guard web data up to date (FORCE=1 to rebuild)"; uv run python -m teacher.guard_web stats; exit 0
fi
nice -n 10 uv run python -m teacher.guard_web build 2>&1 | tee $LOG
