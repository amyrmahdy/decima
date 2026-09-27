#!/usr/bin/env bash
# Phase 1 run: train → predict on the scoreboard item sets → side-by-side with V0 and the competitors.
#   scripts/train_v1.sh <name> [train.train args...]
#   scripts/train_v1.sh v1a --init checkpoints/v0/best --question-in-state --no-question-in-choices --max-state-tokens 512 --epochs 1
# Data: teacher (generate + label) + gold NLI/QA (data/gold, teacher/gold.py). Keep scripts/thermal_guard.sh running.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
NAME="$1"; shift
mkdir -p runs/logs runs/preds
GOLD="${GOLD:-data/gold/*.jsonl}"      # override to change the teacher/gold mix, e.g. GOLD='data/gold-lite/*.jsonl'
uv run python -m train.train --data "data/teacher/generate-*.jsonl" "data/teacher/label-*.jsonl" "$GOLD" \
    --out "checkpoints/$NAME" --log-every 100 --resume "$@" 2>&1 | grep --line-buffered -v -i "warning\|tokenizers\|Loading weights\|Writing model" | tee "runs/logs/train-$NAME.log"
[ -f "checkpoints/$NAME/best/decima.json" ] || { echo "TRAIN FAILED $NAME"; exit 1; }
scripts/eval_sets.sh "$NAME" | tee -a "runs/logs/run-eval-$NAME.log"
echo "RUN DONE $NAME"
