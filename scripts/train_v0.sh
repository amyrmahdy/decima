#!/usr/bin/env bash
# V0 pipeline once the teacher data is complete: train → bench → compare → ONNX export.
#   scripts/train_v0.sh [name]      default name: v0
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
NAME="${1:-v0}"
mkdir -p runs/logs
uv run python -m teacher.stats "data/teacher/generate-*.jsonl" "data/teacher/label-*.jsonl" | tee "runs/logs/data-stats-$NAME.txt"
uv run python -m train.train --data "data/teacher/generate-*.jsonl" "data/teacher/label-*.jsonl" \
    --out "checkpoints/$NAME" --epochs 3 --log-every 100 2>&1 | grep -v -i "warning\|tokenizers\|Loading weights\|Writing model" | tee "runs/logs/train-$NAME.log"
uv run python -m bench.run --system decima --checkpoint "checkpoints/$NAME/best" --device cuda --limit 1000 --no-latency \
    --out "runs/$NAME-decima-1000.json" 2>&1 | grep -v -i "warning\|tokenizers\|Loading weights\|Repo card" | tee "runs/logs/bench-$NAME.log"
uv run python -m bench.compare runs/baseline-e5-small-500-part1.json runs/baseline-e5-small-500-part2.json -- "runs/$NAME-decima-1000.json" | tee "runs/logs/compare-$NAME.txt"
uv run python -m decima.export "checkpoints/$NAME/best" --out "export/$NAME" 2>&1 | grep -i "exported"
