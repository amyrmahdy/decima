#!/usr/bin/env bash
# Full evaluation of one run: ONNX fp32 + int8, CPU latency (1 thread), Jev Decision Index, BTZSC.
#   scripts/eval_winner.sh <name>          (checkpoints/<name>/best must exist)
# Outputs: export/<name>{,-int8}, runs/<name>-jdi.json, runs/<name>-btzsc.json, runs/logs/{jdi-score,btzsc-score,latency}-<name>.*
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
N=$1; CK=checkpoints/$N/best
say() { echo "$(date '+%F %T') $*" | tee -a runs/logs/auto.log; }

say "export $N"
uv run python -m decima.export $CK --out export/$N > runs/logs/export-$N.log 2>&1 || say "FAILED export $N"
nice uv run python -m bench.latency export/$N --threads 1 --choices 4 77 --reps 40 --out runs/latency-$N-gx10-indicative.json \
    > runs/logs/latency-$N.log 2>&1 || say "FAILED latency $N"

[ -f runs/items/jdi-panelall.jsonl ] || cat runs/items/jdi-panel.jsonl runs/items/jdi-panel-rest.jsonl > runs/items/jdi-panelall.jsonl
say "JDI preds $N"
[ -s runs/preds/jdi-all--$N.jsonl ] || uv run python -m bench.predict --items runs/items/jdi-panelall.jsonl --system decima --checkpoint $CK \
    --device cuda --out runs/preds/jdi-all--$N.jsonl > runs/logs/jdi--$N.log 2>&1 || say "FAILED jdi preds $N"
nice uv run python -m bench.jdi_index --items runs/items/jdi.jsonl \
    --preds v0-official=runs/preds/jdi-all--v0.jsonl $N-truncated=runs/preds/jdi-all--$N.jsonl $N-official=runs/preds/jdi-all--$N.jsonl \
    --decima-checkpoint $CK --budget-systems $N-official v0-official --out runs/$N-jdi.json > runs/logs/jdi-score-$N.txt 2>&1 || say "FAILED jdi score $N"

say "BTZSC preds $N"
[ -s runs/preds/btzsc--$N.jsonl ] || uv run python -m bench.predict --items runs/items/btzsc.jsonl --system decima --checkpoint $CK --device cuda \
    --out runs/preds/btzsc--$N.jsonl > runs/logs/btzsc--$N.log 2>&1 || say "FAILED btzsc $N"
uv run python -m bench.score --items runs/items/btzsc.jsonl --metric macro_f1 \
    --preds v0=runs/preds/btzsc--v0.jsonl e5=runs/preds/btzsc--e5.jsonl $N=runs/preds/btzsc--$N.jsonl \
    --out runs/$N-btzsc.json > runs/logs/btzsc-score-$N.txt 2>&1
say "eval done $N"
