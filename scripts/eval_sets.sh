#!/usr/bin/env bash
# Scoreboard for one checkpoint: predict (skipped when preds exist) + score against V0 and the competitors.
#   scripts/eval_sets.sh <name> [sets...]        default sets: kev laya decima jevtyped btzsc
# Writes runs/preds/<set>--<name>.jsonl, runs/<name>-<set>.json and appends to runs/logs/score-<name>.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NAME=$1; shift
SETS="${*:-kev laya decima jevtyped btzsc}"
for s in $SETS; do
  [ -s runs/preds/$s--$NAME.jsonl ] || uv run python -m bench.predict --items runs/items/$s.jsonl --system decima \
      --checkpoint "checkpoints/$NAME/best" --device cuda --out runs/preds/$s--$NAME.jsonl > runs/logs/$s--$NAME.log 2>&1 \
      || echo "PREDICT FAILED $s $NAME"
  A="v0=runs/preds/$s--v0.jsonl $NAME=runs/preds/$s--$NAME.jsonl"
  for c in kev-0.5b kev-0.8b laya laya-multilingual; do [ -s runs/preds/$s--$c.jsonl ] && A="$A $c=runs/preds/$s--$c.jsonl"; done
  M=acc; [ $s = btzsc ] && M=macro_f1 && A="$A e5=runs/preds/btzsc--e5.jsonl"
  uv run python -m bench.score --items runs/items/$s.jsonl --metric $M --preds $A --out runs/$NAME-$s.json 2>&1 \
      | sed -n '/side by side/,$p' >> "runs/logs/score-$NAME.txt"
done
echo "SETS DONE $NAME"
