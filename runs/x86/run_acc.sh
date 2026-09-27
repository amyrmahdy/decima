#!/usr/bin/env bash
# x86 accuracy pass for Decima-small (v1i): int8 ONNX on all 5 release sets, fp32 ONNX on kev/laya/decima.
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
run() { # prec set
  local d=export/v1i; [ "$1" = int8 ] && d=export/v1i-int8
  uv run --no-sync python -m bench.predict --items runs/x86/items/$2-eval.jsonl --system onnx --checkpoint $d --threads 2 \
     --out runs/x86/preds/$2--v1i-x86-$1.jsonl > runs/x86/logs/$2-$1.log 2>&1 && echo "DONE $1 $2" || echo "FAILED $1 $2"
}
for s in decima kev laya jevtyped btzsc; do run int8 $s & done
for s in decima kev laya; do run fp32 $s & done
wait
echo ALL_FINISHED
