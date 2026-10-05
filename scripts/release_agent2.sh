#!/usr/bin/env bash
# Release build for decima-agent 2.1 (agent2): int8, agentbench on the int8 runtime, int8-vs-fp32 check. CPU-only, niced.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
LOG=runs/logs/release-build.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
[ -f export/agent2-int8/decima.json ] || nice uv run python -m decima.quantize export/agent2 --out export/agent2-int8 > runs/logs/quantize-agent2.log 2>&1 || say "FAILED quantize agent2"
say "agent2 int8 built"
nice uv run python -m bench.agent_eval --checkpoint export/agent2-int8 --name agent2-int8 --max-proc 1000 > runs/logs/agent-eval-agent2-int8.txt 2>&1
say "agent2 int8 agentbench: $(grep -m1 ' gold ' runs/logs/agent-eval-agent2-int8.txt)"
nice uv run python scripts/int8_check.py agent2 >> runs/final/determinism.txt 2> runs/logs/int8-check-agent2.log || say "FAILED int8 check agent2"
say "agent2 int8 check: $(tail -2 runs/final/determinism.txt | tr '\n' ' ')"
say "agent2 release build done"
