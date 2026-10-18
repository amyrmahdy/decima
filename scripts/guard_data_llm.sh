#!/usr/bin/env bash
# decima-guard, OWASP LLM Top 10 data: teacher generation (offline, local Gemma via a local OpenAI-compatible gateway — never a
# paid API) + rebuild of data/guard/{train,test}-llm*.jsonl. Resumable: re-running continues where it stopped.
#
#   bash scripts/guard_data_llm.sh              # full run: ~12.9k writer + ~8.4k labeler calls ≈ 21k calls → ~60-80k rows; 7.7 h at 0.77 calls/s, token-bound worst case ~20 h (MAX_HOURS=30 cap)
#   SCALE=0.1 bash scripts/guard_data_llm.sh    # a tenth of every job (quick check)
#
# Order: (1) frozen public test/train (build; test-llm.jsonl is never rewritten once it exists), (2) held-out teacher test
# rows (test-llm-gen-*.jsonl, test-only scenario domains), (3) train generation, (4) decontam of train + gen against every
# test-*.jsonl. Waits for the gateway when it is down and retries a job until all its calls are done (max PASSES passes).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SCALE=${SCALE:-1}
PASSES=${PASSES:-6}
STREAMS=${STREAMS:-8}
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-30} * 3600 ))
log() { echo "[$(date '+%F %T')] $*"; }

# writer calls per job (synth jobs: +1 labeler call each, ~5 items → ~6-9 rows kept; translate: 1 call → ~4 rows)
declare -A CALLS=( [translate]=4500 [inject]=1500 [indirect]=1100 [leak]=1000 [output]=900 [agency]=1000 [ground]=900 [dos]=500 [toxic]=900 [misc]=600 )
declare -A TEST_CALLS=( [translate]=150 [inject]=50 [indirect]=40 [leak]=40 [output]=40 [agency]=40 [ground]=40 [dos]=25 [toxic]=40 [misc]=30 )
ORDER="inject indirect leak agency output translate ground toxic dos misc"

gateway_up() {
  uv run python - <<'PY' > /dev/null 2>&1
import asyncio
from teacher.client import Teacher
async def m():
    async with Teacher(model="rande-fast-local", log_path=None) as t:
        assert "rande-fast-local" in await t.models()
asyncio.run(m())
PY
}
wait_gateway() {
  local n=0
  until gateway_up; do
    [ $(date +%s) -gt $DEADLINE ] && { log "deadline reached while waiting for the gateway"; return 1; }
    [ $((n % 12)) -eq 0 ] && log "gateway / rande-fast-local not reachable — waiting (docker ps | grep infer)"
    n=$((n + 1)); sleep 300
  done
}
run_job() {   # job split calls
  local job=$1 split=$2 calls=$3 p
  calls=$(python3 -c "print(max(1, int(round($calls * $SCALE))))")
  for p in $(seq 1 $PASSES); do
    [ $(date +%s) -gt $DEADLINE ] && { log "deadline reached — stopping before $job/$split"; return 1; }
    wait_gateway || return 1
    log "gen $job/$split calls=$calls pass=$p"
    uv run python -m teacher.guard_llm gen --job $job --split $split --calls $calls --streams $STREAMS && return 0
    log "$job/$split: some calls failed (exit $?) — retrying the missing ones"; sleep 60
  done
}

log "== 1. public build (test-llm.jsonl stays frozen if present)"
[ -f data/guard/train-llm.jsonl ] && [ -f data/guard/test-llm.jsonl ] || uv run python -m teacher.guard_llm build || exit 1

log "== 2. held-out teacher test rows"
for j in $ORDER; do run_job $j test ${TEST_CALLS[$j]} || break; done

log "== 3. train generation"
for j in $ORDER; do run_job $j train ${CALLS[$j]} || break; done

log "== 4. rebuild public train (decontam against all test files) + filter gen rows"
uv run python -m teacher.guard_llm build          # test-llm.jsonl untouched; train-llm.jsonl re-filtered
uv run python -m teacher.guard_llm decontam
uv run python -m teacher.guard_llm stats
log "done"
