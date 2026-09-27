#!/usr/bin/env bash
# Full V0 teacher run — launch only after the pilot is approved.
#   generate: 9,000 calls ≈ 85k examples   (≈ 186 completion tokens / example)
#   label:    ~210k examples over bench TRAIN splits + generated states re-asked (≈ 33 tokens / example)
# Both resumable: re-running this script continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p runs/logs
nohup uv run python -m teacher.generate --calls 9000 --concurrency 24 --log-every 50 >> runs/logs/full-generate.log 2>&1 &
echo "generate pid $!"
# label starts on the bench sources right away; a second pass with --generated picks up our states once they exist
nohup uv run python -m teacher.label --sources banking77/en clinc150/en massive/en massive/fa massive/ar massive/ru agnews/en sst5/en xnli/en xnli/ar xnli/ru \
    --per-source 6000 --asks 2 --concurrency 12 --log-every 50 >> runs/logs/full-label.log 2>&1 &
echo "label pid $!"
