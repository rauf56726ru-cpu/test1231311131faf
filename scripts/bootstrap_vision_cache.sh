#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${LOG_PATH:-logs/pipeline.log}"

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

python -m scripts.bootstrap_vision_cache "$@" | tee -a "$LOG_PATH"
