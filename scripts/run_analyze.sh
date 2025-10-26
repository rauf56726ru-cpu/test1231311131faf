#!/usr/bin/env bash

set -Eeuo pipefail

LOG_PATH="${LOG_PATH:-logs/pipeline.log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

export LOG_PATH
export LOG_LEVEL

CLI_ARGS=("$@")
COMMAND=("$PYTHON_BIN" -m src.cli.smc72 "${CLI_ARGS[@]}")
COMMAND_STRING="$(printf '%q ' "${COMMAND[@]}")"

printf '[%s] run_analyze.sh -> %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$COMMAND_STRING" | tee -a "$LOG_PATH"

stdbuf -oL -eL -- "${COMMAND[@]}" 2>&1 | tee -a "$LOG_PATH"
status=${PIPESTATUS[0]}

exit "$status"
