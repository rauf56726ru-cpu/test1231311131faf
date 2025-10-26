#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_PATH="${LOG_PATH:-logs/pipeline.log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

export LOG_PATH
export LOG_LEVEL

INGEST_ARGS=()
ANALYZE_ARGS=()
TARGET_ARRAY=ingest

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        TARGET_ARRAY=analyze
        continue
    fi
    if [[ "$TARGET_ARRAY" == "ingest" ]]; then
        INGEST_ARGS+=("$arg")
    else
        ANALYZE_ARGS+=("$arg")
    fi
done

if [[ "${#ANALYZE_ARGS[@]}" -eq 0 ]]; then
    ANALYZE_ARGS=("${INGEST_ARGS[@]}")
fi

printf '[%s] run_all.sh -> ingest args: %s | analyze args: %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "$(printf '%q ' "${INGEST_ARGS[@]}")" \
    "$(printf '%q ' "${ANALYZE_ARGS[@]}")" | tee -a "$LOG_PATH"

"${SCRIPT_DIR}/run_ingest.sh" "${INGEST_ARGS[@]}"
ingest_status=$?

if [[ "$ingest_status" -ne 0 ]]; then
    printf '[%s] run_all.sh -> ingest failed (status=%s)\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ingest_status" | tee -a "$LOG_PATH"
    exit "$ingest_status"
fi

"${SCRIPT_DIR}/run_analyze.sh" "${ANALYZE_ARGS[@]}"
analyze_status=$?

printf '[%s] run_all.sh -> analyze finished (status=%s)\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$analyze_status" | tee -a "$LOG_PATH"

exit "$analyze_status"
