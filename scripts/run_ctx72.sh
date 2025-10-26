#!/usr/bin/env bash
set -euo pipefail

SYMBOL=${1:-BTCUSDT}
python -m src.cli.context_72h "$SYMBOL" | tee -a logs/pipeline.log
