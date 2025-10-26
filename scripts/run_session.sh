#!/usr/bin/env bash
set -euo pipefail

SYMBOL=${1:-BTCUSDT}

python -m src.cli.analyze_session "$SYMBOL" | tee -a logs/pipeline.log
