"""Generate backtest reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult
from .metrics import per_regime_metrics, summary


def generate_report(
    result: BacktestResult,
    output_dir: Path | str = Path("reports/last_run"),
    regimes: pd.DataFrame | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / "equity.csv"
    equity_path.write_text(result.equity.to_csv())
    trades_path = output_dir / "trades.csv"
    result.trades.to_csv(trades_path, index=False)

    stats = summary(result.equity, result.trades)
    stats.update(result.stats)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        result.equity.plot(ax=ax)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Balance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "equity.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting optional
        (output_dir / "equity_plot.txt").write_text(f"Failed to render plot: {exc}")

    if regimes is not None and not result.trades.empty:
        try:
            regime_metrics = per_regime_metrics(result.trades, regimes)
            (output_dir / "regime_metrics.json").write_text(json.dumps(regime_metrics, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            (output_dir / "regime_metrics.txt").write_text(f"Failed to compute regime metrics: {exc}")

    return output_dir
