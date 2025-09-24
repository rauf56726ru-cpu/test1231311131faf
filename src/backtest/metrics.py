"""Performance metric helpers for backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1).fillna(0.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period).mean().bfill().fillna(0.0)


def rolling_volatility(values: pd.Series, window: int = 14) -> pd.Series:
    returns = values.pct_change().fillna(0.0)
    return returns.rolling(window=window, min_periods=1).std().fillna(0.0)


def compute_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def cagr(equity: pd.Series, periods_per_year: int = 365 * 24 * 60) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / max(len(returns), 1)
    std = excess.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(len(returns)) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf") if returns.mean() > 0 else 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    avg_excess = returns.mean() - risk_free / max(len(returns), 1)
    return float(avg_excess / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return float(drawdowns.min())


def calmar_ratio(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    return float(cagr(equity) / dd)


def hit_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return float(wins / len(trades))


def average_return(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades:
        return 0.0
    return float(trades["pnl"].mean())


def summary(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    returns = compute_returns(equity)
    return {
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "volatility": float(returns.std()),
        "hit_rate": hit_rate(trades),
        "average_trade": average_return(trades),
    }


def per_regime_metrics(trades: pd.DataFrame, regimes: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if trades.empty or regimes is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    merged = trades.merge(regimes, left_on="entry_time", right_index=True, how="left")
    for column in regimes.columns:
        subset = merged[merged[column] > 0.5]
        if subset.empty:
            continue
        out[column] = {
            "hit_rate": hit_rate(subset),
            "avg_trade": average_return(subset),
        }
    return out
