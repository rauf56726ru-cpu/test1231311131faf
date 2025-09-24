"""Risk management utilities used by policy runtime."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class RiskState:
    balance: float
    equity: float
    daily_pnl: float
    daily_drawdown: float
    open_position: Optional[str] = None


def compute_risk_state(trades: pd.DataFrame) -> RiskState:
    if trades.empty:
        return RiskState(balance=1.0, equity=1.0, daily_pnl=0.0, daily_drawdown=0.0)
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["timestamp"], unit="s")
    today = trades["date"].dt.date.max()
    daily = trades[trades["date"].dt.date == today]
    pnl = daily["pnl"].sum() if "pnl" in daily else 0.0
    equity = trades.get("equity", pd.Series(dtype=float)).iloc[-1] if "equity" in trades else 1.0 + trades.get("pnl", pd.Series(dtype=float)).cumsum().iloc[-1]
    balance = trades.get("balance", pd.Series(dtype=float)).iloc[-1] if "balance" in trades else equity
    drawdown = min(daily.get("drawdown", pd.Series([0.0])).min(), 0.0)
    return RiskState(balance=balance, equity=equity, daily_pnl=pnl, daily_drawdown=abs(drawdown))
