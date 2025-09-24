"""Vectorised backtest engine with execution costs and latency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..policy.runtime import PolicyDecision, PolicyRuntime
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    holding_period: int


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: Dict[str, float]


def _execution_price(
    price: float,
    direction: int,
    commission_bps: float,
    spread: float,
    slippage: float,
    maker: bool = False,
) -> float:
    fee = price * commission_bps / 10_000.0
    if maker:
        fee = -fee
    adj = price + direction * (spread / 2.0 + slippage) + direction * fee
    return float(adj)


def _infer_vol_bucket(regimes: Optional[pd.DataFrame], ts: pd.Timestamp) -> str:
    if regimes is None or ts not in regimes.index:
        return "med"
    for bucket in ("vol_bucket_high", "vol_bucket_med", "vol_bucket_low"):
        if regimes.loc[ts].get(bucket, 0.0) > 0.5:
            return bucket.rsplit("_", 1)[-1]
    return "med"


def run_backtest(
    market: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    policy: PolicyRuntime,
    regimes: Optional[pd.DataFrame] = None,
    costs: Optional[Dict[str, object]] = None,
    execution: Optional[Dict[str, object]] = None,
    initial_balance: float = 10_000.0,
) -> BacktestResult:
    market = market.copy().sort_index()
    if "is_synthetic" in market.columns:
        market = market.loc[~market["is_synthetic"].astype(bool)].drop(columns=["is_synthetic"])
    features = features.reindex(market.index, method="ffill").fillna(0.0)
    if isinstance(predictions, pd.Series):
        predictions = predictions.to_frame(name="prob_up")
    predictions = predictions.reindex(market.index, method="ffill")
    regimes = regimes.reindex(market.index) if regimes is not None else None

    costs = costs or {}
    commission_bps = float(costs.get("commission_bps", 7))
    maker_bps = float(costs.get("maker_bps", -1.0))
    taker_bps = float(costs.get("taker_bps", commission_bps))
    spread_series = (market["high"] - market["low"]).rolling(20, min_periods=1).median().fillna(0.0)
    slip_cfg = costs.get("slip_k", {"low": 0.2, "med": 0.4, "high": 0.8})

    execution = execution or {}
    latency = int(execution.get("latency_bars", 1))
    funding_bps = float(execution.get("funding_bps", 0.0))
    max_fill = float(execution.get("max_fill_ratio", 1.0))

    atr = (market["high"] - market["low"]).rolling(14, min_periods=1).mean().fillna(0.0)
    avg_volume = market.get("volume", pd.Series(1.0, index=market.index)).rolling(50, min_periods=1).mean().replace(0.0, 1.0)

    balance = initial_balance
    equity_curve: List[float] = []
    equity_index: List[pd.Timestamp] = []
    trades: List[Trade] = []

    position_side: Optional[str] = None
    position_size = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    pending_decisions: List[Tuple[int, PolicyDecision, pd.Timestamp]] = []
    accrual_pnl = 0.0

    for i, ts in enumerate(market.index):
        price = float(market.loc[ts, "close"])
        feature_row = features.loc[ts].to_dict()
        pred_row = predictions.loc[ts].to_dict() if ts in predictions.index else {}
        if "prob_up" in pred_row and "prob_down" not in pred_row:
            prob_up = float(pred_row.get("prob_up", 0.5))
            pred_row.setdefault("prob_down", 1.0 - prob_up)
            pred_row.setdefault("prob_flat", 0.0)
            pred_row.setdefault("pred_conf", max(prob_up, pred_row["prob_down"]))
        context = {
            "features": feature_row,
            "state": {"balance": balance, "position": position_side},
            "costs": {"commission_bps": commission_bps, "spread_bps": costs.get("spread_bps", commission_bps)},
        }
        decision = policy.apply(pred_row, context)
        pending_decisions.append((latency, decision, ts))

        # process queue
        executable: Optional[PolicyDecision] = None
        while pending_decisions and pending_decisions[0][0] <= 0:
            _, executable, _ = pending_decisions.pop(0)
        pending_decisions = [(delay - 1, dec, t) for delay, dec, t in pending_decisions]

        if executable is None:
            equity_curve.append(balance if position_side is None else balance + position_size * (price - entry_price) * (1 if position_side == "long" else -1))
            equity_index.append(ts)
            continue

        side = executable.side
        size = executable.size
        vol_bucket = _infer_vol_bucket(regimes, ts)
        slip_mult = float(slip_cfg.get(vol_bucket, slip_cfg.get("med", 0.4)))
        spread = spread_series.loc[ts]
        volume = float(market.get("volume", pd.Series(index=market.index)).fillna(0.0).loc[ts] or 1.0)
        liquidity_ratio = min(1.0, float(volume / avg_volume.loc[ts])) if volume else 1.0
        fill_ratio = min(max_fill, liquidity_ratio)
        slippage = spread * slip_mult * min(1.0, abs(decision.size) / max(volume, 1e-6))

        if side == "flat" and position_side is not None:
            direction = -1 if position_side == "long" else 1
            exit_price = _execution_price(price, direction, taker_bps, spread, slippage, maker=False)
            pnl = position_size * (exit_price - entry_price) * (1 if position_side == "long" else -1)
            balance += pnl
            accrual_pnl = 0.0
            trades.append(
                Trade(
                    entry_time=entry_time or ts,
                    exit_time=ts,
                    side=position_side,
                    size=position_size,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    holding_period=i - market.index.get_loc(entry_time) if entry_time is not None else 0,
                )
            )
            position_side = None
            position_size = 0.0
            entry_price = 0.0
            entry_time = None
        elif side in {"long", "short"}:
            direction = 1 if side == "long" else -1
            is_taker = abs(pred_row.get("pred_conf", 0.5) - 0.5) > 0.1
            fee_rate = taker_bps if is_taker else maker_bps
            exec_price = _execution_price(price, direction, fee_rate, spread, slippage, maker=not is_taker)
            executed_size = abs(size) * fill_ratio
            if position_side == side:
                position_size = executed_size
            else:
                if position_side is not None:
                    exit_dir = -1 if position_side == "long" else 1
                    exit_price = _execution_price(price, exit_dir, taker_bps, spread, slippage, maker=False)
                    pnl = position_size * (exit_price - entry_price) * (1 if position_side == "long" else -1)
                    balance += pnl
                    trades.append(
                        Trade(
                            entry_time=entry_time or ts,
                            exit_time=ts,
                            side=position_side,
                            size=position_size,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            holding_period=i - market.index.get_loc(entry_time) if entry_time is not None else 0,
                        )
                    )
                position_side = side
                position_size = executed_size
                entry_price = exec_price
                entry_time = ts

        unrealised = 0.0
        if position_side is not None:
            direction = 1 if position_side == "long" else -1
            unrealised = position_size * (price - entry_price) * direction
            accrual_pnl -= position_size * price * (funding_bps / 10_000.0)
            balance += accrual_pnl
            accrual_pnl = 0.0
        equity_curve.append(balance + unrealised)
        equity_index.append(ts)

    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_index))
    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    stats = {
        "final_equity": float(equity_series.iloc[-1]) if not equity_series.empty else initial_balance,
        "total_return": float((equity_series.iloc[-1] / initial_balance) - 1.0) if not equity_series.empty else 0.0,
        "num_trades": int(len(trades_df)),
    }
    return BacktestResult(trades=trades_df, equity=equity_series, stats=stats)
