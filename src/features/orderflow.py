"""Order-flow and microstructure feature helpers."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def cumulative_volume_delta(trades: pd.DataFrame | None) -> pd.Series:
    trades = _ensure_frame(trades)
    if trades.empty:
        return pd.Series(dtype=float)
    required = {"quantity", "is_buyer_maker"}
    if not required.issubset(trades.columns):
        missing = required - set(trades.columns)
        raise ValueError(f"Trades frame missing required columns: {missing}")
    direction = np.where(trades["is_buyer_maker"], -1.0, 1.0)
    signed_qty = trades["quantity"].astype(float).values * direction
    return pd.Series(signed_qty, index=trades.index, dtype=float).cumsum()


def order_flow_imbalance(book: pd.DataFrame | None) -> pd.Series:
    book = _ensure_frame(book)
    if book.empty:
        return pd.Series(dtype=float)
    required = {"bid_size", "ask_size", "bid_price", "ask_price"}
    if not required.issubset(book.columns):
        missing = required - set(book.columns)
        raise ValueError(f"Order book missing columns: {missing}")
    bid_change = book["bid_size"].astype(float).diff().fillna(0.0)
    ask_change = book["ask_size"].astype(float).diff().fillna(0.0)
    bid_price_change = book["bid_price"].astype(float).diff().fillna(0.0)
    ask_price_change = book["ask_price"].astype(float).diff().fillna(0.0)
    ofi = bid_change.where(bid_price_change >= 0, 0.0) - ask_change.where(ask_price_change <= 0, 0.0)
    return ofi.cumsum()


def imbalance_levels(book: pd.DataFrame | None, levels: Iterable[int]) -> pd.DataFrame:
    book = _ensure_frame(book)
    if book.empty:
        return pd.DataFrame()
    out: dict[str, pd.Series] = {}
    for level in levels:
        bid_col = f"bid_size_{level}"
        ask_col = f"ask_size_{level}"
        if bid_col in book.columns and ask_col in book.columns:
            denom = book[bid_col].astype(float) + book[ask_col].astype(float)
            imbalance = (book[bid_col].astype(float) - book[ask_col].astype(float)) / denom.replace(0.0, np.nan)
            out[f"imbalance_{level}"] = imbalance.fillna(0.0)
    return pd.DataFrame(out, index=book.index)


def microprice(book: pd.DataFrame | None) -> pd.Series:
    book = _ensure_frame(book)
    if book.empty:
        return pd.Series(dtype=float)
    bid_price = book["bid_price"].astype(float)
    ask_price = book["ask_price"].astype(float)
    bid_size = book.get("bid_size", pd.Series(1.0, index=book.index)).astype(float)
    ask_size = book.get("ask_size", pd.Series(1.0, index=book.index)).astype(float)
    total = bid_size + ask_size
    micro = (bid_price * ask_size + ask_price * bid_size) / total.replace(0.0, np.nan)
    return micro.ffill().bfill()


def roll_impact(trades: pd.DataFrame | None, window: int = 50) -> pd.Series:
    trades = _ensure_frame(trades)
    if trades.empty or "price" not in trades.columns:
        return pd.Series(dtype=float)
    price_change = trades["price"].astype(float).diff().abs()
    volume = trades.get("quantity", pd.Series(1.0, index=trades.index)).astype(float)
    impact = (price_change / volume.replace(0.0, np.nan)).rolling(window, min_periods=1).mean()
    return impact.fillna(0.0)


def vpin_proxy(trades: pd.DataFrame | None, window: int = 50) -> pd.Series:
    trades = _ensure_frame(trades)
    if trades.empty or "quantity" not in trades.columns:
        return pd.Series(dtype=float)
    is_sell = trades.get("is_buyer_maker", pd.Series(False, index=trades.index))
    buy_volume = trades["quantity"].astype(float).where(~is_sell, 0.0)
    sell_volume = trades["quantity"].astype(float).where(is_sell, 0.0)
    volume_bucket = trades["quantity"].astype(float).rolling(window, min_periods=1).sum()
    imbalance = (buy_volume - sell_volume).abs().rolling(window, min_periods=1).sum()
    return (imbalance / volume_bucket.replace(0.0, np.nan)).fillna(0.0)


def build_orderflow_features(
    trades: pd.DataFrame | None,
    book: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate order-flow features and align them with the candle index."""

    index = pd.DatetimeIndex(index)
    features: dict[str, pd.Series] = {}

    cvd = cumulative_volume_delta(trades)
    if not cvd.empty:
        features["cvd"] = cvd

    ofi = order_flow_imbalance(book)
    if not ofi.empty:
        features["ofi"] = ofi

    micro = microprice(book)
    if not micro.empty:
        features["microprice"] = micro

    roll = roll_impact(trades)
    if not roll.empty:
        features["roll_impact"] = roll

    vpin = vpin_proxy(trades)
    if not vpin.empty:
        features["vpin"] = vpin

    if book is not None and not _ensure_frame(book).empty:
        imbalance_df = imbalance_levels(book, levels=[1, 5, 10])
        for column in imbalance_df.columns:
            features[column] = imbalance_df[column]

    frame = pd.DataFrame({name: series for name, series in features.items()})
    if frame.empty:
        frame = pd.DataFrame(index=index)
    frame = frame.reindex(index).ffill().fillna(0.0)
    return frame.astype(float)
