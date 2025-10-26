"""Legacy JSON storage backend (read-only, deprecated)."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .base import Storage, StorageError
from .parquet import BAR_COLUMNS

UTC = timezone.utc
_DEPRECATION_MSG = "JSONStorage is deprecated; migrate datasets to ParquetStorage"


def _parse_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _iter_days(start_ts: int, end_ts: int) -> Iterable[datetime]:
    current = datetime.fromtimestamp(start_ts / 1000, tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    final = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
    while current <= final:
        yield current
        current += pd.Timedelta(days=1)


@dataclass(slots=True)
class _JSONStorageConfig:
    root: Path
    market: str


class JSONStorage(Storage):
    """Read-only storage that retrieves OHLCV bars from JSON files."""

    _warned: bool = False

    def __init__(self, root: Path | str = "data", *, market: str = "default") -> None:
        self._config = _JSONStorageConfig(root=Path(root), market=market)
        if not JSONStorage._warned:
            warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            JSONStorage._warned = True

    def _resolve_path(self, symbol: str, interval: str, day: datetime) -> Path:
        symbol_part = symbol.upper()
        interval_part = interval.lower()
        relative = Path(self._config.market) / symbol_part / interval_part / day.strftime("%Y") / day.strftime("%m")
        return self._config.root / relative / f"{day.strftime('%d')}.json"

    @staticmethod
    def _normalise_payload(payload: object) -> list[Mapping[str, object]]:
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("rows", "candles", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [entry for entry in value if isinstance(entry, Mapping)]
        return []

    @staticmethod
    def _normalise_row(entry: Mapping[str, object]) -> dict[str, object] | None:
        ts_candidates = (
            entry.get("ts_open"),
            entry.get("ts"),
            entry.get("open_time"),
            entry.get("openTime"),
            entry.get("time"),
        )
        ts_open = next((value for value in ts_candidates if _parse_int(value) is not None), None)
        if ts_open is None:
            return None
        ts_open = int(ts_open)
        open_price = _parse_float(entry.get("open") or entry.get("o"))
        high_price = _parse_float(entry.get("high") or entry.get("h"))
        low_price = _parse_float(entry.get("low") or entry.get("l"))
        close_price = _parse_float(entry.get("close") or entry.get("c"))
        volume = _parse_float(entry.get("volume") or entry.get("v"))
        taker_buy_vol = _parse_float(entry.get("taker_buy_vol") or entry.get("takerBuyBase"))
        taker_buy_quote = _parse_float(entry.get("taker_buy_quote") or entry.get("takerBuyQuote"))
        trades = _parse_int(entry.get("trades") or entry.get("trade_count") or entry.get("trades_cnt"))

        fields = (open_price, high_price, low_price, close_price, volume)
        if any(value is None for value in fields):
            return None

        return {
            "ts_open": ts_open,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
            "taker_buy_vol": float(taker_buy_vol) if taker_buy_vol is not None else 0.0,
            "taker_buy_quote": float(taker_buy_quote) if taker_buy_quote is not None else 0.0,
            "trades": int(trades) if trades is not None else 0,
        }

    def load_window(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for day in _iter_days(start_ts, end_ts):
            path = self._resolve_path(symbol, interval, day)
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem edge
                raise StorageError(f"Failed to read {path}") from exc
            try:
                payload = json.loads(raw) if raw else []
            except json.JSONDecodeError as exc:
                raise StorageError(f"Invalid JSON payload in {path}") from exc
            for entry in self._normalise_payload(payload):
                normalised = self._normalise_row(entry)
                if normalised is None:
                    continue
                if start_ts <= normalised["ts_open"] <= end_ts:
                    records.append(normalised)

        if not records:
            return pd.DataFrame(columns=columns or list(BAR_COLUMNS), dtype="float64")

        df = pd.DataFrame.from_records(records)
        df.sort_values("ts_open", inplace=True)
        if columns:
            missing = [column for column in columns if column not in df.columns]
            if missing:
                raise StorageError(f"Unknown columns requested: {missing}")
            df = df.loc[:, list(columns)]
        return df.reset_index(drop=True)
