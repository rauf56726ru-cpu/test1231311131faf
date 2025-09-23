"""Daily backfill utilities with deterministic pagination for self-train."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import httpx

from ..io.history import _klines_to_frame
from ..utils.logging import get_logger
from ..utils.timeframes import interval_to_seconds

LOGGER = get_logger(__name__)

FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


@dataclass(slots=True)
class BackfillDayResult:
    """Container describing the outcome of a daily backfill step."""

    symbol: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp
    candles: pd.DataFrame
    day_slice: pd.DataFrame
    rows_in_day: int
    rows_total: int


class SelfTrainBackfill:
    """Fetch Binance klines with strict backwards pagination semantics."""

    def __init__(
        self,
        data_root: Path | str = Path("data") / "raw",
        *,
        limit: int = 1000,
        cooldown: float = 0.15,
    ) -> None:
        self.data_root = Path(data_root)
        self.limit = max(1, min(int(limit), 1000))
        self.cooldown = max(0.0, float(cooldown))
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        safe_symbol = (symbol or "").upper()
        safe_interval = interval or "1m"
        return self.data_root / f"{safe_symbol}_{safe_interval}.parquet"

    async def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            empty_index = pd.DatetimeIndex([], tz="UTC")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=empty_index
            )
        frame = await asyncio.to_thread(pd.read_parquet, path)
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        return frame

    async def _save(self, path: Path, frame: pd.DataFrame) -> None:
        await asyncio.to_thread(frame.to_parquet, path)

    async def _fetch_page(
        self, symbol: str, interval: str, end_ms: int
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": self.limit,
            "endTime": int(end_ms),
        }
        attempt = 0
        delay = 1.0
        max_attempts = 5
        while attempt < max_attempts:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(FUTURES_BASE_URL, params=params)
                if response.status_code in {418, 429}:
                    LOGGER.warning(
                        "Binance futures rate limit for %s %s (attempt %s/%s)",
                        symbol,
                        interval,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {418, 429} and attempt < max_attempts:
                    LOGGER.warning(
                        "HTTP %s from Binance futures, retrying %s/%s",
                        status,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(f"Binance futures REST error: {exc}") from exc
            except (httpx.RequestError, httpx.HTTPError) as exc:
                if attempt >= max_attempts:
                    LOGGER.warning("Binance futures REST request failed after retries: %s", exc)
                    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                LOGGER.warning("Binance futures request failed (%s), retrying", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue

            payload = response.json()
            if not isinstance(payload, list):
                LOGGER.warning("Unexpected Binance futures payload for %s %s", symbol, interval)
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            frame = _klines_to_frame(payload)
            return frame

        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    async def _fetch_backward(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch klines moving backwards from ``end_ms`` until ``start_ms``."""

        pages: list[pd.DataFrame] = []
        loops = 0
        max_loops = 2000  # safeguards against runaway pagination
        interval_seconds = max(1.0, interval_to_seconds(interval))
        interval_ms = int(interval_seconds * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        day_end_ms = int(end_ms) + interval_ms - 1
        current_end = min(now_ms, day_end_ms)
        earliest_open = int(start_ms)
        while loops < max_loops and current_end >= earliest_open:
            loops += 1
            frame = await self._fetch_page(symbol, interval, current_end)
            if frame.empty:
                break
            frame = frame.sort_index()
            if frame.index.tz is None:  # pragma: no cover - defensive
                frame.index = frame.index.tz_localize("UTC")
            now_cap = pd.Timestamp(datetime.now(timezone.utc))
            max_allowed = now_cap + pd.Timedelta(milliseconds=interval_ms)
            frame = frame.loc[frame.index <= max_allowed]
            frame = frame.loc[frame.index >= pd.Timestamp(earliest_open, unit="ms", tz="UTC")]

            if frame.empty:
                break

            first_open = frame.index[0]
            last_open = frame.index[-1]
            LOGGER.info(
                "[BACKFILL] TF=%s D=%s page_n=%s first_ts=%s last_ts=%s",
                interval,
                pd.Timestamp(start_ms, unit="ms", tz="UTC").date().isoformat(),
                loops,
                first_open.isoformat(),
                last_open.isoformat(),
            )
            pages.append(frame)
            first_open_ms = int(first_open.timestamp() * 1000)
            if first_open_ms <= earliest_open:
                break
            current_end = first_open_ms - 1
            if self.cooldown:
                await asyncio.sleep(self.cooldown)

        if not pages:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        merged = pd.concat(pages).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        return merged

    def _resample_10m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        optional_sum = [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        for column in optional_sum:
            if column in frame.columns:
                agg_map[column] = "sum"
        resampled = (
            frame.resample("10T", origin="epoch", label="left", closed="left").agg(agg_map)
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        if resampled.index.tz is None:  # pragma: no cover - defensive
            resampled.index = resampled.index.tz_localize("UTC")
        else:
            resampled.index = resampled.index.tz_convert("UTC")
        interval_ms = int(10 * 60 * 1000)
        close_ms = resampled.index.view("int64") // 1_000_000 + interval_ms - 1
        resampled["close_time"] = close_ms
        return resampled

    async def backfill_day(
        self,
        symbol: str,
        interval: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
    ) -> BackfillDayResult:
        """Backfill an exact day range with idempotent parquet updates."""

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        interval_seconds = max(1.0, interval_to_seconds(interval))
        interval_ms = int(interval_seconds * 1000)
        fetch_interval = "1m" if interval == "10m" else interval
        fetch_interval_ms = int(max(1.0, interval_to_seconds(fetch_interval)) * 1000)
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        path = self._path(symbol, interval)
        existing = await self._load(path)

        # Skip pagination if we already have full coverage for this day
        expected_index = pd.date_range(
            start=start_ts,
            end=end_ts,
            freq=pd.to_timedelta(interval_seconds, unit="s"),
            tz="UTC",
        )
        slice_existing = existing.loc[(existing.index >= start_ts) & (existing.index <= end_ts)]
        missing = expected_index.difference(slice_existing.index)

        new_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if not missing.empty:
            fetch_end_ms = end_ms + fetch_interval_ms
            fetched = await self._fetch_backward(
                symbol,
                fetch_interval,
                start_ms=start_ms,
                end_ms=fetch_end_ms,
            )
            if interval == "10m" and not fetched.empty:
                fetched = self._resample_10m(fetched)
            if not fetched.empty:
                fetched = fetched.loc[(fetched.index >= start_ts) & (fetched.index <= end_ts)]
                new_data = fetched

        if not new_data.empty:
            combined = pd.concat([existing, new_data])
        else:
            combined = existing.copy()

        if not combined.empty:
            combined = combined.loc[~combined.index.duplicated(keep="last")]  # drop duplicates
            combined.sort_index(inplace=True)
            max_allowed = pd.Timestamp(datetime.now(timezone.utc)) + pd.Timedelta(milliseconds=interval_ms)
            combined = combined.loc[combined.index <= max_allowed]

        if existing.shape != combined.shape or not existing.index.equals(combined.index) or not existing.equals(combined):
            await self._save(path, combined)

        day_slice = combined.loc[(combined.index >= start_ts) & (combined.index <= end_ts)]
        rows_in_day = int(len(day_slice))
        rows_total = int(len(combined))

        LOGGER.debug(
            "Backfilled %s %s for %s rows_day=%s rows_total=%s",
            symbol,
            interval,
            start_ts.date().isoformat(),
            rows_in_day,
            rows_total,
        )

        return BackfillDayResult(
            symbol=symbol,
            interval=interval,
            start=start_ts,
            end=end_ts,
            candles=combined,
            day_slice=day_slice,
            rows_in_day=rows_in_day,
            rows_total=rows_total,
        )

