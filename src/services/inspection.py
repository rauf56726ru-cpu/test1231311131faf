"""Inspection payload assembly and UI rendering."""
from __future__ import annotations

import html as html_utils
import json
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, DefaultDict

from .ohlc import TIMEFRAME_WINDOWS, TIMEFRAME_TO_MS, normalise_ohlcv
from ..meta import Meta

Snapshot = Dict[str, Any]

DEFAULT_SYMBOL = "BTCUSDT"

_MAX_STORED_SNAPSHOTS = 16
_SNAPSHOT_STORE: "OrderedDict[str, Snapshot]" = OrderedDict()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        _SNAPSHOT_STORE.popitem(last=False)


def build_placeholder_snapshot(*, symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Snapshot:
    """Create a synthetic snapshot used to populate the inspection UI by default."""

    timeframe_key = timeframe.lower()
    if timeframe_key not in TIMEFRAME_TO_MS:
        timeframe_key = "1m"
    interval_ms = TIMEFRAME_TO_MS[timeframe_key]

    total_candles = 120
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (total_candles - 1) * interval_ms

    base_price = 100_000.0
    candles = []
    rolling_price = base_price
    for idx in range(total_candles):
        ts = start_ms + idx * interval_ms
        wave = math.sin(idx / 6.0) * 140.0
        drift = idx * 6.5
        open_price = rolling_price + wave + drift
        close_variation = math.sin(idx / 3.5) * 60.0 + math.cos(idx / 5.0) * 35.0
        close_price = max(1.0, open_price + close_variation)
        high_price = max(open_price, close_price) + abs(math.sin(idx / 4.5)) * 55.0
        low_price = min(open_price, close_price) - abs(math.cos(idx / 3.8)) * 55.0
        volume = 180.0 + abs(math.sin(idx / 4.2)) * 90.0

        candles.append(
            {
                "t": ts,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(low_price, 2),
                "c": round(close_price, 2),
                "v": round(volume, 2),
            }
        )
        rolling_price = close_price

    selection = None
    if candles:
        window = min(40, len(candles))
        selection = {"start": candles[-window]["t"], "end": candles[-1]["t"]}

    agg_trades: List[Dict[str, Any]] = []
    sample = candles[-80:] if candles else []
    for candle in sample:
        ts = int(candle["t"])
        close = float(candle.get("c", candle.get("o", 0.0)))
        open_ = float(candle.get("o", close))
        qty = max(0.01, abs(close - open_) / max(1.0, interval_ms / 60_000))
        trade_time = ts + interval_ms // 2
        agg_trades.append({
            "t": trade_time,
            "p": round(close, 2),
            "q": round(qty, 4),
            "side": "buy" if close >= open_ else "sell",
        })

    agg_payload = {
        "symbol": symbol.upper(),
        "agg": agg_trades,
    }

    return {
        "id": "placeholder",
        "symbol": symbol.upper(),
        "tf": timeframe_key,
        "frames": {timeframe_key: {"tf": timeframe_key, "candles": candles}},
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "agg_trades": agg_payload,
        "meta": {"source": {"kind": "placeholder", "generated": True}},
    }




def _in_session(moment: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= moment < end
    return moment >= start or moment < end


def _compute_vwap_value(entries: Iterable[Mapping[str, Any]]) -> float | None:
    total_pv = 0.0
    total_volume = 0.0
    for entry in entries:
        high = float(entry.get("h", entry.get("high", 0.0)))
        low = float(entry.get("l", entry.get("low", 0.0)))
        close = float(entry.get("c", entry.get("close", 0.0)))
        volume = float(entry.get("v", entry.get("volume", 0.0)))
        typical_price = (high + low + close) / 3.0
        total_pv += typical_price * volume
        total_volume += volume
    if total_volume <= 0.0:
        return None
    return total_pv / total_volume


def compute_session_vwaps(symbol: str, candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute VWAP for daily and configured sessions across recent days."""

    if not candles:
        return {"symbol": symbol.upper(), "vwap": []}

    bars: List[Tuple[int, float, float, float, float]] = []
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        raw_ts = (
            candle.get("t")
            or candle.get("time")
            or candle.get("openTime")
        )
        if raw_ts is None:
            continue
        try:
            open_ms = int(raw_ts)
            high = float(candle.get("h", candle.get("high", 0.0)))
            low = float(candle.get("l", candle.get("low", 0.0)))
            close = float(candle.get("c", candle.get("close", 0.0)))
            volume = float(candle.get("v", candle.get("volume", 0.0)))
        except (TypeError, ValueError):
            continue
        bars.append((open_ms, high, low, close, volume))

    if not bars:
        return {"symbol": symbol.upper(), "vwap": []}

    bars.sort(key=lambda item: item[0])
    last_date = datetime.fromtimestamp(bars[-1][0] / 1000.0, tz=timezone.utc).date()
    lookback = max(1, int(Meta.VWAP_LOOKBACK_DAYS))
    start_date = last_date - timedelta(days=lookback - 1)
    sessions = list(Meta.iter_vwap_sessions())

    daily_buckets: defaultdict = defaultdict(list)  # type: ignore[var-annotated]
    session_buckets: DefaultDict[Tuple[datetime.date, str], List[Mapping[str, Any]]] = defaultdict(list)

    for open_ms, high, low, close, volume in bars:
        dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
        if dt.date() < start_date:
            continue
        entry = {"h": high, "l": low, "c": close, "v": volume}
        daily_buckets[dt.date()].append(entry)
        moment = dt.time()
        for session_name, start_time, end_time in sessions:
            if _in_session(moment, start_time, end_time):
                session_buckets[(dt.date(), session_name)].append(entry)

    ordered_dates = sorted(daily_buckets.keys())
    if len(ordered_dates) > lookback:
        ordered_dates = ordered_dates[-lookback:]

    results: List[Dict[str, object]] = []
    for date_key in ordered_dates:
        daily_value = _compute_vwap_value(daily_buckets[date_key])
        if daily_value is not None:
            results.append({"date": date_key.isoformat(), "session": "daily", "value": daily_value})
        for session_name, _, _ in sessions:
            entries = session_buckets.get((date_key, session_name))
            if not entries:
                continue
            session_value = _compute_vwap_value(entries)
            if session_value is None:
                continue
            results.append({"date": date_key.isoformat(), "session": session_name, "value": session_value})

    return {"symbol": symbol.upper(), "vwap": results}
def _coerce_frame(tf_key: str, frame: Mapping[str, Any] | Sequence[Any]) -> Dict[str, Any]:
    if tf_key not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {tf_key}")

    if isinstance(frame, Mapping):
        raw_candles = frame.get("candles", [])
    else:
        raw_candles = frame

    try:
        candles = list(raw_candles)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Frame candles must be iterable") from exc

    return {"tf": tf_key, "candles": candles}


def _extract_frames(snapshot: Mapping[str, Any], primary_tf: str) -> Dict[str, Dict[str, Any]]:
    frames: Dict[str, Dict[str, Any]] = {}
    raw_frames = snapshot.get("frames")
    if isinstance(raw_frames, Mapping):
        for key, frame in raw_frames.items():
            tf_value = None
            if isinstance(frame, Mapping):
                tf_value = frame.get("tf")
            tf_key = str(tf_value or key or primary_tf).lower()
            frames[tf_key] = _coerce_frame(tf_key, frame)  # type: ignore[arg-type]
    elif "candles" in snapshot:
        try:
            candles = list(snapshot["candles"])  # type: ignore[index]
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Snapshot candles must be iterable") from exc
        frames[primary_tf] = {"tf": primary_tf, "candles": candles}
    else:
        raise ValueError("Snapshot must include candles or frames")

    if not frames:
        raise ValueError("Snapshot did not include any frames")

    return frames


def register_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Store a snapshot captured by the chart frontend or inspection UI."""

    symbol = str(snapshot.get("symbol") or snapshot.get("ticker") or "UNKNOWN").upper()
    primary_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()
    snapshot_id = str(
        snapshot.get("id")
        or snapshot.get("snapshot_id")
        or snapshot.get("snapshot")
        or f"snap-{int(datetime.now(timezone.utc).timestamp()*1000)}"
    )

    frames = _extract_frames(snapshot, primary_tf)
    if primary_tf not in frames:
        primary_tf = next(iter(frames))

    meta: MutableMapping[str, Any] = {}
    for key in ("meta", "diagnostics", "source"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            meta[key] = dict(value)

    selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    selection_data = None
    if selection is not None:
        try:
            selection_data = {
                "start": int(selection.get("start", 0)),
                "end": int(selection.get("end", 0)),
            }
        except (TypeError, ValueError):
            selection_data = None

    stored: Snapshot = {
        "id": snapshot_id,
        "symbol": symbol,
        "tf": primary_tf,
        "frames": frames,
        "captured_at": snapshot.get("captured_at") or _now_iso(),
        "meta": meta,
    }

    if selection_data:
        stored["selection"] = selection_data

    for key in ("delta", "vwap", "zones", "smt", "agg_trades"):
        if key in snapshot:
            stored[key] = snapshot[key]

    _SNAPSHOT_STORE[snapshot_id] = stored
    _SNAPSHOT_STORE.move_to_end(snapshot_id)
    _ensure_snapshot_limit()
    return snapshot_id


def get_snapshot(snapshot_id: str) -> Snapshot | None:
    """Return a stored snapshot if present."""

    snapshot = _SNAPSHOT_STORE.get(snapshot_id)
    if snapshot is not None:
        _SNAPSHOT_STORE.move_to_end(snapshot_id)
    return snapshot


def list_snapshots() -> List[Dict[str, Any]]:
    """Return metadata about stored snapshots ordered from newest to oldest."""

    entries: List[Dict[str, Any]] = []
    for snapshot in reversed(_SNAPSHOT_STORE.values()):
        entries.append(
            {
                "id": snapshot.get("id"),
                "symbol": snapshot.get("symbol"),
                "tf": snapshot.get("tf"),
                "captured_at": snapshot.get("captured_at"),
                "selection": snapshot.get("selection"),
            }
        )
    return entries


def _filter_by_selection(
    candles: Sequence[Mapping[str, Any]],
    *,
    start: int | None,
    end: int | None,
) -> List[Dict[str, Any]]:
    if start is None and end is None:
        return [dict(candle) for candle in candles]
    filtered: List[Dict[str, Any]] = []
    for candle in candles:
        ts = int(candle.get("t", 0))
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        filtered.append(dict(candle))
    return filtered


def _compute_delta_series(candles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    cumulative = 0.0
    for candle in candles:
        open_price = float(candle.get("o", 0.0))
        close_price = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        net = (close_price - open_price) * volume
        cumulative += net
        delta_pct = ((close_price - open_price) / open_price * 100.0) if open_price else 0.0
        series.append(
            {
                "t": candle.get("t"),
                "delta": net,
                "deltaPct": delta_pct,
                "cvd": cumulative,
            }
        )
    return series


def _compute_vwap(candles: Sequence[Mapping[str, Any]]) -> float:
    total_pv = 0.0
    total_volume = 0.0
    for candle in candles:
        high = float(candle.get("h", 0.0))
        low = float(candle.get("l", 0.0))
        close = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        typical_price = (high + low + close) / 3.0
        total_pv += typical_price * volume
        total_volume += volume
    if total_volume <= 0:
        return 0.0
    return total_pv / total_volume


def build_inspection_payload(snapshot: Snapshot) -> Dict[str, Any]:
    """Build a combined inspection payload from a stored snapshot."""

    symbol = snapshot.get("symbol", "UNKNOWN")
    frames: Mapping[str, Mapping[str, Any]] = snapshot.get("frames", {})  # type: ignore[assignment]
    selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    start = int(selection.get("start")) if selection and selection.get("start") else None
    end = int(selection.get("end")) if selection and selection.get("end") else None

    normalised_frames: Dict[str, Dict[str, Any]] = {}
    diagnostics_frames: Dict[str, Any] = {}
    delta_frames: Dict[str, List[Dict[str, Any]]] = {}
    vwap_frames: Dict[str, Dict[str, Any]] = {}

    for tf_key, frame in frames.items():
        candles = frame.get("candles", []) if isinstance(frame, Mapping) else []
        if not isinstance(candles, Sequence):
            try:
                candles = list(candles)  # type: ignore[arg-type]
            except TypeError:
                candles = []

        window_limit = None
        interval_ms = TIMEFRAME_TO_MS.get(tf_key.lower())
        if interval_ms and start is not None and end is not None:
            span_ms = abs(end - start)
            window_limit = max(1, math.ceil(span_ms / interval_ms) + 1)

        result = normalise_ohlcv(
            symbol,
            tf_key,
            candles,
            include_diagnostics=True,
            window_limit=window_limit,
        )
        diagnostics = result.pop("diagnostics", {})

        filtered_candles = _filter_by_selection(result.get("candles", []), start=start, end=end)
        result["candles"] = filtered_candles

        if isinstance(diagnostics, Mapping):
            diagnostics_series = diagnostics.get("series") if isinstance(diagnostics.get("series"), Sequence) else []
            diagnostics_missing = diagnostics.get("missing_bars") if isinstance(diagnostics.get("missing_bars"), Sequence) else []
            diagnostics = dict(diagnostics)
            diagnostics["series"] = _filter_by_selection(diagnostics_series, start=start, end=end)
            diagnostics["missing_bars"] = _filter_by_selection(diagnostics_missing, start=start, end=end)
        else:
            diagnostics = {}

        normalised_frames[tf_key] = result
        diagnostics_frames[tf_key] = diagnostics
        delta_frames[tf_key] = _compute_delta_series(filtered_candles)
        vwap_frames[tf_key] = {
            "selection": {"start": start, "end": end},
            "value": _compute_vwap(filtered_candles),
        }

    base_candles: Sequence[Mapping[str, Any]] = normalised_frames.get("1m", {}).get("candles", [])
    if not base_candles and normalised_frames:
        first_key = next(iter(normalised_frames))
        base_candles = normalised_frames[first_key].get("candles", [])
    session_vwap = compute_session_vwaps(symbol, base_candles)

    data_section = {
        "symbol": symbol,
        "frames": normalised_frames,
        "selection": selection,
        "session_vwap": session_vwap,
        "agg_trades": snapshot.get("agg_trades")
        or {
            "status": "unavailable",
            "detail": "Agg trade data is not present in the snapshot.",
        },
        "delta_cvd": delta_frames,
        "vwap_tpo": vwap_frames,
        "zones": snapshot.get("zones")
        or {
            "status": "unavailable",
            "detail": "Zones provider is not configured in the snapshot.",
        },
        "smt": snapshot.get("smt")
        or {
            "status": "unavailable",
            "detail": "SMT provider is not configured in the snapshot.",
        },
        "meta": {
            "requested": {
                "symbol": symbol,
                "frames": sorted(normalised_frames),
            },
            "source": snapshot.get("meta", {}),
        },
    }

    diagnostics_section = {
        "generated_at": _now_iso(),
        "snapshot_id": snapshot.get("id"),
        "captured_at": snapshot.get("captured_at"),
        "frames": diagnostics_frames,
    }

    return {"DATA": data_section, "DIAGNOSTICS": diagnostics_section}


def render_inspection_page(
    payload: Dict[str, Any],
    *,
    snapshot_id: str | None,
    symbol: str,
    timeframe: str,
    snapshots: List[Dict[str, Any]],
) -> str:
    """Render the inspection dashboard HTML."""

    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    snapshots_json = json.dumps(snapshots, ensure_ascii=False).replace("</", "<\\/")

    symbol_clean = (symbol or "").strip().upper()
    if not symbol_clean or symbol_clean in {"—", "-", "UNKNOWN"}:
        symbol_clean = DEFAULT_SYMBOL
    symbol_value = html_utils.escape(symbol_clean)
    timeframe_value = html_utils.escape(timeframe)
    snapshot_value = html_utils.escape(snapshot_id or "")

    timeframe_options = []
    for tf_key in TIMEFRAME_WINDOWS:
        selected = " selected" if tf_key == timeframe else ""
        timeframe_options.append(
            f'<option value="{html_utils.escape(tf_key)}"{selected}>{html_utils.escape(tf_key)}</option>'
        )


    data_section = payload.get("DATA") if isinstance(payload, Mapping) else None
    diagnostics_section = payload.get("DIAGNOSTICS") if isinstance(payload, Mapping) else None

    def _format_json_block(value: Any) -> str:
        try:
            formatted = json.dumps(value if value is not None else None, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            formatted = json.dumps(None, ensure_ascii=False, indent=2)
        return html_utils.escape(formatted)

    frames_section: Dict[str, Any] = {}
    if isinstance(data_section, Mapping):
        raw_frames = data_section.get("frames")
        if isinstance(raw_frames, Mapping):
            frames_section = dict(raw_frames)

    timeframe_key = str(timeframe) if timeframe is not None else ""
    metric_section = None
    if frames_section:
        candidate = frames_section.get(timeframe_key)
        metric_section = candidate if candidate else frames_section

    data_json_initial = _format_json_block(data_section)
    diagnostics_json_initial = _format_json_block(diagnostics_section)
    metric_json_initial = _format_json_block(metric_section)

    style_block = """
    :root {
      color-scheme: dark;
      --bg: #020617;
      --fg: #e2e8f0;
      --muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.24);
      --accent: #38bdf8;
      --accent-strong: #0ea5e9;
      --panel: rgba(15, 23, 42, 0.72);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at 20% -10%, #1e293b 0%, #0f172a 40%, #020617 100%);
      color: var(--fg);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 2rem 1.5rem 1rem;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
    }
    header h1 {
      margin: 0 0 0.35rem;
      font-size: clamp(1.8rem, 2.8vw, 2.6rem);
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    main {
      width: min(1200px, 96vw);
      margin: 0 auto 3rem;
      flex: 1;
      display: grid;
      grid-template-columns: minmax(320px, 360px) 1fr;
      gap: 1.5rem;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1.5rem;
      box-shadow: 0 28px 70px rgba(2, 6, 23, 0.45);
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
    }
    h2 {
      margin: 0;
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.9);
    }
    label span {
      display: block;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.3rem;
      color: rgba(148, 163, 184, 0.75);
    }
    input, select, button {
      font: inherit;
    }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.55rem 0.75rem;
      color: var(--fg);
    }
    button {
      cursor: pointer;
      border-radius: 999px;
      border: none;
      padding: 0.55rem 1.1rem;
      font-weight: 600;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    button.primary {
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #0f172a;
      box-shadow: 0 16px 40px rgba(14, 165, 233, 0.35);
    }
    button.secondary {
      background: rgba(148, 163, 184, 0.15);
      color: var(--fg);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      box-shadow: none;
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(8, 47, 73, 0.4);
    }
    .controls-grid {
      display: grid;
      gap: 1rem;
    }
    .timeframes {
      display: grid;
      gap: 0.4rem;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .timeframes label {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.4rem 0.6rem;
      border-radius: 10px;
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.18);
      font-size: 0.85rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-size: 0.75rem;
      background: rgba(148, 163, 184, 0.18);
      color: var(--muted);
    }
    .snapshot-select {
      display: flex;
      gap: 0.65rem;
      align-items: center;
      flex-wrap: wrap;
    }
    .snapshot-select select {
      flex: 1;
      min-width: 200px;
    }
    .chart-shell {
      height: 420px;
      min-height: 360px;
      width: 100%;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: rgba(15, 23, 42, 0.9);
      position: relative;
    }
    .chart-shell::after {
      content: attr(data-selection-label);
      position: absolute;
      inset: auto 1rem 1rem auto;
      background: rgba(8, 47, 73, 0.85);
      border-radius: 999px;
      padding: 0.35rem 0.8rem;
      font-size: 0.75rem;
      color: rgba(248, 250, 252, 0.88);
      pointer-events: none;
    }
    .json-panels {
      display: grid;
      gap: 1rem;
    }
    .collapse {
      border-radius: 14px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      overflow: hidden;
      background: rgba(15, 23, 42, 0.9);
    }
    .collapse header {
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.85rem 1rem;
      background: rgba(14, 165, 233, 0.18);
      cursor: pointer;
      gap: 1rem;
    }
    .collapse header h3 {
      margin: 0;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .collapse pre {
      margin: 0;
      padding: 1rem;
      max-height: 260px;
      overflow: auto;
      font-size: 0.85rem;
      background: rgba(15, 23, 42, 0.78);
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }
    .collapse.collapsed pre {
      display: none;
    }
    .metrics-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
    }
    .metrics-bar button.active {
      background: rgba(56, 189, 248, 0.22);
      color: #f8fafc;
    }
    .meta-grid {
      display: grid;
      gap: 0.8rem;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .meta-tile {
      padding: 0.75rem 1rem;
      border-radius: 12px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .meta-tile span {
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(148, 163, 184, 0.7);
      margin-bottom: 0.4rem;
    }
    .status-banner {
      padding: 0.5rem 0.75rem;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(15, 23, 42, 0.6);
      font-size: 0.85rem;
    }
    .status-banner[data-tone="success"] {
      border-color: rgba(34, 197, 94, 0.6);
      background: rgba(22, 101, 52, 0.3);
    }
    .status-banner[data-tone="error"] {
      border-color: rgba(248, 113, 113, 0.6);
      background: rgba(127, 29, 29, 0.35);
    }
    .status-banner[data-tone="warning"] {
      border-color: rgba(250, 204, 21, 0.6);
      background: rgba(113, 63, 18, 0.35);
    }

    .main-preview-panel {
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(8, 15, 32, 0.6);
    }
    .index-preview {
      display: flex;
      flex-direction: column;
      gap: 1.4rem;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(15, 23, 42, 0.72);
      padding: 1.4rem;
    }
    .index-preview .page-header {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding-bottom: 0.6rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }
    .index-preview .header-top {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .index-preview .header-controls {
      display: inline-flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .index-preview .app-meta {
      display: inline-flex;
      gap: 0.45rem;
      align-items: center;
      padding: 0.4rem 0.75rem;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.22);
      font-variant-numeric: tabular-nums;
    }
    .index-preview .app-meta__label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(148, 163, 184, 0.75);
    }
    .index-preview .app-meta__value {
      font-weight: 600;
      font-size: 0.95rem;
    }
    .index-preview .page-header p {
      margin: 0;
      color: rgba(148, 163, 184, 0.75);
    }
    .index-preview .page-main {
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
    }
    .index-preview .controls-card,
    .index-preview .chart-card {
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      background: rgba(11, 22, 38, 0.78);
      padding: 1.25rem;
      box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.12);
    }
    .index-preview .controls-form {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem 1.4rem;
      align-items: flex-end;
    }
    .index-preview .form-field {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      min-width: 160px;
      flex: 1 1 220px;
    }
    .index-preview .form-field span {
      font-size: 0.85rem;
      color: rgba(148, 163, 184, 0.78);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .index-preview .form-field input,
    .index-preview .form-field select {
      padding: 0.7rem 0.9rem;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: rgba(15, 23, 42, 0.68);
      color: var(--fg);
      font: inherit;
    }
    .index-preview .btn-primary,
    .index-preview .btn-secondary {
      cursor: pointer;
      font-weight: 600;
      border-radius: 0.85rem;
      border: none;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .index-preview .btn-primary {
      padding: 0.8rem 1.6rem;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #0b1120;
      box-shadow: 0 12px 32px rgba(56, 189, 248, 0.35);
    }
    .index-preview .btn-secondary {
      padding: 0.65rem 1.3rem;
      background: rgba(148, 163, 184, 0.16);
      color: var(--fg);
    }
    .index-preview .btn-primary:hover,
    .index-preview .btn-secondary:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 30px rgba(8, 47, 73, 0.45);
    }
    .index-preview .chart-wrapper {
      width: 100%;
      height: 420px;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: rgba(7, 14, 30, 0.85);
      overflow: hidden;
    }
    .index-preview .chart-area {
      width: 100%;
      height: 100%;
    }
    .index-preview .chart-info {
      margin-top: 1.1rem;
      display: grid;
      gap: 0.85rem;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .index-preview .chart-info > div {
      padding: 0.75rem 1rem;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.68);
      border: 1px solid rgba(148, 163, 184, 0.2);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    .preview-selection {
      margin-top: 1.1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .preview-selection__controls {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.95rem;
      font-variant-numeric: tabular-nums;
    }
    .preview-actions {
      margin-top: 1.1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
    }
    #preview-status {
      flex: 1 1 260px;
      min-height: 0;
    }
    .index-preview .info-label {
      color: rgba(148, 163, 184, 0.75);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .index-preview .info-value {
      font-size: 1.05rem;
      font-variant-numeric: tabular-nums;
    }
    .index-preview #status-message {
      margin-top: 1.2rem;
      padding: 0.85rem 1rem;
      border-radius: 12px;
      border-left: 4px solid rgba(148, 163, 184, 0.25);
      background: rgba(148, 163, 184, 0.14);
      font-weight: 600;
    }
    @media (max-width: 960px) {
      main {
        grid-template-columns: 1fr;
      }
      .snapshot-select {
        flex-direction: column;
        align-items: stretch;
      }
    }
    """

    script_block = (
        "window.__INSPECTION_INITIAL__ = {\n"
        f"  payload: {payload_json},\n"
        f"  snapshotId: {json.dumps(snapshot_id or '')},\n"
        f"  symbol: {json.dumps(symbol_value)},\n"
        f"  timeframe: {json.dumps(timeframe_value)},\n"
        f"  snapshots: {snapshots_json},\n"
        f"  defaultSymbol: {json.dumps(DEFAULT_SYMBOL)}\n"
        "};\n"
    )

    ui_script = """
(function () {
  const TIMEFRAME_TO_MS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
  };
  const DEFAULT_TEST_TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"];


  function toChartBars(candles) {
    return (candles || []).map((candle) => ({
      time: Math.floor(Number(candle.t || candle.time || 0) / 1000),
      open: Number(candle.o ?? candle.open ?? 0),
      high: Number(candle.h ?? candle.high ?? 0),
      low: Number(candle.l ?? candle.low ?? 0),
      close: Number(candle.c ?? candle.close ?? 0),
    }));
  }

  async function fetchSnapshots() {
    const response = await fetch("/inspection/snapshots");
    if (!response.ok) throw new Error("Failed to fetch snapshots");
    return response.json();
  }

  function formatTs(ts) {
    if (!Number.isFinite(ts)) return "—";
    const date = new Date(ts);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toISOString().replace("T", " ").replace(".000Z", "Z");
  }

  function setJson(pre, data) {
    if (!pre) return;
    pre.textContent = JSON.stringify(data ?? null, null, 2);
  }

  function selectionLabel(start, end) {
    if (!start || !end) return "Выделите диапазон";
    const from = formatTs(start);
    const to = formatTs(end);
    return `${from} → ${to}`;
  }

  function normaliseSymbol(input) {
    const trimmed = (input || "").trim().toUpperCase();
    const cleaned = trimmed.replace(/[^A-Z0-9]/g, "");
    return cleaned;
  }

  async function fetchCandles(symbol, interval, startMs, endMs) {
    const results = [];
    const url = new URL("https://api.binance.com/api/v3/klines");
    url.searchParams.set("symbol", symbol.toUpperCase());
    url.searchParams.set("interval", interval);
    url.searchParams.set("startTime", Math.floor(Math.min(startMs, endMs)));
    url.searchParams.set("endTime", Math.floor(Math.max(startMs, endMs)));
    const intervalMs = TIMEFRAME_TO_MS[interval] || 60000;
    const span = Math.max(1, Math.ceil(Math.abs(endMs - startMs) / intervalMs) + 3);
    url.searchParams.set("limit", String(Math.min(1000, Math.max(span, 100))));
    const resp = await fetch(url.toString());
    if (!resp.ok) {
      throw new Error(`klines ${resp.status}`);
    }
    const rows = await resp.json();
    if (!Array.isArray(rows)) return [];
    for (const row of rows) {
      if (!row) continue;
      const openMs = Number(row[0]);
      const open = Number(row[1]);
      const high = Number(row[2]);
      const low = Number(row[3]);
      const close = Number(row[4]);
      const volume = Number(row[5]);
      if (!Number.isFinite(openMs)) continue;
      results.push({ t: openMs, o: open, h: high, l: low, c: close, v: volume });
    }
    results.sort((a, b) => a.t - b.t);
    return results;
  }

  async function postSnapshot(payload) {
    const response = await fetch("/inspection/snapshot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`Snapshot failed: ${response.status} ${text}`);
    }
    return response.json();
  }

  async function fetchPayload(id) {
    const response = await fetch(`/inspection?snapshot=${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("Failed to fetch payload");
    }
    return response.json();
  }

  function initCollapsibles() {
    document.querySelectorAll("[data-collapse-toggle]").forEach((toggle) => {
      toggle.addEventListener("click", () => {
        const target = toggle.closest(".collapse");
        if (!target) return;
        target.classList.toggle("collapsed");
      });
    });
    document.querySelectorAll("[data-copy-target]").forEach((button) => {
      button.addEventListener("click", async () => {
        const id = button.getAttribute("data-copy-target");
        if (!id) return;
        const el = document.getElementById(id);
        if (!el) return;
        try {
          await navigator.clipboard.writeText(el.textContent || "");
          const original = button.textContent;
          button.textContent = "Скопировано";
          setTimeout(() => (button.textContent = original), 1200);
        } catch (error) {
          console.warn("copy failed", error);
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const initial = window.__INSPECTION_INITIAL__ || {};
    const defaultSymbol = normaliseSymbol(initial.defaultSymbol) || "BTCUSDT";
    const snapshotSelect = document.getElementById("snapshot-select");
    const refreshButton = document.getElementById("refresh-snapshot");
    const dataPre = document.getElementById("data-json");
    const diagnosticsPre = document.getElementById("diagnostics-json");
    const metricPre = document.getElementById("metric-json");
    const snapshotMeta = document.getElementById("snapshot-meta");
    const frameSelect = document.getElementById("frame-select");
    const chartContainer = document.getElementById("inspection-chart");
    const selectionInfo = document.getElementById("selection-info");
    const buildButton = document.getElementById("build-session");
    const clearSelection = document.getElementById("clear-selection");
    const timeframeCheckboxes = Array.from(document.querySelectorAll("[data-tf-checkbox]"));
    const statusEl = document.getElementById("inspection-status");
    const metricButtons = Array.from(document.querySelectorAll("[data-metric]"));
    const symbolInput = document.getElementById("symbol-input");
    const checkAllDatasButton = document.getElementById("btn-check-all-datas");
    const dataCheckJson = document.getElementById("data-check-json");
    const dataCheckPanel = document.getElementById("data-check-panel");
    const dataCheckStatus = document.getElementById("data-check-status");

    initCollapsibles();

    const state = {
      payload: initial.payload || null,
      snapshotId: initial.snapshotId || null,
      selection: initial.payload?.DATA?.selection || null,
      frame: initial.timeframe || initial.payload?.DATA?.meta?.requested?.frames?.[0] || "1m",
      chart: null,
      series: null,
    };

    function setDataCheckStatus(message, tone = "info") {
      if (!dataCheckStatus) return;
      dataCheckStatus.textContent = message || "";
      dataCheckStatus.dataset.tone = tone;
      dataCheckStatus.hidden = !message;
    }

    if (symbolInput) {
      const initialSymbol =
        normaliseSymbol(initial.payload?.DATA?.symbol) || normaliseSymbol(initial.symbol) || defaultSymbol;
      symbolInput.value = initialSymbol;
    }

    function updateStatus(message, tone = "info") {
      if (!statusEl) return;
      statusEl.textContent = message || "";
      statusEl.dataset.tone = tone;
      statusEl.hidden = !message;
    }

    function updateSelectionLabel() {
      const start = state.selection && state.selection.start;
      const end = state.selection && state.selection.end;
      const label = selectionLabel(start, end);
      if (selectionInfo) selectionInfo.textContent = label;
      if (chartContainer) chartContainer.setAttribute("data-selection-label", label);
    }

    function populateSnapshots(list) {
      if (!snapshotSelect) return;
      snapshotSelect.innerHTML = "";
      for (const item of list || []) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.id} • ${item.symbol || "-"} • ${item.tf || "-"}`;
        snapshotSelect.append(option);
      }
      if (state.snapshotId && list.some((item) => item.id === state.snapshotId)) {
        snapshotSelect.value = state.snapshotId;
      }
    }

    function populateFrames(payload) {
      if (!frameSelect) return;
      frameSelect.innerHTML = "";
      const frames = payload?.DATA?.frames || {};
      const keys = Object.keys(frames);
      for (const key of keys) {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = key;
        frameSelect.append(option);
      }
      if (keys.length) {
        const target = keys.includes(state.frame) ? state.frame : keys[0];
        frameSelect.value = target;
        state.frame = target;
      }
    }

    function renderMeta(payload) {
      if (!snapshotMeta) return;
      const symbolRaw = payload?.DATA?.symbol || initial.symbol;
      const symbol = normaliseSymbol(symbolRaw) || "—";
      const frames = payload?.DATA?.meta?.requested?.frames || [];
      const found = (initial.snapshots || []).find((item) => item.id === state.snapshotId);
      const captured = found && found.captured_at;
      const selection = payload?.DATA?.selection;
      snapshotMeta.innerHTML = `
        <div class=\"meta-grid\">
          <div class=\"meta-tile\"><span>Snapshot</span><strong>${state.snapshotId || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Symbol</span><strong>${symbol || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Таймфреймы</span><strong>${frames.join(", ") || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Захват</span><strong>${captured || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Диапазон</span><strong>${selectionLabel(selection?.start, selection?.end)}</strong></div>
        </div>
      `;
    }

    function renderJson(payload) {
      setJson(dataPre, payload?.DATA);
      setJson(diagnosticsPre, payload?.DIAGNOSTICS);
    }

    async function fetchCheckAllDatasOverview() {
      if (!checkAllDatasButton) return;
      const url = new URL("/inspection/check-all-datas", window.location.origin);
      url.searchParams.set("hours", "4");
      setDataCheckStatus("Загрузка агрегированных данных…", "info");
      checkAllDatasButton.disabled = true;
      if (dataCheckJson) dataCheckJson.textContent = "null";
      if (dataCheckPanel) dataCheckPanel.innerHTML = "";
      try {
        const jsonResponse = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
        });
        let hasData = false;
        if (jsonResponse.status === 204) {
          setDataCheckStatus("Нет данных для агрегирования", "warning");
        } else if (jsonResponse.ok) {
          const payload = await jsonResponse.json();
          setJson(dataCheckJson, payload);
          hasData = true;
        } else {
          const text = await jsonResponse.text().catch(() => "");
          throw new Error(`JSON ${jsonResponse.status} ${text}`);
        }

        const htmlResponse = await fetch(url.toString(), {
          headers: { Accept: "text/html" },
        });

        if (htmlResponse.status === 204) {
          if (!hasData) {
            setDataCheckStatus("Нет данных для агрегирования", "warning");
          }
        } else if (htmlResponse.ok) {
          const html = await htmlResponse.text();
          if (dataCheckPanel) {
            dataCheckPanel.innerHTML = html;
          }
          hasData = true;
        } else {
          const text = await htmlResponse.text().catch(() => "");
          throw new Error(`HTML ${htmlResponse.status} ${text}`);
        }

        if (hasData) {
          setDataCheckStatus("", "info");
        }
      } catch (error) {
        console.error("check-all-datas", error);
        setDataCheckStatus("Ошибка загрузки агрегированных данных", "error");
      } finally {
        checkAllDatasButton.disabled = false;
      }
    }

    function ensureChart() {
      if (!chartContainer) return;
      const ensureLibrary = () => {
        if (window.LightweightCharts) {
          initialiseChart();
        }
      };

      function initialiseChart() {
        if (state.chart) return;
        const baseHeight = Math.max(
          320,
          chartContainer.clientHeight ||
            chartContainer.offsetHeight ||
            (chartContainer.parentElement && chartContainer.parentElement.clientHeight) ||
            320,
        );
        state.chart = LightweightCharts.createChart(chartContainer, {
          autoSize: true,
          height: baseHeight,
          layout: {
            background: { color: "#0f172a" },
            textColor: "#e2e8f0",
          },
          rightPriceScale: {
            borderColor: "rgba(148, 163, 184, 0.4)",
          },
          timeScale: {
            borderColor: "rgba(148, 163, 184, 0.4)",
            timeVisible: true,
            secondsVisible: true,
          },
          crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
          },
          grid: {
            vertLines: { color: "rgba(15, 23, 42, 0.6)" },
            horzLines: { color: "rgba(15, 23, 42, 0.6)" },
          },
        });
        state.series = state.chart.addCandlestickSeries({
          upColor: "#22c55e",
          downColor: "#ef4444",
          wickUpColor: "#f8fafc",
          wickDownColor: "#f8fafc",
          borderUpColor: "#22c55e",
          borderDownColor: "#ef4444",
          borderVisible: true,
        });

        const seedInitialFrame = () => {
          const frameKey = state.frame;
          const frameCandles = state.payload?.DATA?.frames?.[frameKey]?.candles || [];
          const bars = toChartBars(frameCandles);
          state.series.setData(bars);
          if (bars.length && state.chart) {
            state.chart.timeScale().fitContent();
          }
        };
        seedInitialFrame();

        const resize = () => {
          if (!state.chart) return;
          const nextHeight = Math.max(
            320,
            chartContainer.clientHeight ||
              chartContainer.offsetHeight ||
              (chartContainer.parentElement && chartContainer.parentElement.clientHeight) ||
              baseHeight,
          );
          state.chart.applyOptions({ height: nextHeight });
        };

        resize();
        if (window.ResizeObserver) {
          const observer = new ResizeObserver(resize);
          observer.observe(chartContainer);
        } else {
          window.addEventListener("resize", resize);
        }

        state.chart.subscribeClick((param) => {
          if (!param || typeof param.time === "undefined") return;
          const ts = Math.floor(Number(param.time) * 1000);
          if (!state.selection || !state.selection.start || state.selection.end) {
            state.selection = { start: ts, end: null };
          } else {
            state.selection.end = ts;
            if (state.selection.end < state.selection.start) {
              const tmp = state.selection.start;
              state.selection.start = state.selection.end;
              state.selection.end = tmp;
            }
          }
          updateSelectionLabel();
        });
      }

      if (window.LightweightCharts) {
        initialiseChart();
        return;
      }

      let loader = document.getElementById("lw-chart-loader");
      if (!loader) {
        loader = document.createElement("script");
        loader.src = "https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js";
        loader.id = "lw-chart-loader";
        loader.async = false;
        loader.onload = ensureLibrary;
        loader.onerror = () => updateStatus("Не удалось загрузить библиотеку графика", "error");
        document.head.appendChild(loader);
      }
    }

    function renderChart() {
      if (!chartContainer) return;
      ensureChart();
      if (!state.series) return;
      const frame = state.frame;
      const candles = state.payload?.DATA?.frames?.[frame]?.candles || [];
      const bars = toChartBars(candles);
      state.series.setData(bars);
      if (bars.length && state.chart) {
        state.chart.timeScale().fitContent();
      }
      updateSelectionLabel();
    }

    async function refreshSnapshots() {
      try {
        const list = await fetchSnapshots();
        initial.snapshots = list;
        populateSnapshots(list);
      } catch (error) {
        console.error(error);
        updateStatus("Не удалось загрузить список снэпшотов", "error");
      }
    }

    async function loadSnapshot(id) {
      if (!id) return;
      updateStatus("Загружаем данные снэпшота...", "info");
      try {
        const payload = await fetchPayload(id);
        state.payload = payload;
        state.snapshotId = id;
        state.selection = payload?.DATA?.selection || null;
        const nextSymbol = payload?.DATA?.symbol || initial.symbol;
        if (symbolInput) {
          const resolved = normaliseSymbol(nextSymbol) || normaliseSymbol(initial.symbol) || defaultSymbol;
          symbolInput.value = resolved;
        }
        populateFrames(payload);
        renderJson(payload);
        renderMeta(payload);
        renderChart();
        updateSelectionLabel();
        updateStatus("Снэпшот загружен", "success");
      } catch (error) {
        console.error(error);
        updateStatus("Ошибка загрузки снэпшота", "error");
      }
    }

    if (snapshotSelect) {
      snapshotSelect.addEventListener("change", (event) => {
        const value = event.target.value;
        loadSnapshot(value);
      });
    }

    if (frameSelect) {
      frameSelect.addEventListener("change", () => {
        state.frame = frameSelect.value;
        renderChart();
      });
    }

    if (refreshButton) {
      refreshButton.addEventListener("click", () => {
        if (state.snapshotId) {
          loadSnapshot(state.snapshotId);
        }
      });
    }

    if (clearSelection) {
      clearSelection.addEventListener("click", () => {
        state.selection = null;
        updateSelectionLabel();
      });
    }

    if (checkAllDatasButton) {
      checkAllDatasButton.addEventListener("click", () => {
        fetchCheckAllDatasOverview();
      });
    }

    async function createSnapshotFromSelection({ symbolValue, selection, frames, source = "inspection-ui" }) {
      if (!selection || !selection.start || !selection.end) {
        throw new Error('selection-missing');
      }
      const uniqueFrames = Array.from(new Set((frames || []).filter(Boolean)));
      if (!uniqueFrames.length) {
        throw new Error('frames-missing');
      }
      const resolvedSymbol =
        normaliseSymbol(symbolValue) || normaliseSymbol(initial.symbol) || defaultSymbol;
      if (!resolvedSymbol) {
        throw new Error('symbol-invalid');
      }
      const selectionStart = Math.floor(Number(selection.start));
      const selectionEnd = Math.floor(Number(selection.end));
      const framesPayload = {};
      for (const tf of uniqueFrames) {
        const candles = await fetchCandles(resolvedSymbol, tf, selectionStart, selectionEnd);
        framesPayload[tf] = { tf, candles };
      }
      const baseFrame = uniqueFrames[0] || Object.keys(framesPayload)[0];
      const baseCandles = (baseFrame && framesPayload[baseFrame]?.candles) || [];
      const intervalMs = TIMEFRAME_TO_MS[baseFrame] || 60000;
      const aggTrades = [];
      for (const candle of baseCandles || []) {
        const rawTs = Number(candle?.t ?? candle?.time ?? 0);
        if (!Number.isFinite(rawTs)) continue;
        const open = Number(candle?.o ?? candle?.open ?? 0);
        const close = Number(candle?.c ?? candle?.close ?? open);
        const qty = Math.max(0.01, Math.abs(close - open) / Math.max(1, intervalMs / 60_000));
        aggTrades.push({
          t: rawTs + Math.floor(intervalMs / 2),
          p: Number.isFinite(close) ? Number(close.toFixed(2)) : Number(open.toFixed(2)),
          q: Number(qty.toFixed(4)),
          side: close >= open ? "buy" : "sell",
        });
      }
      const payload = {
        id: `test-${Date.now()}`,
        symbol: resolvedSymbol,
        frames: framesPayload,
        selection: { start: selectionStart, end: selectionEnd },
        agg_trades: { symbol: resolvedSymbol, agg: aggTrades },
        meta: {
          source: {
            kind: source,
            frames: uniqueFrames,
            generated_at: new Date().toISOString(),
          },
        },
      };
      const result = await postSnapshot(payload);
      return { snapshotId: result.snapshot_id, payload };
    }

    if (buildButton) {
      buildButton.addEventListener("click", async () => {
        if (!state.selection || !state.selection.start || !state.selection.end) {
          updateStatus("Select a range on the chart before creating the test environment", "warning");
          return;
        }
        const selectedFrames = timeframeCheckboxes
          .filter((checkbox) => checkbox.checked)
          .map((checkbox) => checkbox.value);
        if (!selectedFrames.length) {
          updateStatus("No timeframes selected for testing", "warning");
          return;
        }
        const symbolValue = symbolInput ? symbolInput.value : initial.symbol;
        updateStatus("Collecting Binance data...", "info");
        buildButton.disabled = true;
        try {
          const { snapshotId } = await createSnapshotFromSelection({
            symbolValue,
            selection: state.selection,
            frames: selectedFrames,
            source: "inspection-panel",
          });
          state.snapshotId = snapshotId;
          await refreshSnapshots();
          if (snapshotSelect) snapshotSelect.value = state.snapshotId;
          await loadSnapshot(state.snapshotId);
          updateStatus("Test environment created", "success");
        } catch (error) {
          console.error(error);
          const detail = error && typeof error.message === "string" ? error.message : "";
          updateStatus(`Failed to create test environment${detail ? `: ${detail}` : ""}`, "error");
        } finally {
          buildButton.disabled = false;
        }
      });
    }

    function initPreviewPanel() {
      const previewRoot = document.querySelector("[data-preview-root]");
      if (!previewRoot) return;

      const form = document.getElementById("preview-chart-controls");
      const symbolField = document.getElementById("preview-symbol");
      const intervalField = document.getElementById("preview-interval");
      const chartEl = document.getElementById("preview-chart");
      const lastTimeEl = document.getElementById("preview-last-time");
      const lastPriceEl = document.getElementById("preview-last-price");
      const lastRangeEl = document.getElementById("preview-last-range");
      const statusEl = document.getElementById("preview-status");
      const selectionLabelEl = document.getElementById("preview-selection-label");
      const clearSelectionBtn = document.getElementById("preview-clear-selection");
      const createSessionBtn = document.getElementById("preview-create-session");
      const versionEl = document.getElementById("preview-app-version");
      const openInspectionBtn = document.getElementById("preview-open-inspection");

      if (openInspectionBtn) {
        openInspectionBtn.addEventListener("click", (event) => {
          event.preventDefault();
          window.open("/inspection", "_blank", "noopener,noreferrer");
        });
      }

      const previewState = {
        chart: null,
        series: null,
        symbol: normaliseSymbol(symbolField ? symbolField.value : initial.symbol) || defaultSymbol,
        interval: (intervalField && intervalField.value) || "1m",
        selection: null,
      };

      function setPreviewStatus(message, tone = "info") {
        if (!statusEl) return;
        statusEl.textContent = message || "";
        statusEl.dataset.tone = tone;
        statusEl.hidden = !message;
      }

      function updatePreviewSelectionLabel() {
        const start = previewState.selection && previewState.selection.start;
        const end = previewState.selection && previewState.selection.end;
        const label = selectionLabel(start, end);
        if (selectionLabelEl) selectionLabelEl.textContent = label;
      }

      function updatePreviewInfo(candle) {
        if (!candle) {
          if (lastTimeEl) lastTimeEl.textContent = "-";
          if (lastPriceEl) lastPriceEl.textContent = "-";
          if (lastRangeEl) lastRangeEl.textContent = "-";
          return;
        }
        if (lastTimeEl) lastTimeEl.textContent = formatTs(Number(candle.t));
        if (lastPriceEl) lastPriceEl.textContent = Number(candle.c ?? candle.close ?? 0).toFixed(2);
        const high = Number(candle.h ?? candle.high ?? 0);
        const low = Number(candle.l ?? candle.low ?? 0);
        const base = Number(candle.c ?? candle.close ?? 0) || 1;
        const rangeValue = Math.max(0, high - low);
        const percent = base ? (rangeValue / base) * 100 : 0;
        if (lastRangeEl) lastRangeEl.textContent = `${rangeValue.toFixed(2)} (${percent.toFixed(2)}%)`;
      }

      function ensurePreviewChart() {
        if (previewState.chart || !chartEl || !window.LightweightCharts) return;
        previewState.chart = LightweightCharts.createChart(chartEl, {
          autoSize: true,
          layout: { background: { color: "#0f172a" }, textColor: "#e2e8f0" },
          rightPriceScale: { borderColor: "rgba(148, 163, 184, 0.4)" },
          timeScale: { borderColor: "rgba(148, 163, 184, 0.4)", timeVisible: true, secondsVisible: true },
          crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
          grid: {
            vertLines: { color: "rgba(15, 23, 42, 0.6)" },
            horzLines: { color: "rgba(15, 23, 42, 0.6)" },
          },
        });
        previewState.series = previewState.chart.addCandlestickSeries({
          upColor: "#22c55e",
          downColor: "#ef4444",
          wickUpColor: "#f8fafc",
          wickDownColor: "#f8fafc",
          borderUpColor: "#22c55e",
          borderDownColor: "#ef4444",
          borderVisible: true,
        });

        const resize = () => {
          if (!previewState.chart) return;
          const height = Math.max(
            320,
            chartEl.clientHeight ||
              chartEl.offsetHeight ||
              (chartEl.parentElement && chartEl.parentElement.clientHeight) ||
              320,
          );
          previewState.chart.applyOptions({ height });
        };
        resize();
        if (window.ResizeObserver) {
          const observer = new ResizeObserver(resize);
          observer.observe(chartEl);
        } else {
          window.addEventListener("resize", resize);
        }

        previewState.chart.subscribeClick((param) => {
          if (!param || typeof param.time === "undefined") return;
          const ts = Math.floor(Number(param.time) * 1000);
          if (!previewState.selection || !previewState.selection.start || previewState.selection.end) {
            previewState.selection = { start: ts, end: null };
          } else {
            previewState.selection.end = ts;
            if (previewState.selection.end < previewState.selection.start) {
              const tmp = previewState.selection.start;
              previewState.selection.start = previewState.selection.end;
              previewState.selection.end = tmp;
            }
          }
          updatePreviewSelectionLabel();
        });
      }

      async function loadPreview(symbolRaw, intervalRaw) {
        const resolvedSymbol =
          normaliseSymbol(symbolRaw) || normaliseSymbol(initial.symbol) || defaultSymbol;
        const resolvedInterval = intervalRaw || "1m";
        previewState.symbol = resolvedSymbol;
        previewState.interval = resolvedInterval;
        previewState.selection = null;
        if (symbolField) symbolField.value = resolvedSymbol;
        if (intervalField) intervalField.value = resolvedInterval;
        updatePreviewSelectionLabel();
        setPreviewStatus("Loading Binance history...", "info");
        try {
          ensurePreviewChart();
          const intervalMs = TIMEFRAME_TO_MS[resolvedInterval] || 60000;
          const endMs = Date.now();
          const startMs = Math.max(0, endMs - intervalMs * 500);
          const candles = await fetchCandles(resolvedSymbol, resolvedInterval, startMs, endMs);
          const bars = candles.map((candle) => ({
            time: Math.floor(Number(candle.t) / 1000),
            open: Number(candle.o ?? candle.open ?? 0),
            high: Number(candle.h ?? candle.high ?? 0),
            low: Number(candle.l ?? candle.low ?? 0),
            close: Number(candle.c ?? candle.close ?? 0),
          }));
          if (previewState.series) {
            previewState.series.setData(bars);
          }
          if (previewState.chart && bars.length) {
            previewState.chart.timeScale().fitContent();
          }
          updatePreviewInfo(candles[candles.length - 1]);
          setPreviewStatus("", "info");
        } catch (error) {
          console.error(error);
          setPreviewStatus("Failed to load Binance history", "error");
        }
      }

      if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener("click", () => {
          previewState.selection = null;
          updatePreviewSelectionLabel();
          setPreviewStatus("", "info");
        });
      }

      if (form) {
        form.addEventListener("submit", (event) => {
          event.preventDefault();
          loadPreview(
            symbolField ? symbolField.value : previewState.symbol,
            intervalField ? intervalField.value : previewState.interval,
          );
        });
      }

      if (createSessionBtn) {
        createSessionBtn.addEventListener("click", async () => {
          if (!previewState.selection || !previewState.selection.start || !previewState.selection.end) {
            setPreviewStatus("Select a range on the chart first", "warning");
            return;
          }
          createSessionBtn.disabled = true;
          setPreviewStatus("Building test environment...", "info");
          try {
            const { snapshotId } = await createSnapshotFromSelection({
              symbolValue: symbolField ? symbolField.value : previewState.symbol,
              selection: previewState.selection,
              frames: DEFAULT_TEST_TIMEFRAMES,
              source: "preview-chart",
            });
            state.snapshotId = snapshotId;
            await refreshSnapshots();
            if (snapshotSelect) snapshotSelect.value = snapshotId;
            await loadSnapshot(snapshotId);
            setPreviewStatus("Test environment created", "success");
          } catch (error) {
            console.error(error);
            const detail = error && typeof error.message === "string" ? error.message : "";
            setPreviewStatus(`Failed to create test environment${detail ? `: ${detail}` : ""}`, "error");
          } finally {
            createSessionBtn.disabled = false;
          }
        });
      }

      if (versionEl) {
        fetch("/version")
          .then((resp) => (resp.ok ? resp.json() : null))
          .then((data) => {
            if (data && typeof data.version === "string" && versionEl) {
              versionEl.textContent = data.version;
            }
          })
          .catch(() => {});
      }

      ensurePreviewChart();
      loadPreview(previewState.symbol, previewState.interval);
    }

    initPreviewPanel();

    metricButtons.forEach((button) => {
      button.addEventListener("click", () => {
        metricButtons.forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        const metric = button.dataset.metric;
        let part = null;
        if (metric === "ohlcv") {
          part = state.payload?.DATA?.frames?.[state.frame] || state.payload?.DATA?.frames;
        } else if (metric === "delta") {
          part = state.payload?.DATA?.delta_cvd?.[state.frame] || state.payload?.DATA?.delta_cvd;
        } else if (metric === "vwap") {
          part = state.payload?.DATA?.vwap_tpo?.[state.frame] || state.payload?.DATA?.vwap_tpo;
        } else if (metric === "zones") {
          part = state.payload?.DATA?.zones;
        } else if (metric === "smt") {
          part = state.payload?.DATA?.smt;
        } else if (metric === "agg") {
          part = state.payload?.DATA?.agg_trades;
        }
        setJson(metricPre, part);
      });
    });

    renderJson(state.payload);
    populateFrames(state.payload);
    populateSnapshots(initial.snapshots || []);
    renderMeta(state.payload);
    renderChart();
    updateSelectionLabel();
    await refreshSnapshots();
    if (state.snapshotId && snapshotSelect) {
      snapshotSelect.value = state.snapshotId;
    }
    if (metricButtons.length) {
      metricButtons[0].click();
    }
  });
})();
"""

    page_html = f"""
    <!DOCTYPE html>
    <html lang=\"ru\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Inspection Dashboard</title>
        <style>{style_block}</style>
      </head>
      <body>
        <header>
          <h1>Панель тестирования данных графика</h1>
          <p>Создание тестовых окружений из собранных свечей, выбор диапазона и проверка расчётов.</p>
        </header>
        <main>
          <section class=\"panel\">
            <h2>Управление</h2>
            <div class=\"snapshot-select\">
              <label style=\"flex:1;\">
                <span>Снэпшоты</span>
                <select id=\"snapshot-select\"></select>
              </label>
              <button id=\"refresh-snapshot\" class=\"secondary\" type=\"button\">Refresh</button>
            </div>
            <div class=\"status-banner\" id=\"inspection-status\" hidden data-tone=\"info\"></div>
            <div class=\"controls-grid\">
              <label>
                <span>Символ</span>
                <input id=\"symbol-input\" type=\"text\" value=\"{symbol_value}\" autocomplete=\"off\" list=\"symbol-suggestions\" />
              </label>
              <datalist id=\"symbol-suggestions\">
                <option value=\"BTCUSDT\"></option>
                <option value=\"ETHUSDT\"></option>
                <option value=\"BNBUSDT\"></option>
                <option value=\"SOLUSDT\"></option>
                <option value=\"XRPUSDT\"></option>
              </datalist>
              <label>
                <span>Таймфрейм для отображения</span>
                <select id=\"frame-select\">{''.join(timeframe_options)}</select>
              </label>
            </div>
            <div>
              <span class=\"badge\">Выделенный диапазон</span>
              <div style=\"margin-top:0.4rem;display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;\">
                <span id=\"selection-info\">—</span>
                <button id=\"clear-selection\" class=\"secondary\" type=\"button\">Сбросить выделение</button>
              </div>
            </div>
            <div>
              <span class=\"badge\">Таймфреймы для теста</span>
              <div class=\"timeframes\">
                <label><input type=\"checkbox\" data-tf-checkbox value=\"1m\" checked />1m</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"3m\" />3m</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"5m\" />5m</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"15m\" />15m</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"1h\" />1h</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"4h\" />4h</label>
                <label><input type=\"checkbox\" data-tf-checkbox value=\"1d\" />1d</label>
              </div>
            </div>
            <button id=\"build-session\" class=\"primary\" type=\"button\">Создать тестовую среду</button>
            <div id=\"snapshot-meta\"></div>
          </section>

          <section class='panel main-preview-panel' data-preview-root>
            <h2>Main Chart Preview</h2>
            <div class='index-preview'>
              <header class='page-header'>
                <div class='header-top'>
                  <h3>Interactive Candlestick Chart</h3>
                  <div class='header-controls'>
                    <div class='app-meta' aria-live='polite'>
                      <span class='app-meta__label'>Version:</span>
                      <span id='preview-app-version' class='app-meta__value'>-</span>
                    </div>
                    <button id='preview-open-inspection' class='btn-secondary' type='button'>Open /inspection</button>
                  </div>
                </div>
                <p>This block mirrors the landing page chart so you can select a range and spawn a test environment directly from the inspection panel.</p>
              </header>
              <main class='page-main'>
                <section class='controls-card'>
                  <form id='preview-chart-controls' class='controls-form'>
                    <label class='form-field'>
                      <span>Symbol</span>
                      <input id='preview-symbol' type='text' value='BTCUSDT' required autocomplete='off' />
                    </label>

                    <label class='form-field'>
                      <span>Interval</span>
                      <select id='preview-interval'>
                        <option value='1s'>1s</option>
                        <option value='1m' selected>1m</option>
                        <option value='3m'>3m</option>
                        <option value='5m'>5m</option>
                        <option value='15m'>15m</option>
                        <option value='30m'>30m</option>
                        <option value='1h'>1h</option>
                        <option value='4h'>4h</option>
                        <option value='1d'>1d</option>
                      </select>
                    </label>

                    <button type='submit' class='btn-primary'>Load Chart</button>
                  </form>
                </section>

                <section class='chart-card'>
                  <div class='chart-wrapper'>
                    <div id='preview-chart' class='chart-area' aria-label='Preview candlestick chart'></div>
                  </div>
                  <aside class='chart-info'>
                    <div>
                      <span class='info-label'>Last candle:</span>
                      <span id='preview-last-time' class='info-value'>-</span>
                    </div>
                    <div>
                      <span class='info-label'>Close price:</span>
                      <span id='preview-last-price' class='info-value'>-</span>
                    </div>
                    <div>
                      <span class='info-label'>Candle range:</span>
                      <span id='preview-last-range' class='info-value'>-</span>
                    </div>
                  </aside>
                  <div class='preview-selection'>
                    <span class='badge'>Selection</span>
                    <div class='preview-selection__controls'>
                      <span id='preview-selection-label'>-</span>
                      <button id='preview-clear-selection' class='btn-secondary' type='button'>Reset</button>
                    </div>
                  </div>
                  <div class='preview-actions'>
                    <button id='preview-create-session' class='btn-primary' type='button'>Create Test Environment</button>
                    <div id='preview-status' class='status-banner' hidden></div>
                  </div>
                </section>
              </main>
            </div>
          </section>


          <section class=\"panel\">
            <h2>Просмотр данных</h2>
            <div id=\"inspection-chart\" class=\"chart-shell\" data-selection-label=\"—\"></div>
            <div class=\"metrics-bar\">
              <button class=\"secondary\" type=\"button\" data-metric=\"ohlcv\">OHLCV</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"delta\">Delta / CVD</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"vwap\">VWAP</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"zones\">Zones</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"smt\">SMT</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"agg\">Agg Trades</button>
              <button id=\"btn-check-all-datas\" class=\"primary\" type=\"button\">Check all datas</button>
            </div>
            <div class=\"json-panels\">
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>DATA</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"data-json\">Copy JSON</button>
                </header>
                <pre id=\"data-json\">{data_json_initial}</pre>
              </div>
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>DIAGNOSTICS</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"diagnostics-json\">Copy JSON</button>
                </header>
                <pre id=\"diagnostics-json\">{diagnostics_json_initial}</pre>
              </div>
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>METRIC</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"metric-json\">Copy JSON</button>
                </header>
                <pre id=\"metric-json\">{metric_json_initial}</pre>
              </div>
            </div>
            <div class=\"data-check-block\">
              <div class=\"data-check-header\">
                <span class=\"badge\">Aggregated market slice</span>
              </div>
              <div id=\"data-check-status\" class=\"status-banner\" hidden data-tone=\"info\"></div>
              <div class=\"data-check-grid\">
                <div class=\"collapse\">
                  <header data-collapse-toggle>
                    <h3>Check all datas JSON</h3>
                    <button class=\"secondary\" type=\"button\" data-copy-target=\"data-check-json\">Copy JSON</button>
                  </header>
                  <pre id=\"data-check-json\">null</pre>
                </div>
                <div id=\"data-check-panel\" class=\"data-check-panel\"></div>
              </div>
            </div>
          </section>
        </main>
        <script>{script_block}</script>
        <script src=\"https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js\"></script>
        <script src="/public/binanceCandles.js"></script>
        <script>{ui_script}</script>
      </body>
    </html>
    """
    return page_html
