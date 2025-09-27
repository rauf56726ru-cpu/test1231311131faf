"""Inspection payload assembly and UI rendering."""
from __future__ import annotations

import html as html_utils
import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .ohlc import TIMEFRAME_WINDOWS, TIMEFRAME_TO_MS, normalise_ohlcv
from .profile import build_profile_package
from .presets import resolve_profile_config
from ..meta import Meta

Snapshot = Dict[str, Any]

DEFAULT_SYMBOL = "BTCUSDT"

_MAX_STORED_SNAPSHOTS = 16
_SNAPSHOT_STORE: "OrderedDict[str, Snapshot]" = OrderedDict()

_DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "var" / "snapshots"
SNAPSHOT_STORAGE_DIR = Path(
    os.environ.get("INSPECTION_SNAPSHOT_DIR", str(_DEFAULT_STORAGE_ROOT))
).expanduser()
_SNAPSHOT_ID_SANITISER = re.compile(r"[^A-Za-z0-9._-]")


def _ensure_storage_dir() -> None:
    try:
        SNAPSHOT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Failing to create the directory should not break the request flow; in-memory
        # storage will continue to function even without persistence on disk.
        pass


def _snapshot_path(snapshot_id: str) -> Path:
    safe_id = _SNAPSHOT_ID_SANITISER.sub("_", snapshot_id or "snapshot")
    return SNAPSHOT_STORAGE_DIR / f"{safe_id}.json"


def _persist_snapshot(snapshot: Snapshot) -> None:
    if not snapshot:
        return

    _ensure_storage_dir()

    snapshot_id = str(snapshot.get("id") or "snapshot")
    path = _snapshot_path(snapshot_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _remove_snapshot_file(snapshot_id: str) -> None:
    path = _snapshot_path(snapshot_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        removed_id, _ = _SNAPSHOT_STORE.popitem(last=False)
        _remove_snapshot_file(removed_id)


def _load_existing_snapshots() -> None:
    if not SNAPSHOT_STORAGE_DIR.exists():
        return

    try:
        files = sorted(
            SNAPSHOT_STORAGE_DIR.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
        )
    except OSError:
        return

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        snapshot_id = str(data.get("id") or path.stem)
        snapshot = dict(data)
        snapshot["id"] = snapshot_id
        _SNAPSHOT_STORE[snapshot_id] = snapshot
        _SNAPSHOT_STORE.move_to_end(snapshot_id)

    _ensure_snapshot_limit()


_load_existing_snapshots()


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
    _persist_snapshot(stored)
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

        result = normalise_ohlcv(
            symbol,
            tf_key,
            candles,
            include_diagnostics=True,
            use_full_span=(tf_key == "1m"),
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

    profile_config = resolve_profile_config(symbol, snapshot.get("meta"))
    preset = profile_config["preset"]
    raw_profile_defaults = profile_config.get("raw_defaults")
    preset_payload = profile_config.get("preset_payload")
    preset_required = profile_config.get("preset_required", False)
    target_tf_key = profile_config.get("target_tf_key", "1m")

    base_candles = normalised_frames.get(target_tf_key, {}).get("candles", [])
    if not base_candles:
        base_candles = normalised_frames.get("1m", {}).get("candles", [])
    if not base_candles and normalised_frames:
        first_key = next(iter(normalised_frames))
        base_candles = normalised_frames[first_key].get("candles", [])

    session_vwap = compute_session_vwaps(symbol, base_candles)

    sessions = list(Meta.iter_vwap_sessions())
    tpo_entries: List[Dict[str, object]] = []
    flattened_profile: List[Dict[str, float]] = []
    zone_items: List[Dict[str, Any]] = []

    tick_size_value = profile_config.get("tick_size")
    adaptive_bins_flag = bool(profile_config.get("adaptive_bins", True))
    value_area_pct = float(profile_config.get("value_area_pct", 0.7))
    profile_last_n = int(profile_config.get("last_n", 3))
    atr_multiplier = float(profile_config.get("atr_multiplier", 0.5))
    target_bins = int(profile_config.get("target_bins", 80))
    clip_threshold = float(profile_config.get("clip_threshold", 0.0))
    smooth_window = int(profile_config.get("smooth_window", 1))

    if preset and base_candles and sessions:
        cache_token = (
            "inspection",
            snapshot.get("id"),
            symbol,
            target_tf_key,
        )
        (tpo_entries, flattened_profile, zone_items) = build_profile_package(
            base_candles,
            sessions=sessions,
            last_n=profile_last_n,
            tick_size=tick_size_value,
            adaptive_bins=adaptive_bins_flag,
            value_area_pct=value_area_pct,
            atr_multiplier=atr_multiplier,
            target_bins=target_bins,
            clip_threshold=clip_threshold,
            smooth_window=smooth_window,
            cache_token=cache_token,
            tf_key=target_tf_key,
        )

    data_section = {
        "symbol": symbol,
        "frames": normalised_frames,
        "selection": selection,
        "session_vwap": session_vwap,
        "tpo": tpo_entries,
        "profile": flattened_profile,
        "zones": zone_items,
        "zones_raw": snapshot.get("zones"),
        "agg_trades": snapshot.get("agg_trades")
        or {
            "status": "unavailable",
            "detail": "Agg trade data is not present in the snapshot.",
        },
        "delta_cvd": delta_frames,
        "vwap_tpo": vwap_frames,
        "smt": snapshot.get("smt")
        or {
            "status": "unavailable",
            "detail": "SMT provider is not configured in the snapshot.",
        },
        "profile_preset": preset_payload,
        "profile_preset_required": bool(preset_required),
        "profile_defaults": raw_profile_defaults,
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
    check_all_json_initial = _format_json_block(None)

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
    .collapse header .actions {
      display: inline-flex;
      gap: 0.5rem;
      align-items: center;
    }
    .collapse header h3 {
      margin: 0;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .checkall-control {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem;
      padding: 0.75rem 1rem 0.5rem;
      background: rgba(15, 23, 42, 0.85);
      border-top: 1px solid rgba(148, 163, 184, 0.16);
    }
    .checkall-control span {
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.78);
    }
    .checkall-control select {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 12px;
      padding: 0.4rem 0.7rem;
      color: var(--fg);
      min-width: 110px;
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
    .preset-controls {
      margin-top: 0.8rem;
      margin-bottom: 0.8rem;
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .preset-chip {
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      background: rgba(14, 165, 233, 0.25);
      color: #f8fafc;
    }
    .preset-chip[data-variant="warning"] {
      background: rgba(249, 115, 22, 0.5);
      color: #0f172a;
    }
    .preset-chip[data-variant="success"] {
      background: rgba(34, 197, 94, 0.55);
      color: #0f172a;
    }
    .preset-chip[data-variant="info"] {
      background: rgba(14, 165, 233, 0.55);
      color: #0f172a;
    }
    .modal {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      z-index: 40;
    }
    .modal[hidden] {
      display: none;
    }
    .modal__backdrop {
      position: absolute;
      inset: 0;
      background: rgba(15, 23, 42, 0.7);
    }
    .modal__dialog {
      position: relative;
      margin: 6vh auto;
      max-width: 640px;
      width: min(92vw, 640px);
      background: rgba(15, 23, 42, 0.96);
      border-radius: 16px;
      padding: 1.5rem;
      border: 1px solid rgba(148, 163, 184, 0.35);
      color: #e2e8f0;
      box-shadow: 0 32px 64px rgba(15, 23, 42, 0.45);
    }
    .modal__header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    .modal__header h3 {
      margin: 0;
      font-size: 1.2rem;
    }
    .modal__close {
      background: none;
      border: none;
      color: #94a3b8;
      font-size: 1.5rem;
      cursor: pointer;
      padding: 0;
      line-height: 1;
    }
    .modal__body {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: 70vh;
      overflow-y: auto;
    }
    .preset-form-grid {
      display: grid;
      gap: 0.8rem;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .preset-form-grid label {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-size: 0.85rem;
    }
    .preset-form-grid input,
    .preset-form-grid select {
      padding: 0.45rem 0.6rem;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: rgba(15, 23, 42, 0.8);
      color: #e2e8f0;
    }
    .preset-form-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }
    .preset-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .preset-list__item {
      padding: 0.75rem 0.9rem;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.25);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.5rem;
    }
    .preset-list__title {
      font-weight: 600;
      font-size: 1rem;
    }
    .preset-list__meta {
      font-size: 0.82rem;
      color: #94a3b8;
    }
    .preset-list__actions {
      display: flex;
      gap: 0.5rem;
    }
    .preset-list__empty {
      margin: 0;
      color: #94a3b8;
    }
    .btn-danger {
      background: rgba(239, 68, 68, 0.85);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      padding: 0.45rem 0.75rem;
      cursor: pointer;
      font-weight: 600;
    }
    .btn-danger:disabled {
      opacity: 0.4;
      cursor: not-allowed;
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
    "1s": 1000,
    "3s": 3000,
    "5s": 5000,
    "15s": 15000,
    "30s": 30000,
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "8h": 28800000,
    "12h": 43200000,
    "1d": 86400000,
  };
  const DEFAULT_TEST_TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"];
  const PREVIEW_REFRESH_INTERVAL_MS = 10_000;
  const PREVIEW_SHARED_MAX_BARS = 2000;
  const MINUTE_INTERVAL_KEY = "1m";
  const MINUTE_INTERVAL_MS = TIMEFRAME_TO_MS[MINUTE_INTERVAL_KEY] || 60_000;
  const PREVIEW_MINUTE_LOOKBACK_MS = MINUTE_INTERVAL_MS * 120;
  const PREVIEW_MINUTE_MAX_BARS = 5000;

  const LightweightCharts = window.LightweightCharts || null;
  const BinanceCandles = window.BinanceCandles || null;
  const ChartGapWatcher = window.ChartGapWatcher || null;
  const SharedCandles = window.SharedCandles || null;


  function toChartBars(candles) {
    return (candles || [])
      .map((candle) => {
        const rawTs = Number(candle?.ts_ms_utc ?? candle?.t ?? candle?.time ?? 0);
        const open = Number(candle?.o ?? candle?.open ?? 0);
        const high = Number(candle?.h ?? candle?.high ?? open);
        const low = Number(candle?.l ?? candle?.low ?? open);
        const close = Number(candle?.c ?? candle?.close ?? open);
        if (
          !Number.isFinite(rawTs) ||
          !Number.isFinite(open) ||
          !Number.isFinite(high) ||
          !Number.isFinite(low) ||
          !Number.isFinite(close)
        ) {
          return null;
        }
        const time = Math.floor(rawTs / 1000);
        return {
          time,
          open,
          high,
          low,
          close,
          ts_ms_utc: Math.floor(rawTs),
        };
      })
      .filter((bar) => bar !== null);
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

  function computeCandleDisplayTime(candle, intervalMs, lastUpdateMs) {
    if (!candle) return Number.NaN;
    const openMs = Number(
      candle.ts_ms_utc ?? candle.t ?? candle.time ?? (candle.openTime ?? candle.open_time ?? 0),
    );
    if (!Number.isFinite(openMs)) return Number.NaN;
    const safeInterval = Math.max(1, Number(intervalMs) || 0);
    const closingMs = openMs + safeInterval;
    const specificUpdate = Number(
      candle.last_update_ms ?? candle.lastUpdateMs ?? candle.last_update ?? candle.updated_at ?? 0,
    );
    const referenceSource = Number.isFinite(specificUpdate) ? specificUpdate : lastUpdateMs;
    const reference = Number.isFinite(referenceSource) ? Number(referenceSource) : Date.now();
    const alignedReference = Math.max(openMs, reference);
    return Math.min(closingMs, alignedReference);
  }

  function intervalToMs(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return Math.max(1, numeric) * 60000;
    }
    return TIMEFRAME_TO_MS[value] || 60000;
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const open = Number(bar.open ?? bar.o ?? 0);
    const high = Number(bar.high ?? bar.h ?? open);
    const low = Number(bar.low ?? bar.l ?? open);
    const close = Number(bar.close ?? bar.c ?? open);
    const time = Number(bar.time ?? bar.t ?? 0);
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }
    const ts = Number(bar.ts_ms_utc ?? time * 1000);
    return {
      time: Math.floor(time),
      open,
      high,
      low,
      close,
      ts_ms_utc: Number.isFinite(ts) ? Math.floor(ts) : Math.floor(time * 1000),
    };
  }

  async function fetchHistory(symbol, interval, limit = 1000) {
    if (!BinanceCandles || typeof BinanceCandles.fetchHistory !== "function") {
      throw new Error("Binance helper is unavailable");
    }
    const rows = await BinanceCandles.fetchHistory(symbol, interval, limit);
    return rows.map((bar) => normaliseBar(bar)).filter((bar) => bar !== null);
  }

  async function fetchRange(symbol, interval, startMs, endMs, limit = 1000) {
    const url = new URL("https://api.binance.com/api/v3/klines");
    url.searchParams.set("symbol", symbol);
    url.searchParams.set("interval", interval);
    if (Number.isFinite(startMs)) {
      url.searchParams.set("startTime", Math.floor(startMs));
    }
    if (Number.isFinite(endMs)) {
      url.searchParams.set("endTime", Math.floor(endMs));
    }
    url.searchParams.set("limit", String(Math.max(1, Math.min(limit, 1000))));
    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Failed to fetch range: ${response.status}`);
    }
    const data = await response.json();
    if (!BinanceCandles || typeof BinanceCandles.transformKlines !== "function") {
      throw new Error("Binance helper is unavailable");
    }
    return BinanceCandles.transformKlines(data)
      .map((bar) => normaliseBar(bar))
      .filter((bar) => bar !== null);
  }

  function ensurePreviewBar(input) {
    if (!input) return null;
    const candidateTime = Number(input.time);
    const candidateOpen = Number(input.open);
    const candidateHigh = Number(input.high ?? candidateOpen);
    const candidateLow = Number(input.low ?? candidateOpen);
    const candidateClose = Number(input.close ?? candidateOpen);
    let candidateTs = Number(input.ts_ms_utc ?? input.t ?? 0);
    if (
      Number.isFinite(candidateTime) &&
      Number.isFinite(candidateOpen) &&
      Number.isFinite(candidateHigh) &&
      Number.isFinite(candidateLow) &&
      Number.isFinite(candidateClose)
    ) {
      if (!Number.isFinite(candidateTs)) {
        candidateTs = candidateTime * 1000;
      }
      return {
        time: Math.floor(candidateTime),
        open: candidateOpen,
        high: candidateHigh,
        low: candidateLow,
        close: candidateClose,
        ts_ms_utc: Math.floor(candidateTs),
      };
    }
    const normalized = normaliseBar(input);
    if (!normalized) return null;
    return {
      time: Math.floor(normalized.time),
      open: normalized.open,
      high: normalized.high,
      low: normalized.low,
      close: normalized.close,
      ts_ms_utc: normalized.ts_ms_utc,
      ...(Number.isFinite(normalized.last_update_ms)
        ? { last_update_ms: Number(normalized.last_update_ms) }
        : {}),
    };
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

  async function fetchCandles(symbol, interval, startMs, endMs, options = {}) {
    const collected = [];
    const seen = new Set();
    const intervalMs = TIMEFRAME_TO_MS[interval] || 60000;
    const hintedLimit = Number.isFinite(options.limit) ? Math.floor(options.limit) : null;
    const batchLimit = Math.max(1, Math.min(1000, hintedLimit || 1000));
    const hasStart = Number.isFinite(startMs);
    const hasEnd = Number.isFinite(endMs);
    let startBound = null;
    if (hasStart && hasEnd) {
      startBound = Math.floor(Math.min(startMs, endMs));
    } else if (hasStart) {
      startBound = Math.floor(startMs);
    }
    let endBound = null;
    if (hasStart && hasEnd) {
      endBound = Math.floor(Math.max(startMs, endMs));
    } else if (hasEnd) {
      endBound = Math.floor(endMs);
    }
    let cursor = startBound;
    let guard = 0;
    const guardLimit = 4096;

    while (true) {
      const url = new URL("https://api.binance.com/api/v3/klines");
      url.searchParams.set("symbol", symbol.toUpperCase());
      url.searchParams.set("interval", interval);
      if (Number.isFinite(cursor)) {
        url.searchParams.set("startTime", String(cursor));
      } else if (Number.isFinite(startBound)) {
        url.searchParams.set("startTime", String(startBound));
      }
      if (Number.isFinite(endBound)) {
        url.searchParams.set("endTime", String(endBound));
      }
      url.searchParams.set("limit", String(batchLimit));

      const resp = await fetch(url.toString());
      if (!resp.ok) {
        throw new Error(`klines ${resp.status}`);
      }
      const rows = await resp.json();
      if (!Array.isArray(rows) || !rows.length) {
        break;
      }

      let lastOpen = null;
      for (const row of rows) {
        if (!row) continue;
        const openMs = Number(row[0]);
        const open = Number(row[1]);
        const high = Number(row[2]);
        const low = Number(row[3]);
        const close = Number(row[4]);
        const volume = Number(row[5]);
        if (!Number.isFinite(openMs)) continue;
        lastOpen = openMs;
        if (Number.isFinite(endBound) && openMs > endBound) {
          continue;
        }
        if (seen.has(openMs)) continue;
        seen.add(openMs);
        collected.push({ t: openMs, o: open, h: high, l: low, c: close, v: volume });
      }

      if (!hasStart || !Number.isFinite(cursor)) {
        break;
      }
      if (!Number.isFinite(lastOpen)) {
        break;
      }
      const nextCursor = lastOpen + intervalMs;
      if (!Number.isFinite(nextCursor)) {
        break;
      }
      if (Number.isFinite(endBound) && nextCursor > endBound) {
        break;
      }
      if (nextCursor <= cursor) {
        break;
      }
      cursor = nextCursor;
      guard += 1;
      if (guard >= guardLimit) {
        break;
      }
      if (rows.length < batchLimit) {
        break;
      }
    }

    collected.sort((a, b) => a.t - b.t);
    if (Number.isFinite(startBound) || Number.isFinite(endBound)) {
      return collected.filter((candle) => {
        const ts = Number(candle?.t ?? 0);
        if (!Number.isFinite(ts)) return false;
        if (Number.isFinite(startBound) && ts < startBound) return false;
        if (Number.isFinite(endBound) && ts > endBound) return false;
        return true;
      });
    }
    return collected;
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
    const checkAllPre = document.getElementById("checkall-json");
    const checkAllButton = document.getElementById("fetch-check-all");
    const checkAllHours = document.getElementById("checkall-hours");
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
    const presetChip = document.getElementById("preset-chip");
    const managePresetsButton = document.getElementById("manage-presets");
    const presetModal = document.getElementById("preset-modal");
    const presetBackdrop = document.getElementById("preset-modal-backdrop");
    const presetForm = document.getElementById("preset-form");
    const presetListView = document.getElementById("preset-list-view");
    const presetListContainer = document.getElementById("preset-list");
    const presetFormSection = document.getElementById("preset-form-section");
    const presetModalTitle = document.getElementById("preset-modal-title");
    const presetModeField = document.getElementById("preset-mode-input");
    const presetSymbolField = document.getElementById("preset-symbol");
    const presetTfField = document.getElementById("preset-tf");
    const presetLastNField = document.getElementById("preset-last-n");
    const presetValueAreaField = document.getElementById("preset-value-area");
    const presetBinningModeField = document.getElementById("preset-binning-mode");
    const presetTickSizeField = document.getElementById("preset-tick-size");
    const presetAtrField = document.getElementById("preset-atr-multiplier");
    const presetBinsField = document.getElementById("preset-target-bins");
    const presetClipField = document.getElementById("preset-clip-tail");
    const presetSmoothField = document.getElementById("preset-smooth-window");
    const presetTickRow = document.querySelector("[data-preset-tick]");
    const presetAdaptiveRow = document.querySelector("[data-preset-adaptive]");
    const presetSubmitButton = document.getElementById("preset-submit");
    const presetCancelButtons = Array.from(document.querySelectorAll("[data-close-preset]"));
    const presetCreateButton = document.getElementById("preset-create-button");

    initCollapsibles();

    const resolveHours = (value) => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) return 1;
      return Math.min(4, Math.max(1, Math.floor(parsed)));
    };

    const state = {
      payload: initial.payload || null,
      snapshotId: initial.snapshotId || null,
      selection: initial.payload?.DATA?.selection || null,
      frame: initial.timeframe || initial.payload?.DATA?.meta?.requested?.frames?.[0] || "1m",
      chart: null,
      series: null,
      checkAll: null,
      hours: checkAllHours ? resolveHours(checkAllHours.value) : 1,
      profilePreset: initial.payload?.DATA?.profile_preset || null,
      presetRequired: Boolean(initial.payload?.DATA?.profile_preset_required),
      presetDefaults: initial.payload?.DATA?.profile_defaults || null,
      presetList: [],
      presetModalMode: null,
      managingSymbol: null,
      presetModalOpen: false,
    };

    const AUTO_PRESET_SYMBOLS = new Set(["BTCUSDT", "ETHUSDT", "SOLUSDT"]);

    function autoPresetSymbol(symbol) {
      const normalised = normaliseSymbol(symbol);
      return normalised ? AUTO_PRESET_SYMBOLS.has(normalised) : false;
    }

    function activeSymbol() {
      return (
        normaliseSymbol(state.payload?.DATA?.symbol) ||
        normaliseSymbol(initial.payload?.DATA?.symbol) ||
        normaliseSymbol(initial.symbol) ||
        defaultSymbol
      );
    }

    function renderPresetChip() {
      if (!presetChip) return;
      const symbol = activeSymbol();
      const preset = state.profilePreset;
      if (state.presetRequired && !preset) {
        presetChip.textContent = `Требуется пресет для ${symbol}`;
        presetChip.dataset.variant = "warning";
        presetChip.hidden = false;
        return;
      }
      if (preset) {
        const name = preset.symbol || symbol;
        const suffix = preset.builtin ? " (auto)" : "";
        presetChip.textContent = `Preset: ${name}${suffix}`;
        presetChip.dataset.variant = preset.builtin ? "info" : "success";
        presetChip.hidden = false;
        return;
      }
      if (autoPresetSymbol(symbol)) {
        presetChip.textContent = `Preset: ${symbol} (auto)`;
        presetChip.dataset.variant = "info";
        presetChip.hidden = false;
        return;
      }
      presetChip.hidden = true;
    }

    function handleBinningModeChange() {
      if (!presetBinningModeField) return;
      const mode = presetBinningModeField.value === "tick" ? "tick" : "adaptive";
      if (presetTickRow) presetTickRow.hidden = mode !== "tick";
      if (presetAdaptiveRow) presetAdaptiveRow.hidden = mode !== "adaptive";
    }

    function populatePresetForm(preset, symbol) {
      if (!presetForm) return;
      const defaults = preset || state.presetDefaults || {};
      const binningDefaults = preset?.binning || defaults.binning || {};
      const extrasDefaults = preset?.extras || defaults.extras || {};

      const mode = (binningDefaults.mode || (defaults.adaptive_bins === false ? "tick" : "adaptive")).toLowerCase() === "tick"
        ? "tick"
        : "adaptive";

      if (presetModeField) presetModeField.value = preset ? "edit" : "create";
      if (presetSymbolField) {
        presetSymbolField.value = symbol || preset?.symbol || defaults.symbol || "";
        presetSymbolField.readOnly = Boolean(preset);
      }
      if (presetTfField) presetTfField.value = preset?.tf || defaults.tf || "1m";
      if (presetLastNField) presetLastNField.value = preset?.last_n ?? defaults.last_n ?? 3;
      if (presetValueAreaField) presetValueAreaField.value = preset?.value_area_pct ?? defaults.value_area_pct ?? defaults.value_area ?? 0.7;
      if (presetBinningModeField) presetBinningModeField.value = mode;
      if (presetTickSizeField)
        presetTickSizeField.value =
          mode === "tick"
            ? binningDefaults.tick_size ?? defaults.tick_size ?? ""
            : "";
      if (presetAtrField) presetAtrField.value = binningDefaults.atr_multiplier ?? defaults.atr_multiplier ?? 0.5;
      if (presetBinsField) presetBinsField.value = binningDefaults.target_bins ?? defaults.target_bins ?? 80;
      if (presetClipField) presetClipField.value = extrasDefaults.clip_low_volume_tail ?? defaults.clip_low_volume_tail ?? 0.005;
      if (presetSmoothField) presetSmoothField.value = extrasDefaults.smooth_window ?? defaults.smooth_window ?? 1;

      handleBinningModeChange();
    }

    function openPresetModal(options = {}) {
      if (!presetModal) return;
      const symbol = options.symbol || activeSymbol();
      presetModal.hidden = false;
      if (presetBackdrop) presetBackdrop.hidden = false;
      state.presetModalOpen = true;

      if (options.mode === "list") {
        state.presetModalMode = "list";
        if (presetModalTitle) presetModalTitle.textContent = "Управление пресетами";
        if (presetListView) presetListView.hidden = false;
        if (presetFormSection) presetFormSection.hidden = true;
        refreshPresetList();
      } else {
        state.presetModalMode = "form";
        if (presetModalTitle) presetModalTitle.textContent = options.title || "Настроить пресет TPO";
        if (presetListView) presetListView.hidden = true;
        if (presetFormSection) presetFormSection.hidden = false;
        populatePresetForm(options.preset || null, symbol);
      }
    }

    function closePresetModal() {
      if (!presetModal) return;
      presetModal.hidden = true;
      if (presetBackdrop) presetBackdrop.hidden = true;
      state.presetModalOpen = false;
      state.presetModalMode = null;
    }

    function readPresetForm() {
      if (!presetForm) return null;
      const symbol = (presetSymbolField?.value || "").trim().toUpperCase();
      if (!symbol) {
        throw new Error("Укажите символ");
      }
      const mode = presetBinningModeField?.value === "tick" ? "tick" : "adaptive";
      const payload = {
        symbol,
        tf: (presetTfField?.value || "1m").toLowerCase(),
        last_n: Number(presetLastNField?.value || 3),
        value_area_pct: Number(presetValueAreaField?.value || 0.7),
        binning: {
          mode,
          tick_size: mode === "tick" ? Number(presetTickSizeField?.value || 0) : null,
          atr_multiplier: Number(presetAtrField?.value || 0.5),
          target_bins: Number(presetBinsField?.value || 80),
        },
        extras: {
          clip_low_volume_tail: Number(presetClipField?.value || 0.005),
          smooth_window: Number(presetSmoothField?.value || 1),
        },
      };

      payload.last_n = Math.min(5, Math.max(1, Math.round(payload.last_n)));
      payload.value_area_pct = Math.min(0.95, Math.max(0.01, payload.value_area_pct));
      if (mode === "tick") {
        if (!payload.binning.tick_size || payload.binning.tick_size <= 0) {
          throw new Error("Tick size должен быть положительным числом");
        }
      } else {
        payload.binning.tick_size = null;
        payload.binning.atr_multiplier = Math.min(2, Math.max(0.1, payload.binning.atr_multiplier));
        payload.binning.target_bins = Math.min(200, Math.max(40, Math.round(payload.binning.target_bins)));
      }
      payload.extras.clip_low_volume_tail = Math.min(0.05, Math.max(0, payload.extras.clip_low_volume_tail));
      payload.extras.smooth_window = Math.min(5, Math.max(1, Math.round(payload.extras.smooth_window)));
      return payload;
    }

    async function refreshPresetList() {
      if (!presetListContainer) return;
      try {
        const response = await fetch("/presets", { headers: { Accept: "application/json" } });
        const data = await response.json();
        const list = Array.isArray(data?.presets) ? data.presets : [];
        state.presetList = list;
      } catch (error) {
        console.error("Failed to load presets", error);
        state.presetList = [];
      }
      renderPresetList();
    }

    function renderPresetList() {
      if (!presetListContainer) return;
      presetListContainer.innerHTML = "";
      const entries = Array.isArray(state.presetList) ? state.presetList : [];
      if (!entries.length) {
        const empty = document.createElement("p");
        empty.className = "preset-list__empty";
        empty.textContent = "Сохранённых пресетов пока нет";
        presetListContainer.appendChild(empty);
        return;
      }
      for (const preset of entries) {
        const row = document.createElement("div");
        row.className = "preset-list__item";
        const title = document.createElement("div");
        title.className = "preset-list__title";
        const symbol = (preset?.symbol || "").toUpperCase();
        title.textContent = symbol;
        row.appendChild(title);

        const meta = document.createElement("div");
        meta.className = "preset-list__meta";
        const modeLabel = preset?.binning?.mode === "tick" ? "Tick" : "Adaptive";
        meta.textContent = `TF: ${preset?.tf || "1m"} · ${modeLabel}`;
        row.appendChild(meta);

        const actions = document.createElement("div");
        actions.className = "preset-list__actions";

        const editButton = document.createElement("button");
        editButton.type = "button";
        editButton.className = "btn-secondary";
        editButton.textContent = "Редактировать";
        editButton.disabled = Boolean(preset?.builtin);
        editButton.addEventListener("click", () => {
          populatePresetForm(preset, symbol);
          if (presetModalTitle) presetModalTitle.textContent = `Редактировать пресет ${symbol}`;
          if (presetListView) presetListView.hidden = true;
          if (presetFormSection) presetFormSection.hidden = false;
          state.presetModalMode = "form";
        });
        actions.appendChild(editButton);

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "btn-danger";
        deleteButton.textContent = "Удалить";
        deleteButton.disabled = Boolean(preset?.builtin);
        deleteButton.addEventListener("click", async () => {
          if (!window.confirm(`Удалить пресет ${symbol}?`)) return;
          try {
            const response = await fetch(`/presets/${symbol}`, { method: "DELETE" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            if (normaliseSymbol(symbol) === normaliseSymbol(activeSymbol())) {
              state.profilePreset = null;
              state.presetRequired = autoPresetSymbol(symbol) ? false : true;
            }
            await refreshPresetList();
            renderPresetState();
            updateCheckAllState();
          } catch (error) {
            console.error("Failed to delete preset", error);
            updateStatus("Не удалось удалить пресет", "error");
          }
        });
        actions.appendChild(deleteButton);

        row.appendChild(actions);
        presetListContainer.appendChild(row);
      }
    }

    async function submitPresetForm(event) {
      event.preventDefault();
      if (!presetForm) return;
      try {
        const payload = readPresetForm();
        if (!payload) return;
        const mode = presetModeField?.value === "edit" ? "edit" : "create";
        const url = mode === "edit" ? `/presets/${payload.symbol}` : "/presets";
        const method = mode === "edit" ? "PUT" : "POST";
        const response = await fetch(url, {
          method,
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const stored = data?.preset || payload;
        if (normaliseSymbol(payload.symbol) === activeSymbol()) {
          state.profilePreset = stored;
          state.presetRequired = false;
        }
        closePresetModal();
        renderPresetState();
        updateCheckAllState();
        if (checkAllButton && !checkAllButton.disabled) {
          checkAllButton.click();
        }
      } catch (error) {
        console.error("Failed to save preset", error);
        updateStatus("Не удалось сохранить пресет", "error");
      }
    }

    function renderPresetState() {
      renderPresetChip();
      if (state.presetRequired && !state.profilePreset && !state.presetModalOpen) {
        openPresetModal({ mode: "form", title: "Настроить пресет TPO", symbol: activeSymbol() });
      } else if (!state.presetRequired && state.presetModalOpen && state.presetModalMode === "form") {
        closePresetModal();
      }
    }

    renderPresetState();
    updateCheckAllState();

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

    function updateCheckAllState() {
      if (!checkAllButton) return;
      const hasSelection = Boolean(state.selection && state.selection.start && state.selection.end);
      const hoursValid = Number.isFinite(state.hours) && state.hours >= 1 && state.hours <= 4;
      if (checkAllHours && hoursValid) {
        checkAllHours.value = String(state.hours);
      }
      const presetReady = !state.presetRequired;
      checkAllButton.disabled = !state.snapshotId || !hasSelection || !hoursValid || !presetReady;
    }

    function populateSnapshots(list) {
      if (!snapshotSelect) return;
      snapshotSelect.innerHTML = "";
      const entries = Array.isArray(list) ? list : [];
      if (state.snapshotId && !entries.some((item) => item.id === state.snapshotId)) {
        state.snapshotId = null;
      }
      for (const item of entries) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.id} • ${item.symbol || "-"} • ${item.tf || "-"}`;
        snapshotSelect.append(option);
      }
      if (state.snapshotId && entries.some((item) => item.id === state.snapshotId)) {
        snapshotSelect.value = state.snapshotId;
      }
      updateCheckAllState();
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
      setJson(checkAllPre, state.checkAll);
    }

    async function requestCheckAllData() {
      if (!state.snapshotId) {
        updateStatus("Выберите снэпшот для запроса check-all данных", "warning");
        return;
      }

      const rawStart = state.selection && state.selection.start ? Number(state.selection.start) : null;
      const rawEnd = state.selection && state.selection.end ? Number(state.selection.end) : null;
      if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd)) {
        updateStatus("Выберите две свечи на графике перед сбором подробных данных", "warning");
        updateCheckAllState();
        return;
      }

      const selectionStart = Math.min(Math.floor(rawStart), Math.floor(rawEnd));
      const selectionEnd = Math.max(Math.floor(rawStart), Math.floor(rawEnd));
      state.hours = resolveHours(state.hours);
      if (checkAllHours) {
        checkAllHours.value = String(state.hours);
      }

      if (checkAllButton) {
        checkAllButton.disabled = true;
      }

      try {
        updateStatus("Загружаем check-all данные...", "info");
        const url = new URL("/inspection/check-all", window.location.origin);
        url.searchParams.set("snapshot", state.snapshotId);
        url.searchParams.set("selection_start", String(selectionStart));
        url.searchParams.set("selection_end", String(selectionEnd));
        url.searchParams.set("hours", String(state.hours));
        const response = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
        });
        if (response.status === 204) {
          state.checkAll = null;
          setJson(checkAllPre, null);
          updateStatus("Check-all данные отсутствуют", "warning");
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        state.checkAll = payload;
        setJson(checkAllPre, payload);
        updateStatus("Check-all данные обновлены", "success");
      } catch (error) {
        console.error(error);
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateStatus("Ошибка запроса check-all данных", "error");
      } finally {
        updateCheckAllState();
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
          updateCheckAllState();
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
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateCheckAllState();
        state.profilePreset = payload?.DATA?.profile_preset || null;
        state.presetRequired = Boolean(payload?.DATA?.profile_preset_required);
        state.presetDefaults = payload?.DATA?.profile_defaults || null;
        renderPresetState();
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

    if (managePresetsButton) {
      managePresetsButton.addEventListener("click", () => {
        openPresetModal({ mode: "list" });
      });
    }

    if (presetCreateButton) {
      presetCreateButton.addEventListener("click", () => {
        if (presetModalTitle) presetModalTitle.textContent = "Создать пресет";
        if (presetListView) presetListView.hidden = true;
        if (presetFormSection) presetFormSection.hidden = false;
        state.presetModalMode = "form";
        populatePresetForm(null, activeSymbol());
      });
    }

    if (presetForm) {
      presetForm.addEventListener("submit", submitPresetForm);
    }

    if (presetBinningModeField) {
      presetBinningModeField.addEventListener("change", handleBinningModeChange);
    }

    presetCancelButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        closePresetModal();
      });
    });

    if (presetBackdrop) {
      presetBackdrop.addEventListener("click", () => closePresetModal());
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.presetModalOpen) {
        closePresetModal();
      }
    });

    if (checkAllHours) {
      checkAllHours.value = String(state.hours);
      checkAllHours.addEventListener("change", () => {
        state.hours = resolveHours(checkAllHours.value);
        checkAllHours.value = String(state.hours);
        updateCheckAllState();
      });
    }

    if (checkAllButton) {
      checkAllButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestCheckAllData();
      });
    }

    if (clearSelection) {
      clearSelection.addEventListener("click", () => {
        state.selection = null;
        updateSelectionLabel();
        updateCheckAllState();
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
      const rawStart = Number(selection.start);
      const rawEnd = Number(selection.end);
      if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd)) {
        throw new Error('selection-invalid');
      }
      const selectionStart = Math.floor(Math.min(rawStart, rawEnd));
      const selectionEnd = Math.floor(Math.max(rawStart, rawEnd));
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
          updateCheckAllState();
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
        candles: [],
        minuteCandles: [],
        refreshTimer: null,
        lastFetchedAtMs: null,
        lastUpdateMs: null,
        isFetching: false,
        intervalMs: intervalToMs((intervalField && intervalField.value) || "1m"),
        gapWatcher: null,
      };

      function normaliseMinuteBar(bar) {
        if (!bar) return null;
        const open = Number(bar.open ?? bar.o ?? 0);
        const high = Number(bar.high ?? bar.h ?? open);
        const low = Number(bar.low ?? bar.l ?? open);
        const close = Number(bar.close ?? bar.c ?? open);
        const volume = Number(bar.volume ?? bar.v ?? 0);
        const ts = Number(bar.ts_ms_utc ?? bar.t ?? bar.time ?? 0);
        if (
          !Number.isFinite(ts) ||
          !Number.isFinite(open) ||
          !Number.isFinite(high) ||
          !Number.isFinite(low) ||
          !Number.isFinite(close)
        ) {
          return null;
        }
        const openMs = Math.floor(ts);
        const lastUpdate = openMs + MINUTE_INTERVAL_MS;
        return {
          time: Math.floor(openMs / 1000),
          open,
          high,
          low,
          close,
          ts_ms_utc: openMs,
          last_update_ms: lastUpdate,
          volume: Number.isFinite(volume) ? volume : 0,
        };
      }

      function mergeMinuteBars(bars, { reset = false } = {}) {
        if (reset) previewState.minuteCandles = [];
        if (!Array.isArray(bars) || !bars.length) return false;
        const index = new Map();
        previewState.minuteCandles.forEach((bar, idx) => {
          const key = Number(bar?.ts_ms_utc ?? (bar?.time ?? 0) * 1000);
          if (Number.isFinite(key)) {
            index.set(key, idx);
          }
        });
        let changed = false;
        bars.forEach((entry) => {
          const normalised = normaliseMinuteBar(entry);
          if (!normalised) return;
          const key = Number(normalised.ts_ms_utc);
          if (!Number.isFinite(key)) return;
          if (index.has(key)) {
            previewState.minuteCandles[index.get(key)] = normalised;
          } else {
            index.set(key, previewState.minuteCandles.length);
            previewState.minuteCandles.push(normalised);
          }
          changed = true;
        });
        if (!changed) return false;
        previewState.minuteCandles.sort((a, b) => Number(a.ts_ms_utc) - Number(b.ts_ms_utc));
        if (previewState.minuteCandles.length > PREVIEW_MINUTE_MAX_BARS) {
          previewState.minuteCandles = previewState.minuteCandles.slice(
            previewState.minuteCandles.length - PREVIEW_MINUTE_MAX_BARS,
          );
        }
        return true;
      }

      function aggregateMinuteBuckets(minuteBars, intervalMs) {
        if (!Array.isArray(minuteBars) || !minuteBars.length) return [];
        const safeInterval = Math.max(1, Number(intervalMs) || MINUTE_INTERVAL_MS);
        const buckets = new Map();
        minuteBars.forEach((bar) => {
          if (!bar) return;
          const openMs = Number(bar.ts_ms_utc ?? bar.t ?? bar.time ?? 0);
          if (!Number.isFinite(openMs)) return;
          const bucketStart = Math.floor(openMs / safeInterval) * safeInterval;
          const high = Number(bar.high ?? bar.h ?? bar.open ?? bar.o ?? 0);
          const low = Number(bar.low ?? bar.l ?? bar.open ?? bar.o ?? 0);
          const close = Number(bar.close ?? bar.c ?? bar.open ?? bar.o ?? 0);
          const open = Number(bar.open ?? bar.o ?? close);
          const volume = Number(bar.volume ?? bar.v ?? 0);
          if (!Number.isFinite(bucketStart)) return;
          let bucket = buckets.get(bucketStart);
          if (!bucket) {
            bucket = {
              start: bucketStart,
              open,
              high,
              low,
              close,
              volume: Number.isFinite(volume) ? volume : 0,
              firstTs: openMs,
              lastUpdate: openMs + MINUTE_INTERVAL_MS,
            };
            buckets.set(bucketStart, bucket);
          } else {
            if (openMs < bucket.firstTs) {
              bucket.firstTs = openMs;
              bucket.open = open;
            }
            bucket.high = Math.max(bucket.high, high);
            bucket.low = Math.min(bucket.low, low);
            bucket.close = close;
            if (Number.isFinite(volume)) {
              bucket.volume += volume;
            }
            bucket.lastUpdate = Math.max(bucket.lastUpdate, openMs + MINUTE_INTERVAL_MS);
          }
        });
        return Array.from(buckets.values())
          .map((bucket) => ({
            start: bucket.start,
            open: bucket.open,
            high: bucket.high,
            low: bucket.low,
            close: bucket.close,
            volume: bucket.volume,
            lastUpdate: bucket.lastUpdate,
          }))
          .sort((a, b) => a.start - b.start);
      }

      function applyMinuteAggregation({ persist = true } = {}) {
        if (!previewState.minuteCandles.length) return false;
        const intervalMs = previewState.intervalMs || intervalToMs(previewState.interval);
        const buckets = aggregateMinuteBuckets(previewState.minuteCandles, intervalMs);
        if (!buckets.length) return false;
        let changed = false;
        const previousUpdate = Number(previewState.lastUpdateMs) || 0;
        let maxUpdate = previousUpdate;
        buckets.forEach((bucket) => {
          const startMs = Number(bucket.start);
          if (!Number.isFinite(startMs)) return;
          const idx = previewState.candles.findIndex((bar) => {
            const barTs = Number(bar?.ts_ms_utc ?? (bar?.time ?? 0) * 1000);
            return Number.isFinite(barTs) && Math.floor(barTs) === startMs;
          });
          const baseBar = idx >= 0 ? previewState.candles[idx] : null;
          const normalised = {
            time: Math.floor(startMs / 1000),
            open: Number.isFinite(baseBar?.open) ? Number(baseBar.open) : Number(bucket.open),
            high: Math.max(
              Number.isFinite(baseBar?.high) ? Number(baseBar.high) : Number(bucket.high),
              Number(bucket.high),
            ),
            low: Math.min(
              Number.isFinite(baseBar?.low) ? Number(baseBar.low) : Number(bucket.low),
              Number(bucket.low),
            ),
            close: Number(bucket.close),
            ts_ms_utc: startMs,
            last_update_ms: Number(bucket.lastUpdate),
          };
          if (baseBar) {
            const updated = {
              ...baseBar,
              open: normalised.open,
              high: normalised.high,
              low: normalised.low,
              close: normalised.close,
              ts_ms_utc: normalised.ts_ms_utc,
              last_update_ms: normalised.last_update_ms,
            };
            const baseVolume = Number(baseBar.volume ?? baseBar.v ?? 0);
            const bucketVolume = Number(bucket.volume ?? 0);
            if (Number.isFinite(baseVolume) || Number.isFinite(bucketVolume)) {
              updated.volume = (Number.isFinite(baseVolume) ? baseVolume : 0) +
                (Number.isFinite(bucketVolume) ? bucketVolume : 0);
            }
            const diff =
              Number(baseBar.open) !== updated.open ||
              Number(baseBar.high) !== updated.high ||
              Number(baseBar.low) !== updated.low ||
              Number(baseBar.close) !== updated.close ||
              Number(baseBar.last_update_ms) !== updated.last_update_ms;
            if (diff) {
              previewState.candles[idx] = updated;
              changed = true;
            } else {
              previewState.candles[idx] = updated;
            }
          } else {
            previewState.candles.push({
              time: normalised.time,
              open: normalised.open,
              high: normalised.high,
              low: normalised.low,
              close: normalised.close,
              ts_ms_utc: normalised.ts_ms_utc,
              last_update_ms: normalised.last_update_ms,
            });
            changed = true;
          }
          if (Number.isFinite(normalised.last_update_ms)) {
            maxUpdate = Math.max(maxUpdate, Number(normalised.last_update_ms));
          }
        });
        if (changed) {
          previewState.candles.sort((a, b) => Number(a.time) - Number(b.time));
        }
        if (Number.isFinite(maxUpdate) && maxUpdate > 0) {
          previewState.lastUpdateMs = maxUpdate;
        }
        if ((changed || previewState.lastUpdateMs !== previousUpdate) && persist) {
          persistPreviewCandles({ bars: previewState.candles, lastUpdateMs: previewState.lastUpdateMs });
        }
        return changed;
      }

      async function fetchMinuteWindow({ force = false } = {}) {
        if (!previewState.symbol) return [];
        const nowMs = Date.now();
        const intervalMs = previewState.intervalMs || intervalToMs(previewState.interval);
        let startMs = null;
        if (force || !previewState.minuteCandles.length) {
          const lookback = Math.max(intervalMs, PREVIEW_MINUTE_LOOKBACK_MS);
          startMs = Math.max(0, nowMs - lookback);
        } else {
          const last = previewState.minuteCandles[previewState.minuteCandles.length - 1];
          const lastTs = Number(last?.ts_ms_utc ?? last?.t ?? 0);
          if (Number.isFinite(lastTs)) {
            startMs = Math.max(0, lastTs - MINUTE_INTERVAL_MS);
          }
        }
        return fetchCandles(previewState.symbol, MINUTE_INTERVAL_KEY, startMs, nowMs);
      }

      function findBarAtOrBefore(targetMs) {
        if (!Number.isFinite(targetMs)) return null;
        for (let idx = previewState.candles.length - 1; idx >= 0; idx -= 1) {
          const bar = previewState.candles[idx];
          if (!bar) continue;
          const barTs = Number(bar.ts_ms_utc ?? (bar.time ?? 0) * 1000);
          if (!Number.isFinite(barTs)) continue;
          if (barTs <= targetMs) {
            return bar;
          }
        }
        return null;
      }

      function normaliseSelectionRange(rawStart, rawEnd) {
        let start = Number(rawStart);
        let end = Number(rawEnd);
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
          return null;
        }
        if (end < start) {
          const tmp = start;
          start = end;
          end = tmp;
        }
        start = Math.floor(start);
        end = Math.floor(end);
        const intervalMs = previewState.intervalMs || intervalToMs(previewState.interval);
        const lastBar = findBarAtOrBefore(end);
        if (lastBar) {
          const barUpdate = Number.isFinite(lastBar.last_update_ms)
            ? Number(lastBar.last_update_ms)
            : Number(previewState.lastUpdateMs);
          const displayEnd = computeCandleDisplayTime(lastBar, intervalMs, barUpdate);
          if (Number.isFinite(displayEnd)) {
            end = Math.max(start, Number(displayEnd));
          }
        }
        return { start, end };
      }

      function setPreviewStatus(message, tone = "info") {
        if (!statusEl) return;
        statusEl.textContent = message || "";
        statusEl.dataset.tone = tone;
        statusEl.hidden = !message;
      }

      function mergePreviewBars(
        bars,
        { reset = false, maxBars = PREVIEW_SHARED_MAX_BARS, lastUpdateMs = null } = {},
      ) {
        if (reset) previewState.candles = [];
        if (!Array.isArray(bars) || !bars.length) return false;
        const index = new Map();
        previewState.candles.forEach((bar, idx) => {
          index.set(Number(bar.time), idx);
        });
        let changed = false;
        bars.forEach((candidate) => {
          const bar = ensurePreviewBar(candidate);
          if (!bar) return;
          const time = Number(bar.time);
          if (!Number.isFinite(time)) return;
          if (index.has(time)) {
            previewState.candles[index.get(time)] = bar;
          } else {
            index.set(time, previewState.candles.length);
            previewState.candles.push(bar);
          }
          changed = true;
        });
        if (!changed) return false;

        previewState.candles.sort((a, b) => Number(a.time) - Number(b.time));
        const limit = Math.max(1, Number(maxBars) || PREVIEW_SHARED_MAX_BARS);
        if (previewState.candles.length > limit) {
          previewState.candles = previewState.candles.slice(previewState.candles.length - limit);
        }
        const effectiveUpdate = Number.isFinite(lastUpdateMs) ? Number(lastUpdateMs) : Date.now();
        previewState.lastUpdateMs = effectiveUpdate;
        persistPreviewCandles({ bars, reset, lastUpdateMs: effectiveUpdate });
        return changed;
      }

      function persistPreviewCandles({ bars = null, reset = false, lastUpdateMs = null } = {}) {
        if (!SharedCandles || typeof SharedCandles.merge !== "function") return;
        const payload = Array.isArray(bars) && bars.length ? bars : previewState.candles;
        if (!payload.length) return;
        const effectiveUpdate = Number.isFinite(lastUpdateMs)
          ? Number(lastUpdateMs)
          : Number.isFinite(previewState.lastUpdateMs)
          ? Number(previewState.lastUpdateMs)
          : Date.now();
        try {
          SharedCandles.merge(previewState.symbol, previewState.interval, payload, {
            intervalMs: previewState.intervalMs,
            lastUpdateMs: effectiveUpdate,
            maxBars: PREVIEW_SHARED_MAX_BARS,
            reset,
          });
        } catch (error) {
          console.warn("preview shared store failed", error);
        }
      }

      function restorePreviewFromShared(symbol, interval) {
        if (!SharedCandles || typeof SharedCandles.get !== "function") {
          return false;
        }
        try {
          const stored = SharedCandles.get(symbol, interval);
          if (!stored || !Array.isArray(stored.candles) || !stored.candles.length) {
            return false;
          }
          const bars = stored.candles.map((bar) => ensurePreviewBar(bar)).filter((bar) => bar !== null);
          if (!bars.length) {
            return false;
          }
          const storedInterval = Number(stored.intervalMs);
          if (Number.isFinite(storedInterval)) {
            previewState.intervalMs = storedInterval;
          }
          const lastUpdate = Number(stored.lastUpdateMs) || Number(stored.updatedAt) || Date.now();
          const changed = mergePreviewBars(bars, { reset: true, lastUpdateMs: lastUpdate });
          if (changed) {
            applyPreviewCandles({ fitContent: true });
            return true;
          }
          return false;
        } catch (error) {
          console.warn("preview shared restore failed", error);
          return false;
        }
      }

      function catch_gap() {
        if (previewState.gapWatcher && typeof previewState.gapWatcher.notifyData === "function") {
          previewState.gapWatcher.notifyData();
        }
      }

      async function fill_gap(gap) {
        if (!gap) return false;
        if (!previewState.symbol || !previewState.interval) return false;
        try {
          const intervalMs = previewState.intervalMs || intervalToMs(previewState.interval);
          const startBound = Number(gap.startMs);
          const endBound = Number(gap.endMs);
          if (!Number.isFinite(startBound) || !Number.isFinite(endBound)) {
            return false;
          }
          const rangeWidth = Math.max(intervalMs, endBound - startBound);
          const approxBars = Math.ceil(rangeWidth / intervalMs) + 2;
          const buffer = intervalMs;
          const bars = await fetchRange(
            previewState.symbol,
            previewState.interval,
            Math.max(0, startBound - buffer),
            endBound + buffer,
            Math.min(1000, Math.max(approxBars, 50)),
          );
          const changed = mergePreviewBars(bars, { lastUpdateMs: Date.now() });
          if (changed) {
            applyPreviewCandles({ fitContent: false });
            catch_gap();
          }
          return true;
        } catch (error) {
          console.error("preview gap fill failed", error);
          setPreviewStatus("Failed to fetch missing candles", "error");
          return false;
        }
      }

      function updatePreviewSelectionLabel() {
        const start = previewState.selection && previewState.selection.start;
        const end = previewState.selection && previewState.selection.end;
        const label = selectionLabel(start, end);
        if (selectionLabelEl) selectionLabelEl.textContent = label;
      }

      function updatePreviewInfo(bar) {
        const intervalMs = previewState.intervalMs || intervalToMs(previewState.interval);
        let lastUpdateMs = Number.isFinite(bar?.last_update_ms)
          ? Number(bar.last_update_ms)
          : Number.isFinite(previewState.lastUpdateMs)
          ? Number(previewState.lastUpdateMs)
          : Date.now();
        const selectionEnd = Number(previewState.selection && previewState.selection.end);
        if (Number.isFinite(selectionEnd)) {
          lastUpdateMs = Math.min(lastUpdateMs, Number(selectionEnd));
        }
        if (!bar) {
          if (lastTimeEl) lastTimeEl.textContent = "-";
          if (lastPriceEl) lastPriceEl.textContent = "-";
          if (lastRangeEl) lastRangeEl.textContent = "-";
          return;
        }
        const displayTs = computeCandleDisplayTime(bar, intervalMs, lastUpdateMs);
        if (lastTimeEl) lastTimeEl.textContent = formatTs(displayTs);
        const close = Number(bar.close ?? bar.c ?? 0);
        if (lastPriceEl) lastPriceEl.textContent = Number.isFinite(close) ? close.toFixed(2) : "-";
        const high = Number(bar.high ?? bar.h ?? close);
        const low = Number(bar.low ?? bar.l ?? close);
        const base = close || 1;
        const rangeValue = Math.max(0, high - low);
        const percent = base ? (rangeValue / base) * 100 : 0;
        if (lastRangeEl) lastRangeEl.textContent = `${rangeValue.toFixed(2)} (${percent.toFixed(2)}%)`;
      }

      function stopPreviewRefresh() {
        if (previewState.refreshTimer) {
          clearTimeout(previewState.refreshTimer);
          previewState.refreshTimer = null;
        }
      }

      function schedulePreviewRefresh() {
        stopPreviewRefresh();
        previewState.refreshTimer = setTimeout(async () => {
          previewState.refreshTimer = null;
          await refreshPreviewCandles({ silent: true });
          schedulePreviewRefresh();
        }, PREVIEW_REFRESH_INTERVAL_MS);
      }

      async function loadPreviewCandles({ reset = false, fitContent = false, limit = 1000 } = {}) {
        if (!previewState.symbol || !previewState.interval) return false;
        const intervalMs = intervalToMs(previewState.interval);
        previewState.intervalMs = intervalMs;
        if (reset) {
          previewState.minuteCandles = [];
        }
        const history = await fetchHistory(previewState.symbol, previewState.interval, limit);
        const historyChanged = mergePreviewBars(history, {
          reset,
          lastUpdateMs: Number.isFinite(previewState.lastUpdateMs) ? previewState.lastUpdateMs : Date.now(),
        });
        if (historyChanged || reset || !previewState.candles.length) {
          applyPreviewCandles({ fitContent });
          catch_gap();
        }

        try {
          const minuteBars = await fetchMinuteWindow({ force: true });
          const previousUpdate = Number(previewState.lastUpdateMs) || 0;
          const minuteChanged = mergeMinuteBars(minuteBars, { reset });
          const aggregatedChanged = minuteChanged ? applyMinuteAggregation({ persist: true }) : false;
          if (aggregatedChanged) {
            applyPreviewCandles({ fitContent: false });
            catch_gap();
          } else if (minuteChanged || Number(previewState.lastUpdateMs || 0) !== previousUpdate) {
            const targetBar =
              findBarAtOrBefore(Number(previewState.selection?.end)) ||
              previewState.candles[previewState.candles.length - 1] ||
              null;
            updatePreviewInfo(targetBar);
          }
          if (previewState.selection && previewState.selection.start && previewState.selection.end) {
            const adjustedRange = normaliseSelectionRange(
              previewState.selection.start,
              previewState.selection.end,
            );
            if (adjustedRange) {
              previewState.selection = adjustedRange;
              updatePreviewSelectionLabel();
            }
          }
        } catch (error) {
          console.warn("preview minute fetch failed", error);
        }

        return historyChanged;
      }

      function applyPreviewCandles({ fitContent = false } = {}) {
        if (previewState.series) {
          previewState.series.setData(previewState.candles);
        }
        if (fitContent && previewState.chart && previewState.candles.length) {
          previewState.chart.timeScale().fitContent();
        }
        let lastBar = previewState.candles[previewState.candles.length - 1] || null;
        const selectionEnd = previewState.selection && previewState.selection.end;
        if (Number.isFinite(selectionEnd)) {
          const endMs = Number(selectionEnd);
          for (let idx = previewState.candles.length - 1; idx >= 0; idx -= 1) {
            const candidate = previewState.candles[idx];
            if (!candidate) continue;
            const candidateTs = Number.isFinite(candidate.ts_ms_utc)
              ? Number(candidate.ts_ms_utc)
              : Number(candidate.time) * 1000;
            if (!Number.isFinite(candidateTs)) continue;
            if (candidateTs <= endMs) {
              lastBar = candidate;
              break;
            }
          }
        }
        updatePreviewInfo(lastBar);
      }

      async function refreshPreviewCandles({ silent = false } = {}) {
        if (previewState.isFetching) return;
        if (!previewState.symbol || !previewState.interval) return;
        previewState.isFetching = true;
        try {
          const minuteBars = await fetchMinuteWindow({ force: false });
          const previousUpdate = Number(previewState.lastUpdateMs) || 0;
          const minuteChanged = mergeMinuteBars(minuteBars, { reset: false });
          const aggregatedChanged = minuteChanged ? applyMinuteAggregation({ persist: true }) : false;
          if (aggregatedChanged) {
            applyPreviewCandles({ fitContent: false });
            catch_gap();
          } else if (minuteChanged || Number(previewState.lastUpdateMs || 0) !== previousUpdate) {
            const targetBar =
              findBarAtOrBefore(Number(previewState.selection?.end)) ||
              previewState.candles[previewState.candles.length - 1] ||
              null;
            updatePreviewInfo(targetBar);
          }
          if (previewState.selection && previewState.selection.start && previewState.selection.end) {
            const adjustedRange = normaliseSelectionRange(
              previewState.selection.start,
              previewState.selection.end,
            );
            if (adjustedRange) {
              previewState.selection = adjustedRange;
              updatePreviewSelectionLabel();
            }
          }
          if (!silent) {
            setPreviewStatus("", "info");
          }
        } catch (error) {
          console.error("Failed to refresh preview candles", error);
          if (!silent) {
            setPreviewStatus("Failed to load Binance history", "error");
          }
        } finally {
          previewState.isFetching = false;
        }
      }

      function ensurePreviewChart() {
        if (previewState.chart || !chartEl || !LightweightCharts) return;
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

        if (ChartGapWatcher && typeof ChartGapWatcher.attach === "function") {
          previewState.gapWatcher = ChartGapWatcher.attach({
            chart: previewState.chart,
            interval: previewState.interval,
            intervalMs: previewState.intervalMs,
            getCandles: () => previewState.candles,
            requestGap: fill_gap,
          });
        }

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
          if (!Number.isFinite(ts)) return;
          if (!previewState.selection || !previewState.selection.start || previewState.selection.end) {
            previewState.selection = { start: ts, end: null };
          } else {
            const range = normaliseSelectionRange(previewState.selection.start, ts);
            if (range) {
              previewState.selection = range;
            } else {
              previewState.selection = { start: ts, end: ts };
            }
          }
          updatePreviewSelectionLabel();
          const targetBar =
            previewState.selection && previewState.selection.end
              ? findBarAtOrBefore(Number(previewState.selection.end)) ||
                previewState.candles[previewState.candles.length - 1] ||
                null
              : previewState.candles[previewState.candles.length - 1] || null;
          updatePreviewInfo(targetBar);
        });
      }

      async function loadPreview(symbolRaw, intervalRaw) {
        const resolvedSymbol =
          normaliseSymbol(symbolRaw) || normaliseSymbol(initial.symbol) || defaultSymbol;
        const resolvedInterval = intervalRaw || "1m";
        previewState.symbol = resolvedSymbol;
        previewState.interval = resolvedInterval;
        previewState.intervalMs = intervalToMs(resolvedInterval);
        previewState.minuteCandles = [];
        previewState.lastUpdateMs = null;
        previewState.lastFetchedAtMs = null;
        previewState.selection = null;
        stopPreviewRefresh();
        if (symbolField) symbolField.value = resolvedSymbol;
        if (intervalField) intervalField.value = resolvedInterval;
        updatePreviewSelectionLabel();
        ensurePreviewChart();
        if (previewState.gapWatcher && typeof previewState.gapWatcher.updateContext === "function") {
          previewState.gapWatcher.updateContext({
            symbol: previewState.symbol,
            interval: previewState.interval,
            intervalMs: previewState.intervalMs,
            getCandles: () => previewState.candles,
            requestGap: fill_gap,
            resetRequestedKeys: true,
          });
        }

        const restoredFromShared = restorePreviewFromShared(resolvedSymbol, resolvedInterval);
        if (restoredFromShared && previewState.gapWatcher && typeof previewState.gapWatcher.notifyData === "function") {
          previewState.gapWatcher.notifyData();
        }

        setPreviewStatus("Loading Binance history...", "info");
        try {
          await loadPreviewCandles({ reset: true, fitContent: !restoredFromShared, limit: 1000 });
          setPreviewStatus("", "info");
          schedulePreviewRefresh();
        } catch (error) {
          console.error(error);
          setPreviewStatus("Failed to load Binance history", "error");
          stopPreviewRefresh();
        }
      }

      if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener("click", () => {
          previewState.selection = null;
          updatePreviewSelectionLabel();
          setPreviewStatus("", "info");
          const lastBar = previewState.candles[previewState.candles.length - 1] || null;
          updatePreviewInfo(lastBar);
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

      window.addEventListener("beforeunload", () => {
        stopPreviewRefresh();
      });

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
            <div class=\"preset-controls\">
              <span id=\"preset-chip\" class=\"preset-chip\" hidden></span>
              <button id=\"manage-presets\" class=\"secondary\" type=\"button\">Управление пресетами</button>
            </div>
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
                  <h3>CHECK ALL DATAS</h3>
                  <div class=\"actions\">
                    <button id=\"fetch-check-all\" class=\"secondary\" type=\"button\">Check all datas</button>
                    <button class=\"secondary\" type=\"button\" data-copy-target=\"checkall-json\">Copy JSON</button>
                  </div>
                </header>
                <div class=\"checkall-control\">
                  <span>Собрать подробно информацию за N часов</span>
                  <select id=\"checkall-hours\">
                    <option value=\"1\">1 час</option>
                    <option value=\"2\">2 часа</option>
                    <option value=\"3\">3 часа</option>
                    <option value=\"4\">4 часа</option>
                  </select>
                </div>
                <pre id=\"checkall-json\">{check_all_json_initial}</pre>
              </div>
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>METRIC</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"metric-json\">Copy JSON</button>
                </header>
                <pre id=\"metric-json\">{metric_json_initial}</pre>
              </div>
            </div>
          </section>
        <div id=\"preset-modal\" class=\"modal\" hidden>
          <div id=\"preset-modal-backdrop\" class=\"modal__backdrop\"></div>
          <div class=\"modal__dialog\" role=\"dialog\" aria-modal=\"true\" aria-labelledby=\"preset-modal-title\">
            <header class=\"modal__header\">
              <h3 id=\"preset-modal-title\">Настроить пресет TPO</h3>
              <button class=\"modal__close\" type=\"button\" data-close-preset>&times;</button>
            </header>
            <div class=\"modal__body\">
              <div id=\"preset-list-view\" hidden>
                <div class=\"preset-list\" id=\"preset-list\"></div>
                <button id=\"preset-create-button\" class=\"btn-primary\" type=\"button\">Создать пресет</button>
              </div>
              <form id=\"preset-form\" hidden>
                <input type=\"hidden\" id=\"preset-mode-input\" value=\"create\" />
                <div class=\"preset-form-grid\" id=\"preset-form-section\">
                  <label>
                    <span>Символ</span>
                    <input id=\"preset-symbol\" type=\"text\" required autocomplete=\"off\" />
                  </label>
                  <label>
                    <span>Таймфрейм</span>
                    <select id=\"preset-tf\">
                      <option value=\"1m\">1m</option>
                    </select>
                  </label>
                  <label>
                    <span>Количество сессий</span>
                    <input id=\"preset-last-n\" type=\"number\" min=\"1\" max=\"5\" step=\"1\" />
                  </label>
                  <label>
                    <span>Value area %</span>
                    <input id=\"preset-value-area\" type=\"number\" min=\"0.1\" max=\"0.95\" step=\"0.01\" />
                  </label>
                  <label>
                    <span>Режим биннинга</span>
                    <select id=\"preset-binning-mode\">
                      <option value=\"adaptive\">Adaptive (ATR)</option>
                      <option value=\"tick\">Tick size</option>
                    </select>
                  </label>
                  <label data-preset-tick>
                    <span>Tick size</span>
                    <input id=\"preset-tick-size\" type=\"number\" step=\"0.0001\" min=\"0\" />
                  </label>
                  <label data-preset-adaptive>
                    <span>ATR multiplier</span>
                    <input id=\"preset-atr-multiplier\" type=\"number\" step=\"0.05\" min=\"0.1\" max=\"2\" />
                  </label>
                  <label data-preset-adaptive>
                    <span>Целевые бины</span>
                    <input id=\"preset-target-bins\" type=\"number\" step=\"5\" min=\"40\" max=\"200\" />
                  </label>
                  <label>
                    <span>Отсечение хвоста</span>
                    <input id=\"preset-clip-tail\" type=\"number\" step=\"0.001\" min=\"0\" max=\"0.05\" />
                  </label>
                  <label>
                    <span>Сглаживание</span>
                    <select id=\"preset-smooth-window\">
                      <option value=\"1\">Без сглаживания</option>
                      <option value=\"2\">Окно 2</option>
                      <option value=\"3\">Окно 3</option>
                    </select>
                  </label>
                </div>
                <div class=\"preset-form-actions\">
                  <button id=\"preset-submit\" class=\"btn-primary\" type=\"submit\">Сохранить</button>
                  <button class=\"btn-secondary\" type=\"button\" data-close-preset>Отмена</button>
                </div>
              </form>
            </div>
          </div>
        </div>

        <div id="preset-modal" class="modal" hidden>
          <div id="preset-modal-backdrop" class="modal__backdrop"></div>
          <div class="modal__dialog" role="dialog" aria-modal="true" aria-labelledby="preset-modal-title">
            <header class="modal__header">
              <h3 id="preset-modal-title">Настроить пресет TPO</h3>
              <button class="modal__close" type="button" data-close-preset>&times;</button>
            </header>
            <div class="modal__body">
              <div id="preset-list-view" hidden>
                <div class="preset-list" id="preset-list"></div>
                <button id="preset-create-button" class="btn-primary" type="button">Создать пресет</button>
              </div>
              <form id="preset-form" hidden>
                <input type="hidden" id="preset-mode-input" value="create" />
                <div class="preset-form-grid" id="preset-form-section">
                  <label>
                    <span>Символ</span>
                    <input id="preset-symbol" type="text" required autocomplete="off" />
                  </label>
                  <label>
                    <span>Таймфрейм</span>
                    <select id="preset-tf">
                      <option value="1m">1m</option>
                    </select>
                  </label>
                  <label>
                    <span>Количество сессий</span>
                    <input id="preset-last-n" type="number" min="1" max="5" step="1" />
                  </label>
                  <label>
                    <span>Value area %</span>
                    <input id="preset-value-area" type="number" min="0.1" max="0.95" step="0.01" />
                  </label>
                  <label>
                    <span>Режим биннинга</span>
                    <select id="preset-binning-mode">
                      <option value="adaptive">Adaptive (ATR)</option>
                      <option value="tick">Tick size</option>
                    </select>
                  </label>
                  <label data-preset-tick>
                    <span>Tick size</span>
                    <input id="preset-tick-size" type="number" step="0.0001" min="0" />
                  </label>
                  <label data-preset-adaptive>
                    <span>ATR multiplier</span>
                    <input id="preset-atr-multiplier" type="number" step="0.05" min="0.1" max="2" />
                  </label>
                  <label data-preset-adaptive>
                    <span>Целевые бины</span>
                    <input id="preset-target-bins" type="number" step="5" min="40" max="200" />
                  </label>
                  <label>
                    <span>Отсечение хвоста</span>
                    <input id="preset-clip-tail" type="number" step="0.001" min="0" max="0.05" />
                  </label>
                  <label>
                    <span>Сглаживание</span>
                    <select id="preset-smooth-window">
                      <option value="1">Без сглаживания</option>
                      <option value="2">Окно 2</option>
                      <option value="3">Окно 3</option>
                    </select>
                  </label>
                </div>
                <div class="preset-form-actions">
                  <button id="preset-submit" class="btn-primary" type="submit">Сохранить</button>
                  <button class="btn-secondary" type="button" data-close-preset>Отмена</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        </main>
        <script>{script_block}</script>
        <script src=\"https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js\"></script>
        <script src="/public/binanceCandles.js"></script>
        <script src="/public/chart-gap-watcher.js"></script>
        <script src="/public/shared-candles.js"></script>
        <script>{ui_script}</script>
      </body>
    </html>
    """
    return page_html
