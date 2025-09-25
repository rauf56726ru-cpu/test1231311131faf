"""Inspection payload assembly and UI rendering."""
from __future__ import annotations

import html as html_utils
import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .ohlc import TIMEFRAME_WINDOWS, normalise_ohlcv

Snapshot = Dict[str, Any]

_MAX_STORED_SNAPSHOTS = 16
_SNAPSHOT_STORE: "OrderedDict[str, Snapshot]" = OrderedDict()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        _SNAPSHOT_STORE.popitem(last=False)


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

    for key in ("delta", "vwap", "zones", "smt"):
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

        result = normalise_ohlcv(
            symbol,
            tf_key,
            candles,
            include_diagnostics=True,
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

    data_section = {
        "symbol": symbol,
        "frames": normalised_frames,
        "selection": selection,
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

    symbol_value = html_utils.escape(symbol.upper())
    timeframe_value = html_utils.escape(timeframe)
    snapshot_value = html_utils.escape(snapshot_id or "")

    timeframe_options = []
    for tf_key in TIMEFRAME_WINDOWS:
        selected = " selected" if tf_key == timeframe else ""
        timeframe_options.append(
            f'<option value="{html_utils.escape(tf_key)}"{selected}>{html_utils.escape(tf_key)}</option>'
        )

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
        f"  snapshots: {snapshots_json}\n"
        "};\n"
    )

    ui_script = """
(function () {
  const TIMEFRAME_TO_MS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
  };

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

    initCollapsibles();

    const state = {
      payload: initial.payload || null,
      snapshotId: initial.snapshotId || null,
      selection: initial.payload?.DATA?.selection || null,
      frame: initial.timeframe || initial.payload?.DATA?.meta?.requested?.frames?.[0] || "1m",
      chart: null,
      series: null,
    };

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
        option.textContent = `${item.id} • ${item.symbol || "?"} • ${item.tf || "?"}`;
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
      const symbol = payload?.DATA?.symbol || initial.symbol;
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

    function renderChart() {
      if (!chartContainer) return;
      if (!window.LightweightCharts) {
        console.warn("LightweightCharts not loaded");
        return;
      }
      if (!state.chart) {
        state.chart = LightweightCharts.createChart(chartContainer, {
          layout: {
            background: { color: "rgba(15, 23, 42, 0.05)" },
            textColor: "#e2e8f0",
          },
          rightPriceScale: { borderColor: "rgba(148, 163, 184, 0.4)" },
          timeScale: {
            borderColor: "rgba(148, 163, 184, 0.4)",
            timeVisible: true,
            secondsVisible: true,
          },
          grid: {
            vertLines: { color: "rgba(15, 23, 42, 0.6)" },
            horzLines: { color: "rgba(15, 23, 42, 0.6)" },
          },
          crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        });
        state.series = state.chart.addCandlestickSeries({
          upColor: "#22c55e",
          downColor: "#ef4444",
          wickUpColor: "#f8fafc",
          wickDownColor: "#f8fafc",
          borderUpColor: "#22c55e",
          borderDownColor: "#ef4444",
        });
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
      const frame = state.frame;
      const candles = state.payload?.DATA?.frames?.[frame]?.candles || [];
      const bars = toChartBars(candles);
      state.series.setData(bars);
      if (bars.length) {
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

    if (buildButton) {
      buildButton.addEventListener("click", async () => {
        if (!state.selection || !state.selection.start || !state.selection.end) {
          updateStatus("Сначала выделите диапазон на графике", "warning");
          return;
        }
        const selectedFrames = timeframeCheckboxes
          .filter((checkbox) => checkbox.checked)
          .map((checkbox) => checkbox.value);
        if (!selectedFrames.length) {
          updateStatus("Выберите хотя бы один таймфрейм", "warning");
          return;
        }
        const symbolInput = document.getElementById("symbol-input");
        const symbol = symbolInput ? symbolInput.value.trim().toUpperCase() : (initial.symbol || "BTCUSDT");
        updateStatus("Получаем свечи Binance...", "info");
        try {
          const frames = {};
          for (const tf of selectedFrames) {
            const candles = await fetchCandles(symbol, tf, state.selection.start, state.selection.end);
            frames[tf] = { tf, candles };
          }
          const payload = {
            id: `test-${Date.now()}`,
            symbol,
            frames,
            selection: { start: state.selection.start, end: state.selection.end },
            meta: {
              source: "inspection-ui",
              timeframes: selectedFrames,
              generated_at: new Date().toISOString(),
            },
          };
          const result = await postSnapshot(payload);
          state.snapshotId = result.snapshot_id;
          await refreshSnapshots();
          if (snapshotSelect) snapshotSelect.value = state.snapshotId;
          await loadSnapshot(state.snapshotId);
          updateStatus("Тестовая среда создана", "success");
        } catch (error) {
          console.error(error);
          updateStatus("Не удалось создать тестовую среду", "error");
        }
      });
    }

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
                <input id=\"symbol-input\" type=\"text\" value=\"{symbol_value}\" autocomplete=\"off\" />
              </label>
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
          <section class=\"panel\">
            <h2>Просмотр данных</h2>
            <div id=\"inspection-chart\" class=\"chart-shell\" data-selection-label=\"—\"></div>
            <div class=\"metrics-bar\">
              <button class=\"secondary\" type=\"button\" data-metric=\"ohlcv\">OHLCV</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"delta\">Delta / CVD</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"vwap\">VWAP</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"zones\">Zones</button>
              <button class=\"secondary\" type=\"button\" data-metric=\"smt\">SMT</button>
            </div>
            <div class=\"json-panels\">
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>DATA</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"data-json\">Copy JSON</button>
                </header>
                <pre id=\"data-json\"></pre>
              </div>
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>DIAGNOSTICS</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"diagnostics-json\">Copy JSON</button>
                </header>
                <pre id=\"diagnostics-json\"></pre>
              </div>
              <div class=\"collapse\">
                <header data-collapse-toggle>
                  <h3>METRIC</h3>
                  <button class=\"secondary\" type=\"button\" data-copy-target=\"metric-json\">Copy JSON</button>
                </header>
                <pre id=\"metric-json\"></pre>
              </div>
            </div>
          </section>
        </main>
        <script>{script_block}</script>
        <script src=\"https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js\"></script>
        <script>{ui_script}</script>
      </body>
    </html>
    """
    return page_html
