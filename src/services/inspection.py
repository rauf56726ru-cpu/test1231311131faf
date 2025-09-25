"""Inspection payload assembly based on frontend snapshots."""
from __future__ import annotations

import html
import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

from ..meta import Meta
from .ohlc import TIMEFRAME_WINDOWS, normalise_ohlcv

Snapshot = Dict[str, Any]

_MAX_STORED_SNAPSHOTS = 16
_SNAPSHOT_STORE: "OrderedDict[str, Snapshot]" = OrderedDict()


def _session_options() -> Iterable[Tuple[str, str, str]]:
    for name, start, end in Meta.iter_vwap_sessions():
        yield name, start.strftime("%H:%M"), end.strftime("%H:%M")


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        _SNAPSHOT_STORE.popitem(last=False)


def register_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Store a snapshot captured by the chart frontend."""

    if "candles" not in snapshot:
        raise ValueError("Snapshot must include candles")

    try:
        candles = list(snapshot["candles"])  # type: ignore[index]
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Snapshot candles must be iterable") from exc

    symbol = str(snapshot.get("symbol") or snapshot.get("ticker") or "UNKNOWN").upper()
    timeframe = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()
    snapshot_id = str(
        snapshot.get("id")
        or snapshot.get("snapshot_id")
        or snapshot.get("snapshot")
        or f"snap-{int(datetime.now(timezone.utc).timestamp()*1000)}"
    )

    meta: MutableMapping[str, Any] = {}
    for key in ("meta", "diagnostics", "source"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            meta[key] = dict(value)

    stored: Snapshot = {
        "id": snapshot_id,
        "symbol": symbol,
        "tf": timeframe,
        "candles": candles,
        "captured_at": snapshot.get("captured_at")
        or datetime.now(timezone.utc).isoformat(),
        "meta": meta,
    }

    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    _SNAPSHOT_STORE[snapshot_id] = stored
    _SNAPSHOT_STORE.move_to_end(snapshot_id)
    _ensure_snapshot_limit()
    return snapshot_id


def get_snapshot(snapshot_id: str) -> Snapshot | None:
    """Return a stored snapshot if present."""

    snapshot = _SNAPSHOT_STORE.get(snapshot_id)
    if snapshot is not None:
        # Refresh LRU ordering so recently accessed snapshots persist longer.
        _SNAPSHOT_STORE.move_to_end(snapshot_id)
    return snapshot


def _extract_section(snapshot: Snapshot, key: str) -> Any:
    value = snapshot.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def build_inspection_payload(snapshot: Snapshot) -> Dict[str, Any]:
    """Build a combined inspection payload from a stored snapshot."""

    symbol = snapshot.get("symbol", "UNKNOWN")
    timeframe = snapshot.get("tf", "1m")
    candles = snapshot.get("candles", [])

    ohlc_payload = normalise_ohlcv(
        symbol,
        timeframe,
        candles,
        include_diagnostics=True,
    )

    ohlc_diagnostics = ohlc_payload.pop("diagnostics", None)

    data_section = {
        "ohlcv": ohlc_payload,
        "delta_cvd": _extract_section(snapshot, "delta"),
        "vwap_tpo": _extract_section(snapshot, "vwap"),
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
                "tf": timeframe,
            },
            "source": snapshot.get("meta", {}),
        },
    }

    diagnostics_section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot.get("id"),
        "captured_at": snapshot.get("captured_at"),
        "ohlcv": ohlc_diagnostics,
        "notes": {
            "delta": "Delta/CVD data was not provided in the snapshot." if "delta" not in snapshot else None,
            "vwap": "VWAP/TPO data was not provided in the snapshot." if "vwap" not in snapshot else None,
        },
    }

    return {"DATA": data_section, "DIAGNOSTICS": diagnostics_section}


def render_inspection_page(
    payload: Dict[str, Any],
    *,
    snapshot_id: str,
    symbol: str,
    timeframe: str,
) -> str:
    """Render the inspection dashboard HTML."""

    payload_json = json.dumps(payload, ensure_ascii=False)
    payload_json = payload_json.replace("</", "<\\/")

    symbol_value = html.escape(symbol.upper())
    timeframe_value = html.escape(timeframe)
    snapshot_value = html.escape(snapshot_id)

    timeframe_options = []
    for tf_key in TIMEFRAME_WINDOWS:
        selected = " selected" if tf_key == timeframe else ""
        timeframe_options.append(
            f'<option value="{html.escape(tf_key)}"{selected}>{html.escape(tf_key)}</option>'
        )

    style_block = """
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --fg: #e2e8f0;
      --muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.25);
      --accent: #38bdf8;
      --accent-strong: #0ea5e9;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #1e293b 0%, #0f172a 55%, #020617 100%);
      color: var(--fg);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 2rem 1.5rem 1rem;
      max-width: 1100px;
      margin: 0 auto;
    }
    header h1 {
      margin: 0 0 0.5rem;
      font-size: clamp(1.8rem, 2.8vw, 2.6rem);
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    main {
      width: min(1100px, 95%);
      margin: 0 auto 2.5rem;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .inspection-card {
      background: rgba(15, 23, 42, 0.85);
      border-radius: 1.1rem;
      border: 1px solid var(--border);
      box-shadow: 0 18px 60px rgba(2, 6, 23, 0.45);
      padding: 1.5rem;
    }
    form {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem 1.5rem;
      align-items: end;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      color: var(--muted);
      font-size: 0.85rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    input, select {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.65rem 0.9rem;
      color: var(--fg);
      font-size: 0.95rem;
    }
    button {
      cursor: pointer;
      border-radius: 0.75rem;
      border: none;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.7rem 1.2rem;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .btn-primary {
      background: linear-gradient(120deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #0b1120;
      box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 34px rgba(14, 165, 233, 0.36);
    }
    .btn-secondary {
      background: rgba(30, 41, 59, 0.85);
      color: var(--fg);
      border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .btn-secondary:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 30px rgba(8, 145, 178, 0.28);
    }
    .section {
      border-radius: 1rem;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.72);
      overflow: hidden;
    }
    .section summary {
      margin: 0;
      padding: 1rem 1.25rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
    }
    .section pre {
      margin: 0;
      padding: 1rem 1.25rem 1.5rem;
      background: rgba(8, 47, 73, 0.55);
      border-top: 1px solid rgba(56, 189, 248, 0.15);
      font-family: "Fira Code", "SFMono-Regular", ui-monospace, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.86rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .section-actions {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .status-bar {
      font-size: 0.85rem;
      color: var(--muted);
    }
    .empty {
      color: rgba(148, 163, 184, 0.65);
      font-style: italic;
    }
    @media (max-width: 640px) {
      header {
        padding: 1.5rem 1rem 0.75rem;
      }
      main {
        width: min(100%, 95%);
      }
    }
    """

    script_block = f"""
    const SNAPSHOT_ID = "{snapshot_value}";
    let currentPayload = {payload_json};

    function renderSections(payload) {{
      const container = document.querySelector('#inspection-sections');
      if (!container) return;
      container.innerHTML = '';
      const entries = Object.entries(payload || {{}});
      if (!entries.length) {{
        container.innerHTML = '<p class="empty">Нет данных</p>';
        return;
      }}
      for (const [key, value] of entries) {{
        const details = document.createElement('details');
        details.className = 'section';
        details.open = key === 'ohlcv';
        const summary = document.createElement('summary');
        summary.innerHTML = `<span>${{key}}</span>`;
        const actions = document.createElement('div');
        actions.className = 'section-actions';
        const copyButton = document.createElement('button');
        copyButton.type = 'button';
        copyButton.className = 'btn-secondary';
        copyButton.textContent = 'Copy JSON';
        copyButton.addEventListener('click', () => {{
          navigator.clipboard?.writeText(JSON.stringify(value, null, 2)).catch(() => {{}});
        }});
        actions.appendChild(copyButton);
        summary.appendChild(actions);
        details.appendChild(summary);
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(value, null, 2);
        details.appendChild(pre);
        container.appendChild(details);
      }}
    }}

    function renderDiagnostics(diagnostics) {{
      const target = document.querySelector('#diagnostics');
      if (!target) return;
      target.textContent = JSON.stringify(diagnostics || {{}}, null, 2);
    }}

    async function refreshPayload() {{
      const status = document.querySelector('#status-bar');
      if (status) {{
        status.textContent = 'Обновляем данные...';
      }}
      try {{
        const response = await fetch(`/inspection?snapshot=${{encodeURIComponent(SNAPSHOT_ID)}}`, {{
          headers: {{ 'Accept': 'application/json' }}
        }});
        if (!response.ok) {{
          throw new Error(`HTTP ${{response.status}}`);
        }}
        currentPayload = await response.json();
        renderSections(currentPayload.DATA);
        renderDiagnostics(currentPayload.DIAGNOSTICS);
        if (status) {{
          status.textContent = 'Обновлено: ' + new Date().toLocaleTimeString();
        }}
      }} catch (error) {{
        console.error('Refresh failed', error);
        if (status) {{
          status.textContent = 'Ошибка обновления: ' + error.message;
        }}
      }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      renderSections(currentPayload.DATA);
      renderDiagnostics(currentPayload.DIAGNOSTICS);
      const refreshButton = document.querySelector('#refresh-inspection');
      refreshButton?.addEventListener('click', refreshPayload);
    }});
    """

    return f"""
    <!DOCTYPE html>
    <html lang=\"ru\">
      <head>
        <meta charset=\"utf-8\" />
        <title>Inspection snapshot {snapshot_value}</title>
        <style>{style_block}</style>
      </head>
      <body>
        <header>
          <h1>Inspection snapshot</h1>
          <p>Снимок данных для {symbol_value} · {timeframe_value} · ID {snapshot_value}</p>
        </header>
        <main>
          <section class=\"inspection-card\">
            <form id=\"inspection-form\" onsubmit=\"return false;\">
              <label>
                <span>Символ</span>
                <input value=\"{symbol_value}\" readonly />
              </label>
              <label>
                <span>Таймфрейм</span>
                <select disabled>{''.join(timeframe_options)}</select>
              </label>
              <label>
                <span>Snapshot</span>
                <input value=\"{snapshot_value}\" readonly />
              </label>
              <button id=\"refresh-inspection\" type=\"button\" class=\"btn-primary\">Refresh</button>
            </form>
            <p id=\"status-bar\" class=\"status-bar\">Загружено: {datetime.now(timezone.utc).isoformat()}</p>
          </section>
          <section class=\"inspection-card\">
            <h2>DATA</h2>
            <div id=\"inspection-sections\"></div>
          </section>
          <section class=\"inspection-card\">
            <h2>DIAGNOSTICS</h2>
            <pre id=\"diagnostics\" class=\"diagnostics\"></pre>
          </section>
        </main>
        <script>{script_block}</script>
      </body>
    </html>
    """

