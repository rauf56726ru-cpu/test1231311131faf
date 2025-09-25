"""Utilities for assembling the inspection payload and UI."""
from __future__ import annotations

import asyncio
import html
import json
from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

from ..meta import Meta
from .delta import fetch_bar_delta
from .ohlc import TIMEFRAME_WINDOWS, fetch_ohlcv
from .tpo import MAX_SESSIONS, MIN_SESSIONS, fetch_tpo_profile
from .vwap import fetch_session_vwap


def _session_options() -> Iterable[Tuple[str, str, str]]:
    for name, start, end in Meta.iter_vwap_sessions():
        yield name, start.strftime("%H:%M"), end.strftime("%H:%M")


async def build_inspection_payload(
    symbol: str,
    timeframe: str,
    *,
    session: str = "ny",
    sessions: int = MAX_SESSIONS,
) -> Dict[str, object]:
    """Collect data required for the inspection dashboard."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    session = session.lower()
    sessions = max(MIN_SESSIONS, min(MAX_SESSIONS, sessions))

    ohlc_task = fetch_ohlcv(symbol, timeframe, include_diagnostics=True)
    delta_task = fetch_bar_delta(symbol, timeframe)
    vwap_task = fetch_session_vwap(symbol)
    tpo_task = fetch_tpo_profile(symbol, session=session, sessions=sessions)

    ohlc_payload, delta_payload, vwap_payload, tpo_payload = await asyncio.gather(
        ohlc_task,
        delta_task,
        vwap_task,
        tpo_task,
    )

    ohlc_diagnostics = (
        ohlc_payload.pop("diagnostics") if isinstance(ohlc_payload, dict) else None
    )

    meta_section = {
        "requested": {
            "symbol": symbol.upper(),
            "tf": timeframe,
            "session": session,
            "sessions": sessions,
        },
        "vwap_lookback_days": Meta.VWAP_LOOKBACK_DAYS,
        "sessions": [
            {"name": name, "start": start, "end": end}
            for name, start, end in _session_options()
        ],
    }

    data_section = {
        "ohlcv": ohlc_payload,
        "delta_cvd": delta_payload,
        "vwap_tpo": {"vwap": vwap_payload, "tpo": tpo_payload},
        "zones": {
            "status": "unavailable",
            "detail": "Zones provider is not configured for the test build.",
        },
        "smt": {
            "status": "unavailable",
            "detail": "SMT provider is not configured for the test build.",
        },
        "meta": meta_section,
    }

    diagnostics_section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "request": meta_section["requested"],
        "ohlcv": ohlc_diagnostics,
        "delta_cvd": {
            "bars": len(delta_payload.get("bar_delta", []))
            if isinstance(delta_payload, dict)
            else None
        },
        "vwap": {
            "rows": len(vwap_payload.get("vwap", []))
            if isinstance(vwap_payload, dict)
            else None
        },
        "tpo": {
            "sessions": tpo_payload.get("sessions")
            if isinstance(tpo_payload, dict)
            else None
        },
        "notes": {
            "zones": "Zones data source is not yet implemented.",
            "smt": "SMT data source is not yet implemented.",
        },
    }

    return {"DATA": data_section, "DIAGNOSTICS": diagnostics_section}


def render_inspection_page(
    payload: Dict[str, object],
    *,
    symbol: str,
    timeframe: str,
    session: str,
    sessions: int,
) -> str:
    """Render the inspection dashboard HTML."""

    payload_json = json.dumps(payload, ensure_ascii=False)
    payload_json = payload_json.replace("</", "<\\/")

    symbol_value = html.escape(symbol.upper())
    timeframe_value = html.escape(timeframe)
    session_value = html.escape(session)
    sessions_value = html.escape(str(sessions))

    timeframe_options = []
    for tf_key in TIMEFRAME_WINDOWS:
        selected = " selected" if tf_key == timeframe else ""
        timeframe_options.append(
            f'<option value="{html.escape(tf_key)}"{selected}>{html.escape(tf_key)}</option>'
        )

    session_options = []
    for name, _, _ in _session_options():
        selected = " selected" if name == session else ""
        session_options.append(
            f'<option value="{html.escape(name)}"{selected}>{html.escape(name)}</option>'
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
      gap: 0.4rem;
      font-size: 0.9rem;
      color: var(--muted);
      font-weight: 600;
    }
    input,
    select {
      padding: 0.65rem 0.9rem;
      border-radius: 0.85rem;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(15, 23, 42, 0.65);
      color: var(--fg);
      font-size: 0.95rem;
    }
    .controls-actions {
      display: flex;
      gap: 0.75rem;
    }
    button {
      cursor: pointer;
      border-radius: 0.85rem;
      border: none;
      font-weight: 600;
    }
    .btn-primary {
      padding: 0.7rem 1.5rem;
      color: #0b1120;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      box-shadow: 0 12px 32px rgba(14, 165, 233, 0.35);
    }
    .btn-secondary {
      padding: 0.7rem 1.2rem;
      background: rgba(148, 163, 184, 0.15);
      border: 1px solid rgba(148, 163, 184, 0.35);
      color: var(--fg);
    }
    .payload-group {
      margin-top: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .payload-group h2 {
      margin: 0;
      font-size: 1.2rem;
      color: var(--accent);
      letter-spacing: 0.04em;
    }
    details {
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 0.9rem;
      background: rgba(2, 6, 23, 0.6);
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      list-style: none;
      padding: 0.9rem 1rem;
      font-weight: 600;
      position: relative;
    }
    summary::after {
      content: "";
      position: absolute;
      right: 1rem;
      top: 50%;
      width: 0.6rem;
      height: 0.6rem;
      border-right: 2px solid var(--muted);
      border-bottom: 2px solid var(--muted);
      transform: translateY(-60%) rotate(45deg);
      transition: transform 0.2s ease;
    }
    details[open] summary::after {
      transform: translateY(-20%) rotate(-135deg);
    }
    .section-body {
      padding: 0 1rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .section-toolbar {
      display: flex;
      justify-content: flex-end;
    }
    .copy-btn {
      padding: 0.45rem 1.1rem;
      border-radius: 0.75rem;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(30, 41, 59, 0.9);
      color: var(--fg);
      font-size: 0.85rem;
    }
    pre {
      margin: 0;
      padding: 0.9rem;
      border-radius: 0.75rem;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: #f8fafc;
      overflow-x: auto;
      font-size: 0.85rem;
    }
    .empty-note {
      margin: 0;
      padding: 1rem;
      border-radius: 0.85rem;
      border: 1px dashed rgba(148, 163, 184, 0.4);
      text-align: center;
      color: var(--muted);
      font-size: 0.95rem;
    }
    @media (max-width: 640px) {
      .controls-actions {
        flex-direction: column;
      }
      button {
        width: 100%;
      }
    }
    """

    script_block = """
    (function () {
      const payloadRoot = document.getElementById("payload-root");
      const form = document.getElementById("inspection-form");
      const refreshBtn = document.getElementById("inspection-refresh");
      let sectionCounter = 0;

      function renderSection(container, key, value) {
        const details = document.createElement("details");
        details.className = "inspection-section";
        details.open = true;

        const summary = document.createElement("summary");
        summary.textContent = key;
        details.appendChild(summary);

        const body = document.createElement("div");
        body.className = "section-body";

        const toolbar = document.createElement("div");
        toolbar.className = "section-toolbar";
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "copy-btn";
        copyBtn.textContent = "Copy JSON";
        const preId = `json-section-${sectionCounter++}`;
        copyBtn.dataset.copyTarget = preId;
        toolbar.appendChild(copyBtn);
        body.appendChild(toolbar);

        const pre = document.createElement("pre");
        pre.id = preId;
        pre.textContent = JSON.stringify(value, null, 2);
        body.appendChild(pre);

        details.appendChild(body);
        container.appendChild(details);
      }

      function renderGroup(title, value) {
        const group = document.createElement("section");
        group.className = "payload-group";
        const heading = document.createElement("h2");
        heading.textContent = title;
        group.appendChild(heading);

        const entries = value && typeof value === "object" ? Object.entries(value) : [];
        if (!entries.length) {
          const empty = document.createElement("p");
          empty.className = "empty-note";
          empty.textContent = "Нет данных";
          group.appendChild(empty);
        } else {
          for (const [key, entryValue] of entries) {
            renderSection(group, key, entryValue);
          }
        }

        payloadRoot.appendChild(group);
      }

      function renderPayload(data) {
        payloadRoot.innerHTML = "";
        sectionCounter = 0;
        if (!data || typeof data !== "object") {
          renderGroup("DATA", {});
          renderGroup("DIAGNOSTICS", {});
          return;
        }
        renderGroup("DATA", data.DATA || {});
        renderGroup("DIAGNOSTICS", data.DIAGNOSTICS || {});
      }

      document.addEventListener("click", (event) => {
        const target = event.target;
        if (!target || !target.dataset || !target.dataset.copyTarget) {
          return;
        }
        const pre = document.getElementById(target.dataset.copyTarget);
        if (!pre) return;
        navigator.clipboard?.writeText(pre.textContent || "").then(
          () => {
            const original = target.textContent;
            target.textContent = "Copied!";
            setTimeout(() => {
              target.textContent = original;
            }, 1200);
          },
          () => {
            target.textContent = "Copy failed";
            setTimeout(() => {
              target.textContent = "Copy JSON";
            }, 1500);
          }
        );
      });

      form?.addEventListener("submit", (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        const url = new URL(window.location.pathname, window.location.origin);
        for (const [key, value] of formData.entries()) {
          if (value) {
            url.searchParams.set(key, String(value).trim());
          }
        }
        window.location.href = url.toString();
      });

      refreshBtn?.addEventListener("click", async () => {
        const formData = new FormData(form);
        const url = new URL(window.location.pathname, window.location.origin);
        for (const [key, value] of formData.entries()) {
          if (value) {
            url.searchParams.set(key, String(value).trim());
          }
        }
        try {
          refreshBtn.disabled = true;
          const original = refreshBtn.textContent;
          refreshBtn.textContent = "Refreshing...";
          const response = await fetch(url.toString(), {
            headers: { Accept: "application/json" },
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const json = await response.json();
          window.history.replaceState(null, document.title, url.toString());
          window.__INSPECTION_PAYLOAD__ = json;
          renderPayload(json);
          refreshBtn.textContent = original;
        } catch (error) {
          console.error("Failed to refresh inspection payload", error);
          alert(`Не удалось обновить данные: ${error.message}`);
          refreshBtn.textContent = "Refresh";
        } finally {
          refreshBtn.disabled = false;
        }
      });

      renderPayload(window.__INSPECTION_PAYLOAD__ || {});
    })();
    """

    return f"""<!DOCTYPE html>
<html lang=\"ru\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Inspection Dashboard</title>
    <style>{style_block}</style>
  </head>
  <body>
    <header>
      <h1>Inspection Dashboard</h1>
      <p>Проверка собранных данных для {symbol_value} на таймфрейме {timeframe_value}. Сессия {session_value}, окон {sessions_value}.</p>
    </header>
    <main>
      <section class=\"inspection-card\">
        <form id=\"inspection-form\">
          <label>
            <span>Symbol</span>
            <input name=\"symbol\" value=\"{symbol_value}\" required autocomplete=\"off\" />
          </label>
          <label>
            <span>Timeframe</span>
            <select name=\"tf\">{''.join(timeframe_options)}</select>
          </label>
          <label>
            <span>Session</span>
            <select name=\"session\">{''.join(session_options)}</select>
          </label>
          <label>
            <span>Sessions</span>
            <input type=\"number\" name=\"sessions\" min=\"{MIN_SESSIONS}\" max=\"{MAX_SESSIONS}\" value=\"{sessions_value}\" />
          </label>
          <div class=\"controls-actions\">
            <button type=\"submit\" class=\"btn-primary\">Load</button>
            <button type=\"button\" id=\"inspection-refresh\" class=\"btn-secondary\">Refresh</button>
          </div>
        </form>
      </section>
      <section class=\"inspection-card\" id=\"payload-root\"></section>
    </main>
    <script>window.__INSPECTION_PAYLOAD__ = {payload_json};</script>
    <script>{script_block}</script>
  </body>
</html>"""
