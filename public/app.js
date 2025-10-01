(function () {
  "use strict";

  const LightweightCharts = window.LightweightCharts;
  const BinanceCandles = window.BinanceCandles;
  const ChartGapWatcher = window.ChartGapWatcher;
  const SharedCandles = window.SharedCandles;

  if (!LightweightCharts || !BinanceCandles) {
    console.error("Chart dependencies are missing");
    return;
  }

  const chartContainer = document.getElementById("chart");
  const form = document.getElementById("chart-controls");
  const symbolInput = document.getElementById("input-symbol");
  const intervalInput = document.getElementById("input-interval");
  const statusEl = document.getElementById("status-message");
  const lastTimeEl = document.getElementById("last-time");
  const lastPriceEl = document.getElementById("last-price");
  const rangeEl = document.getElementById("last-range");
  const versionEl = document.getElementById("app-version");
  const inspectionBtn = document.getElementById("inspection-btn") || document.getElementById("show-inspection");
  const analyzeBtn = document.getElementById("analyze-btn");
  const exportBtn = document.getElementById("export-zones");
  const toastContainer = document.getElementById("toast-container");
  const modalEl = document.getElementById("modal-confirm");
  const modalConfirmBtn = modalEl ? modalEl.querySelector("[data-action='confirm']") : null;
  const modalCancelBtn = modalEl ? modalEl.querySelector("[data-action='cancel']") : null;
  const progressBar = document.getElementById("inspection-progress");
  const dashboardEl = document.getElementById("dashboard");
  const presetBadge = document.getElementById("preset-badge");
  const symbolButtons = Array.from(document.querySelectorAll("[data-symbol-option]"));

  if (!chartContainer || !form || !symbolInput || !intervalInput) {
    console.error("Chart container or controls are missing in the DOM");
    return;
  }

  const state = {
    symbol: (symbolInput.value || "BTCUSDT").toUpperCase(),
    interval: intervalInput.value || "1m",
    candles: [],
    chart: null,
    candleSeries: null,
    priceLine: null,
    ws: null,
    reconnectTimer: null,
    reconnectAttempts: 0,
    gapWatcher: null,
    lastUpdateMs: null,
    pocSeries: null,
    vahSeries: null,
    valSeries: null,
    inspectProgressTimer: null,
    lastSnapshotId: null,
  };

  function setActiveSymbolButton(symbol) {
    if (!symbolButtons.length) return;
    symbolButtons.forEach((button) => {
      const isActive = button.dataset.symbolOption === symbol;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  function intervalToMs(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return Math.max(1, numeric) * 60_000;
    }
    const map = {
      "1s": 1_000,
      "3s": 3_000,
      "5s": 5_000,
      "15s": 15_000,
      "30s": 30_000,
      "1m": 60_000,
      "3m": 180_000,
      "5m": 300_000,
      "15m": 900_000,
      "30m": 1_800_000,
      "1h": 3_600_000,
      "2h": 7_200_000,
      "4h": 14_400_000,
      "6h": 21_600_000,
      "8h": 28_800_000,
      "12h": 43_200_000,
      "1d": 86_400_000,
    };
    return map[value] || 60_000;
  }

  function formatNumber(value, digits = 2) {
    if (!Number.isFinite(value)) return "—";
    return Number(value).toFixed(digits);
  }

  function formatUtc(tsMs) {
    const date = new Date(Number(tsMs));
    if (Number.isNaN(date.getTime())) return "—";
    const pad = (num) => String(num).padStart(2, "0");
    const datePart = `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
    const timePart = `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
    return `${datePart} ${timePart}`;
  }

  function showToast(message, variant = "info", timeout = 4000) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast--${variant}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add("is-hidden");
      setTimeout(() => toast.remove(), 250);
    }, timeout);
  }

  function notifyStatus(message, variant = "info") {
    if (!statusEl) return;
    statusEl.textContent = message || "";
    statusEl.dataset.variant = variant;
    statusEl.hidden = !message;
  }

  async function fetchVersion() {
    if (!versionEl) return;
    try {
      const response = await fetch("/version");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      versionEl.textContent = typeof data.version === "string" ? data.version : "—";
    } catch (error) {
      console.warn("Failed to fetch version", error);
      versionEl.textContent = "—";
    }
  }

  async function fetchPreset(symbol) {
    if (!presetBadge) return;
    try {
      const response = await fetch(`/presets/${encodeURIComponent(symbol)}`);
      if (!response.ok) throw new Error("Preset request failed");
      const data = await response.json();
      const preset = data && data.preset ? data.preset : null;
      presetBadge.textContent = preset ? `Preset: ${preset.symbol || symbol}` : `Preset: ${symbol}`;
    } catch (error) {
      console.warn("Failed to load preset", error);
      presetBadge.textContent = `Preset: ${symbol}`;
    }
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const open = Number(bar.open ?? bar.o);
    const high = Number(bar.high ?? bar.h);
    const low = Number(bar.low ?? bar.l);
    const close = Number(bar.close ?? bar.c);
    const time = Number(bar.time ?? bar.t ?? 0);
    if (!Number.isFinite(time) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) {
      return null;
    }
    return {
      time,
      open,
      high,
      low,
      close,
      ts_ms_utc: time * 1000,
    };
  }

  function persistSharedCandles({ bars = null, reset = false, lastUpdateMs = null } = {}) {
    if (!SharedCandles || typeof SharedCandles.merge !== "function") return;
    const payload = Array.isArray(bars) && bars.length ? bars : state.candles;
    if (!Array.isArray(payload) || !payload.length) return;
    const intervalMs = intervalToMs(state.interval);
    const effectiveUpdate = Number.isFinite(lastUpdateMs) ? Number(lastUpdateMs) : state.lastUpdateMs;
    try {
      SharedCandles.merge(state.symbol, state.interval, payload, {
        intervalMs,
        lastUpdateMs: effectiveUpdate,
        maxBars: 2000,
        reset,
      });
    } catch (error) {
      console.warn("SharedCandles merge failed", error);
    }
  }

  function mergeCandles(bars, { reset = false, lastUpdateMs = null } = {}) {
    if (reset) state.candles = [];
    if (!Array.isArray(bars) || !bars.length) return false;
    const index = new Map();
    state.candles.forEach((bar, idx) => {
      index.set(Number(bar.time), idx);
    });
    let changed = false;
    bars.forEach((bar) => {
      if (!bar) return;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) return;
      if (index.has(time)) {
        state.candles[index.get(time)] = bar;
        changed = true;
      } else {
        index.set(time, state.candles.length);
        state.candles.push(bar);
        changed = true;
      }
    });
    if (changed) {
      state.candles.sort((a, b) => Number(a.time) - Number(b.time));
      if (state.candles.length > 2000) {
        state.candles = state.candles.slice(state.candles.length - 2000);
      }
      const effectiveUpdate = Number.isFinite(lastUpdateMs) ? Number(lastUpdateMs) : Date.now();
      state.lastUpdateMs = effectiveUpdate;
      persistSharedCandles({ reset, lastUpdateMs: effectiveUpdate });
    }
    return changed;
  }

  async function restoreFromSharedStore(symbol, interval) {
    let restored = false;
    let applied = false;

    if (SharedCandles && typeof SharedCandles.get === "function") {
      try {
        const stored = SharedCandles.get(symbol, interval);
        if (stored && Array.isArray(stored.candles) && stored.candles.length) {
          const bars = stored.candles.map((bar) => normaliseBar(bar)).filter(Boolean);
          if (bars.length) {
            const lastUpdate = Number(stored.lastUpdateMs) || Number(stored.updatedAt) || Date.now();
            mergeCandles(bars, { reset: true, lastUpdateMs: lastUpdate });
            restored = true;
            applied = true;
          }
        }
      } catch (error) {
        console.warn("SharedCandles restore failed", error);
      }
    }

    if (SharedCandles && typeof SharedCandles.fetchRemote === "function") {
      try {
        const remote = await SharedCandles.fetchRemote(symbol, interval);
        if (remote && Array.isArray(remote.candles) && remote.candles.length) {
          const lastUpdate = Number(remote.lastUpdateMs) || Number(remote.updatedAt) || Date.now();
          const shouldReset = !restored;
          mergeCandles(remote.candles.map((bar) => normaliseBar(bar)).filter(Boolean), {
            reset: shouldReset,
            lastUpdateMs: lastUpdate,
          });
          restored = true;
          applied = true;
        }
      } catch (error) {
        console.warn("SharedCandles remote restore failed", error);
      }
    }

    if (applied) {
      applyCandles();
    }
    return restored;
  }

  function updateInfo(lastBar) {
    const bar = lastBar || state.candles[state.candles.length - 1];
    if (!bar) {
      if (lastTimeEl) lastTimeEl.textContent = "—";
      if (lastPriceEl) lastPriceEl.textContent = "—";
      if (rangeEl) rangeEl.textContent = "—";
      return;
    }
    if (lastTimeEl) lastTimeEl.textContent = formatUtc(bar.ts_ms_utc);
    if (lastPriceEl) lastPriceEl.textContent = formatNumber(bar.close, 2);
    if (rangeEl) {
      const range = bar.high - bar.low;
      rangeEl.textContent = `${formatNumber(range, 2)} (${formatNumber((range / bar.low) * 100, 2)}%)`;
    }
  }

  function ensureOverlaySeries() {
    if (!state.chart || !state.candleSeries) return;
    if (!state.pocSeries) {
      state.pocSeries = state.chart.addLineSeries({ color: "#22c55e", lineWidth: 2, title: "POC" });
    }
    if (!state.vahSeries) {
      state.vahSeries = state.chart.addLineSeries({ color: "#ef4444", lineWidth: 2, title: "VAH" });
    }
    if (!state.valSeries) {
      state.valSeries = state.chart.addLineSeries({ color: "#ef4444", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dotted, title: "VAL" });
    }
  }

  function updateOverlayLevels(levels) {
    ensureOverlaySeries();
    if (!levels || !Array.isArray(state.candles) || !state.candles.length) {
      if (state.pocSeries) state.pocSeries.setData([]);
      if (state.vahSeries) state.vahSeries.setData([]);
      if (state.valSeries) state.valSeries.setData([]);
      return;
    }
    const lastTime = state.candles[state.candles.length - 1].time;
    const linePoints = [
      { time: lastTime - 10, value: levels.POC },
      { time: lastTime, value: levels.POC },
    ];
    if (state.pocSeries && Number.isFinite(levels.POC)) {
      state.pocSeries.setData(linePoints);
    }
    if (state.vahSeries && Number.isFinite(levels.VAH)) {
      state.vahSeries.setData([
        { time: lastTime - 10, value: levels.VAH },
        { time: lastTime, value: levels.VAH },
      ]);
    }
    if (state.valSeries && Number.isFinite(levels.VAL)) {
      state.valSeries.setData([
        { time: lastTime - 10, value: levels.VAL },
        { time: lastTime, value: levels.VAL },
      ]);
    }
  }

  function updatePriceLine(lastBar) {
    if (!state.candleSeries) return;
    const bar = lastBar || state.candles[state.candles.length - 1];
    if (!bar) return;
    if (!state.priceLine) {
      state.priceLine = state.candleSeries.createPriceLine({
        price: bar.close,
        color: "#60a5fa",
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: "last",
      });
    } else {
      state.priceLine.applyOptions({ price: bar.close });
    }
  }

  function applyCandles(lastBar) {
    if (!state.candleSeries) return;
    state.candleSeries.setData(state.candles);
    updatePriceLine(lastBar);
    updateInfo(lastBar);
  }

  function focusOnLastCandle({ preserveSpan = true } = {}) {
    if (!state.chart || !state.candles.length) return;
    const timeScale = state.chart.timeScale();
    const lastIndex = state.candles.length - 1;
    let span = 150;
    if (preserveSpan) {
      const logicalRange = timeScale.getVisibleLogicalRange();
      if (logicalRange) {
        const currentSpan = Number(logicalRange.to) - Number(logicalRange.from);
        if (Number.isFinite(currentSpan) && currentSpan > 0) {
          span = currentSpan;
        }
      }
    }
    const safeSpan = Math.max(25, Math.round(span));
    const padding = Math.max(2, Math.round(safeSpan * 0.05));
    const from = Math.max(0, lastIndex - safeSpan);
    const to = lastIndex + padding;
    timeScale.setVisibleLogicalRange({ from, to });
    timeScale.scrollToRealTime();
  }

  function detachWs() {
    if (state.ws) {
      state.ws.onopen = null;
      state.ws.onmessage = null;
      state.ws.onclose = null;
      state.ws.onerror = null;
      state.ws.close(1000);
    }
    state.ws = null;
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) return;
    const delay = Math.min(30_000, 1_000 * Math.pow(2, state.reconnectAttempts));
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connectWs();
    }, delay);
  }

  function handleWsMessage(event) {
    if (!state.candleSeries) return;
    try {
      const payload = JSON.parse(event.data);
      const bar = BinanceCandles.barFromWs(payload);
      const normalised = normaliseBar(bar);
      if (!normalised) return;
      const last = state.candles[state.candles.length - 1];
      const nowMs = Date.now();
      if (last && Number(last.time) === Number(normalised.time)) {
        state.candles[state.candles.length - 1] = normalised;
        state.candleSeries.update(normalised);
        state.lastUpdateMs = nowMs;
        persistSharedCandles({ bars: [normalised], lastUpdateMs: nowMs });
      } else if (!last || Number(normalised.time) > Number(last.time)) {
        state.candles.push(normalised);
        state.candleSeries.update(normalised);
        state.lastUpdateMs = nowMs;
        persistSharedCandles({ bars: [normalised], lastUpdateMs: nowMs });
      } else {
        mergeCandles([normalised], { lastUpdateMs: nowMs });
        state.candleSeries.setData(state.candles);
      }
      updatePriceLine(normalised);
      updateInfo(normalised);
      if (state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
        state.gapWatcher.notifyData();
      }
    } catch (error) {
      console.error("Failed to parse ws message", error);
    }
  }

  function connectWs() {
    detachWs();
    const symbol = state.symbol.toLowerCase();
    const interval = state.interval;
    const url = `wss://fstream.binance.com/ws/${symbol}@kline_${interval}`;
    const ws = new WebSocket(url);
    state.ws = ws;
    ws.onopen = () => {
      state.reconnectAttempts = 0;
      notifyStatus("Подключено к потоку Binance", "info");
    };
    ws.onmessage = handleWsMessage;
    ws.onerror = (event) => {
      console.error("WebSocket error", event);
      notifyStatus("Ошибка WebSocket, переподключение...", "warning");
      ws.close();
    };
    ws.onclose = () => {
      if (state.ws === ws) {
        state.reconnectAttempts += 1;
        notifyStatus("Соединение закрыто, переподключаемся...", "warning");
        scheduleReconnect();
      }
    };
  }

  async function fetchHistory(symbol, interval, limit = 500) {
    const rows = await BinanceCandles.fetchHistory(symbol, interval, limit);
    return rows.map((bar) => normaliseBar(bar)).filter(Boolean);
  }

  async function fetchRange(symbol, interval, startMs, endMs, limit = 1000) {
    const url = new URL("https://fapi.binance.com/fapi/v1/klines");
    url.searchParams.set("symbol", symbol);
    url.searchParams.set("interval", interval);
    if (Number.isFinite(startMs)) {
      url.searchParams.set("startTime", Math.floor(startMs));
    }
    if (Number.isFinite(endMs)) {
      url.searchParams.set("endTime", Math.floor(endMs));
    }
    url.searchParams.set("limit", String(Math.max(1, Math.min(limit, 1500))));
    const resp = await fetch(url.toString());
    if (!resp.ok) {
      throw new Error(`Failed to fetch gap candles: ${resp.status}`);
    }
    const data = await resp.json();
    return BinanceCandles.transformKlines(data).map((bar) => normaliseBar(bar)).filter(Boolean);
  }

  async function handleGapRequest(gap) {
    try {
      const intervalMs = intervalToMs(state.interval);
      const rangeWidth = Math.max(intervalMs, Number(gap.endMs) - Number(gap.startMs));
      const approxBars = Math.ceil(rangeWidth / intervalMs) + 2;
      const bars = await fetchRange(
        state.symbol,
        state.interval,
        Number(gap.startMs) - intervalMs,
        Number(gap.endMs) + intervalMs,
        Math.min(1000, Math.max(approxBars, 50))
      );
      const changed = mergeCandles(bars, { lastUpdateMs: Date.now() });
      if (changed) {
        applyCandles();
        if (state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
          state.gapWatcher.notifyData();
        }
      }
      return true;
    } catch (error) {
      console.error("Gap request failed", error);
      notifyStatus("Не удалось загрузить пропущенные свечи", "error");
      return false;
    }
  }

  function initChart() {
    if (state.chart) return;
    state.chart = LightweightCharts.createChart(chartContainer, {
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
    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#f8fafc",
      wickDownColor: "#f8fafc",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      borderVisible: true,
    });

    const resize = () => {
      const { clientWidth, clientHeight } = chartContainer;
      state.chart.applyOptions({ width: clientWidth, height: clientHeight });
    };
    resize();
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(resize);
      observer.observe(chartContainer);
    } else {
      window.addEventListener("resize", resize);
    }

    if (ChartGapWatcher && typeof ChartGapWatcher.attach === "function") {
      state.gapWatcher = ChartGapWatcher.attach({
        chart: state.chart,
        interval: state.interval,
        intervalMs: intervalToMs(state.interval),
        getCandles: () => state.candles,
        requestGap: handleGapRequest,
      });
    }
  }

  async function loadSymbol(symbol, interval) {
    detachWs();
    notifyStatus("Загружаем историю...", "info");
    const normalizedSymbol = symbol.trim().toUpperCase();
    const normalizedInterval = interval.trim();
    state.symbol = normalizedSymbol;
    state.interval = normalizedInterval;
    symbolInput.value = normalizedSymbol;
    setActiveSymbolButton(normalizedSymbol);
    fetchPreset(normalizedSymbol);
    initChart();
    const restored = await restoreFromSharedStore(normalizedSymbol, normalizedInterval);
    if (restored && state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
      state.gapWatcher.notifyData();
    }
    try {
      const history = await fetchHistory(normalizedSymbol, normalizedInterval, 1000);
      mergeCandles(history, { reset: true, lastUpdateMs: Date.now() });
      applyCandles();
      focusOnLastCandle({ preserveSpan: false });
      if (state.gapWatcher && typeof state.gapWatcher.updateContext === "function") {
        state.gapWatcher.updateContext({
          symbol: state.symbol,
          interval: state.interval,
          intervalMs: intervalToMs(state.interval),
          getCandles: () => state.candles,
          requestGap: handleGapRequest,
          resetRequestedKeys: true,
        });
        state.gapWatcher.notifyData();
      }
      notifyStatus("История загружена", "success");
      connectWs();
    } catch (error) {
      console.error("Failed to load history", error);
      notifyStatus("Не удалось загрузить данные Binance", "error");
    }
  }

  function renderDashboard(payload) {
    if (!dashboardEl) return;
    if (!payload) {
      dashboardEl.innerHTML = "<p class=\"empty\">Нет данных анализа</p>";
      return;
    }
    const sections = [];
    if (payload.ohlcv_multi) {
      const frames = Object.entries(payload.ohlcv_multi)
        .map(([tf, frame]) => `<li><strong>${tf}</strong>: ${(frame.candles || []).length} свечей</li>`) // summary
        .join("");
      sections.push(`<section><h3>OHLCV</h3><ul>${frames}</ul></section>`);
    }
    if (payload.orderflow && payload.orderflow.footprint) {
      const recent = payload.orderflow.footprint.slice(-3);
      const rows = recent
        .map((row) => `<tr><td>${row.t}</td><td>${formatNumber(row.price)}</td><td>${formatNumber(row.delta)}</td><td>${formatNumber(row.imbalance)}</td></tr>`)
        .join("");
      sections.push(`<section><h3>Footprint</h3><table><thead><tr><th>Время</th><th>Цена</th><th>Δ</th><th>Imb.</th></tr></thead><tbody>${rows}</tbody></table></section>`);
    }
    if (payload.derivatives) {
      const last = payload.derivatives[payload.derivatives.length - 1];
      if (last) {
        sections.push(
          `<section><h3>Derivatives</h3><p>OI: ${formatNumber(last.oi, 0)} | Funding: ${formatNumber(last.funding * 100, 4)}% | Basis: ${formatNumber(last.basis_bps, 2)} bps</p></section>`
        );
      }
    }
    if (payload.liquidity_map) {
      sections.push(
        `<section><h3>Liquidity Map</h3><p>PDH: ${formatNumber(payload.liquidity_map.PDH)} | PDL: ${formatNumber(payload.liquidity_map.PDL)}</p></section>`
      );
    }
    if (payload.news_events) {
      const items = payload.news_events
        .slice(-3)
        .map((event) => `<li><time>${event.time_utc}</time> — <span>${event.title}</span> (${event.impact})</li>`)
        .join("");
      sections.push(`<section><h3>Новости</h3><ul>${items}</ul></section>`);
    }
    dashboardEl.innerHTML = sections.join("");
  }

  function buildInspectionSnapshot() {
    const createdAt = Date.now();
    const id = `snap-${createdAt}-${Math.random().toString(36).slice(2, 8)}`;
    const candles = state.candles.map((bar) => {
      const timeMs = Number(bar.ts_ms_utc || bar.t || bar.time * 1000 || 0);
      const open = Number(bar.open ?? bar.o ?? 0);
      const high = Number(bar.high ?? bar.h ?? open);
      const low = Number(bar.low ?? bar.l ?? open);
      const close = Number(bar.close ?? bar.c ?? open);
      const volume = Number(bar.volume ?? bar.v ?? 0);
      return {
        t: Number.isFinite(timeMs) ? timeMs : 0,
        o: Number.isFinite(open) ? open : close,
        h: Number.isFinite(high) ? high : close,
        l: Number.isFinite(low) ? low : close,
        c: Number.isFinite(close) ? close : open,
        v: Math.max(0, Number.isFinite(volume) ? volume : 0),
      };
    });
    return {
      id,
      symbol: state.symbol,
      tf: state.interval,
      candles,
      lookback_days: 7,
      meta: {
        source: "chart-ui",
        generated_at: new Date(createdAt).toISOString(),
        candle_count: candles.length,
      },
    };
  }

  function toggleInspectionProgress(active) {
    if (!progressBar) return;
    if (active) {
      progressBar.value = 0;
      progressBar.classList.remove("hidden");
      let current = 0;
      clearInterval(state.inspectProgressTimer);
      state.inspectProgressTimer = setInterval(() => {
        current = Math.min(95, current + Math.random() * 7);
        progressBar.value = current;
      }, 250);
    } else {
      clearInterval(state.inspectProgressTimer);
      state.inspectProgressTimer = null;
      progressBar.value = 100;
      setTimeout(() => progressBar.classList.add("hidden"), 300);
    }
  }

  async function submitInspectionSnapshot() {
    const snapshot = buildInspectionSnapshot();
    toggleInspectionProgress(true);
    if (inspectionBtn) inspectionBtn.disabled = true;
    try {
      const response = await fetch("/inspection/snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(snapshot),
      });
      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload?.error || JSON.stringify(errorPayload));
      }
      const data = await response.json();
      const snapshotId = typeof data.snapshot_id === "string" ? data.snapshot_id : snapshot.id;
      state.lastSnapshotId = snapshotId;
      showToast("Snapshot создан", "success");
      await refreshDashboard(snapshotId);
    } catch (error) {
      console.error("Inspection snapshot failed", error);
      showToast(`Ошибка инспекции: ${error.message || error}`, "error", 6000);
    } finally {
      toggleInspectionProgress(false);
      if (inspectionBtn) inspectionBtn.disabled = false;
    }
  }

  async function refreshDashboard(snapshotId) {
    try {
      const url = new URL("/inspection", window.location.origin);
      url.searchParams.set("snapshot", snapshotId);
      const response = await fetch(url.toString(), { headers: { accept: "application/json" } });
      if (!response.ok) throw new Error("Inspection payload unavailable");
      const payload = await response.json();
      const data = payload?.DATA || {};
      const snapshotData = {
        ohlcv_multi: data.ohlcv ?? null,
        orderflow: data.orderflow ?? null,
        derivatives: data.derivatives ?? null,
        liquidity_map: data.liquidity_map ?? null,
        news_events: data.news_events ?? null,
      };
      renderDashboard(snapshotData);
      const tpo = data?.tpo?.sessions || [];
      const pocEntry = data?.tpo?.daily?.[0];
      if (pocEntry) {
        updateOverlayLevels({ POC: pocEntry.POC, VAH: pocEntry.VAH, VAL: pocEntry.VAL });
      }
    } catch (error) {
      console.warn("Failed to refresh dashboard", error);
      renderDashboard(null);
    }
  }

  async function analyzeSnapshot(type) {
    if (!state.lastSnapshotId) {
      showToast("Сначала создайте snapshot", "warning");
      return;
    }
    try {
      const response = await fetch("/inspection/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snapshot_id: state.lastSnapshotId, analysis_type: type }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.error || "Analyze failed");
      }
      const data = await response.json();
      if (type === "tpo" && data?.tpo?.daily?.length) {
        const last = data.tpo.daily[data.tpo.daily.length - 1];
        updateOverlayLevels({ POC: last.POC, VAH: last.VAH, VAL: last.VAL });
      }
      renderDashboard({
        ohlcv_multi: null,
        orderflow: null,
        derivatives: null,
        liquidity_map: type === "liquidity" ? data?.liquidity_map : null,
        news_events: null,
      });
      showToast("Анализ завершен", "success");
    } catch (error) {
      console.error("Analyze request failed", error);
      showToast(`Ошибка анализа: ${error.message || error}`, "error", 6000);
    }
  }

  async function exportZones() {
    if (!state.lastSnapshotId) {
      showToast("Нет данных для экспорта", "warning");
      return;
    }
    try {
      const url = new URL("/profile", window.location.origin);
      url.searchParams.set("snapshot", state.lastSnapshotId);
      url.searchParams.set("tf", state.interval);
      const response = await fetch(url.toString());
      if (!response.ok) throw new Error("Profile export failed");
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `${state.symbol}_${state.interval}_profile.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(downloadUrl);
      showToast("Профиль экспортирован", "success");
    } catch (error) {
      console.error("Export failed", error);
      showToast(`Экспорт не удался: ${error.message || error}`, "error", 6000);
    }
  }

  function openModal() {
    if (!modalEl) return Promise.resolve(false);
    modalEl.classList.remove("hidden");
    return new Promise((resolve) => {
      const cleanup = () => {
        modalEl.classList.add("hidden");
        modalConfirmBtn?.removeEventListener("click", onConfirm);
        modalCancelBtn?.removeEventListener("click", onCancel);
      };
      const onConfirm = () => {
        cleanup();
        resolve(true);
      };
      const onCancel = () => {
        cleanup();
        resolve(false);
      };
      modalConfirmBtn?.addEventListener("click", onConfirm, { once: true });
      modalCancelBtn?.addEventListener("click", onCancel, { once: true });
    });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const symbol = symbolInput.value || "BTCUSDT";
    const interval = intervalInput.value || "1m";
    loadSymbol(symbol, interval);
  });

  if (symbolButtons.length) {
    symbolButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        const nextSymbol = (button.dataset.symbolOption || "").toUpperCase();
        if (!nextSymbol) return;
        if (state.symbol === nextSymbol) {
          symbolInput.value = nextSymbol;
          return;
        }
        loadSymbol(nextSymbol, state.interval);
      });
    });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.ws) {
      connectWs();
    }
  });

  if (inspectionBtn) {
    inspectionBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      const confirmed = await openModal();
      if (!confirmed) return;
      submitInspectionSnapshot();
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", (event) => {
      event.preventDefault();
      analyzeSnapshot("tpo");
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", (event) => {
      event.preventDefault();
      exportZones();
    });
  }

  fetchVersion();
  fetchPreset(state.symbol);
  setActiveSymbolButton(state.symbol);
  loadSymbol(state.symbol, state.interval);
})();
