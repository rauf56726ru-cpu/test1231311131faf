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
  const selectionLabelEl = document.getElementById("selection-label");
  const clearSelectionBtn = document.getElementById("clear-selection");

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
    gapWatcher: null,
    lastUpdateMs: null,
    selectionStartMs: null,
    selectionEndMs: null,
    lastClosedMinute: null,
    lastClosedTimer: null,
  };

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
    return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(
      date.getUTCMinutes()
    )}:${pad(date.getUTCSeconds())}`;
  }

  function timeParamToMs(timeParam) {
    if (typeof timeParam === "number" && Number.isFinite(timeParam)) {
      return Number(timeParam) * 1000;
    }
    if (timeParam && typeof timeParam === "object") {
      const year = Number(timeParam.year);
      const month = Number(timeParam.month);
      const day = Number(timeParam.day);
      if (Number.isFinite(year) && Number.isFinite(month) && Number.isFinite(day)) {
        return Date.UTC(year, month - 1, day);
      }
    }
    return null;
  }

  function formatSelectionLabel(startMs, endMs) {
    const hasStart = Number.isFinite(startMs);
    const hasEnd = Number.isFinite(endMs);
    if (hasStart && hasEnd) {
      const start = Math.min(startMs, endMs);
      const end = Math.max(startMs, endMs);
      return `${formatUtc(start)} → ${formatUtc(end)}`;
    }
    if (hasStart) {
      return `${formatUtc(startMs)} → …`;
    }
    return "—";
  }

  function updateSelectionLabel() {
    if (!selectionLabelEl) return;
    selectionLabelEl.textContent = formatSelectionLabel(state.selectionStartMs, state.selectionEndMs);
  }

  function emitSelectionChange() {
    let detail = null;
    const hasStart = Number.isFinite(state.selectionStartMs);
    const hasEnd = Number.isFinite(state.selectionEndMs);
    if (hasStart && hasEnd) {
      detail = {
        start: Math.min(state.selectionStartMs, state.selectionEndMs),
        end: Math.max(state.selectionStartMs, state.selectionEndMs),
      };
    } else if (hasStart) {
      detail = { start: state.selectionStartMs, end: null };
    }
    try {
      document.dispatchEvent(
        new CustomEvent("chart:selection-change", {
          detail,
        })
      );
    } catch (error) {
      console.warn("Selection event dispatch failed", error);
    }
  }

  function resetSelection() {
    state.selectionStartMs = null;
    state.selectionEndMs = null;
    updateSelectionLabel();
    emitSelectionChange();
  }

  function registerSelectionPoint(tsMs) {
    if (!Number.isFinite(tsMs)) return;
    if (!Number.isFinite(state.selectionStartMs) || Number.isFinite(state.selectionEndMs)) {
      state.selectionStartMs = tsMs;
      state.selectionEndMs = null;
    } else {
      state.selectionEndMs = tsMs;
    }

    if (
      Number.isFinite(state.selectionStartMs) &&
      Number.isFinite(state.selectionEndMs) &&
      state.selectionEndMs < state.selectionStartMs
    ) {
      const tmp = state.selectionStartMs;
      state.selectionStartMs = state.selectionEndMs;
      state.selectionEndMs = tmp;
    }

    updateSelectionLabel();
    emitSelectionChange();
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
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const version = data && typeof data.version === "string" ? data.version : null;
      versionEl.textContent = version || "—";
    } catch (error) {
      console.warn("Failed to fetch version", error);
      versionEl.textContent = "—";
    }
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const open = Number(bar.open);
    const high = Number(bar.high);
    const low = Number(bar.low);
    const close = Number(bar.close);
    const time = Number(bar.time);
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
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

  function restoreFromSharedStore(symbol, interval) {
    if (!SharedCandles || typeof SharedCandles.get !== "function") {
      return false;
    }
    try {
      const stored = SharedCandles.get(symbol, interval);
      if (!stored || !Array.isArray(stored.candles) || !stored.candles.length) {
        return false;
      }
      const bars = stored.candles.map((bar) => normaliseBar(bar)).filter(Boolean);
      if (!bars.length) {
        return false;
      }
      const lastUpdate = Number(stored.lastUpdateMs) || Number(stored.updatedAt) || Date.now();
      mergeCandles(bars, { reset: true, lastUpdateMs: lastUpdate });
      applyCandles();
      return true;
    } catch (error) {
      console.warn("SharedCandles restore failed", error);
      return false;
    }
  }

  function updateInfo(lastBar) {
    let payload = null;

    if (state.lastClosedMinute) {
      const close = Number(state.lastClosedMinute.close);
      const high = Number(state.lastClosedMinute.high);
      const low = Number(state.lastClosedMinute.low);
      const closeTimeMs = Number.isFinite(state.lastClosedMinute.closeTimeMs)
        ? Number(state.lastClosedMinute.closeTimeMs)
        : Number(state.lastClosedMinute.openTimeMs) + 60_000;
      if (
        Number.isFinite(close) &&
        Number.isFinite(high) &&
        Number.isFinite(low) &&
        Number.isFinite(closeTimeMs)
      ) {
        payload = { close, high, low, timeMs: closeTimeMs };
      }
    }

    if (!payload) {
      const bar = lastBar || state.candles[state.candles.length - 1];
      if (bar) {
        const close = Number(bar.close);
        const high = Number(bar.high);
        const low = Number(bar.low);
        const timeMs = Number.isFinite(bar.ts_ms_utc)
          ? Number(bar.ts_ms_utc)
          : Number.isFinite(bar.time)
          ? Number(bar.time) * 1000
          : null;
        if (Number.isFinite(close) && Number.isFinite(high) && Number.isFinite(low) && Number.isFinite(timeMs)) {
          payload = { close, high, low, timeMs };
        }
      }
    }

    if (!payload) {
      if (lastTimeEl) lastTimeEl.textContent = "—";
      if (lastPriceEl) lastPriceEl.textContent = "—";
      if (rangeEl) rangeEl.textContent = "—";
      return;
    }

    if (lastTimeEl) lastTimeEl.textContent = formatUtc(payload.timeMs);
    if (lastPriceEl) lastPriceEl.textContent = formatNumber(payload.close, 2);
    if (rangeEl) {
      const range = payload.high - payload.low;
      const percent = Number.isFinite(payload.low) && payload.low !== 0 ? (range / payload.low) * 100 : null;
      rangeEl.textContent = Number.isFinite(percent)
        ? `${formatNumber(range, 2)} (${formatNumber(percent, 2)}%)`
        : `${formatNumber(range, 2)}`;
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
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connectWs();
    }, 3000);
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
    const url = `wss://stream.binance.com:9443/ws/${symbol}@kline_${interval}`;
    const ws = new WebSocket(url);
    state.ws = ws;
    ws.onopen = () => {
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

  function handleChartClick(param) {
    if (!param) return;
    let targetMs = null;
    if (param.seriesData && typeof param.seriesData.get === "function" && state.candleSeries) {
      const seriesPoint = param.seriesData.get(state.candleSeries);
      if (seriesPoint) {
        if (Number.isFinite(seriesPoint.ts_ms_utc)) {
          targetMs = Number(seriesPoint.ts_ms_utc);
        } else if (Number.isFinite(seriesPoint.time)) {
          targetMs = Number(seriesPoint.time) * 1000;
        }
      }
    }
    if (!Number.isFinite(targetMs)) {
      targetMs = timeParamToMs(param.time);
    }
    if (Number.isFinite(targetMs)) {
      registerSelectionPoint(targetMs);
    }
  }

  function stopLastClosedPolling() {
    if (state.lastClosedTimer) {
      clearTimeout(state.lastClosedTimer);
      state.lastClosedTimer = null;
    }
  }

  async function fetchLastClosedMinute(symbol) {
    if (!symbol) return null;
    const url = new URL("https://api.binance.com/api/v3/klines");
    url.searchParams.set("symbol", symbol);
    url.searchParams.set("interval", "1m");
    url.searchParams.set("limit", "2");
    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Failed to fetch last closed candle: ${response.status}`);
    }
    const rows = await response.json();
    if (!Array.isArray(rows) || !rows.length) {
      return null;
    }
    const lastRow = rows[rows.length - 1];
    const useArray = Array.isArray(lastRow) ? lastRow : null;
    const openTime = useArray ? Number(useArray[0]) : Number(lastRow?.openTime ?? lastRow?.open_time);
    const open = useArray ? Number(useArray[1]) : Number(lastRow?.open);
    const high = useArray ? Number(useArray[2]) : Number(lastRow?.high);
    const low = useArray ? Number(useArray[3]) : Number(lastRow?.low);
    const close = useArray ? Number(useArray[4]) : Number(lastRow?.close);
    const closeTime = useArray ? Number(useArray[6]) : Number(lastRow?.closeTime ?? lastRow?.close_time);
    if (
      !Number.isFinite(openTime) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }
    const closeTimeMs = Number.isFinite(closeTime) ? closeTime : openTime + 60_000;
    return {
      open,
      high,
      low,
      close,
      openTimeMs: openTime,
      closeTimeMs,
    };
  }

  async function updateLastClosedFromApi() {
    try {
      const info = await fetchLastClosedMinute(state.symbol);
      if (info) {
        state.lastClosedMinute = info;
        updateInfo();
      }
    } catch (error) {
      console.warn("Failed to refresh last closed candle", error);
    }
  }

  function scheduleLastClosedPolling() {
    stopLastClosedPolling();
    const tick = async () => {
      await updateLastClosedFromApi();
      const now = Date.now();
      const msUntilNext = 60_000 - (now % 60_000) + 750;
      const delay = Math.min(Math.max(msUntilNext, 20_000), 80_000);
      state.lastClosedTimer = setTimeout(tick, delay);
    };
    void tick();
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

    state.chart.subscribeClick(handleChartClick);
  }

  async function loadSymbol(symbol, interval) {
    stopLastClosedPolling();
    detachWs();
    notifyStatus("Загружаем историю...", "info");
    const normalizedSymbol = symbol.trim().toUpperCase();
    const normalizedInterval = interval.trim();
    state.symbol = normalizedSymbol;
    state.interval = normalizedInterval;
    initChart();
    state.candles = [];
    if (state.candleSeries) {
      state.candleSeries.setData([]);
    }
    state.lastClosedMinute = null;
    resetSelection();
    updateInfo();
    const restored = restoreFromSharedStore(normalizedSymbol, normalizedInterval);
    if (restored && state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
      state.gapWatcher.notifyData();
    }
    scheduleLastClosedPolling();
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

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const symbol = symbolInput.value || "BTCUSDT";
    const interval = intervalInput.value || "1m";
    loadSymbol(symbol, interval);
  });

  if (clearSelectionBtn) {
    clearSelectionBtn.addEventListener("click", () => {
      resetSelection();
    });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.ws) {
      connectWs();
    }
  });

  fetchVersion();
  loadSymbol(state.symbol, state.interval);
})();

(function () {
  "use strict";

  const STORAGE_KEYS = {
    prompt: "copilot.systemPrompt",
    settings: "copilot.chatSettings",
  };

  const elements = {
    chatWindow: document.getElementById("chat-window"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    promptTextarea: document.getElementById("system-prompt"),
    savePrompt: document.getElementById("save-prompt"),
    resetPrompt: document.getElementById("reset-prompt"),
    settingsModal: document.getElementById("settings-modal"),
    openSettings: document.getElementById("open-settings"),
    closeSettings: document.getElementById("close-settings"),
    cancelSettings: document.getElementById("cancel-settings"),
    settingsForm: document.getElementById("settings-form"),
    settingsApiKey: document.getElementById("settings-api-key"),
    settingsModel: document.getElementById("settings-model"),
    settingsTemperature: document.getElementById("settings-temperature"),
    settingsTopP: document.getElementById("settings-top-p"),
    settingsApiBase: document.getElementById("settings-api-base"),
    toast: document.getElementById("toast"),
    refreshCheckAll: document.getElementById("refresh-check-all"),
    createTestEnv: document.getElementById("create-test-env"),
    checkAllOutput: document.getElementById("check-all-output"),
    requestTrade: document.getElementById("request-trade"),
    tradeOutput: document.getElementById("trade-output"),
  };

  const state = {
    defaultPrompt: "",
    systemPrompt: "",
    settings: {
      model: "",
      temperature: 0.2,
      top_p: null,
      api_base: null,
      api_key: "",
    },
    chat: [],
    loadingChat: false,
    loadingCheckAll: false,
    loadingTrade: false,
    selection: null,
  };

  function showToast(message, variant = "info") {
    if (!elements.toast) return;
    elements.toast.textContent = message;
    elements.toast.dataset.variant = variant;
    elements.toast.classList.add("visible");
    window.setTimeout(() => {
      elements.toast.classList.remove("visible");
    }, 2800);
  }

  async function fetchJSON(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = "";
      try {
        const payload = await response.json();
        detail = payload?.detail || payload?.message || JSON.stringify(payload);
      } catch (error) {
        detail = await response.text();
      }
      throw new Error(detail || `Запрос завершился с ошибкой ${response.status}`);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function loadSettings() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEYS.settings);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        state.settings.model = typeof parsed.model === "string" ? parsed.model : state.settings.model;
        state.settings.temperature = Number.isFinite(parsed.temperature)
          ? Number(parsed.temperature)
          : state.settings.temperature;
        state.settings.top_p = Number.isFinite(parsed.top_p) ? Number(parsed.top_p) : null;
        state.settings.api_base = typeof parsed.api_base === "string" && parsed.api_base ? parsed.api_base : null;
        state.settings.api_key = typeof parsed.api_key === "string" ? parsed.api_key : "";
      }
    } catch (error) {
      console.warn("Не удалось прочитать сохранённые настройки", error);
    }
  }

  function persistSettings() {
    const payload = {
      model: state.settings.model,
      temperature: state.settings.temperature,
      top_p: state.settings.top_p,
      api_base: state.settings.api_base,
      api_key: state.settings.api_key,
    };
    try {
      window.localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(payload));
    } catch (error) {
      console.warn("Не удалось сохранить настройки", error);
    }
  }

  function loadPrompt() {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEYS.prompt);
      if (typeof saved === "string" && saved.trim()) {
        state.systemPrompt = saved;
      }
    } catch (error) {
      console.warn("Не удалось прочитать промпт", error);
    }
  }

  function persistPrompt() {
    try {
      window.localStorage.setItem(STORAGE_KEYS.prompt, state.systemPrompt);
    } catch (error) {
      console.warn("Не удалось сохранить промпт", error);
    }
  }

  function renderPrompt() {
    if (elements.promptTextarea) {
      elements.promptTextarea.value = state.systemPrompt;
    }
  }

  function renderChat() {
    if (!elements.chatWindow) return;
    elements.chatWindow.innerHTML = "";
    if (!state.chat.length) {
      const empty = document.createElement("p");
      empty.className = "chat-placeholder";
      empty.textContent = "Сообщений пока нет";
      elements.chatWindow.append(empty);
      return;
    }
    for (const message of state.chat) {
      const node = document.createElement("div");
      node.className = `chat-message ${message.role}`;
      const role = document.createElement("div");
      role.className = "role";
      role.textContent = message.role === "user" ? "Вы" : "Модель";
      node.append(role);

      const content = document.createElement("div");
      if (message.isJson) {
        const pre = document.createElement("pre");
        pre.textContent = message.content;
        content.append(pre);
      } else {
        const paragraph = document.createElement("p");
        paragraph.textContent = message.content;
        content.append(paragraph);
      }
      node.append(content);
      if (message.pending) {
        node.classList.add("pending");
      }
      elements.chatWindow.append(node);
    }
    elements.chatWindow.scrollTop = elements.chatWindow.scrollHeight;
  }

  function renderSettingsForm() {
    if (!elements.settingsForm) return;
    elements.settingsApiKey.value = state.settings.api_key;
    elements.settingsModel.value = state.settings.model;
    elements.settingsTemperature.value = state.settings.temperature;
    elements.settingsTopP.value = state.settings.top_p ?? "";
    elements.settingsApiBase.value = state.settings.api_base ?? "";
  }

  function formatJson(input) {
    if (!input) return "";
    try {
      const parsed = typeof input === "string" ? JSON.parse(input) : input;
      return JSON.stringify(parsed, null, 2);
    } catch (error) {
      return typeof input === "string" ? input : JSON.stringify(input);
    }
  }

  function toIsoString(ms) {
    if (!Number.isFinite(ms)) return null;
    const dt = new Date(Number(ms));
    if (Number.isNaN(dt.getTime())) return null;
    return dt.toISOString();
  }

  function updateSelectionState(detail) {
    if (detail && Number.isFinite(detail.start) && Number.isFinite(detail.end)) {
      state.selection = { start: Number(detail.start), end: Number(detail.end) };
    } else if (detail && Number.isFinite(detail.start)) {
      state.selection = { start: Number(detail.start), end: null };
    } else {
      state.selection = null;
    }
  }

  function setLoading(target, loading) {
    const button = elements[target];
    if (!button) return;
    button.disabled = loading;
    button.classList.toggle("is-loading", loading);
  }

  async function initialiseDefaults() {
    try {
      const defaults = await fetchJSON("/api/chat/defaults");
      if (defaults) {
        state.defaultPrompt = defaults.system_prompt || "";
        if (!state.systemPrompt) {
          state.systemPrompt = state.defaultPrompt;
        }
        if (defaults.settings) {
          state.settings.model = defaults.settings.model || state.settings.model;
          if (typeof defaults.settings.temperature === "number") {
            state.settings.temperature = defaults.settings.temperature;
          }
          if (typeof defaults.settings.top_p === "number") {
            state.settings.top_p = defaults.settings.top_p;
          }
        }
      }
    } catch (error) {
      console.warn("Не удалось загрузить настройки по умолчанию", error);
    }
  }

  function addMessage(role, content, { pending = false } = {}) {
    const isJson = (() => {
      if (typeof content !== "string") return false;
      const trimmed = content.trim();
      return trimmed.startsWith("{") || trimmed.startsWith("[");
    })();
    state.chat.push({ role, content, pending, isJson });
    renderChat();
  }

  function updatePendingMessage(content) {
    for (let idx = state.chat.length - 1; idx >= 0; idx -= 1) {
      const pending = state.chat[idx];
      if (pending && pending.pending) {
        pending.pending = false;
        pending.content = content;
        const trimmed = String(content || "").trim();
        pending.isJson = trimmed.startsWith("{") || trimmed.startsWith("[");
        break;
      }
    }
    renderChat();
  }

  async function sendChatMessage(text) {
    if (state.loadingChat) return;
    const trimmed = text.trim();
    if (!trimmed) return;
    if (!state.settings.api_key) {
      showToast("Укажите OpenAI API key в настройках", "warning");
      return;
    }
    state.loadingChat = true;
    addMessage("user", trimmed);
    addMessage("assistant", "Ожидание ответа...", { pending: true });
    renderChat();

    const payload = {
      system_prompt: state.systemPrompt,
      messages: state.chat.filter((item) => !item.pending).map((item) => ({
        role: item.role,
        content: item.content,
      })),
      settings: {
        model: state.settings.model,
        temperature: state.settings.temperature,
        top_p: state.settings.top_p,
        api_base: state.settings.api_base,
      },
      api_key: state.settings.api_key,
    };

    try {
      const data = await fetchJSON("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      updatePendingMessage(data?.reply || "Ответ без содержимого");
    } catch (error) {
      updatePendingMessage(`Ошибка: ${error.message || error}`);
    } finally {
      state.loadingChat = false;
    }
  }

  async function refreshCheckAll() {
    if (state.loadingCheckAll) return;
    state.loadingCheckAll = true;
    setLoading("refreshCheckAll", true);
    try {
      const data = await fetchJSON("/api/check-all");
      const formatted = formatJson(data?.check_all);
      elements.checkAllOutput.textContent = formatted || "Нет данных";
    } catch (error) {
      elements.checkAllOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingCheckAll = false;
      setLoading("refreshCheckAll", false);
    }
  }

  async function createTestEnvironment() {
    if (state.loadingCheckAll) return;
    const selection = state.selection;
    const hasStart = selection && Number.isFinite(selection.start);
    const hasEnd = selection && Number.isFinite(selection.end);
    if (!hasStart || !hasEnd) {
      showToast("Выделите на графике две свечи для тестовой среды", "warning");
      return;
    }

    const startMs = Math.min(Number(selection.start), Number(selection.end));
    const endMs = Math.max(Number(selection.start), Number(selection.end));
    const startIso = toIsoString(startMs);
    const endIso = toIsoString(endMs);
    if (!startIso || !endIso) {
      showToast("Не удалось преобразовать диапазон в дату", "error");
      return;
    }

    const symbolInputEl = document.getElementById("input-symbol");
    const intervalInputEl = document.getElementById("input-interval");
    const payload = {
      symbol: symbolInputEl?.value?.trim()?.toUpperCase() || "BTCUSDT",
      timeframe: intervalInputEl?.value || "1m",
      start: startIso,
      end: endIso,
    };
    state.loadingCheckAll = true;
    setLoading("createTestEnv", true);
    try {
      const data = await fetchJSON("/api/test-environment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const formatted = formatJson(data?.check_all);
      elements.checkAllOutput.textContent = formatted || "Нет данных";
      if (data?.rendered_html) {
        const popup = window.open("", "_blank", "noopener");
        if (popup) {
          popup.document.write(data.rendered_html);
          popup.document.close();
        } else {
          showToast("Браузер заблокировал новое окно", "warning");
        }
      }
    } catch (error) {
      elements.checkAllOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingCheckAll = false;
      setLoading("createTestEnv", false);
    }
  }

  async function requestTrade() {
    if (state.loadingTrade) return;
    if (!state.settings.api_key) {
      showToast("Укажите OpenAI API key в настройках", "warning");
      return;
    }
    state.loadingTrade = true;
    setLoading("requestTrade", true);
    try {
      const payload = {
        api_key: state.settings.api_key,
        model: state.settings.model,
        system_prompt: state.systemPrompt,
        api_base: state.settings.api_base,
      };
      const data = await fetchJSON("/api/trade-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      elements.tradeOutput.textContent = formatJson(data);
      if (data?.trade_json) {
        addMessage("assistant", JSON.stringify(data.trade_json, null, 2));
      }
    } catch (error) {
      elements.tradeOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingTrade = false;
      setLoading("requestTrade", false);
    }
  }

  function openSettingsModal() {
    elements.settingsModal?.classList.remove("hidden");
    renderSettingsForm();
  }

  function closeSettingsModal() {
    elements.settingsModal?.classList.add("hidden");
  }

  async function init() {
    await initialiseDefaults();
    loadSettings();
    loadPrompt();
    if (!state.systemPrompt) {
      state.systemPrompt = state.defaultPrompt;
    }
    renderPrompt();
    renderSettingsForm();
    renderChat();
    refreshCheckAll();
  }

  if (elements.savePrompt) {
    elements.savePrompt.addEventListener("click", () => {
      state.systemPrompt = elements.promptTextarea?.value || "";
      persistPrompt();
      showToast("Промпт сохранён");
    });
  }

  if (elements.resetPrompt) {
    elements.resetPrompt.addEventListener("click", () => {
      state.systemPrompt = state.defaultPrompt;
      renderPrompt();
      persistPrompt();
      showToast("Промпт сброшен к значениям по умолчанию");
    });
  }

  if (elements.chatForm && elements.chatInput) {
    elements.chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const value = elements.chatInput.value;
      elements.chatInput.value = "";
      sendChatMessage(value);
    });
  }

  if (elements.openSettings) {
    elements.openSettings.addEventListener("click", () => {
      renderSettingsForm();
      openSettingsModal();
    });
  }

  if (elements.closeSettings) {
    elements.closeSettings.addEventListener("click", closeSettingsModal);
  }

  if (elements.cancelSettings) {
    elements.cancelSettings.addEventListener("click", closeSettingsModal);
  }

  if (elements.settingsForm) {
    elements.settingsForm.addEventListener("submit", (event) => {
      event.preventDefault();
      state.settings.api_key = elements.settingsApiKey.value.trim();
      state.settings.model = elements.settingsModel.value.trim() || state.settings.model;
      state.settings.temperature = Number(elements.settingsTemperature.value) || state.settings.temperature;
      const topPValue = elements.settingsTopP.value;
      state.settings.top_p = topPValue === "" ? null : Number(topPValue);
      const apiBaseValue = elements.settingsApiBase.value.trim();
      state.settings.api_base = apiBaseValue ? apiBaseValue : null;
      persistSettings();
      closeSettingsModal();
      showToast("Настройки обновлены");
    });
  }

  if (elements.refreshCheckAll) {
    elements.refreshCheckAll.addEventListener("click", refreshCheckAll);
  }

  if (elements.createTestEnv) {
    elements.createTestEnv.addEventListener("click", createTestEnvironment);
  }

  if (elements.requestTrade) {
    elements.requestTrade.addEventListener("click", requestTrade);
  }

  document.addEventListener("chart:selection-change", (event) => {
    updateSelectionState(event.detail);
  });

  window.addEventListener("DOMContentLoaded", init);
})();
