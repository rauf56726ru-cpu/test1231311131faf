(function () {
  "use strict";

  const LightweightCharts = window.LightweightCharts;
  const BinanceCandles = window.BinanceCandles;
  const ChartGapWatcher = window.ChartGapWatcher;

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

  function mergeCandles(bars, { reset = false } = {}) {
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
    }
    return changed;
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
      if (last && Number(last.time) === Number(normalised.time)) {
        state.candles[state.candles.length - 1] = normalised;
        state.candleSeries.update(normalised);
      } else if (!last || Number(normalised.time) > Number(last.time)) {
        state.candles.push(normalised);
        state.candleSeries.update(normalised);
      } else {
        mergeCandles([normalised]);
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
      const changed = mergeCandles(bars);
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
    initChart();
    try {
      const history = await fetchHistory(normalizedSymbol, normalizedInterval, 1000);
      mergeCandles(history, { reset: true });
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

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.ws) {
      connectWs();
    }
  });

  fetchVersion();
  loadSymbol(state.symbol, state.interval);
})();
