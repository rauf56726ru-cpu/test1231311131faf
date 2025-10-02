(function () {
  "use strict";

  const LightweightCharts = window.LightweightCharts;
  const BinanceCandles = window.BinanceCandles;
  const ChartGapWatcher = window.ChartGapWatcher;
  const SharedCandles = window.SharedCandles;
  const MarketDataStore = window.MarketDataStore || null;

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
    gapWatcher: null,
    lastUpdateMs: null,
    pocSeries: null,
    vahSeries: null,
    valSeries: null,
    inspectProgressTimer: null,
    lastSnapshotId: null,
    shared: {
      pending: new Map(),
      resetPending: false,
      lastUpdateMs: null,
      flushTimer: null,
      inFlight: false,
      dirty: false,
      lastFlushMs: 0,
      flushPromise: null,
    },
  };

  const marketStore =
    MarketDataStore &&
    new MarketDataStore({
      symbol: state.symbol,
      interval: state.interval,
      pollIntervalMs: 1500,
      historyLimit: 1500,
    });

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
      const response = await fetch("/version", { cache: "no-store" });
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
      const response = await fetch(`/presets/${encodeURIComponent(symbol)}`, { cache: "no-store" });
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

  function toSharedBar(bar) {
    if (!bar) return null;
    const time = Number(bar.time ?? bar.t ?? 0);
    const open = Number(bar.open ?? bar.o);
    const high = Number(bar.high ?? bar.h);
    const low = Number(bar.low ?? bar.l);
    const close = Number(bar.close ?? bar.c);
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }
    return { time, open, high, low, close };
  }

  function persistSharedCandles({ bars = null, reset = false, lastUpdateMs = null, immediate = false } = {}) {
    const shared = state.shared;
    if (reset) {
      shared.pending = new Map();
      shared.resetPending = true;
    }
    const entries = Array.isArray(bars) ? bars : [];
    for (const bar of entries) {
      const sharedBar = toSharedBar(bar);
      if (!sharedBar) continue;
      shared.pending.set(sharedBar.time, sharedBar);
    }
    if (Number.isFinite(lastUpdateMs)) {
      shared.lastUpdateMs = Number(lastUpdateMs);
    }
    if (shared.pending.size || shared.resetPending) {
      scheduleSharedFlush({ immediate: immediate || reset });
    }
  }

  function scheduleSharedFlush({ immediate = false, waitMs = null, force = false } = {}) {
    const shared = state.shared;
    const now = Date.now();
    const sinceLast = now - (shared.lastFlushMs || 0);
    let delay;
    if (Number.isFinite(waitMs)) {
      delay = Math.max(0, waitMs);
    } else {
      const baseDelay = Math.max(0, 1000 - sinceLast);
      delay = immediate ? baseDelay : Math.max(250, baseDelay);
    }
    if (delay === 0 && shared.flushTimer) {
      clearTimeout(shared.flushTimer);
      shared.flushTimer = null;
    }
    if (shared.flushTimer) {
      return;
    }
    shared.flushTimer = setTimeout(() => {
      shared.flushTimer = null;
      flushSharedCandles({ force }).catch((error) => {
        console.warn("SharedCandles flush failed", error);
      });
    }, delay);
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
  }

  async function flushSharedCandles({ force = false } = {}) {
    const shared = state.shared;
    if (!shared.pending.size && !shared.resetPending) {
      return;
    }
    if (shared.inFlight) {
      shared.dirty = true;
      return shared.flushPromise || Promise.resolve();
    }

    const execute = async () => {
      const now = Date.now();
      const sinceLast = now - (shared.lastFlushMs || 0);
      if (!force && sinceLast < 1000) {
        scheduleSharedFlush({ waitMs: 1000 - sinceLast });
        return;
      }
      if (force && sinceLast < 1000) {
        await wait(1000 - sinceLast);
      }
      if (shared.flushTimer) {
        clearTimeout(shared.flushTimer);
        shared.flushTimer = null;
      }
      const pendingEntries = Array.from(shared.pending.values());
      const reset = shared.resetPending;
      const lastUpdateMs = Number.isFinite(shared.lastUpdateMs) ? Number(shared.lastUpdateMs) : Date.now();
      shared.pending = new Map();
      shared.resetPending = false;
      shared.inFlight = true;
      shared.dirty = false;

      const payload = pendingEntries
        .map((bar) => ({ ...bar }))
        .sort((a, b) => Number(a.time) - Number(b.time));

      if (SharedCandles && typeof SharedCandles.merge === "function") {
        try {
          SharedCandles.merge(state.symbol, state.interval, payload, {
            intervalMs: intervalToMs(state.interval),
            lastUpdateMs,
            maxBars: 2000,
            reset,
            syncRemote: false,
          });
        } catch (error) {
          console.warn("SharedCandles local merge failed", error);
        }
      }

      try {
        const response = await fetch("/shared-candles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
          body: JSON.stringify({
            symbol: state.symbol,
            interval: state.interval,
            candles: payload,
            reset,
            intervalMs: intervalToMs(state.interval),
            lastUpdateMs,
            maxBars: 2000,
          }),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        shared.lastFlushMs = Date.now();
        shared.lastUpdateMs = lastUpdateMs;
      } catch (error) {
        console.warn("SharedCandles remote sync failed", error);
        const buffer = shared.pending;
        if (reset) {
          buffer.clear();
        }
        for (const bar of payload) {
          buffer.set(Number(bar.time), { ...bar });
        }
        shared.resetPending = reset || shared.resetPending;
        shared.dirty = true;
      } finally {
        shared.inFlight = false;
        if (!shared.dirty) {
          shared.lastFlushMs = shared.lastFlushMs || Date.now();
        }
        if (shared.dirty || shared.pending.size) {
          scheduleSharedFlush();
        }
      }
    };

    shared.flushPromise = execute().finally(() => {
      shared.flushPromise = null;
    });
    return shared.flushPromise;
  }

  function mergeCandles(bars, { reset = false, lastUpdateMs = null, immediate = false } = {}) {
    if (reset) state.candles = [];
    if (!Array.isArray(bars) || !bars.length) return false;
    const index = new Map();
    state.candles.forEach((bar, idx) => {
      index.set(Number(bar.time), idx);
    });
    let changed = false;
    const delta = new Map();
    bars.forEach((bar) => {
      if (!bar) return;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) return;
      if (index.has(time)) {
        state.candles[index.get(time)] = bar;
        changed = true;
        delta.set(time, bar);
      } else {
        index.set(time, state.candles.length);
        state.candles.push(bar);
        changed = true;
        delta.set(time, bar);
      }
    });
    if (changed) {
      state.candles.sort((a, b) => Number(a.time) - Number(b.time));
      if (state.candles.length > 2000) {
        state.candles = state.candles.slice(state.candles.length - 2000);
      }
      const effectiveUpdate = Number.isFinite(lastUpdateMs) ? Number(lastUpdateMs) : Date.now();
      state.lastUpdateMs = effectiveUpdate;
      const changedBars = Array.from(delta.values());
      persistSharedCandles({ bars: changedBars, reset, lastUpdateMs: effectiveUpdate, immediate });
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
          mergeCandles(bars, { reset: true, lastUpdateMs: lastUpdate, immediate: true });
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
            immediate: shouldReset,
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


  function refreshGapWatcherContext() {
    if (!state.gapWatcher || typeof state.gapWatcher.updateContext !== "function") return;
    state.gapWatcher.updateContext({
      symbol: state.symbol,
      interval: state.interval,
      intervalMs: intervalToMs(state.interval),
      getCandles: () => state.candles,
      requestGap: handleGapRequest,
      resetRequestedKeys: false,
    });
    if (typeof state.gapWatcher.notifyData === "function") {
      state.gapWatcher.notifyData();
    }
  }

  function handleStoreEvent(event) {
    if (!event) return;
    const eventSymbol = (event.symbol || "").toUpperCase();
    if (eventSymbol && eventSymbol !== state.symbol) return;
    if (event.interval && event.interval !== state.interval) return;

    if (event.type === "status") {
      if (event.status === "connected") {
        notifyStatus("Подключено к потоку Binance", "info");
      } else if (event.status === "closed") {
        notifyStatus("Соединение закрыто, активирован пуллинг", "warning");
      } else if (event.status === "error") {
        notifyStatus("Ошибка потока, переподключение...", "warning");
      }
      return;
    }

    if (event.type === "error") {
      notifyStatus("Ошибка загрузки данных Binance", "error");
      return;
    }

    const now = Date.now();

    if (event.type === "snapshot") {
      const bars = Array.isArray(event.candles) ? event.candles : [];
      const changed = mergeCandles(bars, { reset: true, lastUpdateMs: now, immediate: true });
      if (changed) {
        applyCandles();
        focusOnLastCandle({ preserveSpan: false });
        refreshGapWatcherContext();
      }
      notifyStatus("История загружена", "success");
    } else if (event.type === "update") {
      const payload = event.candle ? [event.candle] : [];
      const changed = mergeCandles(payload, { reset: false, lastUpdateMs: now, immediate: Boolean(event.isFinal) });
      if (changed) {
        applyCandles(event.candle);
        refreshGapWatcherContext();
      }
    } else if (event.type === "poll") {
      const bars = Array.isArray(event.candles) ? event.candles : [];
      if (mergeCandles(bars, { reset: false, lastUpdateMs: now, immediate: false })) {
        applyCandles();
      }
    }

    if (event.meta && event.meta.last_price != null) {
      updateInfo(event.candle || marketStore?.getLastCandle() || null);
    }
  }

  if (marketStore) {
    marketStore.subscribe(handleStoreEvent);
    marketStore.start();
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
    const resp = await fetch(url.toString(), { cache: "no-store" });
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
      const changed = mergeCandles(bars, { lastUpdateMs: Date.now(), immediate: true });
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
    notifyStatus("Загружаем историю...", "info");
    const normalizedSymbol = symbol.trim().toUpperCase();
    const normalizedInterval = interval.trim();
    state.symbol = normalizedSymbol;
    state.interval = normalizedInterval;
    symbolInput.value = normalizedSymbol;
    setActiveSymbolButton(normalizedSymbol);
    fetchPreset(normalizedSymbol);
    initChart();
    if (state.shared.flushTimer) {
      clearTimeout(state.shared.flushTimer);
      state.shared.flushTimer = null;
    }
    state.shared.pending = new Map();
    state.shared.resetPending = false;
    state.shared.lastUpdateMs = null;
    state.shared.inFlight = false;
    state.shared.dirty = false;
    state.shared.lastFlushMs = 0;
    state.shared.flushPromise = null;
    const restored = await restoreFromSharedStore(normalizedSymbol, normalizedInterval);
    if (restored) {
      applyCandles();
      focusOnLastCandle({ preserveSpan: false });
      refreshGapWatcherContext();
    }
    if (marketStore) {
      marketStore.setSymbol(normalizedSymbol, normalizedInterval);
      marketStore.restart();
    } else {
      notifyStatus("Лайв-хранилище недоступно", "error");
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
    await flushSharedCandles({ force: true });
    const snapshot = buildInspectionSnapshot();
    toggleInspectionProgress(true);
    if (inspectionBtn) inspectionBtn.disabled = true;
    try {
      const response = await fetch("/inspection/snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
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
      const response = await fetch(url.toString(), {
        headers: { accept: "application/json" },
        cache: "no-store",
      });
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
        cache: "no-store",
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
      const response = await fetch(url.toString(), { cache: "no-store" });
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
    if (document.visibilityState === "visible" && marketStore) {
      marketStore.restart();
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
