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
  const analyzeBtn = document.getElementById("run-session-analysis");
  const collectBtn = document.getElementById("collect-context");
  const exportBtn = document.getElementById("export-zones");
  const analysisPanel = document.getElementById("analysis-panel");
  const sessionContentEl = document.getElementById("session-content");
  const contextSectionEl = document.getElementById("context-analysis");
  const contextContentEl = document.getElementById("context-content");
  const analysisStatusEl = document.getElementById("analysis-status");
  const copyAnalysisBtn = document.getElementById("copy-analysis");
  const downloadAnalysisBtn = document.getElementById("download-analysis");
  const sessionCoverageBadge = document.getElementById("session-coverage-badge");
  const sessionSourceBadge = document.getElementById("session-source-badge");
  const ctxCoverageBadge = document.getElementById("ctx-coverage-badge");
  const ctxSourceBadge = document.getElementById("ctx-source-badge");
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
    sessionPayload: null,
    contextPayload: null,
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

  function summariseTouchEvents(events, recentTail = 3) {
    if (!Array.isArray(events) || events.length === 0) {
      return {
        events: [],
        stats: {
          total: 0,
          filled: 0,
          wick_only: 0,
          first_ms: null,
          last_ms: null,
          max_depth: null,
          last_kind: null,
          sampled: 0,
          truncated: false,
        },
      };
    }

    const normalised = events
      .map((entry) => {
        const tsCandidates = [entry.ts_ms, entry.ts, entry.timestamp, entry.t, entry.time];
        let tsValue = null;
        for (const candidate of tsCandidates) {
          const numeric = Number(candidate);
          if (Number.isFinite(numeric)) {
            tsValue = numeric;
            break;
          }
        }
        if (!Number.isFinite(tsValue)) return null;
        const rawKind = String(entry.kind ?? entry.touch_kind ?? entry.status ?? "").toLowerCase();
        const kind = rawKind.includes("fill") ? "filled" : rawKind.includes("wick") ? "wick_only" : rawKind || "wick_only";
        const depthCandidates = [entry.depth, entry.penetration, entry.fill_pct, entry.filled_pct];
        let depthValue = null;
        for (const candidate of depthCandidates) {
          if (candidate == null) continue;
          const numeric = Number(candidate);
          if (Number.isFinite(numeric)) {
            depthValue = Math.max(0, Math.min(1, numeric));
            break;
          }
        }
        return {
          ts_ms: tsValue,
          kind,
          depth: Number.isFinite(depthValue) ? depthValue : null,
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.ts_ms - b.ts_ms);

    if (!normalised.length) {
      return {
        events: [],
        stats: {
          total: 0,
          filled: 0,
          wick_only: 0,
          first_ms: null,
          last_ms: null,
          max_depth: null,
          last_kind: null,
          sampled: 0,
          truncated: false,
        },
      };
    }

    const total = normalised.length;
    const filled = normalised.reduce((acc, item) => (item.kind === "filled" ? acc + 1 : acc), 0);
    const wickOnly = total - filled;
    const depths = normalised.map((item) => item.depth).filter((value) => Number.isFinite(value));
    const maxDepth = depths.length ? Math.max(...depths) : null;

    const keypoints = new Map();

    const mark = (event, label) => {
      const existing = keypoints.get(event.ts_ms);
      if (!existing) {
        keypoints.set(event.ts_ms, {
          ts_ms: event.ts_ms,
          kind: event.kind,
          depth: event.depth,
          labels: [label],
        });
        return;
      }
      if (!existing.labels.includes(label)) {
        existing.labels.push(label);
      }
      if (existing.kind !== "filled" && event.kind === "filled") {
        existing.kind = event.kind;
      }
      if (Number.isFinite(event.depth)) {
        const currentDepth = Number(existing.depth);
        if (!Number.isFinite(currentDepth) || event.depth > currentDepth) {
          existing.depth = event.depth;
        }
      }
    };

    const firstEvent = normalised[0];
    const lastEvent = normalised[normalised.length - 1];
    mark(firstEvent, "first");
    mark(lastEvent, "last");

    const filledIndexes = normalised.reduce((indexes, item, index) => {
      if (item.kind === "filled") {
        indexes.push(index);
      }
      return indexes;
    }, []);
    if (filledIndexes.length) {
      mark(normalised[filledIndexes[0]], "filled_first");
      mark(normalised[filledIndexes[filledIndexes.length - 1]], "filled_last");
    }

    if (maxDepth != null) {
      let deepestIndex = 0;
      let deepestDepth = -Infinity;
      normalised.forEach((item, index) => {
        const depth = Number(item.depth);
        if (Number.isFinite(depth) && depth >= deepestDepth) {
          deepestDepth = depth;
          deepestIndex = index;
        }
      });
      mark(normalised[deepestIndex], "max_depth");
    }

    if (recentTail > 0) {
      const tailCount = Math.max(1, parseInt(recentTail, 10) || 1);
      const tailEvents = normalised.slice(-tailCount);
      tailEvents.forEach((event) => mark(event, "recent"));
    }

    const samples = Array.from(keypoints.values()).sort((a, b) => a.ts_ms - b.ts_ms);
    samples.forEach((item) => {
      if (Array.isArray(item.labels)) {
        item.labels = Array.from(new Set(item.labels)).sort();
      }
    });

    return {
      events: samples,
      stats: {
        total,
        filled,
        wick_only: wickOnly,
        first_ms: firstEvent.ts_ms,
        last_ms: lastEvent.ts_ms,
        max_depth: maxDepth != null ? Number(maxDepth.toFixed(4)) : null,
        last_kind: lastEvent.kind,
        sampled: samples.length,
        truncated: total > samples.length,
      },
    };
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
    if (!SharedCandles || typeof SharedCandles.merge !== "function") {
      return;
    }
    const payload = Array.isArray(bars)
      ? bars.map((bar) => toSharedBar(bar)).filter(Boolean)
      : [];
    try {
      SharedCandles.merge(state.symbol, state.interval, payload, {
        intervalMs: intervalToMs(state.interval),
        lastUpdateMs,
        maxBars: 2000,
        reset,
        immediate,
      });
      if (immediate && SharedCandles.flush) {
        SharedCandles.flush(state.symbol, state.interval, { force: true }).catch((error) => {
          console.warn("SharedCandles flush failed", error);
        });
      }
    } catch (error) {
      console.warn("SharedCandles merge failed", error);
    }
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
    const previousSymbol = state.symbol;
    const previousInterval = state.interval;
    if (SharedCandles && typeof SharedCandles.flush === "function") {
      SharedCandles.flush(previousSymbol, previousInterval, { force: true }).catch((error) => {
        console.warn("SharedCandles flush before switch failed", error);
      });
    }
    const normalizedSymbol = symbol.trim().toUpperCase();
    const normalizedInterval = interval.trim();
    state.symbol = normalizedSymbol;
    state.interval = normalizedInterval;
    symbolInput.value = normalizedSymbol;
    setActiveSymbolButton(normalizedSymbol);
    fetchPreset(normalizedSymbol);
    initChart();
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
    dashboardEl.innerHTML = sections.join("");
  }

  function toggleInspectionProgress() {
    clearInterval(state.inspectProgressTimer);
    state.inspectProgressTimer = null;
    if (progressBar) {
      progressBar.value = 0;
      progressBar.classList.add("hidden");
    }
  }

  async function submitInspectionSnapshot() {
    toggleInspectionProgress(false);
    showToast("Инспекции отключены в этой версии", "warn");
  }

  async function refreshDashboard() {
    renderDashboard(null);
  }

  async function analyzeSnapshot() {
    showToast("Инспекции отключены в этой версии", "warn");
  }

  async function exportZones() {
    showToast("Экспорт профиля недоступен", "warn");
  }

  function openModal() {
    showToast("Инспекции отключены в этой версии", "warn");
    return Promise.resolve(false);
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
  function showAnalysisPanel() {
    if (!analysisPanel) return;
    analysisPanel.hidden = false;
  }

  const ANALYSIS_RETRY_LIMIT = 1;
  const ANALYSIS_RETRY_DELAY_MS = 2500;

  function renderCoverageBadge(badgeEl, coverage, label = "coverage") {
    if (!badgeEl) return;
    const pct = Number(coverage);
    if (!Number.isFinite(pct)) {
      badgeEl.textContent = `${label}: —`;
      badgeEl.classList.remove("badge--warn");
      return;
    }
    const normalized = pct > 1 ? pct / 100 : pct;
    const valuePct = (normalized * 100).toFixed(2);
    badgeEl.textContent = `${label}: ${valuePct}%`;
    badgeEl.classList.toggle("badge--warn", normalized < 0.99);
  }

  function renderSourceBadge(badgeEl, source) {
    if (!badgeEl) return;
    badgeEl.textContent = source || "live";
  }

  function renderSessionPayload(payload) {
    if (!sessionContentEl) return;
    if (!payload || typeof payload !== "object" || payload.status !== "ok") {
      sessionContentEl.innerHTML = '<p class="error">Нет данных по сессии.</p>';
      return;
    }

    const metrics = payload.metrics || {};
    const delta = metrics.delta || {};
    const micro = metrics.microstructure || {};
    const levels = metrics.levels || {};

    const rows = [];
    rows.push(`<div><span class="metric-label">Сессия:</span> <span>${payload.session_used || "—"}</span></div>`);
    rows.push(`<div><span class="metric-label">Диапазон:</span> <span>${formatNumber(metrics.range)}</span></div>`);
    rows.push(`<div><span class="metric-label">ATR(14):</span> <span>${formatNumber(metrics.ATR)}</span></div>`);
    rows.push(`<div><span class="metric-label">VWAP:</span> <span>${formatNumber(metrics.VWAP)}</span></div>`);
    rows.push(`<div><span class="metric-label">RVOL:</span> <span>${formatNumber(metrics.RVOL)}</span></div>`);
    rows.push(`<div><span class="metric-label">Delta Σ:</span> <span>${formatNumber(delta.sum)}</span></div>`);
    rows.push(`<div><span class="metric-label">CVD:</span> <span>${formatNumber(delta.cvd)}</span></div>`);
    if (levels.ib_high != null && levels.ib_low != null) {
      rows.push(
        `<div><span class="metric-label">IB диапазон:</span> <span>${formatNumber(levels.ib_low)} → ${formatNumber(
          levels.ib_high
        )}</span></div>`
      );
    }
    if (levels.pdh != null || levels.pdl != null) {
      rows.push(
        `<div><span class="metric-label">PDH / PDL:</span> <span>${formatNumber(levels.pdl)} → ${formatNumber(
          levels.pdh
        )}</span></div>`
      );
    }
    rows.push(
      `<div><span class="metric-label">Импульсы:</span> <span>${Array.isArray(delta.impulses) ? delta.impulses.length : 0
      }</span></div>`
    );
    rows.push(
      `<div><span class="metric-label">Basis p50:</span> <span>${formatNumber(micro.basis_p50)}</span></div>`
    );
    rows.push(
      `<div><span class="metric-label">Funding:</span> <span>${formatNumber(micro.funding_hint)}</span></div>`
    );

    sessionContentEl.innerHTML = `<div class="metrics-grid">${rows.join("")}</div>`;

    const coverageValue = payload.window?.coverage_pct;
    renderCoverageBadge(sessionCoverageBadge, coverageValue);
    const sources = Array.isArray(payload.window?.source_seq) ? payload.window.source_seq : [];
    renderSourceBadge(sessionSourceBadge, sources.length ? sources.join(" → ") : "live");
  }

  function renderContextPayload(payload) {
    if (!contextContentEl || !contextSectionEl) return;
    contextSectionEl.hidden = false;
    if (!payload || typeof payload !== "object" || payload.status !== "ok") {
      contextContentEl.innerHTML = '<p class="error">Контекст недоступен.</p>';
      return;
    }

    const aggregates = payload.aggregates || {};
    const biasBlock = aggregates.bias && typeof aggregates.bias === "object" ? aggregates.bias : null;
    const vwapContext = aggregates.vwap_context && typeof aggregates.vwap_context === "object" ? aggregates.vwap_context : null;
    const tpoContext = aggregates.tpo_context && typeof aggregates.tpo_context === "object" ? aggregates.tpo_context : null;
    const sessionsContext = aggregates.sessions && typeof aggregates.sessions === "object" ? aggregates.sessions : null;
    const biasMarkup = biasBlock
      ? Object.entries(biasBlock)
          .map(([tf, info]) => {
            const direction = String(info.direction || "—").toUpperCase();
            const change = Number.isFinite(info.change_pct) ? Number(info.change_pct).toFixed(2) : "—";
            const confidence = Number.isFinite(info.confidence) ? (Number(info.confidence) * 100).toFixed(1) : "—";
            return `<li><span class="badge badge--outline">${tf}</span> <strong>${direction}</strong> Δ ${change}% · Conf ${confidence}%</li>`;
          })
          .join("")
      : "";

    const dailyBlock = vwapContext && typeof vwapContext.daily === "object" ? vwapContext.daily : null;
    const sessionsBlock = vwapContext && typeof vwapContext.sessions === "object" ? vwapContext.sessions : null;

    const vwapDailyMarkup = dailyBlock && Object.values(dailyBlock).some((value) => value != null)
      ? (() => {
          const pairs = [
            ["VWAP", dailyBlock.vwap],
            ["POC", dailyBlock.poc],
            ["VAH", dailyBlock.vah],
            ["VAL", dailyBlock.val],
            ["SD1+", dailyBlock.sd1_plus],
            ["SD1-", dailyBlock.sd1_minus],
            ["SD2+", dailyBlock.sd2_plus],
            ["SD2-", dailyBlock.sd2_minus],
          ];
          const items = pairs
            .filter(([, value]) => value != null)
            .map(([label, value]) => `<li><span class="badge badge--muted">${label}</span> ${formatNumber(value)}</li>`)
            .join("");
          const dateLabel = dailyBlock.date ? `<p class="counts">${dailyBlock.date}</p>` : "";
          return `<div><h5>Daily</h5>${dateLabel}<ol class="list-tight">${items || '<li>Нет данных</li>'}</ol></div>`;
        })()
      : "";

    const vwapSessionsMarkup = sessionsBlock
      ? Object.entries(sessionsBlock)
          .map(([name, info]) => {
            if (!info || typeof info !== "object") {
              return `<li><span class="badge badge--muted">${name}</span> —</li>`;
            }
            const label = name.charAt(0).toUpperCase() + name.slice(1);
            const corePairs = [
              ["VWAP", info.vwap],
              ["POC", info.poc],
              ["VAH", info.vah],
              ["VAL", info.val],
            ];
            const rangePairs = [
              ["High", info.high],
              ["Low", info.low],
              ["IBH", info.ib_high],
              ["IBL", info.ib_low],
            ];
            const list = [...corePairs, ...rangePairs]
              .filter(([, value]) => value != null)
              .map(([labelKey, value]) => `<span class="badge badge--outline">${labelKey}</span> ${formatNumber(value)}`)
              .join(" · ");
            return `<li><strong>${label}</strong>: ${list || "—"}</li>`;
          })
          .join("")
      : "";

    const vwapSection = vwapDailyMarkup || vwapSessionsMarkup
      ? `<section>
            <h4>VWAP / TPO</h4>
            ${vwapDailyMarkup}
            ${vwapSessionsMarkup ? `<ol class="list-tight">${vwapSessionsMarkup}</ol>` : ""}
         </section>`
      : "";

    const tpoDays = tpoContext && Array.isArray(tpoContext.days) ? tpoContext.days : [];
    const tpoMarkup = tpoDays.length
      ? `<section>
            <h4>TPO Days</h4>
            <ol class="list-tight">
              ${tpoDays
                .map((entry) => {
                  const date = entry.date || "—";
                  const poc = formatNumber(entry.poc);
                  const vah = formatNumber(entry.vah);
                  const val = formatNumber(entry.val);
                  return `<li><span class="badge badge--muted">${date}</span> POC ${poc} · VAH ${vah} · VAL ${val}</li>`;
                })
                .join("")}
            </ol>
          </section>`
      : "";

    const sessionLast = sessionsContext && typeof sessionsContext.last_closed === "object" ? sessionsContext.last_closed : null;
    const sessionMarkup = sessionLast
      ? `<section>
            <h4>Last Session (UTC)</h4>
            <p class="counts">${formatUtc(sessionLast.session_start_ms)} → ${formatUtc(sessionLast.session_end_ms)}</p>
            <ol class="list-tight">
              <li><span class="badge badge--muted">High</span> ${formatNumber(sessionLast.high)}</li>
              <li><span class="badge badge--muted">Low</span> ${formatNumber(sessionLast.low)}</li>
              <li><span class="badge badge--muted">IB High</span> ${formatNumber(sessionLast.ib_high)}</li>
              <li><span class="badge badge--muted">IB Low</span> ${formatNumber(sessionLast.ib_low)}</li>
            </ol>
          </section>`
      : "";

    const zonesRaw = Array.isArray(payload.zones_top) ? payload.zones_top : [];
    const zones = zonesRaw.map((zone) => {
      const baseEvents = Array.isArray(zone.touches) ? zone.touches : [];
      const hasStats = zone.touch_stats && typeof zone.touch_stats === "object";
      let touchStats;
      let events;
      if (hasStats) {
        touchStats = { ...zone.touch_stats };
        const normalisedEvents = baseEvents
          .map((event) => {
            const tsCandidates = [event.ts_ms, event.ts, event.timestamp, event.t, event.time];
            let tsValue = null;
            for (const candidate of tsCandidates) {
              const numeric = Number(candidate);
              if (Number.isFinite(numeric)) {
                tsValue = numeric;
                break;
              }
            }
            if (!Number.isFinite(tsValue)) return null;
            const rawKind = String(event.kind ?? event.touch_kind ?? "").toLowerCase();
            const kind = rawKind.includes("fill") ? "filled" : rawKind.includes("wick") ? "wick_only" : rawKind || "wick_only";
            const depthNumeric = Number(event.depth);
            const depth = Number.isFinite(depthNumeric) ? depthNumeric : null;
            const labels = Array.isArray(event.labels) ? Array.from(new Set(event.labels)) : [];
            return {
              ts_ms: tsValue,
              kind,
              depth,
              labels,
            };
          })
          .filter(Boolean)
          .sort((a, b) => a.ts_ms - b.ts_ms);
        events = normalisedEvents;
        if (typeof touchStats.total !== "number" && Number.isFinite(Number(touchStats.total))) {
          touchStats.total = Number(touchStats.total);
        }
        if (typeof touchStats.filled !== "number" && Number.isFinite(Number(touchStats.filled))) {
          touchStats.filled = Number(touchStats.filled);
        }
        if (typeof touchStats.wick_only !== "number" && Number.isFinite(Number(touchStats.wick_only))) {
          touchStats.wick_only = Number(touchStats.wick_only);
        }
        if (typeof touchStats.max_depth !== "number" && Number.isFinite(Number(touchStats.max_depth))) {
          touchStats.max_depth = Number(touchStats.max_depth);
        }
      } else {
        const summarised = summariseTouchEvents(baseEvents);
        touchStats = summarised.stats;
        events = summarised.events;
      }
      let touchCount = Number(zone.touch_count);
      if (!Number.isFinite(touchCount) || touchCount < 0) {
        touchCount = Number.isFinite(touchStats.total) ? Number(touchStats.total) : baseEvents.length;
      }
      return {
        ...zone,
        touch_count: touchCount,
        touch_stats: touchStats,
        touches: events,
      };
    });

    const zonesMarkup = zones
      .map((zone) => {
        const strength = typeof zone.strength === "number" ? zone.strength.toFixed(2) : "—";
        const priceLo = formatNumber(zone.price_lo);
        const priceHi = formatNumber(zone.price_hi);
        const stats = zone.touch_stats && typeof zone.touch_stats === "object" ? zone.touch_stats : {};
        const totalTouches = Number.isFinite(stats.total) ? Number(stats.total) : Number(zone.touch_count) || 0;
        const filledTouches = Number.isFinite(stats.filled) ? Number(stats.filled) : 0;
        const touchesBadge =
          totalTouches > 0
            ? `<span class="badge">T=${totalTouches} · F=${filledTouches}</span>`
            : `<span class="badge badge--muted">T=0</span>`;
        const truncatedBadge = stats.truncated ? `<span class="badge badge--outline">compact</span>` : "";
        return `<li><span class="badge badge--muted">${zone.type}</span> <strong>${priceLo} → ${priceHi}</strong> <span class="badge">S=${strength}</span> <span class="badge badge--outline">${zone.status}</span> ${touchesBadge} ${truncatedBadge}</li>`;
      })
      .join("");

    const touches = zones.flatMap((zone) => {
      const events = Array.isArray(zone.touches) ? zone.touches : [];
      const stats = zone.touch_stats && typeof zone.touch_stats === "object" ? zone.touch_stats : {};
      const total = Number.isFinite(Number(stats.total)) ? Number(stats.total) : events.length;
      const truncatedFlag = typeof stats.truncated === "string" ? stats.truncated === "true" : Boolean(stats.truncated);
      const truncated = Boolean(truncatedFlag && total > events.length);
      return events.map((touch) => ({
        id: zone.id,
        total,
        truncated,
        ...touch,
      }));
    });
    const touchesMarkup = touches
      .map((touch) => {
        const depthPct = touch.depth != null ? (Number(touch.depth) * 100).toFixed(1) : "—";
        const labels = Array.isArray(touch.labels) && touch.labels.length ? ` <span class="badge badge--muted">${touch.labels.join(" · ")}</span>` : "";
        const tsLabel = touch.ts_ms != null ? ` <span class="badge badge--muted">${formatUtc(touch.ts_ms)}</span>` : "";
        return `<li><code>${touch.id}</code> — ${touch.kind} (${depthPct}%)${labels}${tsLabel}</li>`;
      })
      .join("");
    const truncatedList = zones
      .filter((zone) => {
        const stats = zone.touch_stats;
        if (!stats || typeof stats !== "object") return false;
        return typeof stats.truncated === "string" ? stats.truncated === "true" : Boolean(stats.truncated);
      })
      .map((zone) => `<code>${zone.id}</code>`)
      .join(", ");
    const touchNote = truncatedList
      ? `<p class="counts">Компактный режим для зон: ${truncatedList}</p>`
      : "";

    const counts = payload.counts || {};

    contextContentEl.innerHTML = `
      <div class="grid-two">
        <section>
          <h4>Top зоны</h4>
          <p class="counts">FVG: ${counts.fvg ?? 0} · OB: ${counts.ob ?? 0} · Others: ${counts.others ?? 0}</p>
          <ol class="list-tight">${zonesMarkup || '<li>Нет активных зон</li>'}</ol>
        </section>
        <section>
          <h4>Касания</h4>
          <ol class="list-tight">${touchesMarkup || '<li>Нет касаний в последней сессии</li>'}</ol>
          ${touchNote}
        </section>
        ${biasMarkup
          ? `<section>
              <h4>Bias</h4>
              <ol class="list-tight">${biasMarkup}</ol>
            </section>`
          : ""}
        ${vwapSection}
        ${tpoMarkup}
        ${sessionMarkup}
      </div>
    `;

    renderCoverageBadge(ctxCoverageBadge, payload.window?.coverage_pct);
    const sources = Array.isArray(payload.window?.source_seq) ? payload.window.source_seq : [];
    renderSourceBadge(ctxSourceBadge, sources.length ? sources.join(" → ") : "live");
  }

  function updateAnalysisStatus(message, variant = "info") {
    if (!analysisStatusEl) return;
    analysisStatusEl.textContent = message || "";
    analysisStatusEl.dataset.variant = variant;
  }

  function serializeAnalysis() {
    return JSON.stringify(
      {
        session: state.sessionPayload,
        context: state.contextPayload,
      },
      null,
      2
    );
  }

  async function copyAnalysis() {
    if (!state.sessionPayload && !state.contextPayload) {
      showToast("Нет данных для копирования", "warn");
      return;
    }
    try {
      await navigator.clipboard.writeText(serializeAnalysis());
      showToast("JSON скопирован", "success");
    } catch (error) {
      console.warn("copy failed", error);
      showToast("Не удалось скопировать", "error");
    }
  }

  function downloadAnalysis() {
    if (!state.sessionPayload && !state.contextPayload) {
      showToast("Нет данных для скачивания", "warn");
      return;
    }
    const blob = new Blob([serializeAnalysis()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${state.symbol}_analysis.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function runSessionAnalysis(retry = 0) {
    const symbol = state.symbol;
    showAnalysisPanel();
    const hadPrevious = Boolean(state.sessionPayload);
    if (!hadPrevious && sessionContentEl) {
      sessionContentEl.innerHTML = '<p class="placeholder">Загружаем сессию…</p>';
    }
    if (retry === 0) {
      updateAnalysisStatus("Получаем данные сессии…", "info");
    }
    try {
      const response = await fetch(`/analyze/session?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      if (payload?.status === "retry" && retry < ANALYSIS_RETRY_LIMIT) {
        updateAnalysisStatus("Еще секундочку…", "info");
        setTimeout(() => {
          void runSessionAnalysis(retry + 1);
        }, ANALYSIS_RETRY_DELAY_MS);
        return;
      }
      if (!payload || payload.status !== "ok") {
        throw new Error(payload?.error || payload?.reason || "session_unavailable");
      }
      state.sessionPayload = payload;
      renderSessionPayload(payload);
      updateAnalysisStatus("Сессия готова", "success");
    } catch (error) {
      console.error("session analysis failed", error);
      if (!hadPrevious) {
        state.sessionPayload = null;
        if (sessionContentEl) {
          sessionContentEl.innerHTML = `<p class="error">Ошибка анализа сессии: ${error.message || error}</p>`;
        }
      }
      updateAnalysisStatus("Ошибка анализа сессии", "error");
    }
  }

  async function runContextAnalysis(retry = 0) {
    const symbol = state.symbol;
    showAnalysisPanel();
    const hadPrevious = Boolean(state.contextPayload);
    if (!hadPrevious && contextContentEl) {
      contextContentEl.innerHTML = '<p class="placeholder">Контекст загружается…</p>';
    }
    if (retry === 0) {
      updateAnalysisStatus("Собираем контекст 72 ч…", "info");
    }
    try {
      const response = await fetch(`/context/72h?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      if (payload?.status === "retry" && retry < ANALYSIS_RETRY_LIMIT) {
        updateAnalysisStatus("Еще секундочку…", "info");
        setTimeout(() => {
          void runContextAnalysis(retry + 1);
        }, ANALYSIS_RETRY_DELAY_MS);
        return;
      }
      if (!payload || payload.status !== "ok") {
        throw new Error(payload?.error || payload?.reason || "context_unavailable");
      }
      state.contextPayload = payload;
      renderContextPayload(payload);
      updateAnalysisStatus("Контекст обновлён", "success");
    } catch (error) {
      console.error("context analysis failed", error);
      if (!hadPrevious) {
        state.contextPayload = null;
        if (contextContentEl) {
          contextContentEl.innerHTML = `<p class="error">Ошибка загрузки контекста: ${error.message || error}</p>`;
        }
      }
      updateAnalysisStatus("Контекст недоступен", "warn");
    }
  }
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", () => {
      void runSessionAnalysis();
    });
  }

  if (collectBtn) {
    collectBtn.addEventListener("click", () => {
      void runContextAnalysis();
    });
  }

  if (copyAnalysisBtn) {
    copyAnalysisBtn.addEventListener("click", copyAnalysis);
  }

  if (downloadAnalysisBtn) {
    downloadAnalysisBtn.addEventListener("click", downloadAnalysis);
  }

})();
