const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const BinanceAdapter = typeof window !== "undefined" ? window.BinanceCandles || null : null;

function toFiniteNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function toTimeMs(value) {
  const num = toFiniteNumber(value);
  if (num === null) return null;
  return num >= 1e12 ? Math.floor(num) : Math.floor(num * 1000);
}

function buildBar(timeMs, open, high, low, close, volume) {
  if ([timeMs, open, high, low, close].some((value) => value === null)) return null;
  const tsMs = Math.floor(timeMs);
  const bar = {
    time: Math.floor(tsMs / 1000),
    ts_ms_utc: tsMs,
    open,
    high,
    low,
    close,
  };
  if (volume !== null) bar.volume = volume;
  return bar;
}

function parseKlineRow(row) {
  if (!Array.isArray(row) || row.length < 5) return null;
  if (BinanceAdapter && typeof BinanceAdapter.transformKlines === "function") {
    try {
      const converted = BinanceAdapter.transformKlines([row]);
      if (converted && converted.length) {
        const base = converted[0];
        const ms = toTimeMs(row[0]) ?? base.time * 1000;
        const volume = row.length > 5 ? toFiniteNumber(row[5]) : null;
        return buildBar(ms, base.open, base.high, base.low, base.close, volume);
      }
    } catch (error) {
      console.warn("Failed to transform kline row", error);
    }
  }
  const timeMs = toTimeMs(row[0]);
  const open = toFiniteNumber(row[1]);
  const high = toFiniteNumber(row[2]);
  const low = toFiniteNumber(row[3]);
  const close = toFiniteNumber(row[4]);
  const volume = row.length > 5 ? toFiniteNumber(row[5]) : null;
  return buildBar(timeMs, open, high, low, close, volume);
}

function parseWsKline(message) {
  if (!message) return null;
  const source = message.k || message;
  if (!source) return null;
  if (BinanceAdapter && typeof BinanceAdapter.barFromWs === "function") {
    try {
      const base = BinanceAdapter.barFromWs(message.k ? message : { k: source });
      if (base) {
        const ms = toTimeMs(source.t ?? source.time) ?? base.time * 1000;
        const volume = source.v !== undefined ? toFiniteNumber(source.v) : toFiniteNumber(source.volume);
        return buildBar(ms, base.open, base.high, base.low, base.close, volume);
      }
    } catch (error) {
      console.warn("Failed to parse ws kline", error);
    }
  }
  const timeMs = toTimeMs(source.t ?? source.time);
  const open = toFiniteNumber(source.o ?? source.open);
  const high = toFiniteNumber(source.h ?? source.high);
  const low = toFiniteNumber(source.l ?? source.low);
  const close = toFiniteNumber(source.c ?? source.close);
  const volume = source.v !== undefined ? toFiniteNumber(source.v) : toFiniteNumber(source.volume);
  return buildBar(timeMs, open, high, low, close, volume);
}

function parseOhlcObject(obj) {
  if (!obj || typeof obj !== "object") return null;
  if (obj.k) return parseWsKline(obj);
  const timeMs = toTimeMs(obj.ts_ms_utc ?? obj.openTime ?? obj.open_time ?? obj.time ?? obj.t);
  const open = toFiniteNumber(obj.open ?? obj.o);
  const high = toFiniteNumber(obj.high ?? obj.h);
  const low = toFiniteNumber(obj.low ?? obj.l);
  const close = toFiniteNumber(obj.close ?? obj.c);
  const volume = obj.volume !== undefined ? toFiniteNumber(obj.volume ?? obj.v) : toFiniteNumber(obj.v);
  return buildBar(timeMs, open, high, low, close, volume);
}

function parseBarLike(input) {
  if (!input) return null;
  if (Array.isArray(input)) return parseKlineRow(input);
  if (typeof input === "object") return parseOhlcObject(input);
  return null;
}

function dedupeByTime(bars) {
  const seen = new Map();
  const result = [];
  for (const bar of Array.isArray(bars) ? bars : []) {
    if (!bar || !Number.isFinite(Number(bar.time))) continue;
    const time = Number(bar.time);
    if (seen.has(time)) {
      result[seen.get(time)] = bar;
    } else {
      seen.set(time, result.length);
      result.push(bar);
    }
  }
  return result;
}

function parseHistory(raw) {
  if (!Array.isArray(raw)) return [];
  const mapped = raw.map((entry) => parseBarLike(entry)).filter(Boolean);
  mapped.sort((a, b) => a.time - b.time);
  return dedupeByTime(mapped);
}

function parseRealtimeBar(payload) {
  if (!payload) return null;
  if (payload.k) {
    const bar = parseWsKline(payload);
    if (bar) return bar;
  }
  if (payload.binance_kline) {
    const bar = parseWsKline({ k: payload.binance_kline });
    if (bar) return bar;
  }
  if (payload.ohlc) {
    const bar = parseBarLike(payload.ohlc);
    if (bar) return bar;
  }
  if (payload.bar) {
    const bar = parseBarLike(payload.bar);
    if (bar) return bar;
  }
  if (payload.candle) {
    const bar = parseBarLike(payload.candle);
    if (bar) return bar;
  }
  return parseBarLike(payload);
}

async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
}

function updateModelOptions(select, models, current) {
  select.innerHTML = "";
  const baseOption = document.createElement("option");
  baseOption.value = "";
  baseOption.textContent = "latest";
  select.appendChild(baseOption);
  (models || []).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name || "auto";
    if (name === current) option.selected = true;
    select.appendChild(option);
  });
  if (current && !select.value) {
    select.value = current;
  }
}

function buildQuery(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, value);
    }
  });
  return search.toString();
}

function formatNumber(value, digits = 4) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 0) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "—";
  return `${Number(value * 100).toFixed(digits)}%`;
}

function formatRange(range, digits = 2, asPercent = false) {
  if (!Array.isArray(range) || range.length < 2) return "—";
  const [start, end] = range;
  if (!Number.isFinite(Number(start)) || !Number.isFinite(Number(end))) return "—";
  if (asPercent) {
    return `${formatPercent(Number(start), digits)} – ${formatPercent(Number(end), digits)}`;
  }
  return `${Number(start).toFixed(digits)} – ${Number(end).toFixed(digits)}`;
}

function formatMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return "—";
  const entries = Object.entries(metrics)
    .slice(0, 4)
    .map(([key, value]) => {
      const label = key.replace(/_/g, " ");
      if (value === null || value === undefined || Number.isNaN(Number(value))) return `${label}:—`;
      if (typeof value === "number") return `${label}:${value.toFixed(2)}`;
      return `${label}:${value}`;
    });
  return entries.join(", ") || "—";
}

function formatMetricPair(dayValue, cumValue, digits = 2) {
  const day = Number(dayValue);
  const cum = Number(cumValue);
  const dayValid = Number.isFinite(day);
  const cumValid = Number.isFinite(cum);
  if (!dayValid && !cumValid) return "—";
  const dayText = dayValid ? day.toFixed(digits) : "—";
  if (!cumValid || Math.abs(day - cum) < 1e-9) return dayText;
  return `${dayText} · avg ${cum.toFixed(digits)}`;
}

function renderList(container, items) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement("li");
    empty.textContent = "нет данных";
    container.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function renderRuleTags(container, tags) {
  container.innerHTML = "";
  const items = Array.isArray(tags) && tags.length ? tags : ["side: flat"];
  items.forEach((raw) => {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "rule-tag";
    const label = typeof raw === "string" ? raw.replace(/:/, ": ") : String(raw ?? "");
    tag.textContent = label;
    li.appendChild(tag);
    container.appendChild(li);
  });
}

function renderGuards(container, guards) {
  container.innerHTML = "";
  if (!Array.isArray(guards)) {
    const empty = document.createElement("li");
    empty.textContent = "нет данных";
    container.appendChild(empty);
    return;
  }
  if (!guards.length) {
    const li = document.createElement("li");
    li.className = "guard-item";
    li.dataset.status = "ok";
    li.innerHTML = "<span>guards</span><span>ok</span>";
    container.appendChild(li);
    return;
  }
  guards.forEach((guard) => {
    const li = document.createElement("li");
    li.className = "guard-item";
    li.dataset.status = guard?.ok ? "ok" : "blocked";
    const name = guard?.name || "guard";
    li.innerHTML = `<span>${name}</span><span>${guard?.ok ? "ok" : "stop"}</span>`;
    container.appendChild(li);
  });
}

function renderMemory(container, memory = {}) {
  container.innerHTML = "";
  const entries = Object.entries(memory || {});
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.textContent = "Нет релевантных материалов";
    container.appendChild(empty);
    return;
  }
  entries.forEach(([rule, hits]) => {
    (hits || []).forEach((hit) => {
      const card = document.createElement("article");
      card.className = "memory-card";
      const header = document.createElement("header");
      const title = document.createElement("strong");
      title.textContent = hit?.title || "Документ";
      header.appendChild(title);
      if (hit?.score !== undefined) {
        const score = document.createElement("span");
        score.textContent = `${(Number(hit.score) * 100).toFixed(1)}%`;
        header.appendChild(score);
      }
      const path = document.createElement("div");
      path.className = "path";
      path.textContent = `${rule} · ${hit?.path || ""}`;
      card.appendChild(header);
      card.appendChild(path);
      container.appendChild(card);
    });
  });
}

function intervalToMs(interval) {
  const mapping = {
    "1s": 1000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
  };
  if (mapping[interval]) return mapping[interval];
  const value = Number.parseFloat(interval);
  if (Number.isFinite(value)) return value * 60_000;
  return 60_000;
}

function formatUtcDate(tsMs) {
  const numeric = Number(tsMs);
  if (!Number.isFinite(numeric)) return "—";
  const date = new Date(numeric);
  if (Number.isNaN(date.getTime())) return "—";
  const formatter = new Intl.DateTimeFormat("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  const parts = formatter.formatToParts(date).reduce((acc, part) => {
    acc[part.type] = part.value;
    return acc;
  }, {});
  const day = parts.day || "00";
  const month = parts.month || "Jan";
  const year = parts.year || "1970";
  const hour = parts.hour || "00";
  const minute = parts.minute || "00";
  const second = parts.second || "00";
  return `${day} ${month} ${year} ${hour}:${minute}:${second}`;
}

function debounce(fn, delay = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function toChartTime(tsMs) {
  const numeric = Number(tsMs);
  if (!Number.isFinite(numeric)) return undefined;
  return Math.floor(numeric / 1000);
}

function formatUtcDate(tsMs) {
  let numeric = Number(tsMs);
  if (!Number.isFinite(numeric)) return "—";
  if (Math.abs(numeric) < 1e12) numeric *= 1000;
  const date = new Date(numeric);
  if (Number.isNaN(date.getTime())) return "—";
  const pad = (num) => String(num).padStart(2, "0");
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const day = pad(date.getUTCDate());
  const month = months[date.getUTCMonth()] || "Jan";
  const year = date.getUTCFullYear();
  const hour = pad(date.getUTCHours());
  const minute = pad(date.getUTCMinutes());
  const second = pad(date.getUTCSeconds());
  return `${day} ${month} ${year} ${hour}:${minute}:${second}`;
}

function initIndexPage() {
  const form = $("#launch-form");
  if (!form) return;
  const familySelect = $("#model-family");
  const modelSelect = $("#model-name");

  async function refreshModels() {
    if (!familySelect || !modelSelect) return;
    const family = familySelect.value;
    if (!family) return;
    try {
      const data = await fetchJSON(`/models?family=${encodeURIComponent(family)}`);
      updateModelOptions(modelSelect, data.models || [], modelSelect.dataset.current || "");
    } catch (err) {
      console.error("Failed to load models", err);
    }
  }

  familySelect?.addEventListener("change", refreshModels);
  void refreshModels();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    if (!formData.has("paper")) formData.append("paper", "0");
    const query = new URLSearchParams(formData);
    window.location.href = `/predict?${query.toString()}`;
  });
}

function initPredictPage() {
  const body = document.body;
  if (body.dataset.page !== "predict") return;

  const params = {
    symbol: body.dataset.symbol || "BTCUSDT",
    interval: body.dataset.interval || "1m",
    modelFamily: body.dataset.modelFamily || "xgb",
    modelName: body.dataset.modelName || "",
    textEmbedder: body.dataset.textEmbedder || "",
    paper: body.dataset.paper === "1" || body.dataset.paper === "true",
    jobKey: body.dataset.jobKey || "",
  };

  const colors = {
    realUpBody: "#f8fafc",
    realDownBody: "#020617",
    realWick: "rgba(148, 163, 184, 0.7)",
    predictUp: "rgba(74, 222, 128, 0.85)",
    predictDown: "rgba(248, 113, 113, 0.85)",
    predictNeutral: "rgba(148, 163, 184, 0.35)",
    knowledge: "#38bdf8",
    structureUp: "#22c55e",
    structureDown: "#f87171",
  };

  const chartUtils = window.ChartUtils || {
    thresholds: { grey: [0.45, 0.55] },
    classifyPrediction(mark) {
      if (!mark) return "flat";
      const conf = Number(mark.confidence ?? mark.prob_up ?? 0.5);
      const grey = Array.isArray(mark.grey_zone) ? mark.grey_zone : this.thresholds.grey;
      if (conf >= (grey?.[1] ?? 0.55)) return "up";
      if (conf <= (grey?.[0] ?? 0.45)) return "down";
      return "flat";
    },
    decorateCandle(bar, mark, palette) {
      const decorated = Object.assign({}, bar);
      if (!mark) return decorated;
      const direction = this.classifyPrediction(mark);
      if (direction === "up") {
        decorated.borderColor = palette.up;
      } else if (direction === "down") {
        decorated.borderColor = palette.down;
      } else {
        decorated.borderColor = palette.neutral;
      }
      decorated.borderUpColor = decorated.borderColor;
      decorated.borderDownColor = decorated.borderColor;
      return decorated;
    },
    applyPredictions(bars, map, palette) {
      if (!Array.isArray(bars)) return [];
      return bars.map((bar) => {
        const mark = map?.get(Number(bar.time));
        return this.decorateCandle(Object.assign({}, bar), mark, palette);
      });
    },
  };

  const predictionPalette = {
    up: colors.predictUp,
    down: colors.predictDown,
    neutral: colors.predictNeutral,
  };

  const defaultGreyZone =
    (chartUtils?.normaliseGreyZone && chartUtils.normaliseGreyZone(chartUtils.thresholds?.grey)) ||
    chartUtils?.thresholds?.grey || [0.45, 0.55];

  function resolveGreyZone(zone) {
    if (chartUtils?.normaliseGreyZone) return chartUtils.normaliseGreyZone(zone);
    if (!Array.isArray(zone) || zone.length < 2) return defaultGreyZone;
    const lower = Number(zone[0]);
    const upper = Number(zone[1]);
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) return defaultGreyZone;
    return lower <= upper ? [lower, upper] : [upper, lower];
  }

  function structureLabel(key) {
    if (!key) return null;
    if (key.startsWith("bos_up")) return "BOS ↑";
    if (key.startsWith("bos_down")) return "BOS ↓";
    if (key.startsWith("fvg_up")) return "FVG ↑";
    if (key.startsWith("fvg_down")) return "FVG ↓";
    if (key.startsWith("swing")) return "Swing";
    if (key.startsWith("impulse")) return "Impulse";
    return key;
  }

  function markerForStructure(time, key) {
    const label = structureLabel(key);
    if (!label) return null;
    const isDown = /down/i.test(key);
    return {
      time,
      position: isDown ? "belowBar" : "aboveBar",
      color: isDown ? colors.structureDown : colors.structureUp,
      shape: isDown ? "arrowDown" : "arrowUp",
      text: label,
    };
  }

  function markerForKnowledge(time, count) {
    const suffix = count > 1 ? `×${count}` : "";
    return {
      time,
      position: "aboveBar",
      color: colors.knowledge,
      shape: "circle",
      text: `RAG${suffix}`,
    };
  }

  function rebuildPatternMarkers() {
    if (!candleSeries) return;
    const markers = [];
    predictionMap.forEach((mark) => {
      const tsMs = Number(mark?.ts_ms_utc);
      if (!Number.isFinite(tsMs) || !mark) return;
      const time = toChartTime(tsMs);
      if (!Number.isFinite(tsMs) || !Number.isFinite(time) || !mark) return;
      const structureHits = Array.isArray(mark.structure) ? mark.structure : [];
      structureHits.forEach((hit) => {
        const marker = markerForStructure(tsMs, hit);
        if (marker) markers.push(marker);
      });
      const knowledgeHits = Array.isArray(mark.knowledge) ? mark.knowledge : [];
      if (knowledgeHits.length) {
        markers.push(markerForKnowledge(tsMs, knowledgeHits.length));
      }
    });
    patternMarkers = markers;
    candleSeries.setMarkers(patternMarkers);
  }

  function renderPatternsPanel(patterns) {
    if (!patternList) return;
    const entries = [];
    const structureHits = Array.isArray(patterns?.structure) ? patterns.structure : [];
    structureHits.forEach((hit) => {
      const label = structureLabel(hit);
      entries.push(`Структура · ${label || hit}`);
    });
    const knowledgeHits = Array.isArray(patterns?.knowledge) ? patterns.knowledge : [];
    knowledgeHits.forEach((hit) => {
      entries.push(`Знание · ${hit}`);
    });
    renderList(patternList, entries);
  }

  const chartContainer = $("#chart");
  let chart = null;
  let gapWatcher = null;
  let gapFillInFlight = false;
  let candleSeries = null;
  let predictionSeries = null;

  const streamLog = $("#stream-log");
  const jobsList = $("#jobs-list");
  const rulesList = $("#rules-list");
  const memoryContainer = $("#memory-hits");
  const uploadStatus = $("#upload-status");
  const logCopyBtn = $("#log-copy");
  const logClearBtn = $("#log-clear");
  const openUploadModalBtn = $("#open-upload-modal");
  const uploadModal = $("#upload-modal");
  const uploadForm = $("#upload-form");
  const uploadSourceInput = $("#upload-source-input");
  const uploadSourcePath = $("#upload-source-path");
  const uploadSourceBrowse = $("#upload-source-browse");
  const uploadExportName = $("#upload-export-name");
  const uploadCancel = $("#upload-cancel");
  const uploadSubmit = $("#upload-submit");
  const uploadError = $("#upload-modal-error");
  const statusIndicator = $("#model-status-indicator");
  const statusLabel = $("#status-label");
  const startBtn = $("#start-stream");
  const stopBtn = $("#stop-stream");
  const trainBtn = $("#train-now");
  const form = $("#stream-form");
  const probEl = $("#state-prob");
  const confEl = $("#state-confidence");
  const ciEl = $("#state-ci");
  const greyEl = $("#state-grey");
  const regimeEl = $("#state-regime");
  const sideEl = $("#state-side");
  const tsEl = $("#state-ts");
  const priceEl = $("#state-price");
  const planTsEl = $("#plan-ts");
  const planPriceEl = $("#plan-price");
  const planProbEl = $("#plan-prob");
  const planConfEl = $("#plan-conf");
  const planSideEl = $("#plan-side");
  const targetEl = $("#plan-target");
  const zoneEl = $("#plan-zone");
  const stopEl = $("#plan-stop");
  const rrEl = $("#plan-rr");
  const planGreyEl = $("#plan-grey");
  const planCiEl = $("#plan-ci");
  const positionEl = $("#state-position");
  const pnlEl = $("#state-pnl");
  const modelStatusEl = $("#state-model-status");
  const modelNameEl = $("#state-model-name");
  const guardsList = $("#state-guards");
  const patternList = $("#pattern-list");
  const entryOverlay = $("#entry-zone-overlay");
  const livePriceIndicator = $("#live-price-indicator");
  const patternLegendGrey = $("#legend-grey-zone");
  const timeframeButtonsContainer = $("#chart-timeframes");
  const timeframeButtons = timeframeButtonsContainer
    ? Array.from(timeframeButtonsContainer.querySelectorAll("button"))
    : [];
  const chartLoader = $("#chart-loading");
  let chartLoaderClaims = 0;
  const intervalSelect = $("#input-interval");
  const openSelfTrainBtn = $("#open-self-train");
  const selfTrainModal = $("#self-train-modal");
  const selfTrainForm = $("#self-train-form");
  const selfTrainClose = $("#self-train-close");
  const selfTrainStop = $("#self-train-stop");
  const selfTrainStatusMsg = $("#self-train-status");
  const selfTrainSymbolEl = $("#self-train-symbol");
  const selfTrainDayEl = $("#self-train-day");
  const selfTrainTfEl = $("#self-train-tf");
  const selfTrainStageEl = $("#self-train-stage");
  const selfTrainRowsEl = $("#self-train-rows");
  const selfTrainMccEl = $("#self-train-mcc");
  const selfTrainEceEl = $("#self-train-ece");
  const selfTrainSharpeEl = $("#self-train-sharpe");
  const selfTrainHitEl = $("#self-train-hit");
  const selfTrainGateEl = $("#self-train-gate");
  const selfTrainProgressBar = $("#self-train-progress-bar");
  const selfTrainProgressText = $("#self-train-progress-text");
  const selfTrainHistoryList = $("#self-train-history");

  const candles = [];
  const predictions = [];
  const priceLines = { stop: null, take: null };
  const jobsLog = [];
  const predictionMap = new Map();
  const logBuffer = [];
  let selectedFiles = [];
  let patternMarkers = [];
  let selfTrainTimer = null;

  let ws = null;
  let reconnectTimer = null;
  let manualStop = false;
  let lastOverlayState = null;
  let lastClosePrice = null;
  let awaitingReload = false;
  let toastTimer = null;
  let lastBarTime = undefined;

  function ensureChart() {
    if (chart || !chartContainer || !window.LightweightCharts) return;
    chart = window.LightweightCharts.createChart(chartContainer, {
      layout: { background: { color: "#101216" }, textColor: "#d9e1f2" },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.05)" },
        horzLines: { color: "rgba(255,255,255,0.05)" },
      },
      crosshair: { mode: 1 },
      localization: {
        timeFormatter: (time) => formatUtcDate(Number(time)),
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
        tickMarkFormatter: (time) => formatUtcDate(Number(time)),
      },
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: colors.realUpBody,
      downColor: colors.realDownBody,
      wickUpColor: colors.realWick,
      wickDownColor: colors.realWick,
      borderUpColor: colors.realWick,
      borderDownColor: colors.realWick,
      borderVisible: true,
    });
    predictionSeries = chart.addCandlestickSeries({
      upColor: colors.predictUp,
      downColor: colors.predictDown,
      wickUpColor: colors.predictUp,
      wickDownColor: colors.predictDown,
      borderVisible: false,
    });
    if (patternLegendGrey) {
      patternLegendGrey.textContent = formatRange(defaultGreyZone, 1, true);
    }
    chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      if (lastOverlayState) requestAnimationFrame(drawEntryZoneOverlay);
      requestAnimationFrame(positionLivePriceIndicator);
    });
    if (window.ChartGapWatcher && typeof window.ChartGapWatcher.attach === "function") {
      if (gapWatcher && typeof gapWatcher.detach === "function") {
        gapWatcher.detach();
      }
      gapWatcher = window.ChartGapWatcher.attach({
        chart,
        getCandles: () => candles,
        symbol: params.symbol,
        interval: params.interval,
        intervalMs: intervalToMs(params.interval),
        requestGap: requestGapFill,
      });
    }
  }

  if (window.ResizeObserver && chartContainer) {
    const observer = new ResizeObserver(() => {
      if (!chart) return;
      const { clientWidth, clientHeight } = chartContainer;
      chart.applyOptions({ width: clientWidth, height: clientHeight });
      requestAnimationFrame(() => {
        drawEntryZoneOverlay();
        positionLivePriceIndicator();
      });
    });
    observer.observe(chartContainer);
  }

  function drawEntryZoneOverlay() {
    if (!entryOverlay || !chart) return;
    if (!lastOverlayState) {
      entryOverlay.classList.add("hidden");
      return;
    }
    const { ts_ms_utc: startMs, from, to, side } = lastOverlayState;
    const intervalMs = intervalToMs(params.interval);
    const startTime = Number.isFinite(startMs) ? toChartTime(startMs) : undefined;
    const endMs = Number.isFinite(startMs) && Number.isFinite(intervalMs) ? startMs + intervalMs : NaN;
    const endTime = Number.isFinite(endMs) ? toChartTime(endMs) : undefined;
    if (!Number.isFinite(startTime) || !Number.isFinite(endTime)) {
      entryOverlay.classList.add("hidden");
      return;
    }
    const timeScale = chart.timeScale();
    const leftCoord = timeScale.timeToCoordinate(startTime);
    const rightCoord = timeScale.timeToCoordinate(endTime);
    if (leftCoord === null || rightCoord === null) {
      entryOverlay.classList.add("hidden");
      return;
    }
    const topPrice = Math.max(from, to);
    const bottomPrice = Math.min(from, to);
    const topCoord = candleSeries.priceToCoordinate(topPrice);
    const bottomCoord = candleSeries.priceToCoordinate(bottomPrice);
    if (
      topCoord === null ||
      topCoord === undefined ||
      bottomCoord === null ||
      bottomCoord === undefined
    ) {
      entryOverlay.classList.add("hidden");
      return;
    }
    const left = Math.min(leftCoord, rightCoord);
    const width = Math.max(4, Math.abs(rightCoord - leftCoord));
    const top = Math.min(topCoord, bottomCoord);
    const height = Math.max(4, Math.abs(bottomCoord - topCoord));

    entryOverlay.style.left = `${left}px`;
    entryOverlay.style.width = `${width}px`;
    entryOverlay.style.top = `${top}px`;
    entryOverlay.style.height = `${height}px`;
    entryOverlay.dataset.side = side;
    entryOverlay.classList.remove("hidden");
  }

  function positionLivePriceIndicator() {
    if (!livePriceIndicator) return;
    if (!chartContainer || !candleSeries || !chart) {
      livePriceIndicator.classList.add("hidden");
      return;
    }
    if (!Number.isFinite(lastClosePrice)) {
      livePriceIndicator.classList.add("hidden");
      return;
    }
    const coordinate = candleSeries.priceToCoordinate(lastClosePrice);
    if (coordinate === null || coordinate === undefined) {
      livePriceIndicator.classList.add("hidden");
      return;
    }
    const { clientHeight } = chartContainer;
    let top = coordinate;
    const wasHidden = livePriceIndicator.classList.contains("hidden");
    if (wasHidden) {
      livePriceIndicator.classList.remove("hidden");
      livePriceIndicator.style.visibility = "hidden";
    }
    const indicatorHeight = livePriceIndicator.offsetHeight || 0;
    if (clientHeight > 0 && indicatorHeight > 0) {
      const half = indicatorHeight / 2;
      top = Math.min(Math.max(coordinate, half), Math.max(half, clientHeight - half));
    }
    livePriceIndicator.style.top = `${top}px`;
    livePriceIndicator.style.visibility = "";
    livePriceIndicator.classList.remove("hidden");
  }

  function updateLivePriceIndicator(price) {
    if (!livePriceIndicator) return;
    const numeric = Number(price);
    if (!Number.isFinite(numeric)) {
      lastClosePrice = null;
      livePriceIndicator.classList.add("hidden");
      return;
    }
    lastClosePrice = numeric;
    livePriceIndicator.textContent = formatNumber(numeric, 2);
    positionLivePriceIndicator();
  }

  function addPrediction(mark) {
    if (!mark) return undefined;
    const tsMs = Number(mark.ts_ms_utc ?? mark.time);
    if (!Number.isFinite(tsMs)) return undefined;
    const timeValue = toChartTime(tsMs);
    if (!Number.isFinite(timeValue)) return undefined;
    const prob = mark.prob_up !== undefined ? Number(mark.prob_up) : undefined;
    const greyZone = resolveGreyZone(mark.grey_zone || mark.greyZone || mark.greyZoneBounds);
    const direction = chartUtils?.resolveDirection ? chartUtils.resolveDirection(mark) : mark.pred_dir || mark.signal;
    const structureHits = Array.isArray(mark.structure)
      ? mark.structure
      : Array.isArray(mark.patterns?.structure)
      ? mark.patterns.structure
      : [];
    const knowledgeHits = Array.isArray(mark.knowledge)
      ? mark.knowledge
      : Array.isArray(mark.patterns?.knowledge)
      ? mark.patterns.knowledge
      : Array.isArray(mark.knowledge_hits)
      ? mark.knowledge_hits
      : [];
    const normalized = {
      ts_ms_utc: tsMs,
      time: tsMs,
      time: timeValue,
      prob_up: Number.isFinite(prob) ? prob : undefined,
      prob_down: Number.isFinite(mark.prob_down) ? Number(mark.prob_down) : undefined,
      prob_flat: Number.isFinite(mark.prob_flat) ? Number(mark.prob_flat) : undefined,
      confidence: Number(mark.confidence ?? mark.pred_conf ?? prob ?? 0.5),
      grey_zone: greyZone,
      confidence_interval: Array.isArray(mark.confidence_interval) ? mark.confidence_interval : undefined,
      pred_dir: direction || "flat",
      signal: direction || "flat",
      structure: structureHits,
      knowledge: knowledgeHits,
    };
    predictionMap.set(tsMs, normalized);
    return normalized;
  }

  function mergePredictions(list, reset = false) {
    if (reset) predictionMap.clear();
    if (!Array.isArray(list)) return;
    list.forEach((item) => addPrediction(item));
    rebuildPatternMarkers();
  }

  function decorateBar(bar) {
    const tsMs = Number(bar.ts_ms_utc ?? bar.time);
    const mark = Number.isFinite(tsMs) ? predictionMap.get(tsMs) : undefined;
    return chartUtils.decorateCandle(bar, mark, predictionPalette);
  }

  function mergeHistoricalCandles(rawBars) {
    if (!Array.isArray(rawBars) || !rawBars.length) return;
    const decorated = rawBars.map((bar) => decorateBar(bar)).filter(Boolean);
    if (!decorated.length) return;
    const indexByTime = new Map();
    for (let i = 0; i < candles.length; i += 1) {
      const time = Number(candles[i]?.time);
      if (Number.isFinite(time) && !indexByTime.has(time)) {
        indexByTime.set(time, i);
      }
    }
    let changed = false;
    decorated.forEach((bar) => {
      if (!bar) return;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) return;
      if (indexByTime.has(time)) {
        const idx = indexByTime.get(time);
        candles[idx] = bar;
        changed = true;
      } else {
        indexByTime.set(time, candles.length);
        candles.push(bar);
        changed = true;
      }
    });
    if (!changed) return;
    candles.sort((a, b) => Number(a.time) - Number(b.time));
    lastBarTime = candles.length ? Number(candles[candles.length - 1].time) : lastBarTime;
    if (candleSeries) candleSeries.setData(candles);
    requestAnimationFrame(positionLivePriceIndicator);
    if (gapWatcher && typeof gapWatcher.notifyData === "function") {
      gapWatcher.notifyData();
    }
  }

  function setUploadStatus(message, ttl = 0) {
    if (!uploadStatus) return;
    uploadStatus.textContent = message || "";
    if (toastTimer) {
      clearTimeout(toastTimer);
      toastTimer = null;
    }
    if (ttl > 0 && message) {
      toastTimer = setTimeout(() => {
        if (uploadStatus && uploadStatus.textContent === message) {
          uploadStatus.textContent = "";
        }
      }, ttl);
    }
  }

  function setSelfTrainStatus(message, variant = "info") {
    if (!selfTrainStatusMsg) return;
    selfTrainStatusMsg.textContent = message || "";
    if (variant && selfTrainStatusMsg.dataset) {
      selfTrainStatusMsg.dataset.variant = variant;
    }
  }

  function renderSelfTrainHistory(history, currentDay, currentTf) {
    if (!selfTrainHistoryList) return;
    selfTrainHistoryList.innerHTML = "";
    const items = Array.isArray(history) ? history : [];
    if (!items.length) {
      const empty = document.createElement("li");
      empty.textContent = "История пока пуста";
      selfTrainHistoryList.appendChild(empty);
      return;
    }
    items
      .slice()
      .sort((a, b) => {
        const dayOrder = (a.day || "").localeCompare(b.day || "");
        if (dayOrder !== 0) return dayOrder;
        return (a.tf || "").localeCompare(b.tf || "");
      })
      .forEach((entry) => {
        const li = document.createElement("li");
        const gate = entry.gate || {};
        li.dataset.state = gate.active ? "active" : "inactive";
        const title = document.createElement("span");
        title.textContent = `${entry.day || "—"} · ${entry.tf || "—"}`;
        if (entry.day === currentDay && entry.tf === currentTf) {
          title.classList.add("current");
        }
        li.appendChild(title);
        const metricsText = formatMetrics(entry.metrics || {});
        const rowsText = Number(entry.rows_in_day || 0).toLocaleString();
        const details = document.createElement("small");
        const gateState = gate.active ? "активна" : "неактивна";
        details.textContent = `${metricsText} · rows:${rowsText} · ${gateState}`;
        li.appendChild(details);
        selfTrainHistoryList.appendChild(li);
      });
  }

  function renderSelfTrainStatus(statusPayload, historyPayload) {
    const status = statusPayload || {};
    const history = Array.isArray(historyPayload) ? historyPayload : [];
    if (selfTrainSymbolEl) selfTrainSymbolEl.textContent = status.symbol || "—";
    if (selfTrainDayEl) selfTrainDayEl.textContent = status.day || "—";
    if (selfTrainTfEl) selfTrainTfEl.textContent = status.tf || "—";
    if (selfTrainStageEl) selfTrainStageEl.textContent = status.stage || "idle";

    const rowsDay = Number(status.rows_in_day || 0);
    const rowsTotal = Number(status.rows_total_tf || 0);
    if (selfTrainRowsEl) {
      selfTrainRowsEl.textContent = `${rowsDay.toLocaleString()} / ${rowsTotal.toLocaleString()}`;
    }

    const metricsDay = status.metrics_day || {};
    const metricsCum = status.metrics_cum || {};
    if (selfTrainMccEl) selfTrainMccEl.textContent = formatMetricPair(metricsDay.MCC, metricsCum.MCC, 2);
    if (selfTrainEceEl) selfTrainEceEl.textContent = formatMetricPair(metricsDay.ECE, metricsCum.ECE, 3);
    if (selfTrainSharpeEl)
      selfTrainSharpeEl.textContent = formatMetricPair(metricsDay.SharpeProxy, metricsCum.SharpeProxy, 2);
    if (selfTrainHitEl) selfTrainHitEl.textContent = formatMetricPair(metricsDay.HitRate, metricsCum.HitRate, 2);

    const gate = status.gate || {};
    if (selfTrainGateEl) {
      const gateState = gate.active ? "active" : gate.ece_pass || gate.mcc_pass ? "partial" : "inactive";
      selfTrainGateEl.textContent = gate.active ? "активна" : gateState === "partial" ? "частично" : "неактивна";
      if (selfTrainGateEl.dataset) {
        selfTrainGateEl.dataset.state = gateState;
      }
    }

    const defaultIntervals = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"];
    const configuredIntervals = Array.isArray(status.intervals) && status.intervals.length
      ? status.intervals
      : defaultIntervals;
    const totalIntervals = Number(status.intervals_total || configuredIntervals.length);
    const currentDay = status.day || null;
    const completed = new Set(
      history
        .filter((entry) => entry.stage === "gate" && entry.day === currentDay)
        .map((entry) => entry.tf)
    );
    const completedCount = completed.size;
    const ratio = totalIntervals > 0 ? Math.min(1, completedCount / totalIntervals) : 0;
    if (selfTrainProgressBar) {
      selfTrainProgressBar.style.width = `${(ratio * 100).toFixed(1)}%`;
    }
    if (selfTrainProgressText) {
      selfTrainProgressText.textContent = `${completedCount} / ${totalIntervals} таймфреймов`;
    }

    renderSelfTrainHistory(history, currentDay, status.tf);
  }

  async function refreshSelfTrainStatus(options = {}) {
    try {
      const [statusData, historyData] = await Promise.all([
        fetchJSON("/self_train/status"),
        fetchJSON("/self_train/history"),
      ]);
      const historyList = historyData && Array.isArray(historyData.history) ? historyData.history : [];
      renderSelfTrainStatus(statusData, historyList);
      if (options.message) {
        setSelfTrainStatus(options.message, "info");
      } else if (statusData && statusData.schema_version) {
        setSelfTrainStatus(`schema v${statusData.schema_version}`, "info");
      }
    } catch (error) {
      console.error("Failed to load self-train status", error);
      if (!options.silent) setSelfTrainStatus("Не удалось получить статус", "error");
    }
  }

  function openSelfTrainModal() {
    if (!selfTrainModal) return;
    selfTrainModal.classList.remove("hidden");
    setSelfTrainStatus("");
    void refreshSelfTrainStatus({ silent: true });
    if (selfTrainTimer) clearInterval(selfTrainTimer);
    selfTrainTimer = setInterval(() => {
      void refreshSelfTrainStatus({ silent: true });
    }, 5000);
  }

  function closeSelfTrainModal() {
    if (!selfTrainModal) return;
    selfTrainModal.classList.add("hidden");
    setSelfTrainStatus("");
    if (selfTrainTimer) {
      clearInterval(selfTrainTimer);
      selfTrainTimer = null;
    }
  }

  function updateStatus(status) {
    if (!statusIndicator || !statusLabel) return;
    statusIndicator.dataset.status = status;
    statusLabel.textContent = status;
  }

  function appendLog(payload) {
    if (!streamLog) return;
    const text = typeof payload === "string" ? payload : JSON.stringify(payload);
    logBuffer.push(text);
    if (logBuffer.length > 500) logBuffer.shift();
    const item = document.createElement("div");
    item.className = "log-entry";
    item.textContent = text;
    streamLog.appendChild(item);
    while (streamLog.children.length > 500) streamLog.removeChild(streamLog.firstChild);
    streamLog.scrollTop = streamLog.scrollHeight;
  }

  function renderJobs() {
    if (!jobsList) return;
    jobsList.innerHTML = "";
    jobsLog.slice(0, 12).forEach((entry) => {
      const item = document.createElement("li");
      item.textContent = `${entry.type} · ${entry.status}`;
      jobsList.appendChild(item);
    });
  }

  function handleBootstrap(payload) {
    if (payload.model_status) {
      updateStatus(payload.model_status);
      if (modelStatusEl) modelStatusEl.textContent = payload.model_status;
    }
    if (payload.version && modelNameEl) {
      modelNameEl.textContent = payload.version;
    }
    mergePredictions(payload.pred_candles, true);
    candles.length = 0;
    lastBarTime = undefined;
    const baseCandles = Array.isArray(payload.ohlc) ? payload.ohlc : [];
    const prepared = parseHistory(baseCandles);
    const chartPredictions = new Map();
    predictionMap.forEach((value) => {
      const tsMs = Number(value?.time ?? value?.ts_ms_utc);
      if (Number.isFinite(tsMs)) chartPredictions.set(tsMs, value);
      const chartTime = Number(value?.time);
      if (Number.isFinite(chartTime)) chartPredictions.set(chartTime, value);
    });
    const decorated = chartUtils.applyPredictions(prepared, chartPredictions, predictionPalette);
    candles.push(...decorated);
    lastBarTime = candles.length ? Number(candles[candles.length - 1].time) : undefined;
    ensureChart();
    if (chart && candleSeries) {
      candleSeries.setData(candles);
      chart.timeScale().fitContent();
    }
    if (predictionSeries) predictionSeries.setData([]);
    rebuildPatternMarkers();
    renderPatternsPanel({});
    const lastBar = candles[candles.length - 1];
    updateLivePriceIndicator(lastBar ? lastBar.close : undefined);
    requestAnimationFrame(positionLivePriceIndicator);
    appendLog({ event: "bootstrap", size: candles.length });
    if (gapWatcher && typeof gapWatcher.updateContext === "function") {
      gapWatcher.updateContext({
        symbol: params.symbol,
        interval: params.interval,
        intervalMs: intervalToMs(params.interval),
        resetRequestedKeys: true,
      });
    }
    if (gapWatcher && typeof gapWatcher.notifyData === "function") {
      gapWatcher.notifyData();
    }
  }

  function setActiveTimeframe(activeInterval) {
    timeframeButtons.forEach((button) => {
      if (!button || !(button instanceof HTMLButtonElement)) return;
      if (button.dataset.interval === activeInterval) {
        button.classList.add("is-active");
      } else {
        button.classList.remove("is-active");
      }
    });
    if (timeframeButtonsContainer) timeframeButtonsContainer.dataset.active = activeInterval;
  }

  function toggleChartLoader(active) {
    if (!chartLoader) return;
    if (active) {
      chartLoaderClaims += 1;
    } else {
      chartLoaderClaims = Math.max(0, chartLoaderClaims - 1);
    }
    if (chartLoaderClaims > 0) {
      chartLoader.classList.remove("hidden");
    } else {
      chartLoader.classList.add("hidden");
    }
  }

  function applyCandlesPayload(payload) {
    if (!payload) return;
    const ohlc = Array.isArray(payload.candles)
      ? payload.candles
      : Array.isArray(payload.ohlc)
      ? payload.ohlc
      : [];
    handleBootstrap({ ...payload, ohlc });
  }

  async function requestGapFill(range) {
    if (!range || gapFillInFlight) return false;
    const startMs = Math.max(0, Math.floor(Number(range.startMs) || 0));
    const endMs = Math.max(startMs + 1, Math.floor(Number(range.endMs) || 0));
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return false;
    const payload = {
      symbol: params.symbol,
      interval: params.interval,
      start_ms: startMs,
      end_ms: endMs,
    };
    let started = true;
    gapFillInFlight = true;
    toggleChartLoader(true);
    try {
      const response = await fetchJSON("/ohlc/backfill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const history = parseHistory(response?.candles || response?.ohlc || []);
      if (history.length) {
        mergeHistoricalCandles(history);
      }
    } catch (error) {
      console.error("Failed to backfill candles", error);
      started = false;
    } finally {
      gapFillInFlight = false;
      toggleChartLoader(false);
      if (gapWatcher && typeof gapWatcher.notifyData === "function") {
        gapWatcher.notifyData();
      }
    }
    return started;
  }

  async function loadActiveFrame(interval, options = {}) {
    const { silent = false } = options;
    try {
      if (!silent) toggleChartLoader(true);
      const query = buildQuery({ symbol: params.symbol, tf_active: interval });
      const data = await fetchJSON(`/predict/data?${query}`);
      applyCandlesPayload({ ...data, interval });
    } catch (error) {
      console.error("Failed to load frame", error);
      throw error;
    } finally {
      if (!silent) toggleChartLoader(false);
    }
  }

  async function switchInterval(nextInterval) {
    if (!nextInterval || nextInterval === params.interval) return;
    const previous = params.interval;
    params.interval = nextInterval;
    if (intervalSelect) intervalSelect.value = nextInterval;
    setActiveTimeframe(nextInterval);
    if (gapWatcher && typeof gapWatcher.updateContext === "function") {
      gapWatcher.updateContext({
        symbol: params.symbol,
        interval: params.interval,
        intervalMs: intervalToMs(params.interval),
        resetRequestedKeys: true,
      });
    }
    try {
      await loadActiveFrame(nextInterval);
      disconnect();
      connect();
    } catch (error) {
      params.interval = previous;
      if (intervalSelect) intervalSelect.value = previous;
      setActiveTimeframe(previous);
      throw error;
    }
  }

  const debouncedSwitchInterval = debounce((value) => {
    void switchInterval(value).catch((error) => {
      console.error("Failed to switch timeframe", error);
    });
  }, 220);

  function updateOverlay(payload, bar) {
    predictions.length = 0;
    const tsMs = Number(bar?.ts_ms_utc);
    const tsSeconds = Number(bar?.time);
    const barTimeMs = Number.isFinite(tsMs)
      ? tsMs
      : Number.isFinite(tsSeconds)
      ? tsSeconds * 1000
      : NaN;
    const intervalMs = intervalToMs(params.interval);
    const nextTimeMs = Number.isFinite(barTimeMs) ? barTimeMs + intervalMs : NaN;
    const nextChartTime = Number.isFinite(nextTimeMs) ? toChartTime(nextTimeMs) : undefined;
    if (
      payload.target_price !== undefined &&
      payload.target_price !== null &&
      Number.isFinite(barTimeMs) &&
      Number.isFinite(intervalMs) &&
      Number.isFinite(nextChartTime)
    ) {
      const entry = payload.entry_plan?.zone || {};
      const high = Math.max(Number(payload.target_price), Number(entry.to ?? payload.target_price));
      const low = Math.min(Number(payload.target_price), Number(entry.from ?? payload.target_price));
      predictions.push({
        time: nextChartTime,
        open: Number(bar.close),
        high,
        low,
        close: Number(payload.target_price),
      });
    }
    if (predictionSeries) predictionSeries.setData(predictions);

    const zone = payload.entry_plan?.zone;
    const from = zone?.from !== undefined ? Number(zone.from) : NaN;
    const to = zone?.to !== undefined ? Number(zone.to) : NaN;
    if (Number.isFinite(from) && Number.isFinite(to) && Number.isFinite(barTimeMs)) {
      lastOverlayState = {
        ts_ms_utc: barTimeMs,
        from,
        to,
        side: payload.side === "short" ? "short" : payload.side === "long" ? "long" : "flat",
      };
    } else {
      lastOverlayState = null;
    }
    requestAnimationFrame(() => {
      drawEntryZoneOverlay();
      positionLivePriceIndicator();
    });

    Object.values(priceLines).forEach((line) => {
      if (line && predictionSeries) predictionSeries.removePriceLine(line);
    });
    priceLines.stop = null;
    priceLines.take = null;
    if (predictionSeries && payload.entry_plan?.stop !== undefined) {
      priceLines.stop = predictionSeries.createPriceLine({
        price: Number(payload.entry_plan.stop),
        color: "#FF6F61",
        lineStyle: 2,
        axisLabelVisible: true,
        title: "STOP",
      });
    }
    if (predictionSeries && payload.entry_plan?.take !== undefined) {
      priceLines.take = predictionSeries.createPriceLine({
        price: Number(payload.entry_plan.take),
        color: "#00C853",
        lineStyle: 0,
        axisLabelVisible: true,
        title: "TAKE",
      });
    }
  }

  function handleTick(payload) {
    ensureChart();
    updateStatus(payload.model_status || "missing");
    if (modelStatusEl) modelStatusEl.textContent = payload.model_status || "missing";
    if (payload.version && modelNameEl) modelNameEl.textContent = payload.version;
    const prob = Number(payload.prob_up);
    if (probEl) probEl.textContent = formatPercent(prob, 1);
    if (planProbEl) planProbEl.textContent = formatPercent(prob, 1);
    const greyZone = resolveGreyZone(payload.grey_zone || payload.prediction?.grey_zone);
    const confValue = Number(
      payload.pred_conf ?? payload.prediction?.confidence ?? payload.prediction?.pred_conf ?? prob ?? 0.0,
    );
    const ciRange = Array.isArray(payload.confidence_interval)
      ? payload.confidence_interval
      : payload.prediction?.confidence_interval;
    if (confEl) confEl.textContent = Number.isFinite(confValue) ? formatPercent(confValue, 1) : "—";
    if (planConfEl) planConfEl.textContent = Number.isFinite(confValue) ? formatPercent(confValue, 1) : "—";
    if (ciEl) ciEl.textContent = formatRange(ciRange, 1, true);
    if (planCiEl) planCiEl.textContent = formatRange(ciRange, 1, true);
    if (greyEl) greyEl.textContent = formatRange(greyZone, 1, true);
    if (planGreyEl) planGreyEl.textContent = formatRange(greyZone, 1, true);
    if (patternLegendGrey) patternLegendGrey.textContent = formatRange(greyZone, 1, true);
    const regimeBucket = payload.regime_bucket || payload.prediction?.regime_bucket;
    if (regimeEl) regimeEl.textContent = regimeBucket || "—";
    const sideValue = payload.side || "flat";
    if (sideEl) sideEl.textContent = sideValue;
    if (planSideEl) planSideEl.textContent = sideValue;
    const serverTime = payload.server_time_ms ?? payload.ts;
    if (tsEl) tsEl.textContent = formatUtcDate(serverTime);
    const bar = parseRealtimeBar(payload);
    if (!bar) {
      console.warn("Tick without valid candle", payload);
      return;
    }
    const barTime = Number(bar.time);
    if (!Number.isFinite(barTime)) return;
    const lastExisting = candles.length ? candles[candles.length - 1] : null;
    const sameTime = lastExisting && Number(lastExisting.time) === barTime;
    if (!sameTime && Number.isFinite(lastBarTime) && barTime < Number(lastBarTime)) {
      return;
    }
    const priceText = formatNumber(bar.close, 2);
    if (priceEl) priceEl.textContent = priceText;
    if (planTsEl && Number.isFinite(bar.ts_ms_utc)) {
      planTsEl.textContent = formatUtcDate(bar.ts_ms_utc);
    }
    if (planPriceEl) planPriceEl.textContent = priceText;
    if (targetEl) targetEl.textContent = formatNumber(payload.target_price, 2);
    if (payload.entry_plan?.zone) {
      const zone = payload.entry_plan.zone;
      if (zoneEl) zoneEl.textContent = `${formatNumber(zone.from, 2)} – ${formatNumber(zone.to, 2)}`;
    } else if (zoneEl) {
      zoneEl.textContent = "—";
    }
    if (stopEl) stopEl.textContent = formatNumber(payload.entry_plan?.stop, 2);
    if (rrEl) rrEl.textContent = payload.entry_plan?.risk_r ? Number(payload.entry_plan.risk_r).toFixed(2) : "—";
    if (positionEl) positionEl.textContent = payload.position || "flat";
    if (pnlEl) pnlEl.textContent = formatNumber(payload.pnl, 2);
    if (guardsList) renderGuards(guardsList, payload.guards || []);
    renderRuleTags(rulesList, payload.rule_tags || payload.explain?.rules || []);
    renderMemory(memoryContainer, payload.memory_hits || payload.explain?.memory_hits || {});

    let shouldResetSeries = false;
    if (Array.isArray(payload.pred_candles)) {
      mergePredictions(payload.pred_candles, true);
      for (let i = 0; i < candles.length; i += 1) {
        candles[i] = decorateBar(candles[i]);
      }
      shouldResetSeries = true;
    } else if (payload.prediction) {
      addPrediction(payload.prediction);
      rebuildPatternMarkers();
    }

    const decoratedBar = decorateBar(bar);
    if (sameTime) {
      candles[candles.length - 1] = decoratedBar;
    } else {
      candles.push(decoratedBar);
      if (candles.length > 1400) candles.shift();
      if (gapWatcher && typeof gapWatcher.notifyData === "function") {
        gapWatcher.notifyData();
      }
    }
    lastBarTime = candles.length ? Number(candles[candles.length - 1].time) : lastBarTime;
    if (candleSeries) {
      if (shouldResetSeries) {
        candleSeries.setData(candles);
      } else {
        candleSeries.update(decoratedBar);
      }
    }

    updateLivePriceIndicator(bar.close);

    updateOverlay(payload, decoratedBar);
    renderPatternsPanel(payload.pattern_hits || payload.prediction || {});
    appendLog({
      event: "tick",
      time: decoratedBar.time,
      close: decoratedBar.close,
      prob_up: prob,
      side: sideValue,
      pred_dir: payload.prediction?.pred_dir || payload.pred_dir,
      pred_conf: Number.isFinite(confValue) ? confValue : undefined,
    });
  }

  function handleJobEvent(event) {
    if (params.jobKey && event.key && event.key !== params.jobKey) return;
    jobsLog.unshift({ type: event.type, status: event.status });
    if (jobsLog.length > 24) jobsLog.length = 24;
    renderJobs();
    if (event.type === "training") {
      if (event.status === "queued") {
        setUploadStatus("Идёт переобучение…");
        awaitingReload = true;
      } else if (event.status === "done" && !awaitingReload) {
        setUploadStatus("Обучение завершено", 4000);
      } else if (event.status === "error") {
        setUploadStatus("Ошибка обучения");
        awaitingReload = false;
      }
    }
  }

  function handleModelReload(event) {
    if (event.family && event.family !== params.modelFamily) return;
    if (modelNameEl && event.version) modelNameEl.textContent = event.version;
    setUploadStatus("Модель обновлена", 4000);
    awaitingReload = false;
    void refreshModels();
  }

  function connect() {
    if (ws) ws.close();
    const query = buildQuery({
      symbol: params.symbol,
      interval: params.interval,
      model_family: params.modelFamily,
      model_name: params.modelName,
      text_embedder: params.textEmbedder,
      paper: params.paper ? "1" : "0",
    });
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${window.location.host}/ws/predict?${query}`);
    ws.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.event === "bootstrap") {
          handleBootstrap(payload);
          return;
        }
        if (payload.event === "tick") {
          handleTick(payload);
          return;
        }
        if (payload.event === "job") {
          handleJobEvent(payload);
          return;
        }
        if (payload.event === "model_reloaded") {
          handleModelReload(payload);
          return;
        }
        if (payload.error) {
          appendLog(payload);
        }
      } catch (error) {
        console.error("Failed to parse WS payload", error);
      }
    });
    ws.addEventListener("open", () => {
      manualStop = false;
      if (startBtn) startBtn.disabled = true;
      if (stopBtn) stopBtn.disabled = false;
    });
    ws.addEventListener("close", () => {
      if (stopBtn) stopBtn.disabled = true;
      if (startBtn) startBtn.disabled = false;
      if (!manualStop) {
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connect, 2000);
      }
    });
    ws.addEventListener("error", (error) => {
      console.error("WebSocket error", error);
      ws.close();
    });
  }

  function disconnect() {
    manualStop = true;
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  async function refreshModels() {
    const familySelect = $("#input-model-family");
    const modelSelect = $("#input-model-name");
    if (!familySelect || !modelSelect) return;
    try {
      const data = await fetchJSON(`/models?family=${encodeURIComponent(familySelect.value)}`);
      updateModelOptions(modelSelect, data.models || [], params.modelName);
    } catch (error) {
      console.error("Failed to load models", error);
    }
  }

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    if (!formData.has("paper")) formData.append("paper", "0");
    const query = new URLSearchParams(formData);
    window.location.href = `/predict?${query.toString()}`;
  });

  startBtn?.addEventListener("click", () => {
    manualStop = false;
    connect();
  });

  stopBtn?.addEventListener("click", () => {
    disconnect();
  });

  trainBtn?.addEventListener("click", async () => {
    trainBtn.disabled = true;
    try {
      const payload = { symbol: params.symbol, interval: params.interval, family: params.modelFamily };
      const resp = await fetch(`/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (resp.status === 202) {
        setUploadStatus("Обучение уже выполняется");
      } else if (resp.ok) {
        setUploadStatus("Запущено обучение");
        awaitingReload = true;
      } else {
        setUploadStatus("Не удалось запустить обучение");
      }
    } catch (error) {
      console.error("Failed to trigger training", error);
      setUploadStatus("Ошибка запуска обучения");
    } finally {
      trainBtn.disabled = false;
    }
  });

  function updateUploadSubmitState() {
    if (!uploadSubmit) return;
    const nameFilled = Boolean(uploadExportName?.value.trim());
    const hasFiles = selectedFiles.length > 0;
    const hasPath = Boolean(uploadSourcePath?.value.trim());
    uploadSubmit.disabled = !(nameFilled && (hasFiles || hasPath));
  }

  function openUploadModal() {
    if (!uploadModal) return;
    uploadModal.classList.remove("hidden");
    uploadError?.classList.add("hidden");
    updateUploadSubmitState();
  }

  function closeUploadModal() {
    if (!uploadModal) return;
    uploadModal.classList.add("hidden");
    uploadError?.classList.add("hidden");
    if (uploadForm) uploadForm.reset();
    selectedFiles = [];
    if (uploadSourceInput) uploadSourceInput.value = "";
    if (uploadSubmit) delete uploadSubmit.dataset.loading;
    updateUploadSubmitState();
  }

  logCopyBtn?.addEventListener("click", async () => {
    const text = logBuffer.join("\n");
    if (!text) {
      setUploadStatus("Стрим пуст", 3000);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setUploadStatus("Стрим скопирован", 4000);
    } catch (error) {
      console.error("Clipboard copy failed", error);
      setUploadStatus("Не удалось скопировать", 3000);
    }
  });

  logClearBtn?.addEventListener("click", () => {
    if (streamLog) streamLog.innerHTML = "";
    logBuffer.length = 0;
    setUploadStatus("Стрим очищен", 3000);
  });

  openUploadModalBtn?.addEventListener("click", () => {
    openUploadModal();
  });

  uploadCancel?.addEventListener("click", () => {
    closeUploadModal();
  });

  uploadModal?.addEventListener("click", (event) => {
    if (event.target === uploadModal || (event.target instanceof HTMLElement && event.target.classList.contains("modal-backdrop"))) {
      closeUploadModal();
    }
  });

  uploadSourceBrowse?.addEventListener("click", () => {
    uploadSourceInput?.click();
  });

  uploadSourceInput?.addEventListener("change", () => {
    selectedFiles = Array.from(uploadSourceInput.files || []);
    if (selectedFiles.length) {
      const first = selectedFiles[0];
      const relative = first.webkitRelativePath || first.name || "";
      if (relative && uploadSourcePath && !uploadSourcePath.value) {
        const root = relative.split("/")[0] || relative;
        uploadSourcePath.value = root;
      }
    } else {
      selectedFiles = [];
    }
    updateUploadSubmitState();
  });

  uploadSourcePath?.addEventListener("input", updateUploadSubmitState);
  uploadExportName?.addEventListener("input", updateUploadSubmitState);

  uploadForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!uploadSubmit) return;
    const exportName = uploadExportName?.value.trim() || "";
    const sourcePathValue = uploadSourcePath?.value.trim() || "";
    if (!exportName || (!selectedFiles.length && !sourcePathValue)) {
      if (uploadError) {
        uploadError.textContent = "Укажите папку и имя";
        uploadError.classList.remove("hidden");
      }
      updateUploadSubmitState();
      return;
    }

    uploadError?.classList.add("hidden");
    uploadSubmit.disabled = true;
    uploadSubmit.dataset.loading = "true";
    try {
      const formData = new FormData();
      formData.append("export_name", exportName);
      if (sourcePathValue) formData.append("source_dir", sourcePathValue);
      selectedFiles.forEach((file) => {
        formData.append("files", file, file.webkitRelativePath || file.name);
      });
      const resp = await fetch("/upload", { method: "POST", body: formData });
      const data = await resp
        .json()
        .catch(() => ({ detail: "Ошибка загрузки" }));
      if (!resp.ok) {
        const message = data?.detail || data?.message || "Ошибка загрузки";
        if (uploadError) {
          uploadError.textContent = message;
          uploadError.classList.remove("hidden");
        }
        return;
      }
      closeUploadModal();
      selectedFiles = [];
      const added = Number(data?.added ?? 0);
      const exportLabel = data?.export_name || exportName;
      setUploadStatus(`Загружено ${added} элементов в ${exportLabel}`, 6000);
      awaitingReload = true;
    } catch (error) {
      console.error("Upload failed", error);
      if (uploadError) {
        uploadError.textContent = "Ошибка загрузки";
        uploadError.classList.remove("hidden");
      }
    } finally {
      if (uploadSubmit) {
        delete uploadSubmit.dataset.loading;
        uploadSubmit.disabled = false;
      }
      updateUploadSubmitState();
    }
  });

  openSelfTrainBtn?.addEventListener("click", () => {
    openSelfTrainModal();
  });

  selfTrainClose?.addEventListener("click", () => {
    closeSelfTrainModal();
  });

  selfTrainModal?.addEventListener("click", (event) => {
    if (
      event.target === selfTrainModal ||
      (event.target instanceof HTMLElement && event.target.classList.contains("modal-backdrop"))
    ) {
      closeSelfTrainModal();
    }
  });

  selfTrainForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selfTrainForm) return;
    const dateFrom = selfTrainForm.date_from?.value || "2022-01-01";
    const dateToRaw = selfTrainForm.date_to?.value || "";
    const resume = selfTrainForm.resume ? selfTrainForm.resume.checked : true;
    const symbols = Array.from(
      selfTrainForm.querySelectorAll('input[name="self-train-symbols"]:checked'),
    )
      .map((input) => input.value)
      .filter(Boolean);
    const symbol = symbols[0] || params.symbol;
    const payload = {
      symbol,
      date_from: dateFrom,
      date_to: dateToRaw || null,
      resume,
    };
    try {
      setSelfTrainStatus("Запуск…");
      await fetchJSON("/self_train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setSelfTrainStatus("Самообучение запущено");
      void refreshSelfTrainStatus({ silent: false });
    } catch (error) {
      console.error("Failed to start self-train", error);
      setSelfTrainStatus("Не удалось запустить", "error");
    }
  });

  selfTrainStop?.addEventListener("click", async () => {
    try {
      setSelfTrainStatus("Остановка…");
      await fetchJSON("/self_train/stop", { method: "POST" });
      setSelfTrainStatus("Остановлено");
    } catch (error) {
      console.error("Failed to stop self-train", error);
      setSelfTrainStatus("Не удалось остановить", "error");
    } finally {
      void refreshSelfTrainStatus({ silent: false });
    }
  });

  setActiveTimeframe(params.interval);
  timeframeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const nextInterval = button.dataset.interval;
      if (nextInterval && nextInterval !== params.interval) {
        debouncedSwitchInterval(nextInterval);
      }
    });
  });

  void loadActiveFrame(params.interval, { silent: true }).catch((error) => {
    console.error("Initial frame load failed", error);
  });
  void refreshModels();
  connect();
}

document.addEventListener("DOMContentLoaded", () => {
  initIndexPage();
  initPredictPage();
});
