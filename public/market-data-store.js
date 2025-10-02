(function (global) {
  "use strict";

  const BinanceCandles = global.BinanceCandles;

  if (!BinanceCandles) {
    console.error("BinanceCandles dependency is missing");
    return;
  }

  const DEFAULT_HISTORY_LIMIT = 600;
  const DEFAULT_POLL_INTERVAL_MS = 2000;
  const WS_HOST = "wss://fstream.binance.com";

  function clampLimit(limit) {
    const numeric = Number(limit);
    if (!Number.isFinite(numeric)) return DEFAULT_HISTORY_LIMIT;
    return Math.max(50, Math.min(1500, Math.floor(numeric)));
  }

  function normaliseSymbol(symbol) {
    return (symbol || "").trim().toUpperCase();
  }

  function normaliseInterval(interval) {
    return (interval || "1m").trim();
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const open = Number(bar.open ?? bar.o);
    const high = Number(bar.high ?? bar.h);
    const low = Number(bar.low ?? bar.l);
    const close = Number(bar.close ?? bar.c);
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
    return {
      time,
      open,
      high,
      low,
      close,
      ts_ms_utc: time * 1000,
    };
  }

  function inferTickSize(price) {
    if (!Number.isFinite(price) || price <= 0) {
      return 0.5;
    }
    const absPrice = Math.abs(price);
    if (absPrice >= 1000) return 0.5;
    if (absPrice >= 100) return 0.1;
    if (absPrice >= 10) return 0.01;
    if (absPrice >= 1) return 0.001;
    if (absPrice >= 0.1) return 0.0001;
    return 0.00001;
  }

  function computeMeta({ candles, interval, lastStreamPrice, lastStreamTs }) {
    const last = candles.length ? candles[candles.length - 1] : null;
    if (!last) {
      return {
        interval,
        last_price: null,
        last_ts_ms: null,
        last_tf: interval,
        age_sec: null,
        stale: true,
        mismatch: false,
      };
    }
    const now = Date.now();
    const ageSec = Math.max(0, Math.round((now - last.ts_ms_utc) / 1000));
    const stale = ageSec > 300;
    let mismatch = false;
    if (Number.isFinite(lastStreamPrice) && Number.isFinite(lastStreamTs)) {
      const diff = Math.abs(lastStreamPrice - last.close);
      const tick = inferTickSize(last.close);
      const lagMs = now - Number(lastStreamTs);
      mismatch = diff > tick + 1e-9 && lagMs > 2000;
    }
    return {
      interval,
      last_price: last.close,
      last_ts_ms: last.ts_ms_utc,
      last_tf: interval,
      age_sec: ageSec,
      stale,
      mismatch,
    };
  }

  class MarketDataStore {
    constructor(options = {}) {
      this.symbol = normaliseSymbol(options.symbol) || "BTCUSDT";
      this.interval = normaliseInterval(options.interval || "1m");
      this.historyLimit = clampLimit(options.historyLimit || DEFAULT_HISTORY_LIMIT);
      this.pollIntervalMs = Math.max(1000, Number(options.pollIntervalMs) || DEFAULT_POLL_INTERVAL_MS);
      this.listeners = new Set();
      this.candles = [];
      this.ws = null;
      this.wsTimer = null;
      this.pollTimer = null;
      this.loadingHistory = false;
      this.connected = false;
      this.lastStreamPrice = null;
      this.lastStreamTs = null;
      this._initialised = false;
      this._pendingHistory = null;
    }

    subscribe(handler) {
      if (typeof handler !== "function") return () => {};
      this.listeners.add(handler);
      return () => {
        this.listeners.delete(handler);
      };
    }

    _emit(event) {
      for (const handler of Array.from(this.listeners)) {
        try {
          handler(event, this);
        } catch (error) {
          console.warn("MarketDataStore listener failed", error);
        }
      }
    }

    start() {
      if (this._initialised) return;
      this._initialised = true;
      this._loadHistory().catch((error) => {
        console.error("MarketDataStore history failed", error);
        this._emit({ type: "error", error });
      });
      this._ensureRealtime();
      this._ensurePolling();
    }

    stop() {
      this._initialised = false;
      this._teardownWs();
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    }

    async _loadHistory(force = false) {
      if (this.loadingHistory && !force) {
        return this._pendingHistory;
      }
      this.loadingHistory = true;
      const loadPromise = BinanceCandles.fetchHistory(this.symbol, this.interval, this.historyLimit)
        .then((rows) => rows.map((bar) => normaliseBar(bar)).filter(Boolean))
        .then((bars) => {
          this.candles = bars;
          this.loadingHistory = false;
          const meta = computeMeta({
            candles: this.candles,
            interval: this.interval,
            lastStreamPrice: this.lastStreamPrice,
            lastStreamTs: this.lastStreamTs,
          });
          this._emit({ type: "snapshot", symbol: this.symbol, interval: this.interval, candles: this.candles.slice(), meta });
          return bars;
        })
        .catch((error) => {
          this.loadingHistory = false;
          throw error;
        });
      this._pendingHistory = loadPromise;
      return loadPromise;
    }

    _ensureRealtime() {
      this._teardownWs();
      const urlSymbol = this.symbol.toLowerCase();
      const streamUrl = `${WS_HOST}/ws/${urlSymbol}@kline_${this.interval}`;
      try {
        const ws = new WebSocket(streamUrl);
        this.ws = ws;
        ws.onopen = () => {
          this.connected = true;
          this._emit({ type: "status", status: "connected", symbol: this.symbol, interval: this.interval });
        };
        ws.onerror = (event) => {
          console.error("MarketDataStore websocket error", event);
          this._emit({ type: "status", status: "error", symbol: this.symbol, interval: this.interval });
          ws.close();
        };
        ws.onclose = () => {
          if (this.ws === ws) {
            this.connected = false;
            this._emit({ type: "status", status: "closed", symbol: this.symbol, interval: this.interval });
            this._scheduleReconnect();
          }
        };
        ws.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            const bar = BinanceCandles.barFromWs(payload);
            const normalised = normaliseBar(bar);
            if (!normalised) return;
            this.lastStreamPrice = normalised.close;
            this.lastStreamTs = Date.now();
            this._mergeBars([normalised]);
            const meta = computeMeta({
              candles: this.candles,
              interval: this.interval,
              lastStreamPrice: this.lastStreamPrice,
              lastStreamTs: this.lastStreamTs,
            });
            this._emit({
              type: "update",
              symbol: this.symbol,
              interval: this.interval,
              candle: normalised,
              candles: this.candles.slice(-10),
              meta,
            });
          } catch (error) {
            console.error("MarketDataStore message parse failed", error);
          }
        };
      } catch (error) {
        console.error("MarketDataStore websocket setup failed", error);
        this._scheduleReconnect();
      }
    }

    _scheduleReconnect() {
      if (this.wsTimer) return;
      this.wsTimer = setTimeout(() => {
        this.wsTimer = null;
        if (!this._initialised) return;
        this._ensureRealtime();
      }, 3000);
    }

    _teardownWs() {
      if (this.ws) {
        try {
          this.ws.onopen = null;
          this.ws.onclose = null;
          this.ws.onerror = null;
          this.ws.onmessage = null;
          this.ws.close(1000);
        } catch (error) {
          console.warn("MarketDataStore ws teardown failed", error);
        }
      }
      this.ws = null;
      if (this.wsTimer) {
        clearTimeout(this.wsTimer);
        this.wsTimer = null;
      }
    }

    _ensurePolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
      }
      this.pollTimer = setInterval(() => {
        this._pollLatest().catch((error) => {
          console.warn("MarketDataStore poll failed", error);
        });
      }, this.pollIntervalMs);
    }

    async _pollLatest() {
      const last = this.candles.length ? this.candles[this.candles.length - 1] : null;
      const startTime = last ? last.ts_ms_utc - 5 * 60 * 1000 : undefined;
      const params = new URLSearchParams();
      params.set("symbol", this.symbol);
      params.set("interval", this.interval);
      if (Number.isFinite(startTime)) {
        params.set("startTime", String(Math.max(0, Math.floor(startTime))));
      }
      params.set("limit", String(Math.min(150, Math.max(10, Math.floor(this.historyLimit / 4)))));
      const url = `https://fapi.binance.com/fapi/v1/klines?${params.toString()}`;
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`poll klines ${response.status}`);
      }
      const rows = await response.json();
      const bars = BinanceCandles.transformKlines(rows).map((bar) => normaliseBar(bar)).filter(Boolean);
      if (!bars.length) return;
      this._mergeBars(bars);
      const meta = computeMeta({
        candles: this.candles,
        interval: this.interval,
        lastStreamPrice: this.lastStreamPrice,
        lastStreamTs: this.lastStreamTs,
      });
      this._emit({ type: "poll", symbol: this.symbol, interval: this.interval, candles: bars, meta });
    }

    _mergeBars(bars) {
      if (!Array.isArray(bars) || !bars.length) return;
      const index = new Map();
      this.candles.forEach((bar, idx) => {
        index.set(Number(bar.time), idx);
      });
      let changed = false;
      for (const bar of bars) {
        if (!bar) continue;
        const key = Number(bar.time);
        if (!Number.isFinite(key)) continue;
        if (index.has(key)) {
          const targetIdx = index.get(key);
          const existing = this.candles[targetIdx];
          if (
            existing.open !== bar.open ||
            existing.high !== bar.high ||
            existing.low !== bar.low ||
            existing.close !== bar.close
          ) {
            this.candles[targetIdx] = bar;
            changed = true;
          }
        } else {
          index.set(key, this.candles.length);
          this.candles.push(bar);
          changed = true;
        }
      }
      if (changed) {
        this.candles.sort((a, b) => Number(a.time) - Number(b.time));
        if (this.candles.length > this.historyLimit) {
          this.candles = this.candles.slice(this.candles.length - this.historyLimit);
        }
      }
    }

    setSymbol(symbol, interval) {
      const nextSymbol = normaliseSymbol(symbol) || this.symbol;
      const nextInterval = normaliseInterval(interval) || this.interval;
      const symbolChanged = nextSymbol !== this.symbol;
      const intervalChanged = nextInterval !== this.interval;
      if (!symbolChanged && !intervalChanged) {
        return;
      }
      this.symbol = nextSymbol;
      this.interval = nextInterval;
      this.candles = [];
      this.lastStreamPrice = null;
      this.lastStreamTs = null;
      this._emit({ type: "status", status: "restarting", symbol: this.symbol, interval: this.interval });
      this._loadHistory(true).catch((error) => {
        console.error("MarketDataStore reload failed", error);
      });
      this._ensureRealtime();
      this._ensurePolling();
    }

    restart() {
      this._loadHistory(true).catch((error) => {
        console.error("MarketDataStore restart failed", error);
      });
      this._ensureRealtime();
      this._ensurePolling();
    }

    getMeta() {
      return computeMeta({
        candles: this.candles,
        interval: this.interval,
        lastStreamPrice: this.lastStreamPrice,
        lastStreamTs: this.lastStreamTs,
      });
    }

    getLastCandle() {
      return this.candles.length ? this.candles[this.candles.length - 1] : null;
    }
  }

  global.MarketDataStore = MarketDataStore;
})(typeof window !== "undefined" ? window : this);
