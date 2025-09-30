(function (global) {
  "use strict";

  const STORAGE_KEY = "shared-candles-store";
  const DEFAULT_MAX_BARS = 2000;
  const REMOTE_ENDPOINT = "/shared-candles";
  const memoryCache = new Map();
  const remoteQueue = new Map();
  let storageAvailable = null;
  let remoteTimer = null;

  function canUseLocalStorage() {
    if (storageAvailable !== null) {
      return storageAvailable;
    }
    try {
      if (!global || typeof global.localStorage === "undefined") {
        storageAvailable = false;
        return storageAvailable;
      }
      const testKey = "__shared_candles_test__";
      global.localStorage.setItem(testKey, "1");
      global.localStorage.removeItem(testKey);
      storageAvailable = true;
    } catch (error) {
      console.warn("SharedCandles: localStorage unavailable", error);
      storageAvailable = false;
    }
    return storageAvailable;
  }

  function loadStore() {
    if (!canUseLocalStorage()) {
      return {};
    }
    try {
      const raw = global.localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
      console.warn("SharedCandles: failed to parse store", error);
      return {};
    }
  }

  function saveStore(store) {
    if (!canUseLocalStorage()) {
      return;
    }
    try {
      global.localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    } catch (error) {
      console.warn("SharedCandles: failed to persist store", error);
    }
  }

  function makeKey(symbol, interval) {
    const safeSymbol = (symbol || "").trim().toUpperCase();
    const safeInterval = (interval || "").trim().toLowerCase();
    return `${safeSymbol}|${safeInterval}`;
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const timeSeconds = Number(bar.time ?? bar.t ?? Math.floor(Number(bar.ts_ms_utc ?? 0) / 1000));
    const open = Number(bar.open ?? bar.o);
    const high = Number(bar.high ?? bar.h ?? open);
    const low = Number(bar.low ?? bar.l ?? open);
    const close = Number(bar.close ?? bar.c ?? open);
    let tsMs = Number(bar.ts_ms_utc ?? bar.t ?? 0);
    if (!Number.isFinite(tsMs)) {
      tsMs = Number.isFinite(timeSeconds) ? timeSeconds * 1000 : NaN;
    }
    const lastUpdate = Number(bar.last_update_ms ?? bar.lastUpdateMs ?? bar.last_update ?? bar.lastupdate);
    if (
      !Number.isFinite(timeSeconds) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }
    return {
      time: Math.floor(timeSeconds),
      open,
      high,
      low,
      close,
      ts_ms_utc: Math.floor(tsMs),
      ...(Number.isFinite(lastUpdate) ? { last_update_ms: Math.floor(lastUpdate) } : {}),
    };
  }

  function mergeBars(existing, incoming, maxBars) {
    const index = new Map();
    const merged = [];
    for (const bar of existing) {
      if (!bar) continue;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) continue;
      index.set(time, merged.length);
      merged.push(bar);
    }
    for (const bar of incoming) {
      if (!bar) continue;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) continue;
      if (index.has(time)) {
        merged[index.get(time)] = bar;
      } else {
        index.set(time, merged.length);
        merged.push(bar);
      }
    }
    merged.sort((a, b) => a.time - b.time);
    const limit = Math.max(1, Number(maxBars) || DEFAULT_MAX_BARS);
    return merged.length > limit ? merged.slice(merged.length - limit) : merged;
  }

  function readEntry(key) {
    if (memoryCache.has(key)) {
      return memoryCache.get(key);
    }
    const store = loadStore();
    const entry = store[key] || null;
    if (entry) {
      memoryCache.set(key, entry);
    }
    return entry;
  }

  function writeEntry(key, entry) {
    if (entry) {
      memoryCache.set(key, entry);
    } else {
      memoryCache.delete(key);
    }
    if (!canUseLocalStorage()) {
      return;
    }
    const store = loadStore();
    if (entry) {
      store[key] = entry;
    } else {
      delete store[key];
    }
    saveStore(store);
  }

  function scheduleRemoteFlush() {
    if (remoteTimer) return;
    remoteTimer = setTimeout(() => {
      remoteTimer = null;
      flushRemoteQueue().catch((error) => {
        console.warn("SharedCandles: remote sync failed", error);
      });
    }, 250);
  }

  function queueRemoteSync(symbol, interval, candles, options = {}) {
    if (!Array.isArray(candles) || !candles.length) return;
    if (!symbol || !interval) return;
    const key = makeKey(symbol, interval);
    const normalizedSymbol = symbol.trim().toUpperCase();
    const normalizedInterval = interval.trim().toLowerCase();
    const intervalMs = Number.isFinite(Number(options.intervalMs))
      ? Number(options.intervalMs)
      : null;
    const lastUpdateMs = Number.isFinite(Number(options.lastUpdateMs))
      ? Number(options.lastUpdateMs)
      : null;
    const maxBars = Number.isFinite(Number(options.maxBars))
      ? Number(options.maxBars)
      : null;
    const reset = Boolean(options.reset);

    const existing = remoteQueue.get(key);
    if (reset) {
      remoteQueue.set(key, {
        symbol: normalizedSymbol,
        interval: normalizedInterval,
        candles: candles.slice(),
        reset: true,
        intervalMs,
        lastUpdateMs,
        maxBars,
      });
    } else if (existing) {
      const mergedCandles = mergeBars(
        existing.candles,
        candles,
        maxBars || existing.maxBars || DEFAULT_MAX_BARS,
      );
      remoteQueue.set(key, {
        symbol: existing.symbol,
        interval: existing.interval,
        candles: mergedCandles,
        reset: existing.reset,
        intervalMs: intervalMs ?? existing.intervalMs ?? null,
        lastUpdateMs: lastUpdateMs ?? existing.lastUpdateMs ?? null,
        maxBars: maxBars ?? existing.maxBars ?? null,
      });
    } else {
      remoteQueue.set(key, {
        symbol: normalizedSymbol,
        interval: normalizedInterval,
        candles: candles.slice(),
        reset: false,
        intervalMs,
        lastUpdateMs,
        maxBars,
      });
    }

    scheduleRemoteFlush();
  }

  async function flushRemoteQueue() {
    if (!remoteQueue.size) return;
    const entries = Array.from(remoteQueue.values());
    remoteQueue.clear();
    for (const entry of entries) {
      const body = {
        symbol: entry.symbol,
        interval: entry.interval,
        candles: entry.candles,
        reset: Boolean(entry.reset),
        intervalMs: entry.intervalMs,
        lastUpdateMs: entry.lastUpdateMs,
      };
      if (Number.isFinite(Number(entry.maxBars))) {
        body.maxBars = Number(entry.maxBars);
      }
      try {
        const response = await fetch(REMOTE_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        console.warn("SharedCandles: remote sync error", error);
      }
    }
  }

  async function fetchRemote(symbol, interval) {
    if (!symbol || !interval) return null;
    const params = new URLSearchParams();
    params.set("symbol", symbol.trim().toUpperCase());
    params.set("interval", interval.trim().toLowerCase());
    try {
      const response = await fetch(`${REMOTE_ENDPOINT}?${params.toString()}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        if (response.status === 404) return null;
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (!data || typeof data !== "object") return null;
      const bars = Array.isArray(data.candles)
        ? data.candles.map((bar) => normaliseBar(bar)).filter(Boolean)
        : [];
      return {
        candles: bars,
        intervalMs: Number(data.intervalMs) || null,
        lastUpdateMs: Number(data.lastUpdateMs) || null,
        updatedAt: Number(data.updatedAt) || null,
      };
    } catch (error) {
      console.warn("SharedCandles: remote fetch failed", error);
      return null;
    }
  }

  function get(symbol, interval) {
    const key = makeKey(symbol, interval);
    const entry = readEntry(key);
    if (!entry) return null;
    const bars = Array.isArray(entry.candles)
      ? entry.candles.map((bar) => normaliseBar(bar)).filter(Boolean)
      : [];
    if (!bars.length) return null;
    return {
      candles: bars,
      intervalMs: Number(entry.intervalMs) || null,
      lastUpdateMs: Number(entry.lastUpdateMs) || null,
      updatedAt: Number(entry.updatedAt) || null,
    };
  }

  function merge(symbol, interval, candles, options = {}) {
    const key = makeKey(symbol, interval);
    const normalizedIncoming = Array.isArray(candles)
      ? candles.map((bar) => normaliseBar(bar)).filter(Boolean)
      : [];
    if (!normalizedIncoming.length) {
      return get(symbol, interval)?.candles || [];
    }
    const reset = Boolean(options.reset);
    const syncRemote = options.syncRemote !== false;
    const existingEntry = reset ? null : readEntry(key);
    const existingBars = existingEntry && Array.isArray(existingEntry.candles)
      ? existingEntry.candles.map((bar) => normaliseBar(bar)).filter(Boolean)
      : [];
    const mergedBars = mergeBars(existingBars, normalizedIncoming, options.maxBars);
    const nextEntry = {
      candles: mergedBars,
      intervalMs: Number.isFinite(Number(options.intervalMs))
        ? Number(options.intervalMs)
        : existingEntry?.intervalMs ?? null,
      lastUpdateMs: Number.isFinite(Number(options.lastUpdateMs))
        ? Number(options.lastUpdateMs)
        : existingEntry?.lastUpdateMs ?? null,
      updatedAt: Date.now(),
    };
    writeEntry(key, nextEntry);
    if (syncRemote) {
      queueRemoteSync(symbol, interval, normalizedIncoming, {
        reset,
        intervalMs: nextEntry.intervalMs,
        lastUpdateMs: nextEntry.lastUpdateMs,
        maxBars: options.maxBars,
      });
    }
    return mergedBars;
  }

  function clear(symbol, interval) {
    if (symbol || interval) {
      const key = makeKey(symbol, interval);
      writeEntry(key, null);
      return;
    }
    memoryCache.clear();
    if (canUseLocalStorage()) {
      saveStore({});
    }
  }

  global.SharedCandles = {
    get,
    merge,
    clear,
    fetchRemote,
  };
})(typeof window !== "undefined" ? window : globalThis);
