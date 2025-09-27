(function (global) {
  "use strict";

  const STORAGE_KEY = "shared-candles-store";
  const DEFAULT_MAX_BARS = 2000;
  const memoryCache = new Map();
  let storageAvailable = null;

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
  };
})(typeof window !== "undefined" ? window : globalThis);
