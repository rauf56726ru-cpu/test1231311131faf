(function (global) {
  "use strict";

  const STORAGE_KEY = "shared-candles-store";
  const LEADER_PREFIX = "shared-candles-leader:";
  const REMOTE_ENDPOINT = "/shared-candles";
  const DEFAULT_MAX_BARS = 2000;
  const DEBOUNCE_MIN_MS = 1000;
  const DEBOUNCE_MAX_MS = 2000;
  const LEADER_TTL_MS = 4000;
  const LEADER_REFRESH_THRESHOLD_MS = 800;
  const instanceId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

  const memoryCache = new Map();
  const states = new Map();
  let storageAvailable = null;

  function now() {
    return Date.now();
  }

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
    if (!safeSymbol || !safeInterval) {
      throw new Error("symbol and interval are required");
    }
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
    const volume = Number(bar.volume ?? bar.v);
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
    const payload = {
      time: Math.floor(timeSeconds),
      open,
      high,
      low,
      close,
      ts_ms_utc: Math.floor(tsMs),
    };
    if (Number.isFinite(volume)) {
      payload.volume = Number(volume);
    }
    if (Number.isFinite(lastUpdate)) {
      payload.last_update_ms = Math.floor(lastUpdate);
    }
    return payload;
  }

  function mergeBars(existing, incoming, maxBars) {
    const index = new Map();
    const merged = [];
    for (const bar of existing) {
      if (!bar) continue;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) continue;
      const normalised = normaliseBar(bar);
      if (!normalised) continue;
      index.set(normalised.time, normalised);
    }
    for (const bar of incoming) {
      if (!bar) continue;
      const normalised = normaliseBar(bar);
      if (!normalised) continue;
      index.set(normalised.time, normalised);
    }
    for (const entry of Array.from(index.entries()).sort((a, b) => a[0] - b[0])) {
      merged.push(entry[1]);
    }
    const limit = Math.max(1, Number.isFinite(maxBars) ? Number(maxBars) : DEFAULT_MAX_BARS);
    return merged.length > limit ? merged.slice(merged.length - limit) : merged;
  }

  function readEntry(key) {
    if (memoryCache.has(key)) {
      return memoryCache.get(key);
    }
    const store = loadStore();
    const entry = store[key];
    if (!entry || typeof entry !== "object") {
      return null;
    }
    const cloned = {
      candles: Array.isArray(entry.candles) ? entry.candles.slice() : [],
      intervalMs: Number.isFinite(entry.intervalMs) ? Number(entry.intervalMs) : null,
      lastUpdateMs: Number.isFinite(entry.lastUpdateMs) ? Number(entry.lastUpdateMs) : null,
      updatedAt: Number.isFinite(entry.updatedAt) ? Number(entry.updatedAt) : null,
      maxBars: Number.isFinite(entry.maxBars) ? Number(entry.maxBars) : null,
    };
    memoryCache.set(key, cloned);
    return cloned;
  }

  function writeEntry(key, entry) {
    if (entry) {
      const payload = {
        candles: Array.isArray(entry.candles) ? entry.candles.map((bar) => ({ ...bar })) : [],
        intervalMs: Number.isFinite(entry.intervalMs) ? Number(entry.intervalMs) : null,
        lastUpdateMs: Number.isFinite(entry.lastUpdateMs) ? Number(entry.lastUpdateMs) : null,
        updatedAt: Number.isFinite(entry.updatedAt) ? Number(entry.updatedAt) : now(),
        maxBars: Number.isFinite(entry.maxBars) ? Number(entry.maxBars) : null,
      };
      memoryCache.set(key, payload);
      if (canUseLocalStorage()) {
        const store = loadStore();
        store[key] = payload;
        saveStore(store);
      }
    } else {
      memoryCache.delete(key);
      if (canUseLocalStorage()) {
        const store = loadStore();
        delete store[key];
        saveStore(store);
      }
    }
  }

  function removeLeader(key) {
    if (!canUseLocalStorage()) return;
    try {
      global.localStorage.removeItem(`${LEADER_PREFIX}${key}`);
    } catch (error) {
      console.warn("SharedCandles: failed to remove leader lock", error);
    }
  }

  function ensureState(key, symbol, interval) {
    if (states.has(key)) {
      return states.get(key);
    }
    const entry = readEntry(key);
    const state = {
      key,
      symbol: (symbol || "").trim().toUpperCase(),
      interval: (interval || "").trim().toLowerCase(),
      pending: new Map(),
      resetPending: false,
      pendingLastUpdateMs: null,
      flushTimer: null,
      flushDueTime: null,
      inFlight: false,
      dirty: false,
      flushPromise: null,
      lastFlushMs: 0,
      lastUpdateMs: entry && Number.isFinite(entry.lastUpdateMs) ? Number(entry.lastUpdateMs) : null,
      intervalMs: entry && Number.isFinite(entry.intervalMs) ? Number(entry.intervalMs) : null,
      maxBars: entry && Number.isFinite(entry.maxBars) ? Number(entry.maxBars) : DEFAULT_MAX_BARS,
      debounceMs:
        Math.floor(Math.random() * (DEBOUNCE_MAX_MS - DEBOUNCE_MIN_MS + 1)) + DEBOUNCE_MIN_MS,
      leaderExpiresAt: 0,
    };
    states.set(key, state);
    return state;
  }

  function trimPending(state) {
    const limit = Math.max(1, Number.isFinite(state.maxBars) ? Number(state.maxBars) : DEFAULT_MAX_BARS);
    const entries = Array.from(state.pending.entries()).sort((a, b) => Number(a[0]) - Number(b[0]));
    const trimmed = entries.length > limit ? entries.slice(entries.length - limit) : entries;
    state.pending = new Map(trimmed.map(([time, bar]) => [Number(time), { ...bar }]));
  }

  function ensureLeader(state) {
    if (!canUseLocalStorage()) {
      return true;
    }
    const lockKey = `${LEADER_PREFIX}${state.key}`;
    const nowMs = now();
    let ownerId = null;
    let expiresAt = 0;
    try {
      const raw = global.localStorage.getItem(lockKey);
      if (raw) {
        const parts = String(raw).split("|");
        if (parts.length === 2) {
          ownerId = parts[0];
          expiresAt = Number(parts[1]);
        }
      }
    } catch (error) {
      console.warn("SharedCandles: failed to read leader lock", error);
      return true;
    }

    const refresh = () => {
      const nextExpiry = nowMs + LEADER_TTL_MS;
      try {
        global.localStorage.setItem(lockKey, `${instanceId}|${nextExpiry}`);
      } catch (error) {
        console.warn("SharedCandles: failed to extend leader lock", error);
      }
      state.leaderExpiresAt = nextExpiry;
    };

    if (!ownerId || !Number.isFinite(expiresAt) || expiresAt <= nowMs) {
      refresh();
      return true;
    }
    if (ownerId === instanceId) {
      if (expiresAt - nowMs <= LEADER_REFRESH_THRESHOLD_MS) {
        refresh();
      } else {
        state.leaderExpiresAt = expiresAt;
      }
      return true;
    }
    return false;
  }

  function scheduleFlush(state, { immediate = false, delayMs = null } = {}) {
    let delay;
    if (immediate) {
      delay = 0;
    } else if (Number.isFinite(delayMs)) {
      delay = Math.max(0, Number(delayMs));
    } else {
      delay = state.debounceMs;
    }
    const dueTime = now() + delay;
    if (state.flushTimer && state.flushDueTime !== null && state.flushDueTime <= dueTime) {
      return;
    }
    if (state.flushTimer) {
      clearTimeout(state.flushTimer);
    }
    state.flushDueTime = dueTime;
    state.flushTimer = setTimeout(() => {
      state.flushTimer = null;
      state.flushDueTime = null;
      triggerFlush(state).catch((error) => {
        console.warn("SharedCandles: remote flush failed", error);
      });
    }, delay);
  }

  function restoreSnapshot(state, snapshot) {
    const buffer = new Map(state.pending);
    if (snapshot.reset) {
      buffer.clear();
      state.resetPending = true;
    }
    for (const [time, bar] of snapshot.entries) {
      buffer.set(Number(time), { ...bar });
    }
    state.pending = buffer;
    trimPending(state);
    if (Number.isFinite(snapshot.lastUpdateMs)) {
      const candidate = Number(snapshot.lastUpdateMs);
      const current = Number.isFinite(state.pendingLastUpdateMs)
        ? Number(state.pendingLastUpdateMs)
        : -Infinity;
      state.pendingLastUpdateMs = Math.max(current, candidate);
    }
    state.dirty = true;
  }

  function triggerFlush(state, { force = false } = {}) {
    if (state.inFlight) {
      state.dirty = true;
      return state.flushPromise || Promise.resolve();
    }
    if (!state.pending.size && !state.resetPending) {
      if (state.flushTimer) {
        clearTimeout(state.flushTimer);
        state.flushTimer = null;
        state.flushDueTime = null;
      }
      return force && state.flushPromise ? state.flushPromise : Promise.resolve();
    }
    if (!ensureLeader(state)) {
      scheduleFlush(state);
      return Promise.resolve();
    }

    const pendingEntries = Array.from(state.pending.entries());
    const resetFlag = state.resetPending;
    const payloadBars = pendingEntries.map(([, bar]) => ({ ...bar }));
    const sortedBars = payloadBars.sort((a, b) => Number(a.time) - Number(b.time));
    const trimmedBars = sortedBars.length > state.maxBars
      ? sortedBars.slice(sortedBars.length - state.maxBars)
      : sortedBars;
    const fallbackUpdate = Number.isFinite(state.lastUpdateMs) ? Number(state.lastUpdateMs) : now();
    const pendingUpdate = Number.isFinite(state.pendingLastUpdateMs)
      ? Number(state.pendingLastUpdateMs)
      : fallbackUpdate;
    const body = {
      symbol: state.symbol,
      interval: state.interval,
      candles: trimmedBars,
      reset: Boolean(resetFlag),
      lastUpdateMs: pendingUpdate,
    };
    if (Number.isFinite(state.intervalMs)) {
      body.intervalMs = Number(state.intervalMs);
    }
    if (Number.isFinite(state.maxBars)) {
      body.maxBars = Number(state.maxBars);
    }

    const snapshot = {
      entries: pendingEntries,
      reset: resetFlag,
      lastUpdateMs: pendingUpdate,
    };

    state.pending = new Map();
    state.resetPending = false;
    state.pendingLastUpdateMs = null;
    state.inFlight = true;
    state.dirty = false;

    const execute = async () => {
      let shouldRestore = false;
      let retryDelay = null;
      let rateLimited = false;
      try {
        const response = await fetch(REMOTE_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        let payload = null;
        try {
          payload = await response.json();
        } catch (error) {
          payload = null;
        }
        const status = payload && typeof payload.status === "string" ? payload.status : "ok";
        if (status === "rate_limited") {
          shouldRestore = true;
          rateLimited = true;
          retryDelay = Number.isFinite(payload?.retryAfterMs)
            ? Number(payload.retryAfterMs)
            : state.debounceMs;
          return;
        }
        const responseUpdate = Number.isFinite(payload?.lastUpdateMs)
          ? Number(payload.lastUpdateMs)
          : pendingUpdate;
        if (Number.isFinite(responseUpdate)) {
          state.lastUpdateMs = responseUpdate;
        }
        state.lastFlushMs = now();
      } catch (error) {
        shouldRestore = true;
        throw error;
      } finally {
        if (shouldRestore) {
          restoreSnapshot(state, snapshot);
        }
        state.inFlight = false;
        state.flushPromise = null;
        if (rateLimited) {
          scheduleFlush(state, { delayMs: retryDelay });
        } else if (state.dirty || state.pending.size || state.resetPending) {
          scheduleFlush(state);
        }
      }
    };

    state.flushPromise = execute();
    return state.flushPromise;
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
    let entry;
    try {
      const key = makeKey(symbol, interval);
      entry = readEntry(key);
    } catch (error) {
      return null;
    }
    if (!entry) return null;
    const bars = Array.isArray(entry.candles)
      ? entry.candles.map((bar) => normaliseBar(bar)).filter(Boolean)
      : [];
    if (!bars.length) return null;
    return {
      candles: bars,
      intervalMs: Number.isFinite(entry.intervalMs) ? Number(entry.intervalMs) : null,
      lastUpdateMs: Number.isFinite(entry.lastUpdateMs) ? Number(entry.lastUpdateMs) : null,
      updatedAt: Number.isFinite(entry.updatedAt) ? Number(entry.updatedAt) : null,
    };
  }

  function merge(symbol, interval, candles, options = {}) {
    let key;
    try {
      key = makeKey(symbol, interval);
    } catch (error) {
      console.warn("SharedCandles: merge skipped due to invalid key", error);
      return [];
    }
    const syncRemote = options.syncRemote !== false;
    const reset = Boolean(options.reset);
    const incomingLastUpdate = Number.isFinite(Number(options.lastUpdateMs))
      ? Number(options.lastUpdateMs)
      : null;
    const intervalMs = Number.isFinite(Number(options.intervalMs))
      ? Number(options.intervalMs)
      : null;
    const maxBars = Number.isFinite(Number(options.maxBars)) ? Number(options.maxBars) : null;

    const state = ensureState(key, symbol, interval);
    const entry = readEntry(key);
    const existingBars = entry && Array.isArray(entry.candles) ? entry.candles : [];
    const existingLastUpdate = Number.isFinite(entry?.lastUpdateMs) ? Number(entry.lastUpdateMs) : null;
    if (state.lastUpdateMs === null && existingLastUpdate !== null) {
      state.lastUpdateMs = existingLastUpdate;
    }

    const normalisedIncoming = Array.isArray(candles)
      ? candles.map((bar) => normaliseBar(bar)).filter(Boolean)
      : [];

    if (!reset && incomingLastUpdate !== null && state.lastUpdateMs !== null) {
      if (incomingLastUpdate <= state.lastUpdateMs) {
        return mergeBars(existingBars, [], maxBars ?? state.maxBars);
      }
    }
    if (!reset && !normalisedIncoming.length) {
      return mergeBars(existingBars, [], maxBars ?? state.maxBars);
    }

    const limit = maxBars ?? state.maxBars ?? DEFAULT_MAX_BARS;
    const mergedBars = mergeBars(reset ? [] : existingBars, normalisedIncoming, limit);
    const updatedAt = now();
    const nextLastUpdate = incomingLastUpdate ?? (reset ? null : state.lastUpdateMs);
    const nextInterval = intervalMs ?? state.intervalMs ?? null;

    writeEntry(key, {
      candles: mergedBars,
      intervalMs: nextInterval,
      lastUpdateMs: nextLastUpdate,
      updatedAt,
      maxBars: limit,
    });

    state.maxBars = limit;
    state.intervalMs = nextInterval;
    state.lastUpdateMs = Number.isFinite(nextLastUpdate) ? Number(nextLastUpdate) : state.lastUpdateMs;

    if (syncRemote && (normalisedIncoming.length || reset)) {
      if (reset) {
        state.pending.clear();
        state.resetPending = true;
      }
      for (const bar of normalisedIncoming) {
        state.pending.set(Number(bar.time), { ...bar });
      }
      trimPending(state);
      if (incomingLastUpdate !== null) {
        state.pendingLastUpdateMs = Number(incomingLastUpdate);
      } else if (state.pendingLastUpdateMs === null && state.lastUpdateMs !== null) {
        state.pendingLastUpdateMs = Number(state.lastUpdateMs);
      }
      if (state.inFlight) {
        state.dirty = true;
      }
      const immediate = Boolean(options.immediate || options.immediateFlush || reset);
      if (immediate) {
        if (state.flushTimer) {
          clearTimeout(state.flushTimer);
          state.flushTimer = null;
          state.flushDueTime = null;
        }
        triggerFlush(state, { force: true });
      } else {
        const delayMs = Number.isFinite(options.flushDelayMs) ? Number(options.flushDelayMs) : null;
        scheduleFlush(state, { delayMs });
      }
    }

    return mergedBars;
  }

  function clear(symbol, interval) {
    if (!symbol && !interval) {
      memoryCache.clear();
      if (canUseLocalStorage()) {
        saveStore({});
      }
      for (const key of states.keys()) {
        removeLeader(key);
      }
      states.clear();
      return;
    }
    let key;
    try {
      key = makeKey(symbol, interval);
    } catch (error) {
      return;
    }
    writeEntry(key, null);
    removeLeader(key);
    states.delete(key);
  }

  function flush(symbol, interval, options = {}) {
    let key;
    try {
      key = makeKey(symbol, interval);
    } catch (error) {
      return Promise.resolve();
    }
    const state = states.get(key);
    if (!state) {
      return Promise.resolve();
    }
    const force = Boolean(options.force || options.immediate);
    if (force && state.flushTimer) {
      clearTimeout(state.flushTimer);
      state.flushTimer = null;
      state.flushDueTime = null;
    }
    return triggerFlush(state, { force });
  }

  global.SharedCandles = {
    get,
    merge,
    clear,
    fetchRemote,
    flush,
  };
})(typeof window !== "undefined" ? window : globalThis);
