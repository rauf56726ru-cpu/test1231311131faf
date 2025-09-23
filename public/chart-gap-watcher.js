(function (global) {
  const DEFAULT_DELAY_MS = 3000;
  const DEFAULT_MAX_BARS = 2000;
  const DEFAULT_LEFT_THRESHOLD = 0.3;
  const DEFAULT_EXTRA_BARS = 5;

  function intervalToMs(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value !== "string") {
      return 60_000;
    }
    const match = value.trim().match(/^(\d+)([a-zA-Z]+)$/);
    if (!match) {
      return 60_000;
    }
    const amount = Number(match[1]);
    if (!Number.isFinite(amount)) {
      return 60_000;
    }
    const unit = match[2].toLowerCase();
    switch (unit.charAt(0)) {
      case "s":
        return amount * 1_000;
      case "h":
        return amount * 3_600_000;
      case "d":
        return amount * 86_400_000;
      default:
        return amount * 60_000;
    }
  }

  function isFiniteNumber(value) {
    return Number.isFinite(Number(value));
  }

  function toMs(seconds) {
    if (!isFiniteNumber(seconds)) return NaN;
    return Number(seconds) * 1_000;
  }

  function createKey(state, gap) {
    const start = Math.floor(Number(gap.startMs) || 0);
    const end = Math.floor(Number(gap.endMs) || 0);
    return `${state.symbol}|${state.interval}|${start}|${end}`;
  }

  function detectBackwardGap(state) {
    const candles = typeof state.getCandles === "function" ? state.getCandles() : [];
    if (!Array.isArray(candles) || candles.length === 0) {
      return null;
    }
    const first = candles[0];
    const firstMs = Number(first?.ts_ms_utc ?? (Number(first?.time) || 0) * 1000);
    if (!Number.isFinite(firstMs)) {
      return null;
    }
    const intervalMs = Number(state.intervalMs) || intervalToMs(state.interval);
    if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
      return null;
    }

    const timeRange = state.timeScale?.getVisibleRange?.();
    if (timeRange && isFiniteNumber(timeRange.from)) {
      const fromMs = Math.floor(Number(timeRange.from) * 1000);
      const gapMs = Math.floor(firstMs) - fromMs;
      if (Number.isFinite(gapMs) && gapMs > intervalMs) {
        const missingBars = Math.max(
          1,
          Math.min(state.maxBarsPerRequest, Math.ceil(gapMs / intervalMs))
        );
        const endMs = Math.floor(firstMs);
        const startMs = Math.max(0, endMs - (missingBars + state.extraBars) * intervalMs);
        if (endMs > startMs) {
          return {
            direction: "backward",
            startMs,
            endMs,
          };
        }
      }
    }

    const logical = state.timeScale?.getVisibleLogicalRange?.();
    if (!logical || !isFiniteNumber(logical.from)) {
      return null;
    }
    const fromLogical = Number(logical.from);
    const threshold = Number(state.minLeftLogical) || DEFAULT_LEFT_THRESHOLD;
    if (fromLogical >= -threshold) {
      return null;
    }
    const missingBars = Math.ceil(Math.abs(fromLogical + threshold));
    const bars = Math.max(1, Math.min(state.maxBarsPerRequest, missingBars));
    const endMs = Math.floor(firstMs);
    const startMs = Math.max(0, endMs - (bars + state.extraBars) * intervalMs);
    if (endMs <= startMs) {
      return null;
    }
    return {
      direction: "backward",
      startMs,
      endMs,
    };
  }

  function attach(options = {}) {
    const chart = options.chart;
    if (!chart || typeof chart.timeScale !== "function") {
      return null;
    }
    const requestGap = typeof options.requestGap === "function" ? options.requestGap : null;
    if (!requestGap) {
      return null;
    }

    const state = {
      chart,
      timeScale: chart.timeScale(),
      getCandles: typeof options.getCandles === "function" ? options.getCandles : () => [],
      symbol: options.symbol || "",
      interval: options.interval || "1m",
      intervalMs: Number(options.intervalMs) || intervalToMs(options.interval),
      delayMs: Number.isFinite(Number(options.delayMs)) ? Number(options.delayMs) : DEFAULT_DELAY_MS,
      minLeftLogical: Number.isFinite(Number(options.minLeftLogical))
        ? Number(options.minLeftLogical)
        : DEFAULT_LEFT_THRESHOLD,
      maxBarsPerRequest: Math.max(1, Number(options.maxBarsPerRequest) || DEFAULT_MAX_BARS),
      extraBars: Math.max(0, Number(options.extraBars) || DEFAULT_EXTRA_BARS),
      timerId: null,
      pendingKey: null,
      pendingGap: null,
      requestedKeys: new Set(),
      requestGap,
    };

    if (!state.timeScale || typeof state.timeScale.subscribeVisibleTimeRangeChange !== "function") {
      return null;
    }

    function clearTimer() {
      if (state.timerId) {
        clearTimeout(state.timerId);
        state.timerId = null;
      }
      state.pendingGap = null;
      state.pendingKey = null;
    }

    function runRequest(gap) {
      const key = createKey(state, gap);
      if (state.requestedKeys.has(key)) return;
      state.requestedKeys.add(key);
      Promise.resolve(state.requestGap({ ...gap }))
        .then((started) => {
          if (started === false) {
            state.requestedKeys.delete(key);
          }
        })
        .catch((error) => {
          console.error("gap watcher request failed", error);
          state.requestedKeys.delete(key);
        })
        .finally(() => {
          state.requestedKeys.delete(key);
          evaluate();
        });
    }

    function schedule(gap) {
      const key = createKey(state, gap);
      if (state.requestedKeys.has(key) || state.pendingKey === key) {
        return;
      }
      clearTimer();
      state.pendingGap = gap;
      state.pendingKey = key;
      state.timerId = setTimeout(() => {
        state.timerId = null;
        if (state.pendingGap) {
          runRequest(state.pendingGap);
          state.pendingGap = null;
          state.pendingKey = null;
        }
      }, state.delayMs);
    }

    function evaluate() {
      const gap = detectBackwardGap(state);
      if (gap) {
        schedule(gap);
      } else {
        clearTimer();
      }
    }

    const onRangeChange = () => {
      evaluate();
    };

    state.timeScale.subscribeVisibleTimeRangeChange(onRangeChange);
    setTimeout(evaluate, 0);

    return {
      detach() {
        clearTimer();
        if (state.timeScale && typeof state.timeScale.unsubscribeVisibleTimeRangeChange === "function") {
          state.timeScale.unsubscribeVisibleTimeRangeChange(onRangeChange);
        }
      },
      updateContext(context = {}) {
        if (context.symbol) state.symbol = context.symbol;
        if (context.interval) state.interval = context.interval;
        if (context.intervalMs) state.intervalMs = Number(context.intervalMs) || state.intervalMs;
        if (context.delayMs) state.delayMs = Number(context.delayMs) || state.delayMs;
        if (typeof context.getCandles === "function") state.getCandles = context.getCandles;
        if (typeof context.requestGap === "function") state.requestGap = context.requestGap;
        if (context.resetRequestedKeys) state.requestedKeys.clear();
        evaluate();
      },
      notifyData() {
        evaluate();
      },
    };
  }

  global.ChartGapWatcher = {
    attach(options) {
      const controller = attach(options);
      if (controller) return controller;
      return {
        detach() {},
        updateContext() {},
        notifyData() {},
      };
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
