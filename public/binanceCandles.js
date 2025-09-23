(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.BinanceCandles = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const BASE_URL = "https://api.binance.com/api/v3/klines";

  /**
   * @typedef {Object} Bar
   * @property {number} time Seconds since epoch expected by Lightweight Charts
   * @property {number} open
   * @property {number} high
   * @property {number} low
   * @property {number} close
   */

  /**
   * @typedef {[number, string, string, string, string, string, number, ...any[]]} KlineRow
   * Binance REST kline payload.
   */

  /**
   * @typedef {Object} WsKlinePayload
   * @property {"kline"} e
   * @property {number} E
   * @property {string} s
   * @property {{ t: number; o: string; h: string; l: string; c: string; x: boolean; i: string }} k
   */

  function toNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : NaN;
  }

  function toTimeSeconds(ms) {
    return Math.floor(ms / 1000);
  }

  function dedupeByTime(bars) {
    const seen = new Map();
    const result = [];
    for (const bar of Array.isArray(bars) ? bars : []) {
      if (!bar || !Number.isFinite(bar.time)) continue;
      const time = Number(bar.time);
      if (!Number.isFinite(time)) continue;
      if (seen.has(time)) {
        result[seen.get(time)] = bar;
      } else {
        seen.set(time, result.length);
        result.push(bar);
      }
    }
    return result;
  }

  function transformKlines(rows) {
    if (!Array.isArray(rows)) return [];
    const mapped = [];
    for (const row of rows) {
      if (!row) continue;
      const openTime = toNumber(Array.isArray(row) ? row[0] : row.openTime ?? row.open_time);
      const open = toNumber(Array.isArray(row) ? row[1] : row.open);
      const high = toNumber(Array.isArray(row) ? row[2] : row.high);
      const low = toNumber(Array.isArray(row) ? row[3] : row.low);
      const close = toNumber(Array.isArray(row) ? row[4] : row.close);
      if (
        !Number.isFinite(openTime) ||
        !Number.isFinite(open) ||
        !Number.isFinite(high) ||
        !Number.isFinite(low) ||
        !Number.isFinite(close)
      ) {
        continue;
      }
      mapped.push({
        time: toTimeSeconds(openTime),
        open,
        high,
        low,
        close,
      });
    }
    mapped.sort((a, b) => a.time - b.time);
    return dedupeByTime(mapped);
  }

  function barFromWs(msg) {
    if (!msg || typeof msg !== "object") {
      throw new Error("Invalid kline payload");
    }
    const source = msg.k || msg;
    const time = toNumber(source.t ?? source.time);
    const open = toNumber(source.o ?? source.open);
    const high = toNumber(source.h ?? source.high);
    const low = toNumber(source.l ?? source.low);
    const close = toNumber(source.c ?? source.close);
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      throw new Error("Incomplete kline payload");
    }
    return {
      time: toTimeSeconds(time),
      open,
      high,
      low,
      close,
    };
  }

  async function fetchHistory(symbol, interval, limit = 500) {
    if (!symbol || !interval) {
      throw new Error("symbol and interval are required");
    }
    const url = `${BASE_URL}?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&limit=${encodeURIComponent(
      limit,
    )}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`klines ${resp.status}`);
    }
    const rows = /** @type {KlineRow[]} */ (await resp.json());
    return transformKlines(rows);
  }

  return {
    transformKlines,
    dedupeByTime,
    barFromWs,
    fetchHistory,
  };
});
