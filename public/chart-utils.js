(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ChartUtils = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const DEFAULT_GREY = Object.freeze([0.45, 0.55]);
  const DEFAULT_PALETTE = Object.freeze({
    up: "rgba(34, 197, 94, 0.9)",
    down: "rgba(239, 68, 68, 0.9)",
    neutral: "rgba(148, 163, 184, 0.4)",
  });

  function normaliseGreyZone(grey) {
    if (!Array.isArray(grey) || grey.length < 2) return DEFAULT_GREY;
    const lower = Number(grey[0]);
    const upper = Number(grey[1]);
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) return DEFAULT_GREY;
    if (lower > upper) return [upper, lower];
    return [lower, upper];
  }

  function resolveConfidence(mark) {
    if (!mark) return NaN;
    const candidates = [mark.pred_conf, mark.confidence, mark.prob_up, mark.prob_down, mark.prob_flat]
      .map(Number)
      .filter((value) => Number.isFinite(value));
    if (!candidates.length) return NaN;
    return Math.max(...candidates.map((value) => Math.abs(value)));
  }

  function resolveDirection(mark) {
    if (!mark) return "flat";
    if (typeof mark.pred_dir === "string") return mark.pred_dir;
    if (typeof mark.signal === "string") return mark.signal;
    const up = Number(mark.prob_up);
    const down = Number(mark.prob_down);
    const flat = Number(mark.prob_flat);
    if (Number.isFinite(up) || Number.isFinite(down) || Number.isFinite(flat)) {
      const triples = [
        { value: up, label: "up" },
        { value: flat, label: "flat" },
        { value: down, label: "down" },
      ];
      const best = triples.reduce((acc, item) => {
        if (!Number.isFinite(item.value)) return acc;
        if (!acc || item.value > acc.value) return item;
        return acc;
      }, null);
      return best ? best.label : "flat";
    }
    return "flat";
  }

  function shouldHighlight(mark, confidence, grey) {
    if (!mark) return false;
    const direction = resolveDirection(mark);
    if (direction === "flat" || direction === "neutral") return false;
    if (!Number.isFinite(confidence)) return false;
    if (!Array.isArray(grey) || grey.length < 2) return confidence >= 0.55 || confidence <= 0.45;
    return confidence >= grey[1] || confidence <= grey[0];
  }

  function decorateCandle(bar, prediction, palette) {
    const output = Object.assign({}, bar);
    if (!prediction) return output;
    const mark = typeof prediction === "object" ? prediction : {};
    const colours = palette || DEFAULT_PALETTE;
    const grey = normaliseGreyZone(mark.grey_zone || mark.greyZone || DEFAULT_GREY);
    const confidence = resolveConfidence(mark);
    const direction = resolveDirection(mark);
    if (shouldHighlight(mark, confidence, grey)) {
      const colour = colours[direction] || colours.neutral;
      output.borderColor = colour;
      output.borderUpColor = colour;
      output.borderDownColor = colour;
    } else if (colours.neutral) {
      output.borderColor = colours.neutral;
      output.borderUpColor = colours.neutral;
      output.borderDownColor = colours.neutral;
    }
    return output;
  }

  function buildPredictionMap(predictions) {
    const map = new Map();
    if (!predictions) return map;
    if (predictions instanceof Map) {
      predictions.forEach((value, key) => {
        const timeKey = Number(key);
        if (Number.isFinite(timeKey)) map.set(timeKey, value);
      });
      return map;
    }
    if (Array.isArray(predictions)) {
      predictions.forEach((item) => {
        if (!item) return;
        const timeKey = Number(item.ts_ms_utc ?? item.time);
        if (Number.isFinite(timeKey)) map.set(timeKey, item);
      });
      return map;
    }
    if (typeof predictions === "object") {
      Object.keys(predictions).forEach((key) => {
        const timeKey = Number(key);
        if (Number.isFinite(timeKey)) map.set(timeKey, predictions[key]);
      });
    }
    return map;
  }

  function applyPredictions(bars, predictions, palette) {
    const map = buildPredictionMap(predictions);
    if (!Array.isArray(bars)) return [];
    return bars.map((bar) => {
      const timeKey = Number(bar && (bar.ts_ms_utc ?? bar.time));
      const mark = Number.isFinite(timeKey) ? map.get(timeKey) : undefined;
      return decorateCandle(bar, mark, palette);
    });
  }

  return {
    thresholds: { grey: DEFAULT_GREY },
    normaliseGreyZone,
    resolveDirection,
    resolveConfidence,
    shouldHighlight,
    decorateCandle,
    applyPredictions,
  };
});
