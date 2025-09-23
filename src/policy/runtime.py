"""Runtime application of compiled policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd

from ..config import get_settings
from ..backtest.metrics import atr as atr_indicator
from ..utils.logging import get_logger
from .compiler import CompiledPolicy

LOGGER = get_logger(__name__)


@dataclass
class PolicyDecision:
    side: str
    size: float
    reason: Dict[str, object] = field(default_factory=dict)


class PolicyRuntime:
    def __init__(self, compiled: CompiledPolicy):
        self.compiled = compiled
        self.settings = get_settings()
        self._smoothed_prob: Optional[float] = None
        self._position_side: str = "flat"
        self._position_age: int = 0
        self._peak_equity: float = 0.0

    def _normalise_prediction(self, prediction) -> Dict[str, object]:
        if isinstance(prediction, dict):
            probs = prediction.get("probs", {})
            p_up = float(probs.get("up", 0.0))
            p_down = float(probs.get("down", 0.0))
            p_flat = float(probs.get("flat", 0.0))
            conf = float(prediction.get("pred_conf", max(p_up, p_down)))
            grey = prediction.get("grey_zone") or (0.45, 0.55)
            ci = prediction.get("confidence_interval")
            bucket = prediction.get("regime_bucket") or "med"
        else:
            p_up = float(prediction)
            p_down = 1.0 - p_up
            p_flat = 0.0
            conf = max(p_up, p_down)
            grey = (0.45, 0.55)
            ci = (conf - 0.1, conf + 0.1)
            bucket = "med"
        return {
            "p_up": p_up,
            "p_down": p_down,
            "p_flat": p_flat,
            "confidence": conf,
            "grey_zone": grey,
            "confidence_interval": ci,
            "bucket": bucket,
        }

    def _delever_multiplier(self, drawdown: float) -> float:
        steps = sorted(self.settings.sizing.drawdown_delever, key=lambda item: item[0])
        multiplier = 1.0
        for threshold, factor in steps:
            if drawdown >= threshold:
                multiplier = min(multiplier, factor)
        return multiplier

    def _regime_thresholds(self, bucket: str, base: Tuple[float, float]) -> Tuple[float, float]:
        overrides = self.settings.policy.regime_confidence_overrides
        if bucket in overrides:
            return overrides[bucket]
        return base

    def apply(self, prediction, context: Dict[str, object]) -> PolicyDecision:
        context = dict(context)
        context.setdefault("features", {})
        state = context.setdefault("state", {})
        account_balance = state.get("balance", 1.0)
        state.setdefault("daily_drawdown", 0.0)
        state.setdefault("daily_loss", 0.0)
        base_risk = account_balance * self.settings.risk.risk_per_trade
        context["base_risk"] = base_risk

        pred_info = self._normalise_prediction(prediction)
        reasons = {"prediction": pred_info, "filters": [], "guards": []}

        for guard in self.compiled.guards:
            ok = guard.fn(context)
            reasons["guards"].append({"name": guard.name, "ok": ok})
            if not ok:
                LOGGER.warning("Risk guard %s triggered action=%s", guard.name, guard.action)
                return PolicyDecision(side="flat", size=0.0, reason=reasons)

        allowed = {"long"}
        if self.settings.policy.allow_short:
            allowed.add("short")

        for flt in self.compiled.filters:
            if flt.fn(context):
                reasons["filters"].append({"name": flt.name, "action": flt.action})
                if flt.action == "block_entries":
                    return PolicyDecision(side="flat", size=0.0, reason=reasons)
                if flt.action == "allow_long_only":
                    allowed = {"long"}
                if flt.action == "allow_short_only":
                    allowed = {"short"}

        conf = float(pred_info["confidence"])
        alpha = self.settings.policy.prob_ema_alpha
        if self._smoothed_prob is None:
            self._smoothed_prob = conf
        else:
            self._smoothed_prob = alpha * conf + (1 - alpha) * self._smoothed_prob
        smoothed = self._smoothed_prob

        base_grey = pred_info.get("grey_zone", (0.45, 0.55))
        regime_grey = self._regime_thresholds(pred_info.get("bucket", "med"), base_grey)
        grey_low, grey_high = regime_grey
        ci = pred_info.get("confidence_interval")
        if grey_low <= conf <= grey_high:
            reasons["abstain"] = "grey_zone"
            return PolicyDecision(side="flat", size=0.0, reason=reasons)
        if ci and len(ci) == 2 and (ci[1] - ci[0]) > self.settings.policy.max_ci_width:
            reasons["abstain"] = "wide_interval"
            return PolicyDecision(side="flat", size=0.0, reason=reasons)

        if abs(conf - 0.5) < self.settings.policy.abstain_margin:
            reasons["abstain"] = "low_margin"
            return PolicyDecision(side="flat", size=0.0, reason=reasons)

        features = context.get("features", {})
        price = float(features.get("close") or 0.0)
        atr_value = float(features.get("atr_14") or features.get("atr") or 0.0)
        if not atr_value and price:
            atr_value = price * 0.004
        reward = self.settings.policy.eu_reward_atr * atr_value
        risk_value = self.settings.policy.eu_risk_atr * atr_value
        cost_cfg = context.get("costs", {})
        commission_bps = float(cost_cfg.get("commission_bps", self.settings.data.fee_bp))
        spread_bps = float(cost_cfg.get("spread_bps", self.settings.data.slippage_bp))
        expected_cost = (commission_bps + spread_bps) / 10_000.0 * price if price else 0.0

        p_up = pred_info["p_up"]
        p_down = pred_info["p_down"]
        eu_long = p_up * reward - p_down * risk_value - expected_cost
        eu_short = p_down * reward - p_up * risk_value - expected_cost
        reasons["expected_utility"] = {"long": eu_long, "short": eu_short}

        min_eu = self.settings.policy.min_expected_utility
        side = "flat"
        if eu_long > min_eu and "long" in allowed:
            side = "long"
        if eu_short > eu_long and eu_short > min_eu and "short" in allowed:
            side = "short"

        for threshold in self.compiled.thresholds:
            if threshold.long_if_prob_gt is not None and smoothed >= threshold.long_if_prob_gt and "long" in allowed:
                side = "long"
            if threshold.short_if_prob_lt is not None and smoothed <= threshold.short_if_prob_lt and "short" in allowed:
                side = "short"

        if side == "long" and eu_long <= min_eu:
            side = "flat"
        if side == "short" and eu_short <= min_eu:
            side = "flat"

        min_hold = max(0, self.settings.policy.min_hold_bars)
        if side != "flat":
            if self._position_side == side:
                if self._position_age < min_hold and max(eu_long, eu_short) > min_eu:
                    self._position_age += 1
                elif max(eu_long, eu_short) <= min_eu:
                    side = "flat"
                else:
                    self._position_age += 1
            elif self._position_side != "flat" and self._position_age < min_hold:
                if max(eu_long, eu_short) > min_eu:
                    side = self._position_side
                    self._position_age += 1
                else:
                    side = "flat"
            else:
                self._position_side = side
                self._position_age = 0
        else:
            self._position_side = "flat"
            self._position_age = 0

        size = 0.0
        if side in {"long", "short"}:
            variance = max(smoothed * (1 - smoothed), 1e-6)
            edge = abs(smoothed - 0.5)
            kelly_fraction = min(edge / variance, self.settings.sizing.kelly_clip)
            self._peak_equity = max(self._peak_equity, account_balance)
            drawdown = 0.0
            if self._peak_equity > 0:
                drawdown = 1.0 - account_balance / self._peak_equity
            delever = self._delever_multiplier(drawdown)
            kelly_fraction = min(kelly_fraction * delever, self.settings.sizing.max_leverage)
            size = base_risk * kelly_fraction
            for sizing in self.compiled.sizing:
                computed = sizing.fn(context)
                if computed:
                    size = float(computed)
            reasons["drawdown"] = drawdown
            reasons["kelly_fraction"] = kelly_fraction
        else:
            reasons["kelly_fraction"] = 0.0

        structure_hits = []
        for key in ("bos_up_5", "bos_down_5", "fvg_up_width", "fvg_down_width"):
            value = context.get("features", {}).get(key)
            if value and abs(float(value)) > 0:
                structure_hits.append(key)
        reasons["structure"] = structure_hits
        reasons["side"] = side
        reasons["size"] = size
        reasons["smoothed_prob"] = smoothed
        reasons["memory_hits"] = context.get("memory_hits", {})
        return PolicyDecision(side=side, size=size, reason=reasons)


def propose_entry_plan(
    ctx: Dict[str, object],
    side: str,
    prob_up: Optional[float],
    settings=None,
) -> Optional[Dict[str, object]]:
    if side not in {"long", "short"}:
        return None
    if prob_up is None:
        return None

    if settings is None:
        settings = get_settings()

    auto_cfg = settings.auto
    candles = ctx.get("candles")
    if not isinstance(candles, pd.DataFrame) or candles.empty:
        return None

    price = float(ctx.get("features", {}).get("close") or candles["close"].iloc[-1])
    atr_series = atr_indicator(
        candles["high"], candles["low"], candles["close"], auto_cfg.target_atr_period
    )
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    if not atr_value:
        atr_value = float(candles["close"].pct_change().std() or 0.0) * price
    atr_value = max(atr_value, price * 0.001)

    k = 0.5 + 1.5 * abs(prob_up - 0.5)
    k = max(0.5, min(2.0, k))
    direction = 1.0 if side == "long" else -1.0
    target_price = price + direction * k * atr_value

    window = max(auto_cfg.target_atr_period, 5)
    recent = candles.tail(window)
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    impulse_range = max(high - low, atr_value)

    fib_low, fib_high = sorted(auto_cfg.fib_entry)
    if side == "long":
        zone_high = price - impulse_range * fib_low
        zone_low = price - impulse_range * fib_high
        stop = min(low, price - impulse_range) - 0.5 * atr_value
    else:
        zone_low = price + impulse_range * fib_low
        zone_high = price + impulse_range * fib_high
        stop = max(high, price + impulse_range) + 0.5 * atr_value

    zone_from = float(min(zone_low, zone_high))
    zone_to = float(max(zone_low, zone_high))
    entry_mid = (zone_from + zone_to) / 2.0
    if side == "long":
        reward = target_price - entry_mid
    else:
        reward = entry_mid - target_price
    risk = abs(entry_mid - stop)
    risk_r = reward / risk if risk > 0 else None

    return {
        "target_price": float(target_price),
        "entry_plan": {
            "zone": {"from": zone_from, "to": zone_to},
            "stop": float(stop),
            "take": float(target_price),
            "risk_r": float(risk_r) if risk_r is not None else None,
        },
    }
