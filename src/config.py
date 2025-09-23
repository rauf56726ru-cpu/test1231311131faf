"""Configuration loading utilities for the autotrader prototype."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class DataSettings(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    window: int = 300
    horizon_min: int = Field(15, ge=1)
    fee_bp: int = Field(7, ge=0)
    slippage_bp: int = Field(5, ge=0)
    lookback_days: int = Field(30, ge=1)
    stream_tick_fraction: float = Field(1.0 / 12.0, gt=0.0)
    stream_tick_min_seconds: float = Field(1.0, ge=0.0)
    stream_tick_max_seconds: float = Field(15.0, ge=0.0)
    market_fetch_fraction: float = Field(0.25, gt=0.0)
    market_fetch_min_seconds: float = Field(1.0, ge=0.0)
    market_fetch_max_seconds: float = Field(15.0, ge=0.0)
    candles_store_path: str = "data/live/candles.sqlite"


class XGBSettings(BaseModel):
    max_depth: int = 5
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class TCNSettings(BaseModel):
    channels: List[int] = Field(default_factory=lambda: [64, 64, 64])
    dropout: float = 0.1
    lr: float = 1e-3


class LGBMSettings(BaseModel):
    learning_rate: float = 0.05
    num_leaves: int = 128
    max_depth: int = -1
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 1e-3
    lambda_l2: float = 1e-3
    validation_fraction: float = 0.3
    min_validation_rows: int = 64
    calibration_method: str = "isotonic"


class ModelSettings(BaseModel):
    kind: str = Field("lgbm", pattern="^(xgb|lgbm|tcn)$")
    xgb: XGBSettings = Field(default_factory=XGBSettings)
    lgbm: LGBMSettings = Field(default_factory=LGBMSettings)
    tcn: TCNSettings = Field(default_factory=TCNSettings)


class PolicySettings(BaseModel):
    conf_up: float = 0.6
    conf_down: float = 0.4
    allow_short: bool = True
    prob_delta_quantile: float = 0.6
    prob_ema_alpha: float = 0.2
    min_hold_bars: int = 3
    abstain_margin: float = 0.03
    max_ci_width: float = 0.25
    regime_confidence_overrides: Dict[str, Tuple[float, float]] = Field(
        default_factory=lambda: {"low": (0.52, 0.48), "med": (0.5, 0.5), "high": (0.48, 0.52)}
    )
    eu_reward_atr: float = 1.5
    eu_risk_atr: float = 1.0
    min_expected_utility: float = 0.0


class SizingSettings(BaseModel):
    kelly_clip: float = 0.5
    max_leverage: float = 1.0
    drawdown_delever: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.05, 0.75), (0.1, 0.5)]
    )


class RiskSettings(BaseModel):
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.03
    take_profit: float = 0.01
    stop_loss: float = 0.005


class MemorySettings(BaseModel):
    faiss_dim_text: int = 384
    faiss_dim_image: int = 512
    text_model: str = "all-MiniLM-L6-v2"
    allowed_text_models: List[str] = Field(
        default_factory=lambda: [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
        ]
    )


class AutoSettings(BaseModel):
    min_bars_for_train: int = Field(5000, ge=1)
    retrain_cooldown_min: int = Field(15, ge=0)
    target_atr_period: int = Field(14, ge=1)
    fib_entry: Tuple[float, float] = (0.236, 0.382)


class CandleColorSettings(BaseModel):
    up_real: str = "#FFFFFF"
    down_real: str = "#000000"
    up_pred: str = "#00C853"
    down_pred: str = "#D50000"


class UISettings(BaseModel):
    candle_colors: CandleColorSettings = Field(default_factory=CandleColorSettings)


class Settings(BaseModel):
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    sizing: SizingSettings = Field(default_factory=SizingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    auto: AutoSettings = Field(default_factory=AutoSettings)
    ui: UISettings = Field(default_factory=UISettings)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path | str = Path("configs/settings.yaml")) -> Settings:
    """Load application settings from YAML and environment variables."""
    load_dotenv()
    path = Path(path)
    raw = _load_yaml(path)
    settings = Settings.parse_obj(raw)

    # allow overriding via environment variables
    symbol = os.getenv("SYMBOL")
    interval = os.getenv("INTERVAL")
    if symbol:
        settings.data.symbol = symbol
    if interval:
        settings.data.interval = interval
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def load_rules(path: Path | str = Path("configs/rules.yaml")) -> Dict[str, Any]:
    """Return raw policy rules configuration."""
    return _load_yaml(Path(path))


DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_INTERVALS = ["1m"]
DEFAULT_HORIZONS = [5, 15, 60]


def load_backtest_config(path: Path | str) -> Dict[str, Any]:
    raw = _load_yaml(Path(path))
    config: Dict[str, Any] = dict(raw)

    if not config.get("symbols"):
        config["symbols"] = list(DEFAULT_SYMBOLS)

    if not config.get("intervals"):
        config["intervals"] = list(DEFAULT_INTERVALS)

    horizons = config.get("horizons") or config.get("horizons_min")
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)
    else:
        horizons = [int(h) for h in horizons]
    config["horizons"] = list(horizons)
    config.setdefault("horizons_min", list(horizons))

    return config
