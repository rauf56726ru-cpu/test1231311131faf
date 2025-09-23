"""Command line interface for autotrader workflows."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import get_settings, load_backtest_config
from .features.builder import build_features, save_feature_sets
from .io.binance_ws import fetch_klines
from .io.store import write_parquet
from .memory.notion_ingest import ingest_notion
from .models.model_registry import get_registry
from .models.tcn_train import run_training as train_tcn_model
from .models.xgb_predict import batch_predict
from .models.xgb_train import run_training as train_xgb_model
from .policy.compiler import load_and_compile
from .policy.runtime import PolicyRuntime
from .self_train import SelfTrainManager
from .utils.logging import configure_logging, get_logger
from .backtest.engine import run_backtest
from .backtest.report import generate_report

configure_logging()
LOGGER = get_logger(__name__)


def cmd_collect(symbol: str, interval: str, limit: int = 1000) -> None:
    async def _collect():
        return await fetch_klines(symbol, interval, limit=limit)

    data = asyncio.run(_collect())
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    path = Path("data") / "raw" / f"{symbol}_{interval}.parquet"
    write_parquet(df, path)
    LOGGER.info(
        "Collected %s candles for %s %s to %s",
        len(df),
        symbol,
        interval,
        path.as_posix(),
    )


def cmd_build_features(symbol: str, interval: str, config_path: Path | None = None) -> None:
    cfg = {}
    if config_path is not None:
        cfg = load_backtest_config(config_path)

    settings = get_settings()

    if config_path is not None:
        symbols: List[str] = list(cfg.get("symbols", []))
        intervals: List[str] = list(cfg.get("intervals", []))
        horizons_raw: Iterable[int] = cfg.get("horizons") or cfg.get("horizons_min", [])
    else:
        symbols = [symbol or settings.data.symbol]
        intervals = [interval or settings.data.interval]
        horizons_raw = [settings.data.horizon_min]

    if not symbols:
        symbols = [symbol or settings.data.symbol]
    if not intervals:
        intervals = [interval or settings.data.interval]

    horizons = sorted({int(h) for h in horizons_raw})
    if not horizons:
        horizons = [settings.data.horizon_min]

    fee_bp = cfg.get("costs", {}).get("fee_bp", settings.data.fee_bp)
    spread_model = cfg.get("spread_model", "median")

    raw_root = Path("data") / "raw"
    any_success = False

    for sym in symbols:
        for tf in intervals:
            path = raw_root / f"{sym}_{tf}.parquet"
            if not path.exists():
                LOGGER.warning(
                    "Raw data not found at %s. Skipping %s %s",
                    path.as_posix(),
                    sym,
                    tf,
                )
                continue
            candles = pd.read_parquet(path)
            candles.index = pd.to_datetime(candles.index)
            feature_sets = build_features(
                candles,
                horizons=horizons,
                symbol=sym,
                interval=tf,
                fee_bp=fee_bp,
                spread_model=spread_model,
            )
            out_dir = Path("data") / "features" / f"{sym}_{tf}"
            save_feature_sets(feature_sets, base_path=out_dir)
            LOGGER.info(
                "Saved features for %s %s horizons=%s to %s",
                sym,
                tf,
                list(feature_sets.keys()),
                out_dir.as_posix(),
            )
            any_success = True

    if not any_success:
        raise FileNotFoundError("No raw data found for requested symbols/intervals")


def cmd_train_xgb(config: Path | None = None) -> None:
    metrics = train_xgb_model(config_path=config or Path("configs/backtest.yaml"))
    LOGGER.info("XGB training metrics: %s", metrics)


def cmd_train_tcn() -> None:
    train_tcn_model()


def cmd_train_all(config: Path | None = None) -> None:
    cmd_train_xgb(config)
    try:
        cmd_train_tcn()
    except Exception as exc:
        LOGGER.warning("TCN training failed: %s", exc)
    LOGGER.info("Completed training pipeline")


def cmd_backtest(config_path: Path) -> None:
    cfg = load_backtest_config(config_path)
    settings = get_settings()
    symbol = cfg.get("symbol", settings.data.symbol)
    interval = cfg.get("interval", settings.data.interval)

    candles_path = Path(cfg.get("candles_path") or (Path("data") / "raw" / f"{symbol}_{interval}.parquet"))
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles not found at {candles_path.as_posix()}")
    candles = pd.read_parquet(candles_path)
    candles.index = pd.to_datetime(candles.index)

    horizons = cfg.get("horizons") or cfg.get("horizons_min") or [settings.data.horizon_min]
    horizons = [int(h) for h in horizons]
    default_horizon = int(cfg.get("default_horizon", horizons[0]))

    feature_candidates = []
    if cfg.get("features_path"):
        feature_candidates.append(Path(cfg["features_path"]))
    feature_root = Path("data") / "features"
    feature_candidates.extend(
        [
            feature_root / f"{symbol}_{interval}" / f"features_h{default_horizon}.parquet",
            feature_root / f"features_h{default_horizon}.parquet",
            feature_root / "features.parquet",
        ]
    )

    feature_path = next((path for path in feature_candidates if path.exists()), None)
    if feature_path is None:
        raise FileNotFoundError("Features not built. Run make build_features")

    features = pd.read_parquet(feature_path)
    features.index = pd.to_datetime(features.index)
    if "target" in features:
        features = features.drop(columns=["target"])

    predictions = pd.DataFrame(
        {
            "prob_up": np.full(len(features), 0.5),
            "prob_down": np.full(len(features), 0.5),
            "prob_flat": np.zeros(len(features)),
        },
        index=features.index,
    )
    if get_settings().model.kind == "xgb":
        registry = get_registry()
        horizon_key = f"h{default_horizon}"
        record = registry.get_active("xgb", horizon=horizon_key) or registry.get_active("xgb", horizon="default")
        model_dir: Path | None = None
        if record is not None:
            candidate = Path(record.path)
            required = [
                candidate / "model.pkl",
                candidate / "feature_list.json",
            ]
            if all(item.exists() for item in required):
                model_dir = candidate
            else:
                LOGGER.info("Using stub probabilities: missing artifacts in %s", candidate.as_posix())
        else:
            LOGGER.info("Using stub probabilities: no active model registered for %s", horizon_key)

        predictions = batch_predict(features.copy(), model_dir=model_dir)

    compiled = load_and_compile(cfg.get("rules_path", "configs/rules.yaml"))
    runtime = PolicyRuntime(compiled)
    costs = {
        "commission_bps": cfg.get("commission_bp", 7),
        "slip_k": cfg.get("slip_k"),
    }
    execution = {
        "latency_bars": cfg.get("latency_bars", 1),
        "funding_bps": cfg.get("funding_bps", 0.0),
    }
    result = run_backtest(
        market=candles.loc[features.index],
        features=features,
        predictions=predictions,
        policy=runtime,
        initial_balance=cfg.get("initial_balance", 10_000),
        costs=costs,
        execution=execution,
    )
    report_dir = generate_report(result)
    LOGGER.info("Backtest report saved to %s", report_dir)


def cmd_ingest_notion(path: Path) -> None:
    retriever = ingest_notion(path)
    LOGGER.info("Ingested Notion export from %s", path)


def cmd_paper() -> None:
    compiled = load_and_compile("configs/rules.yaml")
    runtime = PolicyRuntime(compiled)
    from .live.loop import LiveLoop

    loop = LiveLoop(policy=runtime, model_kind=get_settings().model.kind)
    LOGGER.info("Paper trading loop initialised: %s", loop)


def cmd_self_train(symbol: str, date_from: str, date_to: Optional[str], resume: bool) -> None:
    async def _runner() -> List[Dict[str, object]]:
        manager = SelfTrainManager(Path("data/self_train/progress_cli.json"))
        await manager.start(symbol, date_from, date_to, resume=resume)
        last_stage: Dict[Tuple[str, str], str] = {}
        last_day: Dict[Tuple[str, str], str] = {}
        while manager.active_task_count() > 0:
            snapshot = await manager.status()
            entries = snapshot.get("entries", []) if isinstance(snapshot, dict) else []
            for entry in entries:
                sym = str(entry.get("symbol") or "").upper()
                interval = str(entry.get("interval") or "")
                key = (sym, interval)
                stage = str(entry.get("stage") or entry.get("last_status") or "")
                current_day = str(entry.get("current_day") or entry.get("last_completed_day") or "")
                message = f"Self-train {sym} {interval} day={current_day or '—'} stage={stage}"
                if last_stage.get(key) != stage or last_day.get(key) != current_day:
                    LOGGER.info(message)
                    last_stage[key] = stage
                    last_day[key] = current_day
            await asyncio.sleep(2.0)
        return await manager.status()

    snapshot = asyncio.run(_runner())
    entries = snapshot.get("entries", []) if isinstance(snapshot, dict) else []
    for entry in entries:
        sym = entry.get("symbol")
        interval = entry.get("interval")
        LOGGER.info(
            "Self-train progress: %s %s last_day=%s rows=%s",
            sym,
            interval,
            entry.get("last_completed_day"),
            entry.get("rows"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotrader CLI")
    sub = parser.add_subparsers(dest="command")

    collect = sub.add_parser("collect")
    collect.add_argument("--symbol", default=get_settings().data.symbol)
    collect.add_argument("--interval", default=get_settings().data.interval)
    collect.add_argument("--limit", type=int, default=1000)

    build = sub.add_parser("build-features")
    build.add_argument("--symbol", default=get_settings().data.symbol)
    build.add_argument("--interval", default=get_settings().data.interval)
    build.add_argument("--config", type=Path, default=None)

    train_xgb = sub.add_parser("train-xgb")
    train_xgb.add_argument("--config", type=Path, default=None)
    sub.add_parser("train-tcn")
    train_all = sub.add_parser("train-all")
    train_all.add_argument("--config", type=Path, default=None)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--config", type=Path, default=Path("configs/backtest.yaml"))

    ingest = sub.add_parser("ingest-notion")
    ingest.add_argument("--path", type=Path, required=True)

    sub.add_parser("paper")

    self_train_cmd = sub.add_parser("self-train")
    self_train_cmd.add_argument("--symbol", default=get_settings().data.symbol)
    self_train_cmd.add_argument("--date-from", dest="date_from", default="2022-01-01")
    self_train_cmd.add_argument("--date-to", dest="date_to", default=None)
    self_train_cmd.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args.symbol, args.interval, args.limit)
    elif args.command == "build-features":
        cmd_build_features(args.symbol, args.interval, args.config)
    elif args.command == "train-xgb":
        cmd_train_xgb(args.config)
    elif args.command == "train-tcn":
        cmd_train_tcn()
    elif args.command == "train-all":
        cmd_train_all(args.config)
    elif args.command == "backtest":
        cmd_backtest(args.config)
    elif args.command == "ingest-notion":
        cmd_ingest_notion(args.path)
    elif args.command == "paper":
        cmd_paper()
    elif args.command == "self-train":
        resolved_symbol = args.symbol or get_settings().data.symbol
        cmd_self_train(resolved_symbol, args.date_from, args.date_to, resume=not args.no_resume)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
