"""Self-training orchestration with persistent progress tracking."""

from __future__ import annotations

import asyncio
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
import joblib

from ..config import get_settings
from ..features.builder import FeatureSet, build_features
from ..models.lgbm_train import IncrementalTrainingResult, train_incremental
from ..models.tcn_incremental import TCNUpdateResult, incremental_update
from ..utils.logging import get_logger
from ..utils.timeframes import interval_to_seconds
from .backfill import BackfillDayResult, SelfTrainBackfill
from .progress import ProgressStore, ProgressTracker, StatusRecord, ResumeKey

LOGGER = get_logger(__name__)

DEFAULT_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)
STAGE_SEQUENCE = ("backfill", "features", "train", "calibrate", "eval", "save", "gate")
HIGH_TF_CACHE = ("4h", "1d")


def _now() -> datetime:
    return datetime.utcnow()


class SelfTrainManager:
    """Coordinate the day-by-day self-training workflow."""

    def __init__(
        self,
        store_path: Path,
        *,
        backfill_service: Optional[SelfTrainBackfill] = None,
        feature_builder: Callable[..., Dict[int, FeatureSet]] = build_features,
        lgbm_trainer: Callable[..., IncrementalTrainingResult] = train_incremental,
        tcn_trainer: Callable[..., TCNUpdateResult] = incremental_update,
        intervals: Iterable[str] = DEFAULT_INTERVALS,
    ) -> None:
        self.progress_store = ProgressStore(store_path)
        self.progress = ProgressTracker(self.progress_store)
        self.settings = get_settings()
        self.backfill = backfill_service or SelfTrainBackfill()
        self.feature_builder = feature_builder
        self.lgbm_trainer = lgbm_trainer
        self.tcn_trainer = tcn_trainer
        self.intervals: Tuple[str, ...] = tuple(intervals)
        if tuple(self.intervals) != tuple(DEFAULT_INTERVALS):
            LOGGER.debug("Custom interval set for self-train: %s", self.intervals)
        self._stop_event = asyncio.Event()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._last_booster: Dict[Tuple[str, str], Path] = {}
        self._high_tf_cache: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}

        self.lgbm_root = Path("models/lgbm")
        self.tcn_root = Path("models/tcn")
        self.feature_root = Path("data/self_train/features")
        self.prediction_root = Path("data/self_train/predictions")
        self.cache_root = Path("data/self_train/cache")
        self.cache_feature_root = self.cache_root / "features"
        self.cache_lgbm_root = self.cache_root / "lgbm"
        for path in (
            self.lgbm_root,
            self.tcn_root,
            self.feature_root,
            self.prediction_root,
            self.cache_feature_root,
            self.cache_lgbm_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    async def status(self) -> Dict[str, object]:
        schema_version, record = await self.progress.status_snapshot()
        payload: Dict[str, object] = {"schema_version": schema_version}
        if record is not None:
            payload.update(record.dict(by_alias=False))
        payload["intervals"] = list(self.intervals)
        payload["intervals_total"] = len(self.intervals)
        return payload

    async def history(self) -> Dict[str, object]:
        schema_version, records = await self.progress.history_snapshot()
        return {
            "schema_version": schema_version,
            "history": [record.dict(by_alias=False) for record in records],
        }

    async def stop(self) -> None:
        self._stop_event.set()
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    async def start(
        self,
        symbol: str,
        date_from: str,
        date_to: Optional[str] = None,
        resume: bool = True,
    ) -> None:
        await self.progress.ensure_loaded()
        self._stop_event.clear()

        target_symbol = (symbol or self.settings.data.symbol).upper()
        if not target_symbol:
            raise ValueError("symbol must be provided for self-train")

        start_day = self._parse_date(date_from) or date(2022, 1, 1)
        today = _now().date()
        end_day = self._parse_date(date_to) if date_to else today - timedelta(days=1)
        end_day = min(end_day or today, today - timedelta(days=1))
        if end_day < start_day:
            end_day = start_day

        first_tf = self.intervals[0]
        await self.progress.initialize_symbol(target_symbol, start_day.isoformat(), first_tf)

        task = asyncio.create_task(
            self._run_symbol(target_symbol, start_day, end_day, resume=resume)
        )
        self._tasks[target_symbol] = task



    async def _run_symbol(
        self,
        symbol: str,
        start_day: date,
        end_day: date,
        *,
        resume: bool,
    ) -> None:
        task = asyncio.current_task()
        try:
            resume_key = await self.progress.resume_pointer(symbol) if resume else None
            today = _now().date()
            final_day = min(end_day, today)
            day = start_day
            while day <= final_day and not self._stop_event.is_set():
                day_str = day.isoformat()
                LOGGER.info("Self-train %s → day %s", symbol, day_str)
                LOGGER.info("[SELF-TRAIN] start %s day=%s", symbol, day_str)
                all_tf_completed = True
                for interval in self.intervals:
                    if self._stop_event.is_set():
                        all_tf_completed = False
                        break
                    start_stage, resume_key = self._starting_stage(day_str, interval, resume_key)
                    if start_stage is None:
                        continue
                    success = await self._process_day(symbol, day, interval, start_stage)
                    if not success:
                        all_tf_completed = False
                        LOGGER.warning(
                            "Halting self-train for %s at %s %s due to stage failure",
                            symbol,
                            day_str,
                            interval,
                        )
                        self._stop_event.set()
                        break
                if not all_tf_completed or self._stop_event.is_set():
                    break
                LOGGER.info("[SELF-TRAIN] completed %s day=%s", symbol, day_str)
                day += timedelta(days=1)
            idle_reference = min(day, final_day)
            await self.progress.update_stage(
                symbol,
                day=idle_reference.isoformat(),
                tf=self.intervals[-1],
                stage="idle",
                rows_in_day=0,
                rows_total_tf=0,
                resume_stage="idle",
            )
        except asyncio.CancelledError:
            await self.progress.update_stage(
                symbol,
                day=start_day.isoformat(),
                tf=self.intervals[0],
                stage="cancelled",
                resume_stage="cancelled",
            )
            raise
        finally:
            if task is not None:
                self._tasks.pop(symbol, None)

    async def _process_day(
        self,
        symbol: str,
        day: date,
        interval: str,
        start_stage: Optional[str],
    ) -> bool:
        day_str = day.isoformat()
        if start_stage is None:
            return True
        if start_stage not in STAGE_SEQUENCE:
            start_stage = STAGE_SEQUENCE[0]
        start_ts, end_ts = self._day_range(day, interval)

        stage_index = STAGE_SEQUENCE.index(start_stage)
        backfill_result: Optional[BackfillDayResult] = None
        feature_set: Optional[FeatureSet] = None
        lgbm_result: Optional[IncrementalTrainingResult] = None
        metrics: Dict[str, float] = self._zero_metrics()
        gate = {"ece_pass": False, "mcc_pass": False, "active": False}
        gates = dict(gate)

        status_record = self.progress.state.status.get(symbol)
        if (
            status_record is not None
            and status_record.day == day_str
            and status_record.tf == interval
        ):
            try:
                metrics = {
                    key: float(value)
                    for key, value in status_record.metrics_day.dict().items()
                }
            except Exception:  # pragma: no cover - defensive fallback
                metrics = self._zero_metrics()
            try:
                gate = {
                    "ece_pass": bool(status_record.gate.ece_pass),
                    "mcc_pass": bool(status_record.gate.mcc_pass),
                    "active": bool(status_record.gate.active),
                }
            except Exception:  # pragma: no cover - defensive fallback
                gate = {"ece_pass": False, "mcc_pass": False, "active": False}

        long_running_stages = {"features", "train", "calibrate", "eval", "save", "gate"}

        async def _mark_stage_in_progress(stage_name: str) -> None:
            if stage_name not in long_running_stages:
                return
            rows_in_day = len(feature_set.X) if feature_set is not None else 0
            if backfill_result is not None:
                rows_total_tf = int(backfill_result.rows_total)
            elif feature_set is not None:
                rows_total_tf = rows_in_day
            else:
                rows_total_tf = 0
            record = await self.progress.update_stage(
                symbol,
                day=day_str,
                tf=interval,
                stage=stage_name,
                rows_in_day=int(rows_in_day),
                rows_total_tf=rows_total_tf,
                metrics_day=metrics,
                gate=gate,
            )
            self._log_progress(record)

        if stage_index > STAGE_SEQUENCE.index("backfill"):
            backfill_result = await self._ensure_backfill(
                symbol,
                interval,
                day_str,
                start_ts,
                end_ts,
                metrics,
                gate,
                gates,
                existing=None,
            )
            if backfill_result is None:
                return True
        if stage_index > STAGE_SEQUENCE.index("features"):
            await _mark_stage_in_progress("features")
            feature_set = await self._ensure_feature_set(
                symbol,
                interval,
                day_str,
                start_ts,
                end_ts,
                backfill_result,
                metrics,
                gate,
                gates,
                existing=None,
            )
            if feature_set is None:
                return True
        if stage_index > STAGE_SEQUENCE.index("train"):
            await _mark_stage_in_progress("train")
            lgbm_result, metrics = await self._ensure_lgbm_result(
                symbol,
                interval,
                day_str,
                feature_set,
                backfill_result,
                metrics,
                existing=None,
            )
            if lgbm_result is None:
                return True
            gate = self._gate_from_metrics(metrics)
            gates = dict(gate)

        status_record = self.progress.state.status.get(symbol)
        if (
            status_record is not None
            and status_record.day == day_str
            and status_record.tf == interval
        ):
            try:
                metrics = {
                    key: float(value)
                    for key, value in status_record.metrics_day.dict().items()
                }
            except Exception:  # pragma: no cover - defensive fallback
                metrics = self._zero_metrics()
            try:
                gates = {
                    "ece_pass": bool(status_record.gates.ece_pass),
                    "mcc_pass": bool(status_record.gates.mcc_pass),
                    "active": bool(status_record.gates.active),
                }
            except Exception:  # pragma: no cover - defensive fallback
                gates = {"ece_pass": False, "mcc_pass": False, "active": False}

        # Preload prerequisites when resuming mid-pipeline
        if stage_index >= STAGE_SEQUENCE.index("train"):
            backfill_result = await self._ensure_backfill(
                symbol,
                interval,
                day_str,
                start_ts,
                end_ts,
                metrics,
                gate,
                gates,
                existing=None,
            )
            if backfill_result is None:
                return True
            await _mark_stage_in_progress("features")
            feature_set = await self._ensure_feature_set(
                symbol,
                interval,
                day_str,
                start_ts,
                end_ts,
                backfill_result,
                metrics,
                gate,
                gates,
                existing=None,
            )
            if feature_set is None:
                return True

        if stage_index >= STAGE_SEQUENCE.index("calibrate"):
            await _mark_stage_in_progress("train")
            lgbm_result, metrics = await self._ensure_lgbm_result(
                symbol,
                interval,
                day_str,
                feature_set,
                backfill_result,
                metrics,
                existing=None,
            )
            if lgbm_result is None:
                return True

        for stage in STAGE_SEQUENCE[stage_index:]:
            if self._stop_event.is_set():
                return False
            await _mark_stage_in_progress(stage)
            if stage == "backfill":
                backfill_result = await self._ensure_backfill(
                    symbol,
                    interval,
                    day_str,
                    start_ts,
                    end_ts,
                    metrics,
                    gate,
                    gates,
                    existing=backfill_result,
                )
                if backfill_result is None:
                    return True
            elif stage == "features":
                if backfill_result is None:
                    backfill_result = await self._ensure_backfill(
                        symbol,
                        interval,
                        day_str,
                        start_ts,
                        end_ts,
                        metrics,
                        gate,
                        gates,
                        existing=None,
                    )
                    if backfill_result is None:
                        return True
                feature_set = await self._ensure_feature_set(
                    symbol,
                    interval,
                    day_str,
                    start_ts,
                    end_ts,
                    backfill_result,
                    metrics,
                    gate,
                    gates,
                    existing=None,
                )
                if feature_set is None:
                    return True

            elif stage == "train":
                if feature_set is None:
                    if backfill_result is None:
                        backfill_result = await self._ensure_backfill(
                            symbol,
                            interval,
                            day_str,
                            start_ts,
                            end_ts,
                            metrics,
                            gate,
                            gates,
                            existing=None,
                        )
                        if backfill_result is None:
                            return True
                    feature_set = await self._ensure_feature_set(
                        symbol,
                        interval,
                        day_str,
                        start_ts,
                        end_ts,
                        backfill_result,
                        metrics,
                        gate,
                        gates,
                        existing=None,
                    )
                    if feature_set is None:
                        return True
                lgbm_result, metrics = await self._ensure_lgbm_result(
                    symbol,
                    interval,
                    day_str,
                    feature_set,
                    backfill_result,
                    metrics,
                    existing=None,
                )
                if lgbm_result is None:
                    return True
                gate = self._gate_from_metrics(metrics)
                gates = dict(gate)

            elif stage == "calibrate":
                gate = self._gate_from_metrics(metrics)
                gates = dict(gate)
                record = await self.progress.update_stage(
                    symbol,
                    day=day_str,
                    tf=interval,
                    stage=stage,
                    metrics_day=metrics,
                    gate=gate,
                )
                self._log_progress(record)
            elif stage == "eval":
                record = await self.progress.update_stage(
                    symbol,
                    day=day_str,
                    tf=interval,
                    stage=stage,
                    metrics_day=metrics,
                    gate=gate,
                )
                self._log_progress(record)
            elif stage == "save":
                if feature_set is None or lgbm_result is None:
                    return True
                features_path = self._store_features(symbol, interval, day_str, feature_set)
                predictions_path = self._store_predictions(symbol, interval, day_str, lgbm_result)
                self._promote_lgbm(symbol, interval, lgbm_result.booster_path.parent)
                await self._update_tcn(symbol, interval, day_str, feature_set)
                rows_total_tf = (
                    backfill_result.rows_total if backfill_result is not None else len(feature_set.X)
                )
                record = await self.progress.update_stage(
                    symbol,
                    day=day_str,
                    tf=interval,
                    stage=stage,
                    rows_in_day=len(feature_set.X),
                    rows_total_tf=rows_total_tf,
                    metrics_day=metrics,
                    gate=gate,
                )
                LOGGER.info(
                    "Stored artifacts for %s %s on %s (features=%s predictions=%s)",
                    symbol,
                    interval,
                    day_str,
                    features_path,
                    predictions_path,
                )
                self._log_progress(record)
            elif stage == "gate":
                gate = self._gate_from_metrics(metrics)
                rows_in_day = len(feature_set.X) if feature_set is not None else 0
                rows_total_tf = (
                    backfill_result.rows_total if backfill_result is not None else rows_in_day
                )
                self._clear_day_cache(symbol, interval, day_str)
                record = await self.progress.update_stage(
                    symbol,
                    day=day_str,
                    tf=interval,
                    stage=stage,
                    rows_in_day=rows_in_day,
                    rows_total_tf=rows_total_tf,
                    metrics_day=metrics,
                    gate=gate,
                    resume_stage="gate",
                )
                self._log_progress(record)
                await self.progress.record_history(
                    symbol,
                    day=day_str,
                    tf=interval,
                    rows_in_day=rows_in_day,
                    metrics=metrics,
                    gate=gate,
                    stage="gate",
                )
                self._clear_day_cache(symbol, interval, day_str)
        return True

    def _starting_stage(
        self,
        day_str: str,
        interval: str,
        resume_key: Optional[ResumeKey],
    ) -> Tuple[Optional[str], Optional[ResumeKey]]:
        if resume_key is None or not resume_key.D:
            return STAGE_SEQUENCE[0], None
        resume_day = resume_key.D
        if day_str < resume_day:
            return None, resume_key
        if day_str > resume_day:
            return STAGE_SEQUENCE[0], None
        if resume_key.TF not in self.intervals:
            return STAGE_SEQUENCE[0], None
        current_idx = self.intervals.index(interval)
        resume_idx = self.intervals.index(resume_key.TF)
        if current_idx < resume_idx:
            return None, resume_key
        if current_idx > resume_idx:
            return STAGE_SEQUENCE[0], None
        next_stage = self._stage_after(resume_key.stage)
        if next_stage is None:
            return None, None
        return next_stage, None

    def _stage_after(self, stage: Optional[str]) -> Optional[str]:
        if stage in STAGE_SEQUENCE:
            idx = STAGE_SEQUENCE.index(stage)
            if idx + 1 < len(STAGE_SEQUENCE):
                return STAGE_SEQUENCE[idx + 1]
            return None
        return STAGE_SEQUENCE[0]
    async def _update_tcn(
        self,
        symbol: str,
        interval: str,
        day: str,
        feature_set: FeatureSet,
    ) -> Optional[TCNUpdateResult]:
        if self.tcn_trainer is None:
            return None
        try:
            result = await asyncio.to_thread(
                self.tcn_trainer,
                feature_set.X,
                feature_set.y,
                model_dir=self.tcn_root / symbol / interval,
                day=day,
                symbol=symbol,
                interval=interval,
                window=max(2, int(getattr(self.settings.data, "window", 64))),
                epochs=1,
                batch_size=64,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("TCN incremental update failed for %s %s %s: %s", symbol, interval, day, exc)
            return None
        return result

    async def _cross_timeframe_frames(
        self,
        symbol: str,
        day: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
    ) -> Dict[str, pd.DataFrame]:
        cache_key = (symbol, day)
        cache = self._high_tf_cache.setdefault(cache_key, {})
        base_seconds = interval_to_seconds(interval)
        cross: Dict[str, pd.DataFrame] = {}
        for high_tf in HIGH_TF_CACHE:
            if interval_to_seconds(high_tf) <= base_seconds:
                continue
            if high_tf not in cache:
                result = await self.backfill.backfill_day(symbol, high_tf, start_ts, end_ts)
                cache[high_tf] = result.day_slice
            cross[high_tf] = cache[high_tf]
        return cross

    def _store_features(
        self,
        symbol: str,
        interval: str,
        day: str,
        feature_set: FeatureSet,
    ) -> Path:
        directory = self.feature_root / symbol / interval
        directory.mkdir(parents=True, exist_ok=True)
        frame = feature_set.X.copy()
        frame["target"] = feature_set.y
        frame["regime_label"] = feature_set.regime_labels
        for column in feature_set.regimes.columns:
            frame[column] = feature_set.regimes[column]
        path = directory / f"{day}.parquet"
        frame.to_parquet(path)
        return path

    def _store_predictions(
        self,
        symbol: str,
        interval: str,
        day: str,
        result: IncrementalTrainingResult,
    ) -> Path:
        directory = self.prediction_root / symbol / interval
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{day}.parquet"
        result.predictions.to_parquet(path)
        return path

    def _promote_lgbm(self, symbol: str, interval: str, day_dir: Path) -> None:
        target_dir = self.lgbm_root / symbol / interval / "active"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(day_dir, target_dir)

    async def _ensure_backfill(
        self,
        symbol: str,
        interval: str,
        day: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        metrics: Dict[str, float],
        gate: Dict[str, bool],
        gates: Dict[str, bool],
        *,
        existing: Optional[BackfillDayResult],
    ) -> Optional[BackfillDayResult]:
        if existing is not None:
            return existing
        result = await self.backfill.backfill_day(symbol, interval, start_ts, end_ts)
        record = await self.progress.update_stage(
            symbol,
            day=day,
            tf=interval,
            stage="backfill",
            rows_in_day=result.rows_in_day,
            rows_total_tf=result.rows_total,
        )
        self._log_progress(record)
        if result.rows_in_day == 0:
            LOGGER.warning("No candles retrieved for %s %s %s", symbol, interval, day)
            await self.progress.record_history(
                symbol,
                day=day,
                tf=interval,
                rows_in_day=0,
                metrics=metrics,
                gate=gate,
                gates=gates,
            )
            return None
        return result

    async def _ensure_feature_set(
        self,
        symbol: str,
        interval: str,
        day: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        backfill_result: Optional[BackfillDayResult],
        metrics: Dict[str, float],
        gate: Dict[str, bool],
        gates: Dict[str, bool],
        *,
        existing: Optional[FeatureSet],
    ) -> Optional[FeatureSet]:
        if existing is not None:
            return existing

        cached = self._load_cached_feature_set(symbol, interval, day)
        if cached is not None:
            feature_set, cached_total = cached
            total_rows = (
                backfill_result.rows_total
                if backfill_result is not None
                else cached_total or len(feature_set.X)
            )
            record = await self.progress.update_stage(
                symbol,
                day=day,
                tf=interval,
                stage="features",
                rows_in_day=len(feature_set.X),
                rows_total_tf=total_rows,
            )
            LOGGER.info(
                "Reusing cached features for %s %s on %s (%s rows)",
                symbol,
                interval,
                day,
                len(feature_set.X),
            )
            self._log_progress(record)
            return feature_set

        if backfill_result is None:
            return None

        cross_frames = await self._cross_timeframe_frames(
            symbol, day, start_ts, end_ts, interval
        )
        feature_sets = await asyncio.to_thread(
            self.feature_builder,
            backfill_result.candles,
            horizons=[self.settings.data.horizon_min],
            symbol=symbol,
            interval=interval,
            fee_bp=self.settings.data.fee_bp,
            spread_model=getattr(self.settings.data, "spread_model", "median"),
            cross_timeframes=cross_frames,
        )
        horizon = sorted(feature_sets.keys())[0]
        built = feature_sets[horizon]
        mask = (built.X.index >= start_ts) & (built.X.index <= end_ts)
        if not mask.any():
            LOGGER.warning("Warm-up trimmed all rows for %s %s %s", symbol, interval, day)
            await self.progress.record_history(
                symbol,
                day=day,
                tf=interval,
                rows_in_day=0,
                metrics=metrics,
                gate=gate,
                gates=gates,
            )
            return None
        feature_set = FeatureSet(
            horizon=built.horizon,
            X=built.X.loc[mask],
            y=built.y.loc[mask],
            regimes=built.regimes.loc[mask],
            regime_labels=built.regime_labels.loc[mask],
            metadata=dict(built.metadata),
        )
        record = await self.progress.update_stage(
            symbol,
            day=day,
            tf=interval,
            stage="features",
            rows_in_day=len(feature_set.X),
            rows_total_tf=backfill_result.rows_total,
        )
        LOGGER.info(
            "Collected %s rows for %s %s on %s; starting training.",
            len(feature_set.X),
            symbol,
            interval,
            day,
        )
        print(
            f"\nCollected data for {symbol} {interval} on {day}, starting training...",
            flush=True,
        )
        self._log_progress(record)
        self._cache_feature_set(symbol, interval, day, feature_set, backfill_result.rows_total)
        return feature_set

    async def _ensure_lgbm_result(
        self,
        symbol: str,
        interval: str,
        day: str,
        feature_set: Optional[FeatureSet],
        backfill_result: Optional[BackfillDayResult],
        metrics: Dict[str, float],
        *,
        existing: Optional[IncrementalTrainingResult],
    ) -> Tuple[Optional[IncrementalTrainingResult], Dict[str, float]]:
        if existing is not None:
            return existing, metrics
        if feature_set is None:
            return None, metrics

        cached = self._load_cached_lgbm_result(symbol, interval, day)
        rows_total = (
            backfill_result.rows_total
            if backfill_result is not None
            else len(feature_set.X)
        )
        if cached is not None:
            self._last_booster[(symbol, interval)] = cached.booster_path
            metrics = self._format_metrics(cached.metrics)
            record = await self.progress.update_stage(
                symbol,
                day=day,
                tf=interval,
                stage="train",
                rows_in_day=len(feature_set.X),
                rows_total_tf=rows_total,
                metrics_day=metrics,
            )
            LOGGER.info("Loaded cached LightGBM result for %s %s on %s", symbol, interval, day)
            self._log_progress(record)
            return cached, metrics

        prev_model = self._last_booster.get((symbol, interval))
        if prev_model is None:
            prev_model = self._active_booster_path(symbol, interval)
        model_dir = self.lgbm_root / symbol / interval
        model_dir.mkdir(parents=True, exist_ok=True)
        result = await asyncio.to_thread(
            self.lgbm_trainer,
            feature_set.X,
            feature_set.y,
            feature_set.regime_labels,
            feature_set.regimes,
            day=day,
            symbol=symbol,
            interval=interval,
            output_dir=model_dir,
            prev_model_path=prev_model,
            calibration_method=getattr(self.settings.model.lgbm, "calibration_method", "isotonic"),
            validation_fraction=getattr(self.settings.model.lgbm, "validation_fraction", 0.3),
            min_validation_rows=getattr(self.settings.model.lgbm, "min_validation_rows", 32),
        )
        self._last_booster[(symbol, interval)] = result.booster_path
        metrics = self._format_metrics(result.metrics)
        record = await self.progress.update_stage(
            symbol,
            day=day,
            tf=interval,
            stage="train",
            rows_in_day=len(feature_set.X),
            rows_total_tf=rows_total,
            metrics_day=metrics,
        )
        LOGGER.info("Finished LightGBM training for %s %s on %s", symbol, interval, day)
        self._log_progress(record)
        self._cache_lgbm_result(symbol, interval, day, result)
        return result, metrics

    def _feature_cache_path(self, symbol: str, interval: str, day: str) -> Path:
        return self.cache_feature_root / symbol / interval / f"{day}.joblib"

    def _lgbm_cache_path(self, symbol: str, interval: str, day: str) -> Path:
        return self.cache_lgbm_root / symbol / interval / f"{day}.joblib"

    def _cache_feature_set(
        self,
        symbol: str,
        interval: str,
        day: str,
        feature_set: FeatureSet,
        rows_total: int,
    ) -> None:
        path = self._feature_cache_path(symbol, interval, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"feature_set": feature_set, "rows_total": int(rows_total)}
        try:
            joblib.dump(payload, path)
        except Exception:  # pragma: no cover - caching best-effort
            LOGGER.debug("Failed to cache features for %s %s %s", symbol, interval, day)

    def _cache_lgbm_result(
        self,
        symbol: str,
        interval: str,
        day: str,
        result: IncrementalTrainingResult,
    ) -> None:
        path = self._lgbm_cache_path(symbol, interval, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump({"result": result}, path)
        except Exception:  # pragma: no cover - caching best-effort
            LOGGER.debug("Failed to cache training result for %s %s %s", symbol, interval, day)

    def _load_cached_feature_set(
        self, symbol: str, interval: str, day: str
    ) -> Optional[Tuple[FeatureSet, Optional[int]]]:
        path = self._feature_cache_path(symbol, interval, day)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
        except Exception:  # pragma: no cover - corrupt cache
            LOGGER.debug("Failed to load cached features for %s %s %s", symbol, interval, day)
            return None
        feature_set: Optional[FeatureSet]
        rows_total: Optional[int] = None
        if isinstance(payload, dict):
            feature_set = payload.get("feature_set") if isinstance(payload.get("feature_set"), FeatureSet) else None
            rows_total = payload.get("rows_total")
        elif isinstance(payload, FeatureSet):
            feature_set = payload
        else:
            feature_set = None
        if feature_set is None:
            return None
        if rows_total is not None:
            try:
                rows_total = int(rows_total)
            except Exception:  # pragma: no cover - defensive
                rows_total = None
        return feature_set, rows_total

    def _load_cached_lgbm_result(
        self, symbol: str, interval: str, day: str
    ) -> Optional[IncrementalTrainingResult]:
        path = self._lgbm_cache_path(symbol, interval, day)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
        except Exception:  # pragma: no cover - corrupt cache
            LOGGER.debug(
                "Failed to load cached LightGBM result for %s %s %s", symbol, interval, day
            )
            return None
        result: Optional[IncrementalTrainingResult]
        if isinstance(payload, dict):
            candidate = payload.get("result")
            result = candidate if isinstance(candidate, IncrementalTrainingResult) else None
        elif isinstance(payload, IncrementalTrainingResult):
            result = payload
        else:
            result = None
        return result

    def _clear_day_cache(self, symbol: str, interval: str, day: str) -> None:
        for path in (self._feature_cache_path(symbol, interval, day), self._lgbm_cache_path(symbol, interval, day)):
            try:
                if path.exists():
                    path.unlink()
            except Exception:  # pragma: no cover - best effort cleanup
                LOGGER.debug("Failed to clear cache file %s", path)

    def _active_booster_path(self, symbol: str, interval: str) -> Optional[Path]:
        active_dir = self.lgbm_root / symbol / interval / "active"
        if not active_dir.exists():
            return None
        candidate = active_dir / "model.txt"
        if candidate.exists():
            return candidate
        for path in active_dir.glob("*.txt"):
            return path
        for path in active_dir.glob("*.bin"):
            return path
        for path in active_dir.iterdir():
            if path.is_file():
                return path
        return None

    def _log_progress(self, record: StatusRecord) -> None:
        metrics = record.metrics_day
        gate = record.gate
        message = (
            f"[SELF-TRAIN] sym={record.symbol} D={record.day} TF={record.tf} stage={record.stage} "
            f"rows_day={record.rows_in_day} rows_total_tf={record.rows_total_tf} "
            f"MCC={metrics.MCC:.2f} ECE={metrics.ECE:.3f} SharpeP={metrics.SharpeProxy:.2f} "
            f"gate={'active' if gate.active else 'inactive'}"
        )
        LOGGER.info(message)
        print(f"\r{message}", end="", flush=True)

    def _gate_from_metrics(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        ece = float(metrics.get("ECE", 1.0))
        mcc = float(metrics.get("MCC", 0.0))
        return {
            "ece_pass": ece <= 0.03,
            "mcc_pass": mcc > 0.0,
            "active": ece <= 0.03 and mcc > 0.0,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        formatted = self._zero_metrics()
        mapping = {
            "mcc": "MCC",
            "ece": "ECE",
            "sharpe_proxy": "SharpeProxy",
            "sharpeproxy": "SharpeProxy",
            "hit_rate": "HitRate",
            "hitrate": "HitRate",
        }
        for key, value in metrics.items():
            norm_key = key.lower()
            target = mapping.get(norm_key)
            if not target:
                continue
            formatted[target] = float(value)
        return formatted

    def _zero_metrics(self) -> Dict[str, float]:
        return {"MCC": 0.0, "ECE": 1.0, "SharpeProxy": 0.0, "HitRate": 0.0}

    def _day_range(self, day: date, interval: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.Timestamp(datetime.combine(day, datetime.min.time()), tz="UTC")
        step = pd.to_timedelta(max(1.0, interval_to_seconds(interval)), unit="s")
        end_ts = start_ts + pd.Timedelta(days=1) - step
        return start_ts, end_ts

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

