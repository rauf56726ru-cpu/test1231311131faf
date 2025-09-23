# Руководство для новичка

Эта памятка описывает минимальный цикл от исторических свечей до решения об исполнении сделки. Старайтесь держать процесс воспроизводимым: фиксируйте сиды (`get_settings().seed`), работайте в UTC и сохраняйте промежуточные артефакты (фичи, метрики, отчёты).

```
свечи → фичи → модель → калибровка → политика → исполнение
```

## 1. Свечи

* Источник — `data/raw/<SYMBOL>_<INTERVAL>.parquet` или поток `StreamManager`.
* Исключаем синтетические бары (`is_synthetic = True`).
* Храним индекс в UTC (`DatetimeIndex`).

Пример загрузки:

```python
import pandas as pd
candles = pd.read_parquet("data/raw/BTCUSDT_1m.parquet")
candles.index = pd.to_datetime(candles.index, utc=True)
candles = candles.loc[~candles["is_synthetic"].astype(bool)].drop(columns=["is_synthetic"])
```

## 2. Фичи

Используем `src/features/builder.build_features` — он добавляет структурные паттерны (BOS/FVG/свинги), импульсы, волатильность, сессионные признаки и orderflow-метрики. Target: трёхклассовый (`-1/0/+1`) с «плоской» зоной на величину комиссий+спреда.

```python
from src.features.builder import build_features, save_feature_sets
feature_sets = build_features(candles, horizons=[15], symbol="BTCUSDT", interval="1m")
paths = save_feature_sets(feature_sets, base_path="data/features/BTCUSDT_1m")
```

## 3. Модель

Для XGBoost используем `src/models/xgb_train.run_training(features_path=...)`. Тренировка включает purged walk-forward, пер-режимные метрики и (при наличии optuna) подбор гиперпараметров.

```python
from src.models.xgb_train import run_training
summary = run_training(features_path=paths[15])
```

## 4. Калибровка

Встроена в пайплайн XGB (`RegimeCalibrator`). Результат сохраняется в `calibrator.pkl` и используется в инференсе (`xgb_predict.predict_distribution`) для корректных вероятностей и доверительных интервалов.

## 5. Политика

`PolicyRuntime` сочетает сигналы модели с правилами (Kelly, абстиненция, фильтры RAG, запрет усреднения против тренда, контроль drawdown). Контекст включает фичи, состояние счёта, издержки и попавшие фильтры.

```python
from src.policy.compiler import load_and_compile
from src.policy.runtime import PolicyRuntime
compiled = load_and_compile("configs/rules.yaml")
policy = PolicyRuntime(compiled)
decision = policy.apply(prediction=dict(prob_up=0.6, prob_down=0.2, pred_conf=0.6), context={"features": features.iloc[-1].to_dict()})
```

## 6. Исполнение

`src/backtest/engine.run_backtest` моделирует реальное исполнение: maker/taker-комиссии (`commission_bps`), задержку (`latency_bars`), частичное заполнение, проскальзывание (`spread × f(size, vol)`) и фандинг.

```python
from src.backtest.engine import run_backtest
result = run_backtest(candles, features, predictions, policy, regimes=feature_set.regimes,
                      costs={"commission_bps": 7}, execution={"latency_bars": 1})
```

Следите за метриками: `Sharpe`, `Calmar`, `MDD`, `ECE` (после калибровки ≤3%), `slippage_gap` ≤20%. При ухудшении — выполняйте роллбэк на последнюю «зелёную» модель.

## 7. Самообучение

API `/self_train/start` запускает воркеры, которые:

1. Загружают недостающие свечи до/от указанной даты.
2. Перестраивают фичи и таргеты без синтетики.
3. Обучают модель (XGB), калибруют и прогоняют walk-forward.
4. Обновляют `data/self_train/progress.json` (символ, режим, покрытия, метрики).

Гейты активации остаются строгими: `ECE≤3%`, положительный `MCC`, `Sharpe_WF≥1.0`, `Calmar≥0.5`, `slippage_gap≤20%`. Без выполнения — модель не переводится в прод.

## 8. Визуализация и UI

Реальные свечи отображаются белым/чёрным телом, прогнозные — зелёным/красным контуром при уверенности вне серой зоны. Боковая панель показывает уверенность, доверительный интервал, текущий режим и активные паттерны (BOS/FVG/знания).

## Куда смотреть дальше

* `CHANGELOG.md` — краткая история изменений.
* `tests/` — примеры использования API, бэктеста и self-train.
* `/predict` — живая картина с терминалом, журналом решений и модалами (загрузка знаний, самообучение).
