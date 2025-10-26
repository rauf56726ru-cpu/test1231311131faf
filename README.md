# Binance Chart Viewer

Проект свёрнут до минимального фронтенда и компактного API, которые отвечают за построение свечного графика и тестовую инспекцию данных, собранных на стороне клиента.

## Возможности
- Загрузка исторических свечей Binance для выбранного тикера и интервала.
- Отрисовка графика на базе Lightweight Charts: свечи, оси, масштабирование, навигация.
- Линия последней цены и краткая сводка по последней свече.
- Автозаполнение пропусков в данных через `ChartGapWatcher`.
- Живое обновление за счёт WebSocket-потока Binance.
- Кнопка «Inspection» формирует снимок текущего состояния свечей и открывает инспекционную панель на основе этих данных.
- REST-эндпоинты `/inspection/snapshot` и `/inspection`, работающие со снимками данных, без дополнительных запросов к внешним сервисам.
- Строгий офлайн-пайплайн с прокси-метриками ордерфлоу и зон без сетевых запросов.
- Все этапы пайплайна логируются в консоль и файл `logs/pipeline.log` (переопределяется через `LOG_PATH`, уровень — `LOG_LEVEL`).
- Derivatives заполняются из Binance Vision `metrics` (если архивов funding/liquidations нет, берём прокси прямо из `metrics`).

### REST-эндпоинты быстрого анализа

- `GET /zones/72h?symbol=BTCUSDT&hours=72` — открытые зоны за окно (параметры `hours`, `end`, `market`, `export`).
- `GET /sessions/last?symbol=BTCUSDT` — метрики последней закрытой UTC-сессии с привязкой к открытым зонам.
- `POST /inspection/snapshot`, `GET /inspection` — инспекционный пайплайн (как раньше).

## Как запустить
1. Запустите приложение FastAPI, которое теперь также обслуживает фронтенд-график и статические файлы:
   ```bash
   uvicorn src.api.app:app --reload
   ```
   После запуска откройте в браузере `http://127.0.0.1:8000/` — загрузится страница графика `index.html`.

   Базовые ответы API:
   - `POST /inspection/snapshot` &mdash; принимает снимок с фронтенда и возвращает идентификатор `snapshot_id`:
   ```json
   { "snapshot_id": "snap-1711459200000-a1b2c3" }
   ```
   - `GET /inspection?snapshot=<ID>` &mdash; отдаёт HTML-дашборд. Для JSON-версии запросите с заголовком `Accept: application/json`:
   ```json
   {
     "DATA": {
       "ohlcv": {"symbol": "BTCUSDT", "tf": "1m", "candles": [...]},
       "meta": {"requested": {"symbol": "BTCUSDT", "tf": "1m"}}
     },
     "DIAGNOSTICS": {
       "snapshot_id": "snap-1711459200000-a1b2c3",
       "generated_at": "2024-03-26T12:00:00Z"
     }
   }
   ```
2. Введите тикер (например, `BTCUSDT`) и выберите интервал свечей. График автоматически загрузит историю и переключится в режим реального времени.

### Быстрый анализ без фронтенда

CLI `python -m src.api.quick_analyze` сводит этапы T1–T4 в один запуск и готовит пакеты для LLM:

```bash
python -m src.api.quick_analyze \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m \
  --window 72h \
  --out out/quick
```

Результат:

- `out/quick/zones_72h.json` — открытые зоны за окно;
- `out/quick/session_last.json` — метрики последней UTC-сессии;
- `out/quick/payload_<ts>.json` — консолидированный payload с зонами, сессиями и конфигурацией LLM (пока только сохраняется).

### UM-инжест (T1)

Бэкэнд дополняется сервисом `UMIngestService`, который агрегирует минутную витрину в RAM через WS+REST:

- комбинированный поток Binance UM (`kline_1m`, `aggTrade`, `bookTicker`) собирается микро-батчами (по умолчанию 1 с), с дедупликацией по уникальным идентификаторам и экспоненциальным бэкоффом при реконнекте;
- агрегатор поддерживает дельту/`CVD`, частоты спреда в б.п. (median/p95) и медиану L1-имбаланса на минуту, а также метрики покрытия (`minutes_found/expected`, `largest_gap_min`, `ws_lag_ms`, `batch_sizes`);
- RAM-буфер хранит последний UTC-день (опционально плюс предыдущий), периодически сбрасывая данные в Parquet (`var/um_ingest/<symbol>/date=YYYY-MM-DD/minute.parquet`) с идемпотентным дедупом по `(symbol, ts_min)`;
- REST-хвост подтягивает последний UTC-день для `klines`, `markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`, закрывая возможные пропуски.

Минимальный пример запуска:

```python
import asyncio
from src.services.um_ingest import UMIngestConfig, UMIngestService

async def main():
    service = UMIngestService(UMIngestConfig(("BTCUSDT", "ETHUSDT")))
    await service.start()
    try:
        await asyncio.sleep(3600)
    finally:
        await service.stop()

asyncio.run(main())
```

Все параметры сервиса доступны через `UMIngestConfig`: окно микро-батчей, лимит держателя, частота сброса, путь к Parquet и т.д.

### Быстрые витрины (T2/T3)

- `GET /analyze/session?symbol=BTCUSDT` и CLI `python -m src.cli.analyze_session BTCUSDT` возвращают `SMC_session_v1` (≤200 KB) за последнюю закрытую UTC‑сессию: OHLC, диапазон, ATR(14), VWAP, RVOL, PDH/PDL/PDC, IB‑60, суммарная дельта и CVD, top‑импульсы, микроструктура и перп‑контекст. На фронте кнопка «Быстрый анализ сессии» мгновенно рендерит карточку.
- `GET /context/72h?symbol=BTCUSDT` и CLI `python -m src.cli.context_72h BTCUSDT` лениво подгружают `SMC_72h_ctx_v1`: FVG/OB/Swing/EQH/EQL, касания последней сессии, coverage/gaps и агрегаты окна. Витрина удерживается ≤ 300 KB.
- CLI‑скрипты `scripts/run_session.sh`, `scripts/run_ctx72.sh`, `scripts/run_all.sh` (с `tee -a logs/pipeline.log`) упрощают запуск и логирование витрин.

Доп. параметры:

- `--llm-providers http://endpoint` — список HTTP JSON провайдеров (запрос POST с тем же payload);
- `--zone-export`, `--session-export` — кастомные пути JSON Lines;
- `--market` — указать альтернативный маркет / каталог данных;
- `--end` — сместить конец окна (ISO8601 или ms).

Если указаны провайдеры, для каждого сохраняется `analysis_<name>_<ts>.json`, а агрегированный список статусов пишется в `analysis_summary_<ts>.json`. Заголовки авторизации задаются через переменные окружения `ANALYSIS_HEADERS_<NAME>` (JSON), таймаут на провайдера и общая отсечка контролируются `--timeout-ms`.

В логах появляется `quick_analyze.complete` с `pipeline_total_ms` и количеством провайдеров; повторные запуски с прогретым кешем заметно быстрее.

## UM-only README

Минимальный запуск без Vision‑архивов:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Переменные окружения:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — ключи для WS/REST (опционально);
- `DATA_DIR` (по умолчанию `data/`), `VAR_DIR` (`var/`), `LOG_PATH` (`logs/pipeline.log`);
- `UM_INGEST_SYMBOLS` — список символов для ingest.

Команды:

- `python -m src.services.um_ingest_runner` — запускает `UMIngestService` с config.
- `scripts/run_session.sh BTCUSDT` — быстрая карточка сессии.
- `scripts/run_ctx72.sh BTCUSDT` — контекст 72 ч.
- `scripts/run_all.sh BTCUSDT` — оба пайплайна подряд.

SLA: coverage ≥ 0.99, largest_gap_min ≤ 1, latency analyze-session ≤ 2.5 с, context-72h ≤ 3 с. При превышении сервис логирует `degradations` в JSON.

## Структура
- `public/` &mdash; JS/стили для графика (`app.js`, `binanceCandles.js`, `chart-gap-watcher.js`, `styles.css`).
- `templates/index.html` &mdash; единственная страница с графиком и панелью управления.
- `src/` &mdash; минимальный FastAPI-проект с обработчиками `/inspection/snapshot`, `/inspection`, `health`, `version` и вспомогательными модулями `services/`.

## Live data pipeline

- Базовый поток данных теперь использует официальный `python-binance` (binance-connector) клиент и сразу тянет свечи через UM REST API.
- Архивы Binance Vision и Materialize-утилиты остаются для бэктестов, но рабочая цепочка `/context/72h` и `/analyze/session` их больше не трогает.
- Переменные окружения управляются через `.env`: `DATA_DIR`, `DUCKDB_PATH`, `FIXTURE_DIRS`, `INGEST_POLICY`, `MARKET`. Значение `OFFLINE` игнорируется.
- CLI `src/cli/smc72.py` оставляет аргументы для обратной совместимости, но "оффлайн" режим снят — все запросы выполняются онлайн.

## REST ingestion

- Модуль `src/storage/ensure_window.py` напрямую запрашивает Binance `/fapi/v1/klines`, пока не соберёт ≥99% минут.
- Дополнительные дозагрузки выполняются через повторные REST-запросы с перекрытием; Parquet и DuckDB больше не задействуются в боевой цепочке.
- В логи (`logs/pipeline.log`) пишутся метрики: сколько минут найдено, сколько запросов потрачено, какой coverage достигнут.

## Профиль объёма и TPO

Эндпоинт `/profile` возвращает рассчитанный профиль объёма и уровни VAH/VAL/POC для последних торговых сессий.

```http
GET /profile?snapshot=<SNAPSHOT_ID>&tf=1m&last_n=3&adaptive_bins=false&value_area_pct=0.7
```

Параметры запроса:

- `snapshot` — обязательный идентификатор ранее сохранённого снапшота.
- `tf` — таймфрейм, по умолчанию `1m`.
- `last_n` — количество последних сессий (1–5, по умолчанию 3).
- `tick_size` — фиксированный шаг цены, если известен.
- `adaptive_bins` — переключает адаптивный расчёт шага бина (0.5 × ATR), когда `tick_size` не задан.
- `value_area_pct` — доля объёма в value area (по умолчанию 0.7).

Пример ответа:

```json
{
  "symbol": "BTCUSDT",
  "tf": "1m",
  "tpo": {
    "sessions": [
      {"date": "2024-05-12", "session": "asia", "POC": 61050.0, "VAL": 60920.0, "VAH": 61200.0},
      {"date": "2024-05-12", "session": "london", "POC": 61310.0, "VAH": 61420.0, "VAL": 61210.0}
    ],
    "zones": [
      {"type": "tpo_poc", "price": 61310.0, "date": "2024-05-12", "session": "london", "tf": "1m", "meta": {"value_area_pct": 0.7, "source": "tpo"}},
      {"type": "tpo_vah", "price": 61420.0, "date": "2024-05-12", "session": "london", "tf": "1m", "meta": {"value_area_pct": 0.7, "source": "tpo"}}
    ]
  },
  "profile": [
    {"price": 61200.0, "volume": 15.2},
    {"price": 61250.0, "volume": 18.7}
  ],
  "zones": {
    "symbol": "BTCUSDT",
    "zones": {
      "fvg": [
        {"tf": "1m", "dir": "up", "top": 61350.0, "bot": 61280.0, "fvl": 61315.0, "created_at": 1715515200000, "status": "open"}
      ],
      "ob": [],
      "inducement": [],
      "cisd": []
    }
  },
  "preset": {
    "symbol": "BTCUSDT",
    "tf": "1m",
    "last_n": 3,
    "value_area_pct": 0.7,
    "binning": {"mode": "adaptive", "tick_size": null, "atr_multiplier": 0.5, "target_bins": 80},
    "extras": {"clip_low_volume_tail": 0.005, "smooth_window": 1},
    "builtin": true
  },
  "preset_required": false
}
```

Секция `tpo.sessions` содержит сводку по последним сессиям, `tpo.zones` — уровни POC/VAH/VAL в формате, совместимом с остальными провайдерами зон, `profile` — дискретный профиль объёма для самой свежей сессии, а объект `zones` содержит структурированные зоны FVG/OB/inducement/CISD.

### Система пресетов TPO

Расчёт профиля управляется пресетами:

- Для популярных инструментов `BTCUSDT`, `ETHUSDT` и `SOLUSDT` встроены дефолтные пресеты (адаптивный биннинг 0.5 × ATR, `last_n=3`, `value_area_pct=0.7`). Они применяются автоматически, а интерфейс отображает компактный бейдж «Preset: SYMBOL».
- Для новых символов пресет настраивается один раз через модальное окно на странице `/inspection`. Введённые параметры сохраняются локально и применяются автоматически при следующих обращениях.
- Перед запуском «check all datas» пресет уже определён (встроенный или пользовательский), а рассчитанные уровни TPO добавляются в массив `zones` основного ответа.

Доступные REST-эндпоинты для управления пресетами:

- `GET /presets` — список доступных пресетов (встроенные + пользовательские).
- `GET /presets/{symbol}` — пресет для конкретного символа (или `null`, если пользовательский ещё не создан).
- `POST /presets` — создание/сохранение пресета (`symbol` обязателен).
- `PUT /presets/{symbol}` — частичное обновление существующего пресета.
- `DELETE /presets/{symbol}` — удаление пользовательского пресета (встроенные остаются доступными).

Все пользовательские пресеты сохраняются в локальном JSON (`var/presets.json`) и кэшируются в памяти для быстрого доступа. Сервер валидирует параметры (`last_n` ∈ [1..5], `value_area_pct` ∈ (0,1), ограничения для биннинга и дополнительных опций) и корректирует их при необходимости.

## Строгий офлайн режим

- Запуск `/inspection` с `strict_window=True` и `network_backfill=False` использует 72-часовой срез минутных данных из репозитория без запросов к Binance.
- Прокси-ордерфлоу строится напрямую из минутных свечей: берётся полный 12-часовой хвост, свёртки `3m/5m/15m` считаются локально и сохраняют диагностические длины/coverage.
- VWAP/TPO и вычисление зон используют тот же срез минутных данных; логирование помечает, что метрики восстановлены локально, а строгий режим отражается в `meta.strict_three_day`.
- Отсутствие «liquidity map» явно фиксируется в заметках пайплайна, пока модуль отключён для офлайна.
- Историю прогонов ищите в `logs/pipeline.log`; ротация и путь управляются переменными `LOG_PATH`, `LOG_BACKUP_COUNT` и `LOG_CONSOLE`.
- При ручном `/inspection/snapshot` не обязательно передавать funding/liquidity — бэкенд заполнит их прокси-значениями и пропустит нулевые значения open interest.

## Тесты

- Перед первым прогоном стоит подготовить реальный кеш (примеры см. выше). Для автоматизации можно задать `REAL_BOOTSTRAP=1`
  и запустить `pytest` — conftest вызовет `scripts.bootstrap_vision_cache` для указанных символов.
- Повторные прогоны используют уже скачанные Parquet-файлы, поэтому сеть не требуется. Если вы хотите запускать тесты полностью офлайн,
  установите `OFFLINE=1` — проверки, требующие реальных данных, будут пропущены, если соответствующего кеша нет.
- Пример последовательности:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ./scripts/bootstrap_vision_cache.sh --symbols BTCUSDT --interval 1m --market um --days 3
  pytest -q
  ```
