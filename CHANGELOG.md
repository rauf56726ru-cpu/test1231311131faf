# CHANGELOG
## [2025-09-21]
- Switched market data client to Binance USDT-M futures REST/websocket endpoints with 1000-bar pagination and UTC ms filtering.
- Normalized candle APIs to emit `ts_ms_utc` millisecond timestamps, `source="binance_futures_usdtm"`, and sanity-check last bar time against `server_now_utc`.
- Added predictive bootstrap service ensuring `/predict` warms 1m–1d frames, logs `[BOOT]`/`[VERIFY]`, and exposes readiness via `/predict/data`.
- Refreshed `/predict` UI with in-chart timeframe toggles, debounce, spinner, and UTC tooltips rendering ms timestamps directly.
- Improved 10m aggregation from 1m history and backend pagination so `/predict` cold starts without spot fallback or small-limit probes.
- Expanded bootstrap coverage to load the full 30-day 1m history (~43k bars) so aggregated frames stay aligned with resampled 1m data in regression tests.

## [2025-09-20]
- Обновлён фронтенд `/predict`: реальные свечи рендерятся белым/чёрным, прогнозы подсвечиваются контуром с учётом серой зоны, добавлены легенда, паттерны BOS/FVG и индикаторы знаний.
- Терминал журнала стал шире (≥60vw), поддерживает автопрокрутку, кнопки «копировать/очистить» и полноразмерный resize.
- UI включает модал самообучения (`/self_train`), статус прогресса и обновлённую панель вероятностей/уверенности.
- API `build_predict_payload` возвращает `pred_conf`, доверительный интервал, серую зону, `pattern_hits` и режим волатильности; WebSocket-ответы и тесты адаптированы к трёхклассовой схеме.
- Реализован `SelfTrainManager`: загрузка истории без синтетики, построение фич, обучение XGB, калибровка и walk-forward с метриками (Sharpe, Calmar, положительные месяцы). Прогресс хранится в `data/self_train/progress.json`.
- Добавлено руководство [docs/BEGINNER_GUIDE.md](docs/BEGINNER_GUIDE.md) и обновлён README с ссылкой и описанием новых UX-функций.
