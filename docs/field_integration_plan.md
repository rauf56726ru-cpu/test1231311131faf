# План интеграции новых полей аналитики

## 1. Контекст и цель

Цель — расширить выдачу «72 ч контекста» (endpoint `/context/72h`, CLI `src/cli/context_72h.py`) и родственных пайплайнов полями:

```
bias.1D, bias.4H, bias.1H
vwap_context.daily, vwap_context.session.{asia,london,ny}
tpo_context.{POC,VAH,VAL}
sessions.IB_HighLow
zones.{fvg,ob,mb,bb,pb}.list, zones.validation_flags
orderflow.last_4h.{delta_cvd_summary,footprint_absorption}
structure.last_4h.triggers(15m/5m/1m)
liquidity_targets.PDH/PDL/EQH/EQL/prices
risk.{news_24_72h,event_risk_score,structure_break_prob,timing_risk}
trade.{rr_min,entry_price,sl,tp1,tp2}
```

План ниже описывает: источник данных, вычислительный модуль, изменения API/фронта, даже если часть данных ещё не собирается (отмечаем, как включить в пайплайн).

---

## 2. Пайплайновые этапы

| Этап | Ответственность | Ключевые действия |
|------|-----------------|-------------------|
| T1 Ingest | `src/services/um_ingest`, `src/storage/ensure_window` | Убедиться, что окно `start_ms → end_ms` покрывает ≥ 72 ч минуток и хранит агрегаты (delta, CVD, VWAP, footprint). |
| T2 Storage | DuckDB / RAM кадры | Расширить вьюхи/таблицы для агрегированных серий (H1/H4/D1 свитчи, VWAP по сессиям, TPO). |
| T3 Analytics | `src/services/zones_ctx72.py`, `src/services/zones_context.py`, модули `src/analysis/*` | Добавить вычисление и упаковку новых полей; при отсутствии логики — создать сервисы (bias, structure, risk, trade). |
| T4 Delivery | `/context/72h` и фронт (`public/app.js`) | Обновить JSON-схему, сериализацию и рендер карточки. |

### 2.1 Статусы доступности данных

| Поле | Источник уже в пайплайне? | Что требуется |
|------|---------------------------|---------------|
| bias.{1D,4H,1H} | Да — минутные свечи + агрегаты `tf_windows`. | Реализовать расчёт bias и сериализацию. |
| vwap_context.daily / session.* | Да — `vwap_tpo` формируется в session collector. | Расширить экспорт в контекст. |
| tpo_context.{POC,VAH,VAL} | Да — `calculate_tpo`, `calculate_session_tpo`. | Подключить к payload. |
| sessions.IB_HighLow | Да — `compute_session_metrics` возвращает IB. | Добавить в ответ контекста. |
| zones.{fvg,ob,mb,bb,pb}.list, validation_flags | Да — `detect_zones` и диагностика. | Сгруппировать и сериализовать. |
| orderflow.last_4h.{delta_cvd_summary, footprint_absorption} | Да — per-bar и footprint из orderflow collector. | Сделать 4h агрегацию. |
| structure.last_4h.triggers(15m/5m/1m) | Да — 15m/5m/1m свечи уже передаются. | Написать анализатор триггеров. |
| liquidity_targets.* | Да — `generate_liquidity_map`. | Форматировать блок целей. |
| risk.structure_break_prob | Да — ATR, delta, CVD, зоны в пайплайне. | Определить формулу риска. |
| trade.* | Да — зоны, уровни, риск-параметры доступны. | Построить планировщик сделки. |
| risk.news_24_72h | Нет. | Подключить внешний календарь/фид. |
| risk.event_risk_score, risk.timing_risk | Нет (читаются из `meta`, но не заполняются). | Собрать события/календарь и расчёт. |

---

## 3. Группа: Bias (1D/4H/1H)

**Источник:** минутные свечи (UM REST + WS), уже подхватываются `ensure_window_real`.

**Статус данных:** история `1m` и агрегированные окна `tf_windows` уже доступны; требуется только расчёт и выдача bias.

**Шаги:**
1. Добавить модуль `src/analysis/bias.py`:
   - агрегировать минутки в 1H/4H/1D (resample).
   - рассчитать bias (направление тренда, например close vs open, ma slope).
   - вернуть структуру `{ timeframe: {"direction": "bull/bear/neutral", "confidence": float, "change_pct": float} }`.
2. Вызывать из `build_smc_72h_ctx_v1` (72 ч builder) до конструктора zonов: сохранить в `aggregates["bias"]`.
3. Расширить `/context/72h` JSON (`ctx_payload["aggregates"]["bias"]`).
4. На фронте добавить секцию «Bias» + бейджи.

**Статус:** backend-расчёт и секция Bias на фронте готовы; остаются unit-тесты/валидация.

**Тесты:** snapshot-тест для `bias` на синтетических данных (рост/падение).

---

## 4. Группа: VWAP context (daily + sessions Asia/London/NY)

**Источник:** минутки + календарь сессий (UTC bounds). Схожая логика уже есть для daily VWAP в `session_last`.

**Статус данных:** session collector (`_build_vwap_session_block`) и `vwap_tpo` уже формируют daily/сессионные VWAP; нужно лишь собрать и отдать в контексте.

**Шаги:**
1. В `src/common/session_windows.py` описать UTC-окна для Asia (00-08 UTC), London (08-16), NY (13-21).
2. Создать `src/analysis/vwap_context.py`:
   - расчет VWAP для каждой сессии за 72 ч (возможно, последние N окон).
   - вернуть `{"daily": {...}, "session": {"asia": {...}, ...}}`.
3. Засунуть в `ctx_payload["aggregates"]["vwap_context"]`.
4. UI: таблица из четырех строк (Daily + 3 сессии) с VWAP, отклонениями от текущей цены.

**Валидация:** сравнить с бенчмарком TradingView, вручную для одной даты.

**Статус:** backend, фронтенд и unit-тесты готовы; требуется внешняя валидация.

---

## 5. Группа: TPO context (POC/VAH/VAL)

**Источник:** TPO профиль (`calculate_tpo`, `calculate_session_tpo`, `/profile` endpoint).

**Статус данных:** TPO расчёты уже возвращают POC/VAH/VAL и лежат в `vwap_tpo`; остаётся извлечь и сериализовать.

**Шаги:**
1. Проверить `src/services/profile` — можно переиспользовать `calculate_profile`.
2. Реализовать helper `compute_tpo_context(frame, last_sessions=3)` возвращает `{POC, VAH, VAL}` для последних сессий (или объединённых 72 ч).
3. Пушить в `ctx_payload["aggregates"]["tpo_context"]`.
4. На фронте — карточка с TPO уровнями (цифра + бейдж отклонения).

**Тесты:** unit-тест на синтетике (равномерный профиль) + snapshot.

**Статус:** backend, UI и unit-тесты готовы; требуется интеграционная валидация.

---

## 6. Группа: Sessions (IB High/Low)

**Источник:** `compute_session_metrics` уже считает IB (проверить `levels.ib_high/ib_low`).

**Статус данных:** IB диапазон хранится в `session_payload["metrics"]["levels"]`; нужно подтянуть в контекст и отрисовать.

**Шаги:**
1. Убедиться, что `session_payload["metrics"]["levels"]` содержит IB.
2. В `/context/72h` добавить `ctx_payload["sessions"] = {"IB_HighLow": {...}}` с последней/предпоследней сессией.
3. Фронт: вывести отдельный блок «Sessions» с IB диапазоном.

**Валидация:** сопоставить с `/sessions/last` JSON.

**Статус:** backend/UI/unit-тесты готовы; требуются интеграционная проверка и сравнение с `/sessions/last`.

---

## 7. Группа: Zones (fvg/ob/mb/bb/pb) + validation_flags

**Источник:** `ZoneDetector` уже выдает FVG/OB/Swings; дополнительные типы (mb/bb/pb) нужно расширить.

**Статус данных:** списки зон и диагностические флаги (reasons) уже присутствуют в `src/services/zones.py`; задача — собрать их в структуру `zones.*.list` и экспортировать.

**Шаги:**
1. В `ZoneDetector._detect_order_blocks` добавить классификацию по подтипам (mitigation block, breaker, potential block).
2. Для FVG/OB уже есть list -> `zones_top`. Добавить:
   - `ctx_payload["zones"]["fvg"]["list"]` — top N FVG (id, price range, touches).
   - Аналогично для ob/mb/bb/pb.
3. `validation_flags`: вычислять при построении зоны (например `overlaps_session`, `confirmed_by_delta`, `invalidated`).
4. Фронт: табы или аккордеон, отображающий списки.

**Тесты:** моковые данные с известными зонами, assert на классификацию.

---

## 8. Группа: Orderflow (last_4h)

**Источник:** агрегаты дельты/CVD и footprint (UM ingest → `frame["delta"]`, `frame["volume"]`).

**Статус данных:** per-bar (`orderflow.per_bar`) и агрегаты (`delta_cvd_compact`) уже пишутся collector’ом; нужен компактный расчёт за последние 4 часа.

**Шаги:**
1. Создать `src/analysis/orderflow_last4h.py`:
   - взять окно `end_ms - 4h`.
   - посчитать sum delta, max absorption (по критерию wick vs delta).
   - структура `{ "delta_cvd_summary": {...}, "footprint_absorption": {...} }`.
2. Добавить в `ctx_payload["orderflow"]["last_4h"]`.
3. UI: карточка «Orderflow 4h» с summary + список absorption сигналов.

**Тесты:** Проверить на синтетике (`delta` чередуется знак).

---

## 9. Группа: Structure triggers (last_4h, TF 15m/5m/1m)

**Источник:** 15m/5m/1m свечи отдаются в `tf_windows`; специализированного анализатора пока нет.

**Статус данных:** все свечные серии уже доступны; требуется новый модуль анализа без дополнительного сбора.

**Шаги:**
1. Создать `src/analysis/structure.py`:
   - Разбить 4h окно на 15m/5m/1m.
   - Идентифицировать триггеры (BMS, CHoCH, BOS).
2. Вернуть `{ "triggers": {"15m": [...], "5m": [...], "1m": [...]} }`.
3. Добавить в payload `ctx_payload["structure"]["last_4h"]`.
4. Front: collapsible список с таймфреймами.

**Тесты:** Unit-тест со заранее размеченной серией.

---

## 10. Группа: Liquidity targets (PDH/PDL/EQH/EQL)

**Источник:** уже есть в `session metrics` (`levels.pdh/pdl`, `liquidity` section).

**Статус данных:** `generate_liquidity_map` возвращает PDH/PDL/EQH/EQL; добавить сериализацию в контекст.

**Шаги:**
1. Реиспользовать `compute_liquidity_levels` (либо реализовать helper).
2. Собрать словарь:
   ```json
   "liquidity_targets": {
     "PDH": {"price": float, "age_ms": int},
     "PDL": {...},
     "EQH": [...],
     "EQL": [...]
   }
   ```
3. UI: бейджи с ценой и статусом (пробит/нет).

**Валидация:** сравнить с существующим `/sessions/last`.

---

## 11. Группа: Risk (news, scores, probabilities)

**Источник:** внешние сервисы / внутренние эвенты.

**Статус данных:**
- `risk.news_24_72h`, `risk.event_risk_score`, `risk.timing_risk` — данных нет, требуется новый фид/календарь и сохранение в `meta`.
- `risk.structure_break_prob` — все входы (ATR, delta, CVD, зоны) уже в пайплайне; достаточно реализовать расчёт.

**Шаги:**
1. `risk.news_24_72h`:
   - интеграция с новостным фидом (e.g. saved JSON в `var/news_cache`).
   - добавить провайдер в `src/services/providers/news_provider.py`.
2. `risk.event_risk_score`:
   - таблица маппинга (например, высокий/средний/низкий на основе новостей + календаря).
3. `risk.structure_break_prob`, `risk.timing_risk`:
   - вычислить из structure triggers + волатильность (ATR).
   - создать `RiskAnalyzer` в `src/analysis/risk.py`.
4. В payload добавить `ctx_payload["risk"]`.
5. UI: блок «Risk dashboard» + цветовые индикаторы.

**Логирование:** при отсутствии данных — писать warning, возвращать `status: "pending"`.

---

## 12. Группа: Trade блок (rr_min, entry/sl/tp)

**Источник:** Комбинация зон + риск-менеджмент (риск-профиль в `risk_prefs`, зоны c диапазонами).

**Статус данных:** необходимые входы (зоны, liquidity targets, риск-параметры) уже собираются; требуется расчёт и выдача торгового плана.

**Шаги:**
1. Создать `TradePlanner` в `src/analysis/trade_plan.py`:
   - выбрать приоритетную зону (по strength и bias).
   - рассчитать entry (середина зоны), SL (за границей), TP1/TP2 (по RR 1:1, 1:2).
   - `rr_min` = минимальный RR из TP.
2. Добавить в payload `ctx_payload["trade"]`.
3. UI: карточка с ценами (entry/sl/tp1/tp2) + RR badge.

**Тесты:** Unit-тест на синтетической зоне.

---

## 13. API и сериализация

1. Обновить `src/services/zones_ctx72.py` и `src/services/zones_context.py`:
   - расширить возвращаемый словарь.
   - добавить версионный bump (`SMC_72h_ctx_v2`).
2. `/context/72h` эндпоинт:
   - добавить `schema="SMC_72h_ctx_v2"`.
   - контролировать `CONTEXT_PAYLOAD_LIMIT_BYTES` (при расширении >300 KB — добавить компрессию ключей и compact-режим).
3. CLI `context_72h` писать payload в JSON с indent=2 (для ручной проверки).

---

## 14. Фронтенд (`public/app.js`)

1. Расширить `renderContextPayload`:
   - новые секции: Bias, VWAP, TPO, Liquidity Targets, Risk, Trade.
   - переиспользовать существующие бейджи.
2. Добавить модульные форматтеры (напр. `renderRisk`, `renderTrade`).
3. Обновить CSS (`public/styles.css`) для карточек.
4. Обновить «Копировать/Скачать JSON» — передавать всю новую структуру.

---

## 15. Тестирование и контроль качества

| Категория | Действия |
|-----------|----------|
| Unit | новые тесты в `tests/analysis/` для bias, vwap_context, orderflow, risk, trade. |
| Integration | e2e тест `/context/72h` (pytest + httpx) проверяет схему `SMC_72h_ctx_v2`. |
| UI | обновить Cypress/Playwright сценарий (если есть) для карточки 72 ч. |
| Load | убедиться, что время ответа ≤ 3 с, payload ≤ 300 KB (опционально gzip). |
| Regression | прогнать `scripts/run_all.sh BTCUSDT` — убедиться, что логи без ошибок. |

Логирование:
- Добавить `ctx_payload["diag"]["missing_fields"]` если какие-то блоки `None`.
- В `logs/pipeline.log` вынести `context72.v2.complete` с бэкендом.

---

## 16. Документация и коммуникация

1. README: дописать раздел «Новые поля SMC_72h_ctx_v2» с примером JSON.
2. Wiki/Confluence: диаграмма данных (истоки → пайплайн → фронт).
3. Командная коммуникация:
   - демо для трейдинговой команды (как читать новые поля).
   - синк с DevOps (нагрузка).

---

## 17. Дорожная карта (пример)

| Спринт | Основные задачи |
|--------|-----------------|
| S1 | Bias, VWAP context, обновление схемы v2. |
| S2 | TPO, Sessions IB, Liquidity targets. |
| S3 | Zones расширения, Orderflow. |
| S4 | Structure triggers, Risk. |
| S5 | Trade planner, фронт-финиш, QA. |

---

## 18. Риски и mitigation

- **Отсутствие внешних новостей** → внедрить фолбэк (return `status="no_data"`).
- **Payload > 300 KB** → вариант B: передавать резюме, а полные списки — по отдельному эндпоинту.
- **Недостаток данных для structure triggers** → временно обозначить поле как `experimental`.
- **Сложность фронта** → вынести карточку 72 ч в отдельный React-like компонент (если масштаб растёт).

---

## 19. Итог

После выполнения шагов `/context/72h` будет выдавать `SMC_72h_ctx_v2` с полноценной структурой принятия решений (bias → zonas → risk → trade). Карточка «Собрать 72 ч» станет компактной, но содержательной: все ключи будут отображаться в аналитических блоках, не перегружая пользователя сырым JSON.
