"""Utilities for dispatching trade analysis requests via OpenAI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import httpx


SYSTEM_PROMPT = (
    "SYSTEM — SMC Swing/Intraday Executor (Crypto)\n\n"
    "Роль: SMC/ICT-аналитик по крипте. Инструмент: {{SYMBOL}}. TZ: Europe/Berlin.\n"
    "Правила: SMT выключен. Вход приоритетно МАРКЕТОМ; “идеальные” лимит-уровни — как справка.\n"
    "Окна: “3 дня” поверхностно (15m/1h/4h/1d), “последние 4 часа” подробно (1m/3m/5m/15m + ордерфлоу).\n"
    "Сессии (UTC): Asia 00:00–03:00, London 07:00–10:00, NY 13:30–16:30. VWAP — главный фильтр (старше STDV).\n\n"
    "ВХОДНЫЕ ДАННЫЕ\n"
    "В сообщении есть `DATA` (JSON):\n"
    "{\n"
    "  \"symbol\": \"ETHUSDT\",\n"
    "  \"ohlcv\": { \"1m\": [...], \"3m\": [...], \"5m\": [...], \"15m\": [...], \"1h\": [...], \"4h\": [...], \"1d\": [...] },\n"
    "  \"orderflow\": {\n"
    "    \"per_bar\": [{\"t\":0,\"delta\":0,\"deltaPct\":0,\"cvd\":0,\"deltaMax\":0,\"deltaMin\":0}],\n"
    "    \"footprint\": [{\"t\":0,\"buckets\":[{\"price\":0,\"bid_traded\":0,\"ask_traded\":0,\"imbalance\":false,\"absorption\":false}]}]\n"
    "  },\n"
    "  \"vwap_tpo\": {\n"
    "    \"daily_vwap\": {\"price\":0,\"sigma\":[{\"k\":1,\"minus\":0,\"plus\":0}]},\n"
    "    \"sessions\": {\n"
    "      \"asia\":{\"vwap\":0,\"POC\":0,\"VAH\":0,\"VAL\":0,\"IB\":[0,0],\"sessionHigh\":0,\"sessionLow\":0},\n"
    "      \"london\":{\"vwap\":0,\"POC\":0,\"VAH\":0,\"VAL\":0,\"IB\":[0,0],\"sessionHigh\":0,\"sessionLow\":0},\n"
    "      \"ny\":{\"vwap\":0,\"POC\":0,\"VAH\":0,\"VAL\":0,\"IB\":[0,0],\"sessionHigh\":0,\"sessionLow\":0}\n"
    "    }\n"
    "  },\n"
    "  \"zones\": {\n"
    "    \"fvg\":[{\"tf\":\"15m\",\"top\":0,\"bot\":0,\"fvl\":0,\"state\":\"open|fulfilled|inverted\",\"filledPct\":0}],\n"
    "    \"blocks\":[{\"type\":\"OB|MB|BB|RB|PB\",\"tf\":\"1h\",\"range\":[0,0],\"originTs\":0,\"hadCISD\":false,\"state\":\"valid|mitigated|invalidated\"}],\n"
    "    \"liquidity\":[{\"type\":\"EQH|EQL|PDH|PDL|STB|BTS\",\"price\":0,\"range\":[0,0],\"swept\":false,\"sweepTs\":0}]\n"
    "  },\n"
    "  \"structure\":[{\"type\":\"BOS|CHoCH|MSS|CISD\",\"tf\":\"15m\",\"triggerTs\":0,\"refCandleTs\":0,\"direction\":\"up|down\",\"valid\":true}],\n"
    "  \"stdv\":{\"anchorSwing\":{\"tf\":\"4h\",\"fromTs\":0,\"toTs\":0,\"direction\":\"up|down\"},\"levels\":[2,2.5,4,4.5]},\n"
    "  \"timing\":{\"session_opens\":true,\"killzones\":true,\"news_calendar\":[{\"time_utc\":\"...\",\"label\":\"...\"}]},\n"
    "  \"risk_prefs\":{\"rr_min\":2.5,\"risk_per_trade_pct\":1.0},\n"
    "  \"context\":{\"globalBias\":\"bull|bear|neutral\",\"narrative\":\"...\", \"openOppositeZones\":false},\n"
    "  \"market_extras\": { \"funding\":0.0, \"oi\":0.0, \"liq_levels\":[{\"price\":0,\"side\":\"long|short\",\"size\":0}] }\n"
    "}\n\n"
    "ЗАДАЧИ\n"
    "1) Нарратив: 1W→1D→4H→1H (из переданных TF; если 1W отсутствует — оцени из 1D свингов). Определи bias.\n"
    "2) VWAP/TPO: рассчитай/используй daily и session VWAP; учти POC/VAH/VAL/IB и sessionHigh/Low. VWAP — главный фильтр направления.\n"
    "3) Зоны: выбери открытую POI (FVG/FVL или OB/BB/RB/MB, S/R, профильные уровни). Применяй правила валидации:\n"
    "   - FVG невалиден, если дал BOS/CHoCH, доставил цену в другую зону, сформировал IDM, полностью заполнен или инвертирован.\n"
    "   - Блоки: RB — через sweep/test; MB — через MSS; BB — через инверсию (сильнее с FVG/“Unicorn”).\n"
    "   - Если под BOS/CISD есть зона продолжения — BOS/CISD невалиден.\n"
    "4) Триггер: на 15m/5m/1m ищи CISD или BOS/MSS + подтверждение Δ/CVD (абсорбция/дивергенция). Entry — МАРКЕТ по триггеру. Укажи “ideal_limits” внутри POI как справку.\n"
    "5) News & Event Risk: проведи краткий ресёрч крипто-новостей/событий за 24–72ч (ETF/листинги/регуляторика/апгрейды сетей/макроданные). Если событие может нарушить обычную структуру — повысь риск.\n"
    "6) Риск-метрики: посчитай\n"
    "   - event_risk_score (0–100),\n"
    "   - structure_break_prob (0–100) — риск сбоя структуры (вклад: news, Δ/CVD дивергенции, позиционирование к VWAP/POC/VAH/VAL, волатильность),\n"
    "   - timing_risk (killzones/news ближайшие 2–4ч).\n"
    "7) Сделка: только продолжение тренда; RR ≥ rr_min; SL = за инвалидацию идеи (не внутри новой POI); TP → ближайшие открытые пулы ликвидности/POI (EQH/EQL, PDH/PDL, OB/FVG/POC/VAH/VAL).\n"
    "8) Временная актуальность: оцени “validity_window” (UTC) — период, пока сетап статистически валиден (обычно до конца текущей/следующей сессии).\n\n"
    "ОГРАНИЧЕНИЯ\n"
    "- Если критичных данных не хватает — НЕ фантазируй: верни status=\"insufficient_data\" с перечнем missing_fields.\n"
    "- Числа и времена — абсолютные; время — UTC ISO-8601.\n"
    "- Вывод — строго JSON по схеме ниже. Никакого текста вне JSON.\n\n"
    "ФОРМАТ ОТВЕТА\n"
    "{\n"
    "  \"symbol\": \"{{SYMBOL}}\",\n"
    "  \"status\": \"ok\" | \"insufficient_data\",\n"
    "  \"missing_fields\": [],\n\n"
    "  \"bias\": {\"1D\":\"bull|bear|neutral\",\"4H\":\"...\",\"1H\":\"...\"},\n"
    "  \"poi\": {\n"
    "    \"type\":\"FVG|OB|Range|SR|Profile\",\n"
    "    \"tf\":\"15m|1h|4h\",\n"
    "    \"levels\":{\"top\":0.0,\"bot\":0.0,\"fvl\":0.0,\"range_hi\":0.0,\"range_lo\":0.0},\n"
    "    \"status\":\"open|fulfilled|inverted|mitigated\"\n"
    "  },\n"
    "  \"trigger\": {\n"
    "    \"tf\":\"15m|5m|1m\",\n"
    "    \"type\":\"CISD|BOS|MSS\",\n"
    "    \"delta\":\"buy>sell|sell>buy|mixed\",\n"
    "    \"cvd\":\"up|down|flat\",\n"
    "    \"vwap_filter\":\"above|below|on\"\n"
    "  },\n\n"
    "  \"trade\": {\n"
    "    \"side\":\"long|short\",\n"
    "    \"entry_type\":\"market\",\n"
    "    \"entry_price\":0.0,\n"
    "    \"sl\":0.0,\n"
    "    \"tp1\":0.0,\n"
    "    \"tp2\":0.0,\n"
    "    \"tp3\":0.0,\n"
    "    \"rr_min\":2.5,\n"
    "    \"validity_window\":{\"from_utc\":\"YYYY-MM-DDTHH:MM:SSZ\",\"to_utc\":\"YYYY-MM-DDTHH:MM:SSZ\"}\n"
    "  },\n\n"
    "  \"ideal_limits\":[0.0,0.0],\n"
    "  \"liquidity_targets\":[{\"type\":\"EQH|EQL|PDH|PDL|SessionH|SessionL\",\"price\":0.0}],\n"
    "  \"vwap_context\":{\"daily\":\"above|below|on\",\"session\":{\"asia\":\"above|below|on\",\"london\":\"...\",\"ny\":\"...\"}},\n"
    "  \"tpo_context\":{\"POC\":0.0,\"VAH\":0.0,\"VAL\":0.0},\n\n"
    "  \"risk\": {\n"
    "    \"event_risk_score\": 0,\n"
    "    \"structure_break_prob\": 0,\n"
    "    \"timing_risk\": \"low|mid|high\",\n"
    "    \"news\": [\n"
    "      {\"title\":\"...\", \"source\":\"...\", \"time_utc\":\"...\", \"impact\":\"low|mid|high\"}\n"
    "    ]\n"
    "  },\n\n"
    "  \"brief_logic\": \"≤60 слов: нарратив, POI, триггер, почему SL там, куда TP.\",\n"
    "  \"notes\": \"VWAP priority; SMT disabled; limit levels for reference only\"\n"
    "}\n"
)


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class TradeAnalysisContext:
    """Context for a trade analysis request."""

    symbol: str
    period: str
    last_price: float
    snapshot_payload: Mapping[str, Any]
    timestamp: datetime


@dataclass(slots=True)
class TradeAnalysisResult:
    """Result payload returned to API consumers."""

    status: str
    request_id: str | None
    trade_json: Mapping[str, Any] | None
    raw_text: str | None
    file_path: Path
    latency_ms: int
    attachment_size: int
    attachment_sha256: str


def _normalise_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9:_-]", "", (value or "").strip().upper()) or "UNKNOWN"


def _normalise_period(value: str) -> str:
    cleaned = re.sub(r"\s+", "_", (value or "").strip())
    return re.sub(r"[^A-Za-z0-9_.-]", "-", cleaned) or "custom"


def _format_timestamp_compact(moment: datetime) -> str:
    aligned = moment.astimezone(timezone.utc)
    return aligned.strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not (number == number and number not in {float("inf"), float("-inf")}):
        return default
    return number


def build_attachment_payload(context: TradeAnalysisContext) -> MutableMapping[str, Any]:
    """Construct the JSON payload stored in the attachment file."""

    return {
        "SYMBOL": context.symbol,
        "PERIOD": context.period,
        "LAST_PRICE": context.last_price,
        "DATA": context.snapshot_payload,
    }


def build_attachment_filename(context: TradeAnalysisContext) -> str:
    symbol_part = _normalise_symbol(context.symbol)
    period_part = _normalise_period(context.period)
    price_part = f"{context.last_price:.4f}".rstrip("0").rstrip(".") if context.last_price else "0"
    timestamp_part = _format_timestamp_compact(context.timestamp)
    filename = f"{symbol_part}_{period_part}_{price_part}_{timestamp_part}.json"
    return re.sub(r"__+", "_", filename)


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    json_payload: Mapping[str, Any] | None = None,
    files: Mapping[str, Any] | None = None,
    max_attempts: int = 3,
    backoff_initial: float = 0.75,
) -> httpx.Response:
    attempt = 0
    delay = backoff_initial
    while True:
        attempt += 1
        response = await client.post(
            url,
            headers=headers,
            data=data,
            json=json_payload,
            files=files,
        )
        if response.status_code not in RETRYABLE_STATUS or attempt >= max_attempts:
            return response
        await asyncio.sleep(delay)
        delay *= 2


def _extract_text_from_response(payload: Mapping[str, Any]) -> str | None:
    if not isinstance(payload, Mapping):
        return None

    if "output_text" in payload and isinstance(payload["output_text"], list):
        for item in payload["output_text"]:
            if isinstance(item, str) and item.strip():
                return item

    if "output" in payload and isinstance(payload["output"], list):
        for block in payload["output"]:
            contents = block.get("content") if isinstance(block, Mapping) else None
            if not isinstance(contents, list):
                continue
            for piece in contents:
                if isinstance(piece, Mapping):
                    text = piece.get("text") or piece.get("value") or piece.get("content")
                    if isinstance(text, str) and text.strip():
                        return text

    if "choices" in payload and isinstance(payload["choices"], list):
        for choice in payload["choices"]:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    for segment in content:
                        if isinstance(segment, Mapping):
                            text = segment.get("text") or segment.get("value")
                            if isinstance(text, str) and text.strip():
                                return text

    if "content" in payload and isinstance(payload["content"], str):
        text = payload["content"].strip()
        if text:
            return text

    return None


async def call_openai_with_attachment(
    *,
    api_key: str,
    model: str,
    file_path: Path,
    symbol: str,
    period: str,
    api_base: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> TradeAnalysisResult:
    """Upload the attachment file and request analysis from OpenAI."""

    base_url = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com")
    headers = {"Authorization": f"Bearer {api_key}"}

    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    start = time.perf_counter()
    try:
        with file_path.open("rb") as handle:
            files = {"file": (file_path.name, handle.read(), "application/json")}
        upload_response = await _post_with_retry(
            client,
            f"{base_url.rstrip('/')}/v1/files",
            headers=headers,
            data={"purpose": "assistants"},
            files=files,
        )
        upload_response.raise_for_status()
        upload_body = upload_response.json()
        file_id = upload_body.get("id")
        if not isinstance(file_id, str):
            raise RuntimeError("OpenAI file upload did not return an id")

        user_prompt = f"Analyze {symbol} for period {period}. DATA attached."
        response_payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_file", "file_id": file_id},
                    ],
                },
            ],
        }

        response = await _post_with_retry(
            client,
            f"{base_url.rstrip('/')}/v1/responses",
            headers={**headers, "Content-Type": "application/json"},
            json_payload=response_payload,
        )
        response.raise_for_status()
        body = response.json()
        text = _extract_text_from_response(body)

        trade_json: Mapping[str, Any] | None = None
        raw_text = text.strip() if isinstance(text, str) else None
        if raw_text:
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, Mapping):
                    trade_json = parsed
            except json.JSONDecodeError:
                trade_json = None

        latency_ms = int((time.perf_counter() - start) * 1000)
        size = file_path.stat().st_size
        digest = sha256(file_path.read_bytes()).hexdigest()

        status = "ok" if trade_json else "insufficient_data"
        request_id = body.get("id") if isinstance(body, Mapping) else None

        return TradeAnalysisResult(
            status=status,
            request_id=request_id,
            trade_json=trade_json,
            raw_text=raw_text,
            file_path=file_path,
            latency_ms=latency_ms,
            attachment_size=size,
            attachment_sha256=digest,
        )
    finally:
        if should_close:
            await client.aclose()


async def dispatch_trade_analysis(
    snapshot_payload: Mapping[str, Any],
    *,
    symbol: str,
    period: str,
    last_price: Any,
    upload_dir: Path,
    api_key: str,
    model: str,
    api_base: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> TradeAnalysisResult:
    """Persist the snapshot payload and dispatch the OpenAI analysis request."""

    upload_dir.mkdir(parents=True, exist_ok=True)

    context = TradeAnalysisContext(
        symbol=_normalise_symbol(symbol),
        period=_normalise_period(period),
        last_price=_safe_float(last_price, default=0.0),
        snapshot_payload=dict(snapshot_payload),
        timestamp=datetime.now(timezone.utc),
    )

    file_payload = build_attachment_payload(context)
    file_name = build_attachment_filename(context)
    file_path = upload_dir / file_name
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(file_payload, handle, ensure_ascii=False, separators=(",", ":"))

    result = await call_openai_with_attachment(
        api_key=api_key,
        model=model,
        file_path=file_path,
        symbol=context.symbol,
        period=context.period,
        api_base=api_base,
        client=client,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Trade analysis dispatched",
        extra={
            "symbol": context.symbol,
            "period": context.period,
            "last_price": context.last_price,
            "request_id": result.request_id,
            "latency_ms": result.latency_ms,
            "attachment_size": result.attachment_size,
            "attachment_sha256": result.attachment_sha256,
            "status": result.status,
        },
    )

    return result

