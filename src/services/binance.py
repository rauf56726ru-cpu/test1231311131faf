"""Helpers for interacting with Binance USDT-margined futures (FAPI) endpoints.

This module replaces the legacy python-binance client with wrappers built on top
of the official `binance-connector` SDK that is vendored in the repository under
``binance-connector-python-master``.  The public surface of the module stays
compatible with the previous implementation so the rest of the backend can rely
on the same helpers and exception hierarchy.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse, urlsplit

import requests

# Make sure the auto-generated connector packages are importable.
_CONNECTOR_ROOT = Path(__file__).resolve().parents[2] / "binance-connector-python-master"
_CONNECTOR_PATHS = (
    _CONNECTOR_ROOT / "common" / "src",
    _CONNECTOR_ROOT / "clients" / "derivatives_trading_usds_futures" / "src",
)
for _path in _CONNECTOR_PATHS:
    if _path.is_dir():
        path_str = str(_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

try:  # pragma: no cover - import guarded for environments without connector
    from binance_common.configuration import ConfigurationRestAPI
    from binance_common.errors import (
        BadRequestError,
        ClientError,
        Error as ConnectorError,
        ForbiddenError,
        NetworkError,
        NotFoundError,
        RateLimitBanError,
        ServerError,
        TooManyRequestsError,
        UnauthorizedError,
    )
    from binance_common.models import ApiResponse
    from binance_common.utils import send_request
except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing in environment
    raise RuntimeError(
        "binance-connector SDK is required. Ensure the repository submodule "
        "'binance-connector-python-master' is present or install the relevant "
        "packages from PyPI."
    ) from exc

__all__ = [
    "BINANCE_FAPI_BASE_URL",
    "BINANCE_FAPI_REST",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_TIME_RANGE_MS",
    "BinanceAPIException",
    "BinanceRequestException",
    "BinanceRateLimitBudgetExceeded",
    "fetch_um_klines",
    "fetch_um_mark_price_klines",
    "fetch_um_index_price_klines",
    "fetch_um_premium_index_klines",
    "fetch_um_agg_trades",
    "fetch_um_order_book",
    "fetch_um_klines_sync",
    "fetch_fapi_endpoint",
    "reset_shared_client",
]

# Retain legacy constants for compatibility with existing modules/tests.
BINANCE_FAPI_BASE_URL = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com/fapi/v1")
BINANCE_FAPI_REST = f"{BINANCE_FAPI_BASE_URL}/klines"

DEFAULT_LIMIT = 500
MAX_LIMIT = 1_000
MAX_TIME_RANGE_MS = 7 * 24 * 60 * 60 * 1_000

_REST_CONFIG: ConfigurationRestAPI | None = None
_REST_SESSION: requests.Session | None = None
_REST_CLIENT_LOCK = asyncio.Lock()
_REST_CALL_LOCK = asyncio.Lock()
_REST_CONCURRENCY = int(os.getenv("BINANCE_MAX_CONCURRENCY", "4"))
_REST_THROTTLE = asyncio.Semaphore(max(1, _REST_CONCURRENCY))
_RETRY_STATUS = {418, 429}
_RETRY_BACKOFF_INITIAL = 0.5
_RETRY_BACKOFF_MAX = 5.0
_RETRY_MAX_ATTEMPTS = 5
LOGGER = logging.getLogger(__name__)


def _resolve_rest_caller() -> str:
    """Return the first stack frame outside this module for logging context."""

    try:
        stack = inspect.stack()
    except RuntimeError:
        return "unknown"
    for frame_info in stack[2:]:
        module = frame_info.frame.f_globals.get("__name__", "")
        if not module.startswith(__name__):
            filename = Path(frame_info.filename).name
            return f"{module or filename}:{frame_info.lineno}:{frame_info.function}"
    return "unknown"


def _summarise_rest_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Trim REST parameters to the fields that help identify the window."""

    allowed_keys = (
        "symbol",
        "interval",
        "startTime",
        "endTime",
        "limit",
        "fromId",
        "toId",
        "page",
    )
    summary: Dict[str, Any] = {}
    for key in allowed_keys:
        if key in params:
            summary[key] = params[key]
    return summary


REST_WEIGHT_LIMIT = max(1, int(os.getenv("BINANCE_WEIGHT_LIMIT", "2400")))
REST_WEIGHT_MARGIN = max(0, int(os.getenv("BINANCE_WEIGHT_MARGIN", "10")))
REST_WEIGHT_WINDOW = float(os.getenv("BINANCE_WEIGHT_WINDOW", "60"))
REST_WEIGHT_WAIT_THRESHOLD = float(os.getenv("BINANCE_WEIGHT_WAIT_THRESHOLD", "0.5"))
_REST_WEIGHT_DEFAULT = max(1, int(os.getenv("BINANCE_WEIGHT_DEFAULT", "1")))
_REST_WEIGHT_MAP: Dict[str, int] = {
    "/fapi/v1/aggTrades": int(os.getenv("BINANCE_WEIGHT_AGGTRADES", "20")),
    "/fapi/v1/klines": int(os.getenv("BINANCE_WEIGHT_KLINES", "2")),
    "/fapi/v1/trades": int(os.getenv("BINANCE_WEIGHT_TRADES", "2")),
    "/fapi/v1/historicalTrades": int(os.getenv("BINANCE_WEIGHT_HIST_TRADES", "5")),
    "/fapi/v1/exchangeInfo": int(os.getenv("BINANCE_WEIGHT_EXCHANGE_INFO", "10")),
}
_RAISE_ON_LIMIT_CALLERS = tuple(
    filter(
        None,
        (part.strip() for part in os.getenv("BINANCE_WEIGHT_RAISE_CALLERS", "session_collector").split(",")),
    )
)


class BinanceRequestException(Exception):
    """Compatibility wrapper for connector request failures."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        request_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.status_code = status_code
        self.request: Dict[str, Any] = dict(request_params or {})
        self.message = message or "Binance request error"
        super().__init__(self.message)


class BinanceAPIException(BinanceRequestException):
    """Compatibility wrapper for API failures reported by Binance."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        request_params: Optional[Mapping[str, Any]] = None,
        response: Any = None,
        code: Optional[int] = None,
    ) -> None:
        super().__init__(
            message or "Binance API error",
            status_code=status_code,
            request_params=request_params,
        )
        self.code = code
        self.response = response


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive for misconfiguration
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - defensive for misconfiguration
        return default


def _flag_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _derive_base_path(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Binance base URL: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _parse_proxy_config(proxy_url: str | None) -> Optional[Dict[str, Any]]:
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return None
    proxy: Dict[str, Any] = {
        "protocol": parsed.scheme,
        "host": parsed.hostname,
    }
    if parsed.port:
        proxy["port"] = parsed.port
    if parsed.username or parsed.password:
        proxy["auth"] = {
            "username": parsed.username or "",
            "password": parsed.password or "",
        }
    return proxy


def _custom_headers() -> Dict[str, str]:
    raw_headers = os.getenv("BINANCE_REST_HEADERS")
    if not raw_headers:
        return {}
    try:
        decoded = json.loads(raw_headers)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}
    return {
        str(key): str(value)
        for key, value in decoded.items()
        if isinstance(key, str) and value is not None
    }


class BinanceRateLimitBudgetExceeded(BinanceRequestException):
    """Raised when the shared REST weight budget cannot accommodate a request."""

    def __init__(self, *, path: str, weight: int, retry_after: float, caller: str) -> None:
        message = (
            f"REST weight budget exhausted for {path} (weight={weight}); "
            f"retry after {max(retry_after, 0.0):.2f}s"
        )
        super().__init__(message=message, status_code=429, request_params={"path": path, "weight": weight})
        self.path = path
        self.weight = weight
        self.retry_after = max(retry_after, 0.0)
        self.caller = caller


class _RestWeightLimiter:
    """Sliding-window limiter that tracks aggregate request weights."""

    def __init__(
        self,
        *,
        limit: int,
        margin: int,
        window: float,
        wait_threshold: float,
        raise_callers: tuple[str, ...],
    ) -> None:
        self.limit = max(limit, 1)
        self.margin = max(margin, 0)
        self.window = max(window, 0.1)
        self.wait_threshold = max(wait_threshold, 0.0)
        self.raise_callers = raise_callers
        self._events: deque[tuple[float, int]] = deque()
        self._weight: int = 0
        self._lock = asyncio.Lock()

    def _trim(self, now: float) -> None:
        window = self.window
        while self._events and now - self._events[0][0] >= window:
            _, weight = self._events.popleft()
            self._weight = max(0, self._weight - weight)

    def _budget(self) -> int:
        effective_limit = max(self.limit - self.margin, 0)
        return effective_limit

    async def acquire(self, path: str, caller: str, weight: int) -> None:
        request_weight = max(weight, 1)
        budget = self._budget()
        if budget == 0:
            raise BinanceRateLimitBudgetExceeded(path=path, weight=request_weight, retry_after=self.window, caller=caller)
        request_weight = min(request_weight, budget)

        while True:
            async with self._lock:
                now = time.monotonic()
                self._trim(now)
                available = budget - self._weight
                if request_weight <= max(available, 0):
                    self._events.append((now, request_weight))
                    self._weight += request_weight
                    return

                wait_for = self.window
                if self._events:
                    wait_for = (self._events[0][0] + self.window) - now
                wait_for = max(wait_for, 0.0)

            if wait_for <= self.wait_threshold:
                await asyncio.sleep(wait_for)
                continue

            if caller not in self.raise_callers:
                await asyncio.sleep(wait_for)
                continue

            raise BinanceRateLimitBudgetExceeded(
                path=path,
                weight=request_weight,
                retry_after=wait_for,
                caller=caller,
            )


_REST_WEIGHT_LIMITER = _RestWeightLimiter(
    limit=REST_WEIGHT_LIMIT,
    margin=REST_WEIGHT_MARGIN,
    window=REST_WEIGHT_WINDOW,
    wait_threshold=REST_WEIGHT_WAIT_THRESHOLD,
    raise_callers=_RAISE_ON_LIMIT_CALLERS,
)


def _create_rest_client() -> tuple[ConfigurationRestAPI, requests.Session]:
    base_path = _derive_base_path(os.getenv("BINANCE_FAPI_REST_BASE", BINANCE_FAPI_BASE_URL))
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    timeout_ms = int(_env_float("BINANCE_HTTP_TIMEOUT", 10.0) * 1_000)
    proxy = _parse_proxy_config(os.getenv("BINANCE_HTTPS_PROXY"))
    retries = _env_int("BINANCE_HTTP_RETRIES", 3)
    backoff_ms = _env_int("BINANCE_HTTP_BACKOFF", 1_000)
    keep_alive = not _flag_env("BINANCE_HTTP_DISABLE_KEEPALIVE", False)
    config = ConfigurationRestAPI(
        api_key=api_key,
        api_secret=api_secret,
        base_path=base_path,
        timeout=timeout_ms,
        proxy=proxy,
        keep_alive=keep_alive,
        compression=True,
        retries=max(retries, 0),
        backoff=max(backoff_ms, 0),
        time_unit=os.getenv("BINANCE_TIME_UNIT"),
        custom_headers=_custom_headers(),
    )
    session = requests.Session()
    return config, session


async def _get_rest_client() -> tuple[ConfigurationRestAPI, requests.Session]:
    global _REST_CONFIG, _REST_SESSION
    async with _REST_CLIENT_LOCK:
        if _REST_CONFIG is None or _REST_SESSION is None:
            _REST_CONFIG, _REST_SESSION = _create_rest_client()
        return _REST_CONFIG, _REST_SESSION


async def reset_shared_client() -> None:
    """Dispose of the cached REST client (useful for tests)."""

    global _REST_CONFIG, _REST_SESSION
    async with _REST_CLIENT_LOCK:
        if _REST_SESSION is not None:
            try:
                _REST_SESSION.close()
            except Exception:  # pragma: no cover - defensive
                pass
        _REST_CONFIG = None
        _REST_SESSION = None


def _normalise_symbol(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if not clean:
        raise ValueError("symbol cannot be empty")
    return clean


def _normalise_interval(interval: Optional[str]) -> Optional[str]:
    if interval is None:
        return None
    cleaned = interval.strip()
    if not cleaned:
        raise ValueError("interval cannot be empty")
    return cleaned


def _validate_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be positive")
    return min(limit, MAX_LIMIT)


def _as_int(name: str, value: Optional[int]) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_time_range(start_ms: Optional[int], end_ms: Optional[int]) -> tuple[int, int]:
    start = _as_int("startTime", start_ms)
    end = _as_int("endTime", end_ms)
    if start >= end:
        raise ValueError("startTime must be less than endTime")
    if end - start > MAX_TIME_RANGE_MS:
        raise ValueError("time range cannot exceed 7 days")
    return start, end


def _normalise_from_id(from_id: Optional[int]) -> Optional[int]:
    if from_id is None:
        return None
    if isinstance(from_id, bool) or not isinstance(from_id, int):
        raise TypeError("fromId must be an integer")
    if from_id < 0:
        raise ValueError("fromId must be non-negative")
    return from_id


_API_STATUS_MAP = {
    BadRequestError: 400,
    UnauthorizedError: 401,
    ForbiddenError: 403,
    NotFoundError: 404,
    RateLimitBanError: 418,
    TooManyRequestsError: 429,
}


def _to_public_exception(exc: ConnectorError, params: Mapping[str, Any]) -> Exception:
    message = getattr(exc, "error_message", str(exc))
    status = getattr(exc, "status_code", None)

    if isinstance(exc, ServerError):
        return BinanceAPIException(
            message=message,
            status_code=status or 500,
            request_params=params,
        )
    mapped_status = _API_STATUS_MAP.get(type(exc))
    if mapped_status is not None:
        return BinanceAPIException(
            message=message,
            status_code=status or mapped_status,
            request_params=params,
        )
    if isinstance(exc, (NetworkError, ClientError)):
        return BinanceRequestException(
            message=message,
            status_code=status,
            request_params=params,
        )
    return BinanceRequestException(
        message=message,
        status_code=status,
        request_params=params,
    )


async def _rest_get(path: str, params: Mapping[str, Any]) -> Any:
    """Execute a GET request against the shared REST client."""

    config, session = await _get_rest_client()
    request_params = dict(params)
    caller_context = _resolve_rest_caller()
    params_snapshot = _summarise_rest_params(request_params)
    weight = _REST_WEIGHT_MAP.get(path, _REST_WEIGHT_DEFAULT)
    try:
        await _REST_WEIGHT_LIMITER.acquire(path, caller_context, weight)
    except BinanceRateLimitBudgetExceeded as exc:
        LOGGER.warning(
            "binance.rest.defer",
            extra={
                "path": exc.path,
                "caller": caller_context,
                "retry_after": round(exc.retry_after, 3),
                "weight": exc.weight,
            },
        )
        raise

    LOGGER.info(
        "binance.rest.call",
        extra={
            "path": path,
            "caller": caller_context,
            "params": params_snapshot,
            "weight": weight,
        },
    )

    attempt = 0
    backoff = _RETRY_BACKOFF_INITIAL

    while True:
        attempt += 1
        async with _REST_THROTTLE:
            async with _REST_CALL_LOCK:
                try:
                    def _perform() -> Any:
                        response: ApiResponse[Any] = send_request(
                            session,
                            config,
                            method="GET",
                            path=path,
                            payload=request_params,
                            time_unit=config.time_unit,
                            response_model=None,
                        )
                        return response.data()

                    return await asyncio.to_thread(_perform)
                except ConnectorError as exc:
                    public_exc = _to_public_exception(exc, request_params)
                    status = getattr(public_exc, "status_code", None)
                    if attempt < _RETRY_MAX_ATTEMPTS and status in _RETRY_STATUS:
                        retry_after = _extract_retry_after(public_exc)
                        delay = retry_after or backoff
                        LOGGER.warning(
                            "binance.rest.retry",
                            extra={
                                "path": path,
                                "caller": caller_context,
                                "attempt": attempt,
                                "status": status,
                                "delay": round(delay, 3),
                            },
                        )
                        await asyncio.sleep(delay)
                        backoff = min(backoff * 2, _RETRY_BACKOFF_MAX)
                        continue
                    raise public_exc


def _extract_retry_after(exc: Exception) -> float | None:
    message = getattr(exc, "message", "") or ""
    if not message:
        return None
    digits: list[str] = []
    for token in message.split():
        if token.isdigit():
            digits.append(token)
    if not digits:
        return None
    try:
        timestamp_ms = int(digits[-1])
    except ValueError:
        return None
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    delta_ms = timestamp_ms - now_ms
    if delta_ms <= 0:
        return None
    return min(delta_ms / 1000.0, _RETRY_BACKOFF_MAX)


async def fetch_um_klines(
    symbol: str,
    interval: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Sequence[Any]]:
    """Fetch standard klines from UM futures."""

    symbol_clean = _normalise_symbol(symbol)
    interval_clean = _normalise_interval(interval) or interval
    params: Dict[str, Any] = {"symbol": symbol_clean, "interval": interval_clean}
    if limit is not None:
        params["limit"] = _validate_limit(limit)
    if start_time is not None:
        params["startTime"] = _as_int("startTime", start_time)
    if end_time is not None:
        params["endTime"] = _as_int("endTime", end_time)

    payload = await _rest_get("/fapi/v1/klines", params)
    if not isinstance(payload, list):
        raise TypeError("Unexpected klines payload")
    return payload  # type: ignore[return-value]


async def fetch_um_mark_price_klines(
    symbol: str,
    interval: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Sequence[Any]]:
    """Fetch mark-price klines."""

    symbol_clean = _normalise_symbol(symbol)
    interval_clean = _normalise_interval(interval) or interval
    params: Dict[str, Any] = {"symbol": symbol_clean, "interval": interval_clean}
    if limit is not None:
        params["limit"] = _validate_limit(limit)
    if start_time is not None:
        params["startTime"] = _as_int("startTime", start_time)
    if end_time is not None:
        params["endTime"] = _as_int("endTime", end_time)

    payload = await _rest_get("/fapi/v1/markPriceKlines", params)
    if not isinstance(payload, list):
        raise TypeError("Unexpected mark-price klines payload")
    return payload  # type: ignore[return-value]


async def fetch_um_index_price_klines(
    symbol: str,
    interval: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Sequence[Any]]:
    """Fetch index-price klines."""

    symbol_clean = _normalise_symbol(symbol)
    interval_clean = _normalise_interval(interval) or interval
    params: Dict[str, Any] = {"symbol": symbol_clean, "interval": interval_clean}
    if limit is not None:
        params["limit"] = _validate_limit(limit)
    if start_time is not None:
        params["startTime"] = _as_int("startTime", start_time)
    if end_time is not None:
        params["endTime"] = _as_int("endTime", end_time)

    payload = await _rest_get("/fapi/v1/indexPriceKlines", params)
    if not isinstance(payload, list):
        raise TypeError("Unexpected index-price klines payload")
    return payload  # type: ignore[return-value]


async def fetch_um_premium_index_klines(
    symbol: str,
    interval: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Sequence[Any]]:
    """Fetch premium-index klines."""

    symbol_clean = _normalise_symbol(symbol)
    interval_clean = _normalise_interval(interval) or interval
    params: Dict[str, Any] = {"symbol": symbol_clean, "interval": interval_clean}
    if limit is not None:
        params["limit"] = _validate_limit(limit)
    if start_time is not None:
        params["startTime"] = _as_int("startTime", start_time)
    if end_time is not None:
        params["endTime"] = _as_int("endTime", end_time)

    payload = await _rest_get("/fapi/v1/premiumIndexKlines", params)
    if not isinstance(payload, list):
        raise TypeError("Unexpected premium-index klines payload")
    return payload  # type: ignore[return-value]


async def fetch_um_agg_trades(
    symbol: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
    from_id: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Fetch aggregated trades from UM futures."""

    symbol_clean = _normalise_symbol(symbol)
    params: Dict[str, Any] = {"symbol": symbol_clean}
    if start_time is not None:
        params["startTime"] = _as_int("startTime", start_time)
    if end_time is not None:
        params["endTime"] = _as_int("endTime", end_time)
    if limit is not None:
        params["limit"] = _validate_limit(limit)
    if from_id is not None:
        params["fromId"] = _normalise_from_id(from_id)

    payload = await _rest_get("/fapi/v1/aggTrades", params)
    if not isinstance(payload, list):
        raise TypeError("Unexpected aggTrades payload")
    return payload  # type: ignore[return-value]


async def fetch_um_order_book(symbol: str, *, limit: int = 100) -> Dict[str, Any]:
    """Fetch an order-book snapshot from UM futures."""

    symbol_clean = _normalise_symbol(symbol)
    params: Dict[str, Any] = {
        "symbol": symbol_clean,
        "limit": int(min(max(limit, 5), 5_000)),
    }
    payload = await _rest_get("/fapi/v1/depth", params)
    if not isinstance(payload, Mapping):
        raise TypeError("Unexpected order-book payload")
    return dict(payload)


def _run_coroutine_sync(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Execute an async factory in sync context while handling running loops."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    result_box: List[Any] = []

    def _runner() -> None:
        try:
            result = asyncio.run(factory())
        except BaseException as exc:  # pragma: no cover - re-raise in caller
            result_box.append(exc)
        else:
            result_box.append(result)

    thread = threading.Thread(target=_runner, name="binance-sync-runner", daemon=True)
    thread.start()
    thread.join()

    if not result_box:
        raise RuntimeError("Coroutine did not produce a result")
    outcome = result_box[0]
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


def fetch_um_klines_sync(
    symbol: str,
    interval: str,
    *,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Sequence[Any]]:
    """Synchronous wrapper around ``fetch_um_klines`` for legacy call sites."""

    return _run_coroutine_sync(
        lambda: fetch_um_klines(
            symbol,
            interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    )


async def fetch_fapi_endpoint(
    endpoint: str,
    *,
    symbol: str,
    startTime: int,
    endTime: int,
    interval: Optional[str] = None,
    limit: Optional[int] = None,
    fromId: Optional[int] = None,
    **_: Any,
) -> Any:
    """Compatibility helper bridging existing call sites to the shared client."""

    endpoint_clean = (endpoint or "").strip()
    if not endpoint_clean:
        raise ValueError("endpoint cannot be empty")

    start_ms, end_ms = _validate_time_range(startTime, endTime)
    symbol_clean = _normalise_symbol(symbol)
    interval_clean = _normalise_interval(interval) if interval is not None else None
    limit_value = _validate_limit(limit)

    if endpoint_clean == "klines":
        return await fetch_um_klines(
            symbol_clean,
            interval_clean or "1m",
            start_time=start_ms,
            end_time=end_ms,
            limit=limit_value,
        )
    if endpoint_clean == "aggTrades":
        return await fetch_um_agg_trades(
            symbol_clean,
            start_time=start_ms,
            end_time=end_ms,
            limit=limit_value,
            from_id=fromId,
        )
    if endpoint_clean == "markPriceKlines":
        return await fetch_um_mark_price_klines(
            symbol_clean,
            interval_clean or "1m",
            start_time=start_ms,
            end_time=end_ms,
            limit=limit_value,
        )
    if endpoint_clean == "indexPriceKlines":
        return await fetch_um_index_price_klines(
            symbol_clean,
            interval_clean or "1m",
            start_time=start_ms,
            end_time=end_ms,
            limit=limit_value,
        )
    if endpoint_clean == "premiumIndexKlines":
        return await fetch_um_premium_index_klines(
            symbol_clean,
            interval_clean or "1m",
            start_time=start_ms,
            end_time=end_ms,
            limit=limit_value,
        )

    raise NotImplementedError(f"Unsupported endpoint: {endpoint_clean}")
