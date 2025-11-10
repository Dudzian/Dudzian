"""Wspólne funkcje mapujące błędy API giełd na wyjątki domenowe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeThrottlingError,
)

_HTTP_THROTTLE_STATUS_CODES = {
    408,
    418,
    429,
    444,
    499,
    502,
    503,
    504,
    520,
    521,
    522,
    523,
    524,
    525,
    526,
    527,
    528,
    529,
    530,
    598,
    599,
}


_NETWORK_THROTTLE_KEYWORDS = (
    "gateway timeout",
    "bad gateway",
    "temporarily unavailable",
    "temporarily down",
    "temporarily disabled",
    "temporarily locked",
    "temporarily overloaded",
    "server overloaded",
    "server busy",
    "origin is unreachable",
    "origin unreachable",
    "web server is down",
    "upstream timeout",
    "upstream unavailable",
    "connection timeout",
    "connection timed out",
    "connect timeout",
    "no route to host",
    "host unreachable",
    "network unreachable",
    "network is unreachable",
    "failed to resolve host",
    "temporary failure in name resolution",
    "name resolution",
    "dns lookup failed",
    "getaddrinfo",
    "read timeout",
    "read timed out",
    "write timeout",
    "write timed out",
    "operation timeout",
    "operation timed out",
    "max retries exceeded",
    "econnreset",
    "econnrefused",
    "econnaborted",
    "etimedout",
    "enetunreach",
    "enetdown",
    "ehostunreach",
    "eai_again",
    "broken pipe",
    "tls handshake",
    "ssl handshake",
    "handshake timeout",
    "handshake failed",
    "handshake failure",
    "ssl routines",
    "certificate verify failed",
    "remote end closed connection",
    "remote end closed the connection",
    "remote host closed connection",
    "connection closed without response",
    "server closed connection",
    "server hung up",
    "connection dropped",
    "connection lost",
    "client network socket disconnected", 
    "backend fetch failed",
    "client.timeout exceeded",
    "client timeout exceeded",
    "context deadline exceeded",
    "upstream request timeout",
    "upstream connect error",
    "upstream connection error",
    "upstream connection timeout",
    "upstream prematurely closed",
    "disconnect/reset before headers",
    "err_connection_reset",
    "err_connection_refused",
    "err_connection_aborted",
    "err_connection_closed",
    "err_connection_timed_out",
    "err_timed_out",
    "err_internet_disconnected",
    "err_network_changed",
    "err_failed",
)


_DERIBIT_AUTH_CODES = {
    13001,
    13002,
    13004,
    13005,
    13006,
    13007,
    13008,
    13009,
    13041,
    13042,
    13043,
    13076,
}

_DERIBIT_THROTTLE_CODES = {
    10028,
    10029,
    11046,
    13033,
    13034,
}

_DERIBIT_AUTH_KEYWORDS = (
    "auth",
    "sign",
    "permission",
    "privilege",
    "credential",
    "api key",
    "not allowed",
)

_DERIBIT_THROTTLE_KEYWORDS = (
    "too many",
    "rate limit",
    "burst limit",
    "busy",
    "temporarily",
    "capacity",
) + _NETWORK_THROTTLE_KEYWORDS

_BITMEX_AUTH_NAMES = {
    "unauthorized",
    "authenticationerror",
    "invalidapikey",
    "apikeydisabled",
    "expiredapikey",
    "forbidden",
}

_BITMEX_THROTTLE_NAMES = {
    "ratelimit",
    "ratelimiterror",
}

_BITMEX_THROTTLE_KEYWORDS = (
    "too many",
    "rate limit",
    "busy",
    "retry",
    "throttle",
) + _NETWORK_THROTTLE_KEYWORDS


@dataclass(slots=True)
class _BinanceErrorRule:
    codes: Sequence[int]
    exception: type[ExchangeAPIError]


_BINANCE_AUTH_CODES = (-2008, -2009, -2014, -2015, -1022, -1002)
_BINANCE_THROTTLE_CODES = (-1003, -1015, -1099, -1105)

_BINANCE_AUTH_KEYWORDS = (
    "invalid api-key",
    "invalid api key",
    "api-key format invalid",
    "signature for this request is not valid",
    "signature verification failed",
    "api key expired",
    "expired api key",
    "permission",
)

_BINANCE_THROTTLE_KEYWORDS = (
    "too many requests",
    "too many new orders",
    "ip banned",
    "ip ban",
    "request weight",
    "system busy",
    "system overloaded",
    "service unavailable",
    "exceeded",
    "try again later",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection closed",
    "network error",
) + _NETWORK_THROTTLE_KEYWORDS

_BINANCE_MESSAGE_KEYS = (
    "msg",
    "message",
    "error",
    "errmsg",
    "errorMessage",
    "errorMsg",
    "error_message",
    "description",
    "detail",
)

_BINANCE_CODE_KEYS = ("code", "errorCode", "errno", "errCode")

_BINANCE_RULES: Sequence[_BinanceErrorRule] = (
    _BinanceErrorRule(codes=_BINANCE_AUTH_CODES, exception=ExchangeAuthError),
    _BinanceErrorRule(codes=_BINANCE_THROTTLE_CODES, exception=ExchangeThrottlingError),
)


def _coerce_binance_message(payload: object, default_message: str) -> tuple[str, int | None]:
    """Wyciąga komunikat oraz kod błędu z odpowiedzi Binance."""

    message = default_message
    code: int | None = None

    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - dekodowanie z domyślnym komunikatem
            return default_message, None

    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith(('{', '[')):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                pass
            else:
                return _coerce_binance_message(parsed, default_message)
        return (payload or default_message, None)

    if isinstance(payload, Mapping):
        msg_candidates: list[str] = []
        nested_entries: list[object] = []
        for key in _BINANCE_MESSAGE_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value:
                msg_candidates.append(value)
            elif value is not None:
                nested_entries.append(value)
        if msg_candidates:
            message = msg_candidates[0]
        for key in _BINANCE_CODE_KEYS:
            raw_code = payload.get(key)
            if raw_code is None:
                continue
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                continue
            else:
                break
        if message != default_message or code is not None:
            return message, code

        nested_sources = (
            *nested_entries,
            payload.get("data"),
            payload.get("body"),
            payload.get("errors"),
            payload.get("meta"),
        )
        for entry in nested_sources:
            if entry is None:
                continue
            nested_message, nested_code = _coerce_binance_message(entry, default_message)
            if nested_message != default_message or nested_code is not None:
                return nested_message, nested_code
        return message, code

    if isinstance(payload, Sequence):
        for entry in payload:
            if isinstance(entry, Mapping):
                message, code = _coerce_binance_message(entry, default_message)
                if message != default_message or code is not None:
                    return message, code
            elif isinstance(entry, (str, bytes)):
                message, _ = _coerce_binance_message(entry, default_message)
                if message != default_message:
                    return message, None
        return default_message, None

    return default_message, None


def raise_for_binance_error(
    *,
    status_code: int,
    payload: object,
    default_message: str,
) -> None:
    """Podnosi odpowiedni wyjątek na podstawie kodu błędu Binance."""

    message, code = _coerce_binance_message(payload, default_message)
    if code is not None:
        for rule in _BINANCE_RULES:
            if code in rule.codes:
                raise rule.exception(message=message, status_code=status_code, payload=payload)
    normalized_message = message.lower()
    if code is None:
        if any(keyword in normalized_message for keyword in _BINANCE_AUTH_KEYWORDS):
            raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
        if any(keyword in normalized_message for keyword in _BINANCE_THROTTLE_KEYWORDS):
            raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)
    if status_code == 401 or status_code == 403:
        raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
    if status_code in _HTTP_THROTTLE_STATUS_CODES:
        raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)
    raise ExchangeAPIError(message=message, status_code=status_code, payload=payload)


_KRAKEN_AUTH_KEYWORDS = (
    "invalid key",
    "invalid api key",
    "invalid signature",
    "invalid nonce",
    "key not enabled",
    "key expired",
    "api key expired",
    "api key disabled",
    "missing api key",
    "permission",
    "not allowed",
    "authentication",
)

_KRAKEN_AUTH_PREFIXES = (
    "eapi:invalid key",
    "eapi:invalid signature",
    "eapi:invalid nonce",
    "eapi:key not enabled",
    "eapi:permission",
    "eservice:permission",
)

_KRAKEN_THROTTLE_KEYWORDS = (
    "rate limit",
    "too many",
    "exceeded",
    "temporarily unavailable",
    "busy",
    "lockout",
    "cooldown",
    "throttled",
    "slow down",
    "timeout",
    "timed out",
    "time-out",
    "try again later",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection closed",
    "network error",
) + _NETWORK_THROTTLE_KEYWORDS

_KRAKEN_THROTTLE_PREFIXES = (
    "eapi:rate limit",
    "eservice:rate limit",
    "eservice:temporarily",
    "eservice:busy",
    "equery:busy",
    "eorder:rate limit",
)


_GENERIC_MESSAGE_KEYS = (
    "message",
    "msg",
    "error",
    "errorMessage",
    "errmsg",
    "description",
    "detail",
    "reason",
    "shortMessage",
    "longMessage",
    "title",
    "info",
)

_GENERIC_NESTED_KEYS = ("errors", "details", "data", "body", "messages")


def _iter_error_messages(source: object) -> Iterator[str]:
    """Zwraca kolejne komunikaty błędów z odpowiedzi API."""

    if source is None:
        return
    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensywne
            return
    if isinstance(source, str):
        normalized = source.strip()
        if not normalized:
            return
        if normalized.startswith(('{', '[')):
            try:
                parsed = json.loads(normalized)
            except (TypeError, ValueError):
                pass
            else:
                yield from _iter_error_messages(parsed)
                return
        yield normalized
        return
    if isinstance(source, Mapping):
        for key in _GENERIC_MESSAGE_KEYS:
            value = source.get(key)
            yield from _iter_error_messages(value)
        for key in _GENERIC_NESTED_KEYS:
            nested = source.get(key)
            if nested is not None:
                yield from _iter_error_messages(nested)
        return
    if isinstance(source, Sequence):
        for entry in source:
            yield from _iter_error_messages(entry)
        return
    text = str(source).strip()
    if text:
        yield text


def _parse_int(value: object) -> int | None:
    """Próbuje sparsować wartość do liczby całkowitej."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def raise_for_kraken_error(
    *, payload: Mapping[str, object], default_message: str, status_code: int = 400
) -> None:
    """Analizuje listę błędów zwróconą przez Kraken."""

    errors = payload.get("error")
    messages = list(_iter_error_messages(errors))
    if not messages:
        for key in ("errorMessage", "error_message", "message", "description", "detail"):
            fallback = list(_iter_error_messages(payload.get(key)))
            if fallback:
                messages = fallback
                break
    message = " | ".join(messages) if messages else default_message
    normalized = message.lower()
    if messages:
        if normalized.startswith(tuple(_KRAKEN_AUTH_PREFIXES)) or any(
            keyword in normalized for keyword in _KRAKEN_AUTH_KEYWORDS
        ):
            raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
        if normalized.startswith(tuple(_KRAKEN_THROTTLE_PREFIXES)) or any(
            keyword in normalized for keyword in _KRAKEN_THROTTLE_KEYWORDS
        ):
            throttling_status = (
                status_code
                if status_code in _HTTP_THROTTLE_STATUS_CODES or status_code >= 500
                else 429
            )
            raise ExchangeThrottlingError(
                message=message, status_code=throttling_status, payload=payload
            )
    if status_code in {401, 403}:
        raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
    if status_code in _HTTP_THROTTLE_STATUS_CODES:
        raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)
    if messages or errors:
        raise ExchangeAPIError(message=message, status_code=status_code, payload=payload)


_ZONDA_AUTH_CODES = {4002, 4003, 4014, 4015}
_ZONDA_THROTTLE_CODES = {429, 5004}

_ZONDA_AUTH_KEYWORDS = (
    "invalid api key",
    "api key expired",
    "permission",
    "not allowed",
    "authentication",
)

_ZONDA_THROTTLE_KEYWORDS = (
    "limit",
    "too many",
    "temporarily unavailable",
    "busy",
    "overloaded",
    "slow down",
    "service unavailable",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection closed",
    "network error",
) + _NETWORK_THROTTLE_KEYWORDS


def raise_for_zonda_error(*, status_code: int, payload: Mapping[str, object], default_message: str) -> None:
    """Podnosi specyficzny wyjątek na podstawie odpowiedzi API Zonda."""

    message = default_message
    top_code = _parse_int(payload.get("code"))
    if top_code in _ZONDA_AUTH_CODES:
        top_message = payload.get("message")
        if isinstance(top_message, (str, bytes)):
            message = next(_iter_error_messages(top_message), message)
        raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
    if top_code in _ZONDA_THROTTLE_CODES:
        top_message = payload.get("message")
        if isinstance(top_message, (str, bytes)):
            message = next(_iter_error_messages(top_message), message)
        raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)

    top_level_messages: list[str] = []
    for key in ("message", "error", "errorMessage", "description", "detail"):
        top_level_messages.extend(_iter_error_messages(payload.get(key)))
    if top_level_messages:
        message = " | ".join(top_level_messages)

    errors = payload.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) and errors:
        parts = []
        for entry in errors:
            if isinstance(entry, Mapping):
                nested_message = entry.get("message")
                if isinstance(nested_message, (str, bytes)):
                    parts.extend(_iter_error_messages(nested_message))
                nested_code = _parse_int(entry.get("code"))
                if nested_code in _ZONDA_AUTH_CODES:
                    msg = entry.get("message") or message
                    raise ExchangeAuthError(message=str(msg), status_code=status_code, payload=payload)
                if nested_code in _ZONDA_THROTTLE_CODES:
                    msg = entry.get("message") or message
                    raise ExchangeThrottlingError(
                        message=str(msg), status_code=status_code, payload=payload
                    )
            else:
                parts.extend(_iter_error_messages(entry))
        if parts:
            message = " | ".join(parts)
    elif isinstance(errors, Mapping):
        # Niektóre odpowiedzi zwracają pojedynczy obiekt zamiast listy.
        nested_message = list(_iter_error_messages(errors))
        if nested_message:
            message = " | ".join(nested_message)
        nested_code = _parse_int(errors.get("code"))
        if nested_code in _ZONDA_AUTH_CODES:
            raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
        if nested_code in _ZONDA_THROTTLE_CODES:
            raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)
    elif isinstance(errors, (str, bytes)):
        message = next(_iter_error_messages(errors), message)
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in {"fail", "error"}:
        raise ExchangeAPIError(message=message, status_code=status_code, payload=payload)
    normalized_message = message.lower()
    if status_code in _HTTP_THROTTLE_STATUS_CODES or any(
        keyword in normalized_message for keyword in _ZONDA_THROTTLE_KEYWORDS
    ):
        raise ExchangeThrottlingError(message=message, status_code=status_code, payload=payload)
    if status_code in {401, 403}:
        raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
    if any(keyword in normalized_message for keyword in _ZONDA_AUTH_KEYWORDS):
        raise ExchangeAuthError(message=message, status_code=status_code, payload=payload)
    raise ExchangeAPIError(message=message, status_code=status_code, payload=payload)


def _coerce_payload_mapping(payload: object) -> Mapping[str, object] | None:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:  # pragma: no cover - fallback dekodowania
            return None
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith(('{', '[')):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return None
            if isinstance(parsed, Mapping):
                return parsed
    return None


def raise_for_deribit_error(*, status_code: int, payload: Mapping[str, object], default_message: str) -> None:
    """Mapuje odpowiedź Deribit na wyjątki domenowe."""

    message = default_message
    code = _parse_int(payload.get("code"))
    error_section = payload.get("error")
    if isinstance(error_section, Mapping):
        code = _parse_int(error_section.get("code")) or code
        message = str(error_section.get("message") or message)
        data = error_section.get("data")
        if isinstance(data, Mapping) and not code:
            code = _parse_int(data.get("error")) or _parse_int(data.get("code"))
        elif isinstance(data, (str, bytes)):
            try:
                extra = json.loads(data)
            except (TypeError, ValueError):  # pragma: no cover - dane bez JSON
                extra = None
            if isinstance(extra, Mapping) and not code:
                code = _parse_int(extra.get("code"))

    normalized = message.lower()
    if (
        status_code in {401, 403}
        or code in _DERIBIT_AUTH_CODES
        or any(keyword in normalized for keyword in _DERIBIT_AUTH_KEYWORDS)
    ):
        raise ExchangeAuthError(message=message, status_code=status_code or 401, payload=payload)

    if (
        status_code in {418, 429}
        or code in _DERIBIT_THROTTLE_CODES
        or any(keyword in normalized for keyword in _DERIBIT_THROTTLE_KEYWORDS)
    ):
        raise ExchangeThrottlingError(message=message, status_code=status_code or 429, payload=payload)

    raise ExchangeAPIError(message=message, status_code=status_code or 500, payload=payload)


def raise_for_bitmex_error(*, status_code: int, payload: Mapping[str, object], default_message: str) -> None:
    """Mapuje błędy BitMEX na wyjątki domenowe."""

    message = default_message
    code = _parse_int(payload.get("errorCode"))
    error_section = payload.get("error")
    name = None
    if isinstance(error_section, Mapping):
        name_value = error_section.get("name")
        if isinstance(name_value, str):
            name = name_value.strip().lower()
        message_value = error_section.get("message")
        if isinstance(message_value, (str, bytes)):
            message = next(_iter_error_messages(message_value), message)
        code = _parse_int(error_section.get("code")) or code

    normalized_message = message.lower()
    if (
        status_code in {401, 403}
        or name in _BITMEX_AUTH_NAMES
        or any(keyword in normalized_message for keyword in ("auth", "signature", "permission"))
    ):
        raise ExchangeAuthError(message=message, status_code=status_code or 401, payload=payload)

    if (
        status_code in {418, 429}
        or name in _BITMEX_THROTTLE_NAMES
        or code == 429
        or any(keyword in normalized_message for keyword in _BITMEX_THROTTLE_KEYWORDS)
    ):
        raise ExchangeThrottlingError(message=message, status_code=status_code or 429, payload=payload)

    raise ExchangeAPIError(message=message, status_code=status_code or 500, payload=payload)


__all__ = [
    "raise_for_binance_error",
    "raise_for_kraken_error",
    "raise_for_zonda_error",
    "raise_for_deribit_error",
    "raise_for_bitmex_error",
]
