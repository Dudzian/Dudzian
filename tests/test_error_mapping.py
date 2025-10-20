"""Testy regresyjne dla modułu mapowania błędów giełd."""

import pytest

from bot_core.exchanges.error_mapping import (
    raise_for_binance_error,
    raise_for_kraken_error,
    raise_for_zonda_error,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeThrottlingError,
)


class TestBinanceErrorMapping:
    """Scenariusze mapowania kodów Binance na wyjątki."""

    def test_auth_code_raises_exchange_auth_error(self) -> None:
        payload = {"code": -2015, "msg": "Invalid API-key format"}
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="auth")

    def test_http_429_raises_throttling_error(self) -> None:
        payload = {"code": -1003, "msg": "Too many requests"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=429, payload=payload, default_message="throttle")

    def test_http_418_is_treated_as_throttling(self) -> None:
        payload = {"msg": "IP banned"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=418, payload=payload, default_message="teapot")

    def test_http_503_is_treated_as_throttling(self) -> None:
        payload = {"msg": "System overloaded"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=503, payload=payload, default_message="overload")

    def test_http_521_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Web server is down"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=521, payload=payload, default_message="web down")

    def test_http_522_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Connection timed out"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=522, payload=payload, default_message="timeout")

    def test_http_520_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Web server returned an unknown error"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=520, payload=payload, default_message="unknown")

    def test_http_444_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Connection closed"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=444, payload=payload, default_message="closed")

    def test_http_499_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Client closed request"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=499, payload=payload, default_message="client closed")

    def test_http_523_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Origin is unreachable"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=523, payload=payload, default_message="origin")

    def test_http_527_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Railgun error"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=527, payload=payload, default_message="railgun")

    def test_http_528_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Site overloaded"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=528, payload=payload, default_message="overload")

    def test_http_529_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Site is overloaded"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=529, payload=payload, default_message="overload")

    def test_http_530_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Origin DNS error"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=530, payload=payload, default_message="dns")

    def test_http_598_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Network read timeout"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=598, payload=payload, default_message="read timeout")

    def test_http_599_is_treated_as_throttling(self) -> None:
        payload = {"msg": "Network connect timeout"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=599, payload=payload, default_message="connect timeout")

    def test_unknown_code_falls_back_to_api_error(self) -> None:
        payload = {"code": -3000, "msg": "Unknown error"}
        with pytest.raises(ExchangeAPIError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_string_payload_becomes_error_message(self) -> None:
        with pytest.raises(ExchangeAPIError) as exc:
            raise_for_binance_error(status_code=500, payload="maintenance", default_message="fallback")
        assert "maintenance" in str(exc.value)

    def test_bytes_payload_is_decoded(self) -> None:
        payload = "{\"msg\":\"Maintenance\"}".encode("utf-8")
        with pytest.raises(ExchangeAPIError) as exc:
            raise_for_binance_error(status_code=500, payload=payload, default_message="fallback")
        assert "Maintenance" in str(exc.value)

    def test_sequence_payload_extracts_first_mapping(self) -> None:
        payload = [
            {"code": -1015, "msg": "Too many new orders"},
            {"code": -2015, "msg": "Invalid"},
        ]
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_error_field_is_used_as_message(self) -> None:
        payload = {"error": "IP banned", "code": -1003}
        with pytest.raises(ExchangeThrottlingError) as exc:
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")
        assert "IP banned" in str(exc.value)

    def test_nested_error_mapping_is_resolved(self) -> None:
        payload = {"error": {"msg": "Invalid", "code": -2015}}
        with pytest.raises(ExchangeAuthError) as exc:
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")
        assert "Invalid" in str(exc.value)

    def test_data_container_with_sequence_is_handled(self) -> None:
        payload = {"data": [{"code": -1003, "msg": "Too many"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_message_field_with_nested_payload(self) -> None:
        payload = {"message": {"msg": "Timeout", "code": -1003}}
        with pytest.raises(ExchangeThrottlingError) as exc:
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")
        assert "Timeout" in str(exc.value)

    def test_textual_auth_message_without_code(self) -> None:
        payload = {"msg": "Signature for this request is not valid."}
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_textual_throttle_message_without_code(self) -> None:
        payload = {"msg": "Request weight exceeded"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_json_string_payload_is_parsed(self) -> None:
        payload = '{"code": -2015, "msg": "Signature verification failed"}'
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_bytes_json_payload_is_parsed(self) -> None:
        payload = b'{"code": -1003, "msg": "Too many requests"}'
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_error_message_variant_is_respected(self) -> None:
        payload = {"errorMessage": "API key expired"}
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_error_code_variants_are_normalized(self) -> None:
        payload = {"errorCode": "-1003", "errorMsg": "Try again later"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_errno_field_is_treated_as_code(self) -> None:
        payload = {"errno": -2015, "message": "Signature verification failed"}
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_try_again_later_keyword_triggers_throttle(self) -> None:
        payload = {"msg": "Please try again later"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_api_key_expired_keyword_triggers_auth(self) -> None:
        payload = {"message": "Your API key expired yesterday"}
        with pytest.raises(ExchangeAuthError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_timeout_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Request timeout, please retry"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_connection_reset_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Connection reset by peer"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Connection dropped unexpectedly",
            "Connection lost with upstream server",
            "Client network socket disconnected before secure TLS connection was established",
            "Backend fetch failed: origin error",
            "Client.Timeout exceeded while awaiting headers",
            "Context deadline exceeded during request",
            "Upstream connect error or disconnect/reset before headers",
            "Upstream request timeout from gateway",
        ],
    )
    def test_network_disruption_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"message": message}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Remote end closed connection without response",
            "Server hung up during request",
        ],
    )
    def test_connection_closed_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"message": message}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "net::ERR_CONNECTION_RESET",
            "ERR_CONNECTION_REFUSED",
            "ERR_CONNECTION_ABORTED",
            "ERR_CONNECTION_CLOSED",
            "ERR_CONNECTION_TIMED_OUT",
            "ERR_TIMED_OUT",
            "ERR_INTERNET_DISCONNECTED",
            "ERR_NETWORK_CHANGED",
            "ERR_FAILED",
        ],
    )
    def test_chromium_err_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"message": message}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_failed_to_resolve_host_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Failed to resolve host api.binance.com"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_gateway_timeout_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Gateway timeout from upstream"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_tls_handshake_keyword_triggers_throttle(self) -> None:
        payload = {"message": "TLS handshake timeout"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_certificate_verify_failed_keyword_triggers_throttle(self) -> None:
        payload = {"message": "certificate verify failed: unable to get local issuer certificate"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_read_timed_out_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Read timed out while contacting upstream"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    def test_max_retries_exceeded_keyword_triggers_throttle(self) -> None:
        payload = {"message": "Max retries exceeded with url: /api/v3/order"}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Request failed: [Errno 104] ECONNRESET",
            "Could not connect: [Errno 111] ECONNREFUSED",
            "Connection aborted: [Errno 113] EHOSTUNREACH",
            "Network unreachable: [Errno 101] ENETUNREACH",
            "Operation timed out: [Errno 60] ETIMEDOUT",
            "DNS lookup error: [Errno -3] EAI_AGAIN",
            "Stream write failed: [Errno 32] Broken pipe",
        ],
    )
    def test_errno_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"message": message}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_binance_error(status_code=400, payload=payload, default_message="fallback")


class TestKrakenErrorMapping:
    """Scenariusze mapowania błędów Kraken."""

    def test_invalid_key_is_treated_as_auth_error(self) -> None:
        payload = {"error": ["EAPI:Invalid key"]}
        with pytest.raises(ExchangeAuthError):
            raise_for_kraken_error(payload=payload, default_message="kraken auth")

    def test_rate_limit_triggers_throttling(self) -> None:
        payload = {"error": ["EAPI:Rate limit exceeded"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_empty_error_list_is_ignored(self) -> None:
        payload = {"error": []}
        # Nie powinien być rzucony żaden wyjątek dla pustej listy błędów.
        raise_for_kraken_error(payload=payload, default_message="noop")

    def test_nested_error_object_is_parsed(self) -> None:
        payload = {"error": [{"message": "EAPI:Invalid signature"}]}
        with pytest.raises(ExchangeAuthError):
            raise_for_kraken_error(payload=payload, default_message="kraken auth")

    def test_bytes_payload_is_decoded(self) -> None:
        payload = {"error": [b"EAPI:Rate limit exceeded"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_invalid_nonce_is_treated_as_auth_error(self) -> None:
        payload = {"error": ["EAPI:Invalid nonce"]}
        with pytest.raises(ExchangeAuthError):
            raise_for_kraken_error(payload=payload, default_message="kraken auth")

    def test_lockout_message_is_throttling(self) -> None:
        payload = {"error": ["EGeneral:Temporary lockout"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=503
            )

    def test_json_string_entry_is_decoded(self) -> None:
        payload = {"error": ['{"message": "EAPI:Rate limit exceeded"}']}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_connection_reset_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:Connection reset by peer"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    @pytest.mark.parametrize(
        "message",
        [
            "EGeneral:Remote end closed connection without response",
            "EGeneral:Server hung up during request",
        ],
    )
    def test_connection_closed_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"error": [message]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    @pytest.mark.parametrize(
        "message",
        [
            "EGeneral:Connection dropped unexpectedly",
            "EGeneral:Connection lost with upstream server",
            "EGeneral:Client network socket disconnected before secure TLS connection was established",
            "EGeneral:Backend fetch failed: origin error",
            "EGeneral:Client.Timeout exceeded while awaiting headers",
            "EGeneral:Context deadline exceeded while calling upstream",
            "EGeneral:Upstream connect error or disconnect/reset before headers",
            "EGeneral:Upstream request timeout from gateway",
        ],
    )
    def test_network_disruption_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"error": [message]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    @pytest.mark.parametrize(
        "message",
        [
            "EGeneral:net::ERR_CONNECTION_RESET",
            "EGeneral:ERR_CONNECTION_REFUSED",
            "EGeneral:ERR_CONNECTION_ABORTED",
            "EGeneral:ERR_CONNECTION_CLOSED",
            "EGeneral:ERR_CONNECTION_TIMED_OUT",
            "EGeneral:ERR_TIMED_OUT",
            "EGeneral:ERR_INTERNET_DISCONNECTED",
            "EGeneral:ERR_NETWORK_CHANGED",
            "EGeneral:ERR_FAILED",
        ],
    )
    def test_chromium_err_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"error": [message]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_gateway_timeout_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:Gateway timeout"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_tls_handshake_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:TLS handshake timeout"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_certificate_verify_failed_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:certificate verify failed"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_read_timed_out_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:Read timed out"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_max_retries_exceeded_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:Max retries exceeded while calling API"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    @pytest.mark.parametrize(
        "message",
        [
            "EGeneral:Request failed: [Errno 104] ECONNRESET",
            "EGeneral:Could not connect: [Errno 111] ECONNREFUSED",
            "EGeneral:Connection aborted: [Errno 113] EHOSTUNREACH",
            "EGeneral:Network unreachable: [Errno 101] ENETUNREACH",
            "EGeneral:Operation timed out: [Errno 60] ETIMEDOUT",
            "EGeneral:DNS lookup error: [Errno -3] EAI_AGAIN",
            "EGeneral:Stream write failed: [Errno 32] Broken pipe",
        ],
    )
    def test_errno_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"error": [message]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_name_resolution_failure_keyword_triggers_throttle(self) -> None:
        payload = {"error": ["EGeneral:Temporary failure in name resolution"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_error_message_without_list_is_used(self) -> None:
        payload = {"errorMessage": "API key expired"}
        with pytest.raises(ExchangeAuthError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken auth", status_code=403
            )

    def test_status_based_auth_without_messages(self) -> None:
        payload: dict[str, object] = {}
        with pytest.raises(ExchangeAuthError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken auth", status_code=401
            )

    def test_reason_field_is_parsed(self) -> None:
        payload = {"error": [{"reason": "EAPI:Rate limit exceeded"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(payload=payload, default_message="kraken throttle")

    def test_status_504_triggers_throttling_even_without_messages(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=504
            )

    def test_status_522_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=522
            )

    def test_status_520_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=520
            )

    def test_status_523_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=523
            )

    def test_status_444_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=444
            )

    def test_status_499_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=499
            )

    def test_status_525_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=525
            )

    def test_status_527_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=527
            )

    def test_status_528_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=528
            )

    def test_status_529_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=529
            )

    def test_status_530_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=530
            )

    def test_status_598_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=598
            )

    def test_status_599_without_message_triggers_throttle(self) -> None:
        payload = {"error": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_kraken_error(
                payload=payload, default_message="kraken throttle", status_code=599
            )


class TestZondaErrorMapping:
    """Mapowanie odpowiedzi błędów Zonda."""

    def test_error_code_raises_auth_error(self) -> None:
        payload = {"errors": [{"code": 4014, "message": "Invalid API key"}]}
        with pytest.raises(ExchangeAuthError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="zonda auth")

    def test_http_429_without_codes_raises_throttling(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=429, payload=payload, default_message="too many")

    def test_fail_status_raises_api_error(self) -> None:
        payload = {"status": "Fail", "errors": [{"message": "Generic"}]}
        with pytest.raises(ExchangeAPIError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="generic")

    def test_textual_auth_message_without_code(self) -> None:
        payload = {"errors": ["Permission denied"]}
        with pytest.raises(ExchangeAuthError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_textual_throttle_message_without_code(self) -> None:
        payload = {"errors": [{"message": "Limit temporarily unavailable"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_gateway_timeout_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "Gateway timeout from upstream"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_status_error_string_raises_api_error(self) -> None:
        payload = {"status": "ERROR", "errors": []}
        with pytest.raises(ExchangeAPIError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_detail_field_triggers_auth_error(self) -> None:
        payload = {"errors": {"details": ["Authentication failed"]}}
        with pytest.raises(ExchangeAuthError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_top_level_error_message_is_used(self) -> None:
        payload = {"error": {"detail": "Exchange overloaded"}}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=429, payload=payload, default_message="fallback")

    def test_top_level_code_is_respected(self) -> None:
        payload = {"code": "4015", "message": "Key not enabled"}
        with pytest.raises(ExchangeAuthError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_connection_reset_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "Connection reset by peer"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Remote end closed connection without response",
            "Server hung up during request",
        ],
    )
    def test_connection_closed_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"errors": [{"message": message}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Connection dropped unexpectedly",
            "Connection lost with upstream server",
            "Client network socket disconnected before secure TLS connection was established",
            "Backend fetch failed: origin error",
            "Client.Timeout exceeded while awaiting headers",
            "Context deadline exceeded during request",
            "Upstream connect error or disconnect/reset before headers",
            "Upstream request timeout from gateway",
        ],
    )
    def test_network_disruption_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"errors": [{"message": message}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "net::ERR_CONNECTION_RESET",
            "ERR_CONNECTION_REFUSED",
            "ERR_CONNECTION_ABORTED",
            "ERR_CONNECTION_CLOSED",
            "ERR_CONNECTION_TIMED_OUT",
            "ERR_TIMED_OUT",
            "ERR_INTERNET_DISCONNECTED",
            "ERR_NETWORK_CHANGED",
            "ERR_FAILED",
        ],
    )
    def test_chromium_err_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"errors": [{"message": message}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_tls_handshake_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "SSL handshake failed"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_certificate_verify_failed_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "certificate verify failed"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    @pytest.mark.parametrize(
        "message",
        [
            "Request failed: [Errno 104] ECONNRESET",
            "Could not connect: [Errno 111] ECONNREFUSED",
            "Connection aborted: [Errno 113] EHOSTUNREACH",
            "Network unreachable: [Errno 101] ENETUNREACH",
            "Operation timed out: [Errno 60] ETIMEDOUT",
            "DNS lookup error: [Errno -3] EAI_AGAIN",
            "Stream write failed: [Errno 32] Broken pipe",
        ],
    )
    def test_errno_keywords_trigger_throttle(self, message: str) -> None:
        payload = {"errors": [{"message": message}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_no_route_to_host_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "No route to host"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_string_code_inside_errors(self) -> None:
        payload = {"errors": [{"code": "5004", "message": "Limit reached"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_mapping_errors_with_nested_code(self) -> None:
        payload = {"errors": {"code": "5004", "detail": "Too many requests"}}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_stringified_json_error_entry(self) -> None:
        payload = {"errors": ['{"code": "4014", "message": "Invalid API key"}']}
        with pytest.raises(ExchangeAuthError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_http_503_triggers_throttling_even_without_errors(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=503, payload=payload, default_message="fallback")

    def test_status_522_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=522, payload=payload, default_message="fallback")

    def test_status_520_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=520, payload=payload, default_message="fallback")

    def test_status_444_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=444, payload=payload, default_message="fallback")

    def test_status_499_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=499, payload=payload, default_message="fallback")

    def test_status_523_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=523, payload=payload, default_message="fallback")

    def test_status_526_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=526, payload=payload, default_message="fallback")

    def test_status_527_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=527, payload=payload, default_message="fallback")

    def test_status_528_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=528, payload=payload, default_message="fallback")

    def test_status_529_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=529, payload=payload, default_message="fallback")

    def test_status_530_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=530, payload=payload, default_message="fallback")

    def test_status_598_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=598, payload=payload, default_message="fallback")

    def test_status_599_without_code_triggers_throttle(self) -> None:
        payload = {"errors": []}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=599, payload=payload, default_message="fallback")

    def test_timeout_keyword_triggers_throttling(self) -> None:
        payload = {"errors": ["Request timeout exceeded"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_read_timed_out_keyword_triggers_throttle(self) -> None:
        payload = {"errors": [{"message": "Read timed out during request"}]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")

    def test_max_retries_exceeded_keyword_triggers_throttle(self) -> None:
        payload = {"errors": ["Max retries exceeded contacting exchange"]}
        with pytest.raises(ExchangeThrottlingError):
            raise_for_zonda_error(status_code=400, payload=payload, default_message="fallback")
