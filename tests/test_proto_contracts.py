from __future__ import annotations

from pathlib import Path

import pytest

PROTO_FILE = Path("proto/trading.proto")


@pytest.mark.parametrize(
    "snippet",
    [
        "service MarketDataService",
        "rpc GetOhlcvHistory",
        "rpc StreamOhlcv",
        "service OrderService",
        "rpc SubmitOrder",
        "rpc CancelOrder",
        "service RiskService",
        "rpc StreamRiskState",
        "service MetricsService",
        "MetricsSnapshot",
        "message RiskState",
    ],
)
def test_trading_proto_contains_required_contracts(snippet: str) -> None:
    content = PROTO_FILE.read_text(encoding="utf-8")
    assert snippet in content, f"Expect snippet {snippet!r} in proto contract"


def test_trading_proto_declares_contract_version() -> None:
    content = PROTO_FILE.read_text(encoding="utf-8")
    assert "package botcore.trading.v1;" in content
    assert "syntax = \"proto3\";" in content


def test_trading_proto_documents_no_websocket_policy() -> None:
    readme = Path("proto/README.md").read_text(encoding="utf-8")
    assert "brak" in readme.lower() and "websocket" in readme.lower()
