import pytest

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.base import ExecutionContext
from bot_core.execution.execution_service import build_live_execution_service
from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.exchanges.base import (
    ExchangeAdapter,
    ExchangeCredentials,
    Environment as ExchangeEnvironment,
    OrderRequest,
    OrderResult,
)
from bot_core.portfolio.payouts import require_hardware_wallet_metadata
from bot_core.security.hardware_wallets import LedgerSigner
from bot_core.security.signing import TransactionSignerSelector


class DummyAdapter(ExchangeAdapter):
    name = "dummy"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="dummy"))
        self.last_request: OrderRequest | None = None

    def configure_network(self, *, ip_allowlist=None) -> None:  # type: ignore[override]
        return None

    def fetch_account_snapshot(self):  # type: ignore[override]
        raise NotImplementedError

    def fetch_symbols(self):  # type: ignore[override]
        raise NotImplementedError

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # type: ignore[override]
        raise NotImplementedError

    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
        self.last_request = request
        return OrderResult(
            order_id="withdrawal-1",
            status="accepted",
            filled_quantity=request.quantity,
            avg_price=None,
            raw_response={},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # type: ignore[override]
        return None

    def stream_public_data(self, *, channels):  # type: ignore[override]
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # type: ignore[override]
        raise NotImplementedError


@pytest.fixture(autouse=True)
def enable_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_HW_SIMULATOR", "1")


def test_withdrawal_requires_hardware_signature() -> None:
    adapter = DummyAdapter()
    signer = LedgerSigner(seed=b"integration-test", key_id="ledger")
    selector = TransactionSignerSelector(default=signer)
    router = LiveExecutionRouter(
        adapters={"primary": adapter},
        default_route=("primary",),
        transaction_signers=selector,
        require_hardware_wallet_for_withdrawals=True,
    )

    metadata = require_hardware_wallet_metadata({}, account_id="primary", operation="withdrawal")
    request = OrderRequest(
        symbol="WITHDRAWAL",
        side="withdraw",
        quantity=1.0,
        order_type="market",
        metadata=metadata,
    )
    context = ExecutionContext(
        portfolio_id="portfolio-1",
        risk_profile="default",
        environment="live",
        metadata={"account": "primary"},
    )

    result = router.execute(request, context)
    assert result.status == "accepted"
    assert adapter.last_request is not None
    signed_metadata = dict(adapter.last_request.metadata or {})
    signature_doc = signed_metadata["hardware_wallet_signature"]
    assert signed_metadata["requires_hardware_wallet"] is True
    assert signed_metadata["operation"] == "withdrawal"
    assert signer.verify(
        {
            "exchange": "primary",
            "account": "primary",
            "portfolio": "portfolio-1",
            "risk_profile": "default",
            "symbol": "WITHDRAWAL",
            "side": "withdraw",
            "quantity": 1.0,
            "operation": "withdrawal",
            "timestamp": signed_metadata["hardware_wallet_signed_at"],
        },
        signature_doc,
    )


def test_missing_signer_with_required_wallet_raises() -> None:
    adapter = DummyAdapter()
    router = LiveExecutionRouter(
        adapters={"primary": adapter},
        default_route=("primary",),
        require_hardware_wallet_for_withdrawals=True,
    )

    metadata = require_hardware_wallet_metadata({}, account_id="primary", operation="withdrawal")
    request = OrderRequest(
        symbol="WITHDRAWAL",
        side="withdraw",
        quantity=1.0,
        order_type="market",
        metadata=metadata,
    )
    context = ExecutionContext(
        portfolio_id="portfolio-1",
        risk_profile="default",
        environment="live",
        metadata={"account": "primary"},
    )

    with pytest.raises(RuntimeError):
        router.execute(request, context)


def test_build_live_execution_service_rejects_non_hardware_signer_when_required() -> None:
    adapter = DummyAdapter()

    class Bootstrap:
        def __init__(self) -> None:
            self.adapter = adapter
            self.license_capabilities = type(
                "Caps",
                (),
                {"require_hardware_wallet_for_outgoing": True},
            )()

    class Environment:
        exchange = "primary"
        environment = ExchangeEnvironment.LIVE
        data_cache_path = None

    runtime_settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(
            enabled=True,
            default_route=("primary",),
            signers={
                "default": {"type": "hmac", "key_value": "secret"},
            },
        )
    )

    with pytest.raises(RuntimeError):
        build_live_execution_service(
            bootstrap_ctx=Bootstrap(),
            environment=Environment(),
            runtime_settings=runtime_settings,
        )


def test_build_live_execution_service_logs_key_index_on_debug(caplog: pytest.LogCaptureFixture) -> None:
    adapter = DummyAdapter()

    class Bootstrap:
        def __init__(self) -> None:
            self.adapter = adapter
            self.license_capabilities = type("Caps", (), {})()

    class Environment:
        exchange = "primary"
        environment = ExchangeEnvironment.LIVE
        data_cache_path = None

    runtime_settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(
            enabled=True,
            default_route=("primary",),
            signers={
                "default": {
                    "type": "ledger",
                    "simulate": True,
                    "seed_hex": "01" * 16,
                    "key_id": "ledger-main",
                },
                "accounts": {
                    "backup": {
                        "type": "hmac",
                        "key_value": "secret",
                        "key_id": "shared",
                    }
                },
            },
        )
    )

    with caplog.at_level("DEBUG"):
        build_live_execution_service(
            bootstrap_ctx=Bootstrap(),
            environment=Environment(),
            runtime_settings=runtime_settings,
        )

    messages = "\n".join(record.message for record in caplog.records)
    assert "Indeks key_id ledger-main" in messages
    assert "Indeks key_id shared" in messages
    assert "Podsumowanie wymagań sprzętowych" in messages
    assert "Wykryte problemy konfiguracji podpisów" in messages


def test_router_reuses_existing_valid_signature() -> None:
    adapter = DummyAdapter()
    signer = LedgerSigner(seed=b"reuse-signature", key_id="ledger")
    selector = TransactionSignerSelector(default=signer)
    router = LiveExecutionRouter(
        adapters={"primary": adapter},
        default_route=("primary",),
        transaction_signers=selector,
        require_hardware_wallet_for_withdrawals=True,
    )

    context = ExecutionContext(
        portfolio_id="portfolio-1",
        risk_profile="default",
        environment="live",
        metadata={"account": "primary"},
    )

    timestamp = "2024-01-01T00:00:00Z"
    payload = {
        "exchange": "primary",
        "account": "primary",
        "portfolio": context.portfolio_id,
        "risk_profile": context.risk_profile,
        "symbol": "WITHDRAWAL",
        "side": "withdraw",
        "quantity": 1.0,
        "operation": "withdrawal",
        "timestamp": timestamp,
    }

    signature = signer.sign(payload)

    request = OrderRequest(
        symbol="WITHDRAWAL",
        side="withdraw",
        quantity=1.0,
        order_type="market",
        metadata={
            "operation": "withdrawal",
            "requires_hardware_wallet": True,
            "hardware_wallet_signature": signature,
            "hardware_wallet_algorithm": signature.get("algorithm"),
            "hardware_wallet_signed_at": timestamp,
            "hardware_wallet_account": "primary",
            "hardware_wallet_key_id": signature.get("key_id"),
        },
    )

    result = router.execute(request, context)
    assert result.status == "accepted"
    assert adapter.last_request is not None
    signed_metadata = dict(adapter.last_request.metadata or {})
    assert signed_metadata["hardware_wallet_signed_at"] == timestamp
    assert dict(signed_metadata["hardware_wallet_signature"]) == dict(signature)


def test_router_restores_key_id_and_reuses_signature_from_other_account() -> None:
    adapter = DummyAdapter()
    signer = LedgerSigner(seed=b"reuse-keyid", key_id="ledger-reuse")
    selector = TransactionSignerSelector(overrides={"primary": signer})
    router = LiveExecutionRouter(
        adapters={"primary": adapter},
        default_route=("primary",),
        transaction_signers=selector,
        require_hardware_wallet_for_withdrawals=True,
    )

    context = ExecutionContext(
        portfolio_id="portfolio-1",
        risk_profile="default",
        environment="live",
        metadata={"account": "primary"},
    )

    timestamp = "2024-02-02T00:00:00Z"
    payload = {
        "exchange": "primary",
        "account": "backup",  # podpis przygotowany dla innego konta
        "portfolio": context.portfolio_id,
        "risk_profile": context.risk_profile,
        "symbol": "WITHDRAWAL",
        "side": "withdraw",
        "quantity": 1.0,
        "operation": "withdrawal",
        "timestamp": timestamp,
    }

    signature = signer.sign(payload)
    sanitized_signature = dict(signature)
    sanitized_signature.pop("key_id", None)

    request = OrderRequest(
        symbol="WITHDRAWAL",
        side="withdraw",
        quantity=1.0,
        order_type="market",
        metadata=require_hardware_wallet_metadata(
            {
                "hardware_wallet_signature": sanitized_signature,
                "hardware_wallet_algorithm": signature.get("algorithm"),
                "hardware_wallet_signed_at": timestamp,
                "hardware_wallet_account": "backup",
                "hardware_wallet_key_id": signature.get("key_id"),
            },
            account_id="primary",
            operation="withdrawal",
        ),
    )

    result = router.execute(request, context)
    assert result.status == "accepted"
    assert adapter.last_request is not None
    signed_metadata = dict(adapter.last_request.metadata or {})

    reused_signature = dict(signed_metadata["hardware_wallet_signature"])
    assert reused_signature["key_id"] == "ledger-reuse"
    assert signed_metadata["hardware_wallet_key_id"] == "ledger-reuse"
    assert signed_metadata["hardware_wallet_account"] == "backup"
