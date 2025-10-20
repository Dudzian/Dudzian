from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from bot_core.reporting.paper import generate_daily_paper_report
from bot_core.runtime.bootstrap import build_alert_channels
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)


class _DummySecretManager:
    def load_secret_value(self, *_args: Any, **_kwargs: Any) -> str:
        return "{}"


class _StubExecutionService:
    def ledger(self) -> list[dict[str, object]]:
        return []


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


def _install(payload: dict[str, Any]) -> None:
    effective_date = datetime.now(timezone.utc).date()
    capabilities = build_capabilities_from_payload(payload, effective_date=effective_date)
    install_capability_guard(capabilities)


def test_build_alert_channels_requires_advanced_module(tmp_path: Path) -> None:
    payload = {
        "edition": "pro",
        "modules": {"alerts_advanced": False},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)
    core_config = SimpleNamespace(
        telegram_channels={},
        email_channels={},
        sms_providers={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
    )
    environment = SimpleNamespace(
        name="paper",
        data_cache_path=str(tmp_path),
        alert_channels=("sms:critical",),
        alert_audit=None,
        alert_throttle=None,
        offline_mode=False,
    )
    secret_manager = _DummySecretManager()

    with pytest.raises(LicenseCapabilityError):
        build_alert_channels(
            core_config=core_config,
            environment=environment,
            secret_manager=secret_manager,
        )


def test_generate_daily_paper_report_requires_reporting_module(tmp_path: Path) -> None:
    payload = {
        "edition": "pro",
        "modules": {"reporting_pro": False},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)
    execution_service = _StubExecutionService()

    with pytest.raises(LicenseCapabilityError):
        generate_daily_paper_report(
            execution_service=execution_service,
            output_dir=tmp_path,
            tz=timezone.utc,
        )


def test_paper_auto_trade_app_requires_auto_trader_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "edition": "pro",
        "runtime": {"auto_trader": False},
        "modules": {"futures": True},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    fake_module = ModuleType("nacl")
    fake_exceptions = ModuleType("nacl.exceptions")
    fake_exceptions.BadSignatureError = Exception
    fake_signing = ModuleType("nacl.signing")

    class _VerifyKey:
        def verify(self, *_args: object, **_kwargs: object) -> None:
            return None

    fake_signing.VerifyKey = lambda *_args, **_kwargs: _VerifyKey()
    fake_module.exceptions = fake_exceptions
    fake_module.signing = fake_signing
    monkeypatch.setitem(sys.modules, "nacl", fake_module)
    monkeypatch.setitem(sys.modules, "nacl.exceptions", fake_exceptions)
    monkeypatch.setitem(sys.modules, "nacl.signing", fake_signing)

    PaperAutoTradeApp = import_module("KryptoLowca.auto_trader.paper").PaperAutoTradeApp
    with pytest.raises(LicenseCapabilityError):
        PaperAutoTradeApp(enable_gui=False, use_dummy_feed=False)

    for name in ("nacl", "nacl.exceptions", "nacl.signing"):
        monkeypatch.delitem(sys.modules, name, raising=False)

