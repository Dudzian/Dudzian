from types import SimpleNamespace
from uuid import uuid4

from bot_core.risk.engine import ThresholdRiskEngine


def test_resolve_account_id_prefers_account_id_attribute() -> None:
    account = SimpleNamespace(account_id="abc123", id="secondary")
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "abc123"


def test_resolve_account_id_falls_back_to_id_attribute() -> None:
    account = SimpleNamespace(id=42)
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "42"


def test_resolve_account_id_coerces_uuid() -> None:
    identifier = uuid4()
    account = SimpleNamespace(account_id=identifier)
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == str(identifier)


def test_resolve_account_id_uses_metadata() -> None:
    account = SimpleNamespace(metadata={"account_id": 99})
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "99"


def test_resolve_account_id_accepts_mapping() -> None:
    account = {"account_id": "map-value"}
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "map-value"


def test_resolve_account_id_handles_account_ref_attribute() -> None:
    account = SimpleNamespace(account_ref="ref-123")
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "ref-123"


def test_resolve_account_id_handles_account_key_attribute() -> None:
    account = SimpleNamespace(account_key="key-456")
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "key-456"


def test_resolve_account_id_handles_account_attribute() -> None:
    account = SimpleNamespace(account="acct-name")
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) == "acct-name"


def test_resolve_account_id_swallows_attribute_errors() -> None:
    class Raising:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError("boom")

    account = Raising()
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) is None


def test_resolve_account_id_returns_none_when_missing() -> None:
    account = SimpleNamespace()
    assert ThresholdRiskEngine._resolve_runtime_account_id(account) is None
