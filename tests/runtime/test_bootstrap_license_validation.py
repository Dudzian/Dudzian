import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import yaml

from bot_core.alerts import AlertSeverity, get_alert_dispatcher
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeAdapter, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.runtime.bootstrap import bootstrap_environment
from bot_core.security import SecretManager, SecretStorage
from bot_core.security.signing import build_hmac_signature


LICENSE_KEY = bytes.fromhex("7d" * 32)
FINGERPRINT_KEY = bytes.fromhex("4c" * 32)
REVOCATION_KEY = bytes.fromhex("3a" * 32)


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


class DummyAdapter(ExchangeAdapter):
    def configure_network(self, *, ip_allowlist=None) -> None:  # type: ignore[override]
        self.ip_allowlist = ip_allowlist

    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self):  # type: ignore[override]
        return []

    def fetch_ohlcv(self, symbol: str, interval: str, start=None, end=None, limit=None):  # type: ignore[override]
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
        return OrderResult(order_id="1", status="filled", filled_quantity=0.0, avg_price=None, raw_response={})

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # type: ignore[override]
        return None

    def stream_public_data(self, *, channels):  # type: ignore[override]
        return self

    def stream_private_data(self, *, channels):  # type: ignore[override]
        return self


def _write_keys(path: Path, *, key_id: str, secret: bytes) -> Path:
    payload = {"keys": {key_id: f"hex:{secret.hex()}"}}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_revocations(
    path: Path,
    *,
    revoked: Sequence[object],
    generated_at: str | None = None,
    sign: bool = False,
    key_id: str = "rev-1",
) -> Path:
    entries: list[dict[str, object]] = []
    for value in revoked:
        if isinstance(value, Mapping):
            entry = {**value}
        else:
            entry = {"license_id": value}
        entries.append(entry)
    payload: dict[str, object] = {"revoked": entries}
    if generated_at:
        payload["generated_at"] = generated_at
    if sign:
        signature = build_hmac_signature(
            payload,
            key=REVOCATION_KEY,
            algorithm="HMAC-SHA384",
            key_id=key_id,
        )
        document: dict[str, object] = {"payload": payload, "signature": signature}
    else:
        document = payload
    revocation_path = path / "revocations.json"
    revocation_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return revocation_path


def _write_revocation_keys(path: Path, *, key_id: str = "rev-1") -> Path:
    payload = {"keys": {key_id: f"hex:{REVOCATION_KEY.hex()}"}}
    output = path / "revocation_keys.json"
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def _build_signed_license(
    base: Path,
    *,
    fingerprint_value: str = "abcd",
    issued_at: str | None = None,
    expires_at: str | None = None,
    license_id: str = "lic-bootstrap",
) -> Mapping[str, str]:
    fingerprint_payload = {
        "version": 1,
        "collected_at": "2024-02-01T12:00:00Z",
        "components": {},
        "component_digests": {},
        "fingerprint": {"algorithm": "sha256", "value": fingerprint_value},
    }
    fingerprint_signature = build_hmac_signature(
        fingerprint_payload,
        key=FINGERPRINT_KEY,
        algorithm="HMAC-SHA384",
        key_id="fp-1",
    )
    issued = issued_at or "2024-02-01T00:00:00Z"
    expires = expires_at or "2040-02-01T00:00:00Z"
    license_payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "issued_at": issued,
        "expires_at": expires,
        "issuer": "qa",
        "profile": "paper",
        "license_id": license_id,
        "fingerprint": fingerprint_payload["fingerprint"],
        "fingerprint_payload": fingerprint_payload,
        "fingerprint_signature": fingerprint_signature,
    }
    license_signature = build_hmac_signature(
        license_payload,
        key=LICENSE_KEY,
        algorithm="HMAC-SHA384",
        key_id="lic-1",
    )
    license_document = {"payload": license_payload, "signature": license_signature}
    fingerprint_document = {"payload": fingerprint_payload, "signature": fingerprint_signature}

    license_path = base / "license.json"
    fingerprint_path = base / "fingerprint.json"
    license_path.write_text(json.dumps(license_document, ensure_ascii=False), encoding="utf-8")
    fingerprint_path.write_text(json.dumps(fingerprint_document, ensure_ascii=False), encoding="utf-8")

    license_keys_path = _write_keys(base / "license_keys.json", key_id="lic-1", secret=LICENSE_KEY)
    fingerprint_keys_path = _write_keys(base / "fingerprint_keys.json", key_id="fp-1", secret=FINGERPRINT_KEY)

    return {
        "license_path": str(license_path),
        "fingerprint_path": str(fingerprint_path),
        "license_keys_path": str(license_keys_path),
        "fingerprint_keys_path": str(fingerprint_keys_path),
    }


def _write_core_config(tmp_path: Path, *, license_paths: Mapping[str, object]) -> Path:
    config = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.05,
                "max_position_pct": 0.1,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 3,
                "hard_drawdown_pct": 0.1,
            }
        },
        "environments": {
            "paper_stub": {
                "exchange": "stub",
                "environment": "paper",
                "keychain_key": "paper_key",
                "credential_purpose": "trading",
                "data_cache_path": str(tmp_path / "cache"),
                "risk_profile": "balanced",
                "alert_channels": [],
            }
        },
        "alerts": {},
        "permission_profiles": {},
        "license": dict(license_paths),
    }
    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _secret_manager_with_credentials() -> SecretManager:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage)
    manager.store_exchange_credentials(
        "paper_key",
        ExchangeCredentials(
            key_id="demo",
            secret="secret",
            environment=Environment.PAPER,
            permissions=("trade",),
        ),
    )
    return manager


def _bootstrap(tmp_path: Path, *, mutate_license=None, monkeypatch=None, license_settings=None):
    license_paths = _build_signed_license(tmp_path)
    if mutate_license is not None:
        mutate_license(Path(license_paths["license_path"]))
    combined_license = dict(license_paths)
    if license_settings:
        combined_license.update(license_settings)
    config_path = _write_core_config(tmp_path, license_paths=combined_license)
    adapter_factories = {
        "stub": lambda credentials, **kwargs: DummyAdapter(credentials),
    }
    if monkeypatch is not None:
        class _RouterStub:
            def register(self, *_args, **_kwargs):
                return None

            def dispatch(self, *_args, **_kwargs):
                return None

        monkeypatch.setattr(
            "bot_core.runtime.bootstrap.build_alert_channels",
            lambda **_kwargs: ({}, _RouterStub(), None),
        )
    return bootstrap_environment(
        "paper_stub",
        config_path=config_path,
        secret_manager=_secret_manager_with_credentials(),
        adapter_factories=adapter_factories,
    )


def test_bootstrap_environment_rejects_invalid_license(tmp_path: Path, monkeypatch) -> None:
    def _corrupt_signature(path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["signature"]["value"] = "tampered"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    dispatcher = get_alert_dispatcher()
    captured: list = []
    token = dispatcher.register(lambda event: captured.append(event))
    try:
        with pytest.raises(RuntimeError):
            _bootstrap(tmp_path, mutate_license=_corrupt_signature, monkeypatch=monkeypatch)
    finally:
        dispatcher.unregister(token)
    assert captured
    assert captured[0].severity == AlertSeverity.CRITICAL


def test_bootstrap_environment_rejects_expired_license(tmp_path: Path, monkeypatch) -> None:
    def _expire_license(path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["payload"]["expires_at"] = "2010-01-01T00:00:00Z"
        document["payload"]["issued_at"] = "2009-01-01T00:00:00Z"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    dispatcher = get_alert_dispatcher()
    captured: list = []
    token = dispatcher.register(lambda event: captured.append(event))
    try:
        with pytest.raises(RuntimeError):
            _bootstrap(tmp_path, mutate_license=_expire_license, monkeypatch=monkeypatch)
    finally:
        dispatcher.unregister(token)

    assert captured
    assert any(
        any("wygas" in err.lower() for err in (event.context or {}).get("errors", []))
        for event in captured
    )


def test_bootstrap_environment_accepts_valid_license(tmp_path: Path, monkeypatch) -> None:
    dispatcher = get_alert_dispatcher()
    events: list = []
    token = dispatcher.register(lambda event: events.append(event))
    try:
        context = _bootstrap(tmp_path, monkeypatch=monkeypatch)
    finally:
        dispatcher.unregister(token)
    assert context.environment.name == "paper_stub"
    assert context.license_validation.is_valid
    assert context.license_validation.fingerprint
    assert context.license_validation.profile == "paper"
    assert context.license_validation.issuer == "qa"
    assert context.license_validation.schema == "core.oem.license"
    assert context.license_validation.schema_version == "1.0"
    assert context.license_validation.license_id == "lic-bootstrap"
    assert context.license_validation.revocation_status == "skipped"
    assert context.license_validation.revocation_checked is False
    assert context.license_validation.revocation_reason is None
    assert context.license_validation.revocation_revoked_at is None
    assert context.license_validation.revocation_actor is None
    assert any(event.severity in {AlertSeverity.INFO, AlertSeverity.WARNING} for event in events)


def test_bootstrap_environment_rejects_revoked_license(tmp_path: Path, monkeypatch) -> None:
    revocation_path = _write_revocations(
        tmp_path,
        revoked=[
            {
                "license_id": "lic-bootstrap",
                "reason": "Klucz unieważniony",
                "revoked_at": "2024-06-30T18:30:00Z",
                "actor": "compliance-team",
            }
        ],
        generated_at="2024-07-01T00:00:00Z",
        sign=True,
        key_id="rev-main",
    )
    revocation_keys_path = _write_revocation_keys(tmp_path, key_id="rev-main")
    dispatcher = get_alert_dispatcher()
    captured: list = []
    token = dispatcher.register(lambda event: captured.append(event))
    try:
        with pytest.raises(RuntimeError):
            _bootstrap(
                tmp_path,
                monkeypatch=monkeypatch,
                license_settings={
                    "revocation_list_path": str(revocation_path),
                    "revocation_required": True,
                    "revocation_list_max_age_hours": 43800,
                    "revocation_keys_path": str(revocation_keys_path),
                    "revocation_signature_required": True,
                },
            )
    finally:
        dispatcher.unregister(token)

    assert captured
    assert any(
        any("odwołań" in err for err in (event.context or {}).get("errors", []))
        for event in captured
    )
    assert any(
        (event.context or {}).get("revocation_reason") == "Klucz unieważniony"
        for event in captured
    )
    assert any(
        (event.context or {}).get("revocation_revoked_at") == "2024-06-30T18:30:00+00:00"
        for event in captured
    )
    assert any(
        (event.context or {}).get("revocation_actor") == "compliance-team"
        for event in captured
    )


def test_bootstrap_environment_accepts_clear_revocation_list(tmp_path: Path, monkeypatch) -> None:
    revocation_path = _write_revocations(
        tmp_path,
        revoked=["other"],
        generated_at="2024-07-01T12:00:00Z",
        sign=True,
        key_id="rev-main",
    )
    revocation_keys_path = _write_revocation_keys(tmp_path, key_id="rev-main")
    dispatcher = get_alert_dispatcher()
    events: list = []
    token = dispatcher.register(lambda event: events.append(event))
    try:
        context = _bootstrap(
            tmp_path,
            monkeypatch=monkeypatch,
            license_settings={
                "revocation_list_path": str(revocation_path),
                "revocation_required": True,
                "revocation_list_max_age_hours": 43800,
                "revocation_keys_path": str(revocation_keys_path),
                "revocation_signature_required": True,
            },
        )
    finally:
        dispatcher.unregister(token)

    assert context.license_validation.revocation_status == "clear"
    assert context.license_validation.revocation_checked is True
    assert context.license_validation.revocation_generated_at is not None
    assert context.license_validation.revocation_signature_key == "rev-main"
    assert context.license_validation.revocation_reason is None
    assert context.license_validation.revocation_revoked_at is None
    assert context.license_validation.revocation_actor is None
    assert any(event.severity in {AlertSeverity.INFO, AlertSeverity.WARNING} for event in events)


def test_bootstrap_environment_requires_keys_for_signed_revocations(tmp_path: Path, monkeypatch) -> None:
    revocation_path = _write_revocations(
        tmp_path,
        revoked=[],
        generated_at="2024-07-01T12:00:00Z",
        sign=True,
    )
    dispatcher = get_alert_dispatcher()
    captured: list = []
    token = dispatcher.register(lambda event: captured.append(event))
    try:
        with pytest.raises(RuntimeError):
            _bootstrap(
                tmp_path,
                monkeypatch=monkeypatch,
                license_settings={
                    "revocation_list_path": str(revocation_path),
                    "revocation_required": True,
                    "revocation_list_max_age_hours": 43800,
                    "revocation_signature_required": True,
                },
            )
    finally:
        dispatcher.unregister(token)

    assert captured
    assert any("revocation_keys_path" in str(event) for event in captured)


def test_bootstrap_environment_rejects_disallowed_profile(tmp_path: Path, monkeypatch) -> None:
    def _change_profile(path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document["payload"]
        payload["profile"] = "vip"
        document["signature"] = build_hmac_signature(
            payload,
            key=LICENSE_KEY,
            algorithm="HMAC-SHA384",
            key_id="lic-1",
        )
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    dispatcher = get_alert_dispatcher()
    captured: list = []
    token = dispatcher.register(lambda event: captured.append(event))
    try:
        with pytest.raises(RuntimeError):
            _bootstrap(
                tmp_path,
                mutate_license=_change_profile,
                monkeypatch=monkeypatch,
                license_settings={"allowed_profiles": ["paper"]},
            )
    finally:
        dispatcher.unregister(token)

    assert captured
    assert any(
        any("Profil licencji" in err for err in (event.context or {}).get("errors", []))
        for event in captured
    )
