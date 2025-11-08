from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from textwrap import dedent
from types import SimpleNamespace
from dataclasses import is_dataclass

import pytest
import yaml

from bot_core.config.loader import load_core_config

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.config.models import DecisionEngineTCOConfig, SMSProviderSettings
from bot_core.alerts import EmailChannel, SMSChannel, TelegramChannel
from bot_core.decision.models import DecisionCandidate, RiskSnapshot
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
)
from bot_core.exchanges.binance import BinanceSpotAdapter
from bot_core.exchanges.bybit import BybitSpotAdapter
from bot_core.exchanges.coinbase import CoinbaseSpotAdapter
from bot_core.exchanges.kucoin import KuCoinSpotAdapter
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter
from bot_core.exchanges.okx import OKXSpotAdapter
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.repository import FileRiskRepository
from bot_core.runtime.metadata import RiskManagerSettings
from bot_core.runtime.bootstrap import (
    BootstrapContext,
    _DEFAULT_ADAPTERS,
    _instantiate_adapter,
    _apply_adapter_factory_specs,
    _load_initial_tco_costs,
    bootstrap_environment,
    catalog_runtime_entrypoints,
    extract_live_readiness_metadata,
    get_registered_adapter_factories,
    register_adapter_factory,
    register_adapter_factory_from_path,
    resolve_runtime_entrypoint,
    parse_adapter_factory_cli_specs,
    unregister_adapter_factory,
    temporary_adapter_factories,
)
import bot_core.runtime.bootstrap as bootstrap_module
from bot_core.runtime.metrics_alerts import DEFAULT_UI_ALERTS_JSONL_PATH
from bot_core.security import SecretManager, SecretStorage, SecretStorageError
from bot_core.security.signing import build_hmac_signature


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"sha256": "  ABCDEF123456  "}, "abcdef123456"),
        ({"hashes": {"sha256": "ABCDEF123456"}}, "abcdef123456"),
        ({"document": {"sha256": "ABCDEF123456"}}, "abcdef123456"),
        (
            {"document": {"hashes": {"sha256": "ABCDEF123456"}}},
            "abcdef123456",
        ),
    ),
)
def test_extract_payload_sha256_normalizes_candidates(payload, expected) -> None:
    assert bootstrap_module._extract_payload_sha256(payload) == expected


def test_extract_payload_sha256_handles_missing_candidates() -> None:
    assert bootstrap_module._extract_payload_sha256({"sha512": "noop"}) is None


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


_BASE_CONFIG = dedent(
    """
    risk_profiles:
      balanced:
        max_daily_loss_pct: 0.015
        max_position_pct: 0.05
        target_volatility: 0.11
        max_leverage: 3.0
        stop_loss_atr_multiple: 1.5
        max_open_positions: 5
        hard_drawdown_pct: 0.10
      aggressive:
        max_daily_loss_pct: 0.05
        max_position_pct: 0.20
        target_volatility: 0.35
        max_leverage: 5.0
        stop_loss_atr_multiple: 2.0
        max_open_positions: 12
        hard_drawdown_pct: 0.25
    permission_profiles:
      trading_default:
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
      read_only:
        required_permissions: [read]
        forbidden_permissions: []
    runtime_entrypoints:
      auto_trader:
        environment: binance_paper
        controller: autotrader_default
        strategy: ai_autotrader
        risk_profile: balanced
        description: "AutoTrader"
        compliance:
          live_allowed: true
          risk_profiles: [balanced, aggressive]
          signoffs: [stage5_audit]
          signed: true
          require_signoff: true
      trading_gui:
        environment: binance_paper
        controller: trading_gui
        risk_profile: manual
        bootstrap: false
      paper_cli:
        environment: binance_paper
        controller: trading_cli
        risk_profile: balanced
        description: "Paper CLI"
        tags: [cli, paper]
        bootstrap: false
    environments:
      binance_paper:
        exchange: binance_spot
        environment: paper
        keychain_key: binance_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/binance_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary", "email:ops", "sms:orange_local"]
        ip_allowlist: ["127.0.0.1"]
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
        alert_throttle:
          window_seconds: 60
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
      zonda_paper:
        exchange: zonda_spot
        environment: paper
        keychain_key: zonda_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary"]
        ip_allowlist: ["127.0.0.1"]
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
        alert_throttle:
          window_seconds: 120
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
      nowa_gielda_paper:
        exchange: nowa_gielda_spot
        environment: paper
        keychain_key: nowa_gielda_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/nowa_gielda_paper
        risk_profile: balanced
        permission_profile: trading_default
        alert_channels: ["telegram:primary"]
        ip_allowlist: ["127.0.0.1"]
        alert_throttle:
          window_seconds: 45
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 16
      coinbase_offline:
        exchange: coinbase_spot
        environment: paper
        keychain_key: coinbase_offline_key
        credential_purpose: trading
        data_cache_path: ./var/data/coinbase_offline
        risk_profile: balanced
        alert_channels: ["telegram:primary", "email:ops", "sms:orange_local"]
        offline_mode: true
        report_storage:
          backend: file
          directory: ./audit/offline/reports
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123456789"
          token_secret: telegram_token
      email_channels:
        ops:
          host: smtp.example.com
          port: 587
          from_address: bot@example.com
          recipients: ["ops@example.com"]
          credential_secret: smtp_credentials
      sms_providers:
        orange_local:
          provider: orange_pl
          api_base_url: https://api.orange.pl/sms/v1
          from_number: BOT-ORANGE
          recipients: ["+48555111222"]
          allow_alphanumeric_sender: true
          sender_id: BOT-ORANGE
          credential_key: sms_orange
    """
)


_LICENSE_KEY = bytes.fromhex("6b" * 32)
_FINGERPRINT_KEY = bytes.fromhex("5a" * 32)


def _write_hmac_keys_file(path: Path, *, key_id: str, secret: bytes) -> Path:
    payload = {"keys": {key_id: f"hex:{secret.hex()}"}}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _prepare_signed_license_bundle(base_dir: Path) -> Mapping[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        "version": 1,
        "collected_at": "2024-01-01T00:00:00Z",
        "components": {},
        "component_digests": {},
        "fingerprint": {"algorithm": "sha256", "value": "abc123"},
    }
    fingerprint_signature = build_hmac_signature(
        fingerprint_payload,
        key=_FINGERPRINT_KEY,
        algorithm="HMAC-SHA384",
        key_id="fp-1",
    )
    license_payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "issued_at": "2024-01-01T00:00:00Z",
        "expires_at": "2040-01-01T00:00:00Z",
        "issuer": "qa",
        "profile": "paper",
        "license_id": "lic-valid",
        "fingerprint": fingerprint_payload["fingerprint"],
        "fingerprint_payload": fingerprint_payload,
        "fingerprint_signature": fingerprint_signature,
    }
    license_signature = build_hmac_signature(
        license_payload,
        key=_LICENSE_KEY,
        algorithm="HMAC-SHA384",
        key_id="lic-1",
    )
    license_path = base_dir / "license.json"
    license_document = {"payload": license_payload, "signature": license_signature}
    license_path.write_text(
        json.dumps(license_document, ensure_ascii=False), encoding="utf-8"
    )

    fingerprint_path = base_dir / "fingerprint.json"
    fingerprint_document = {
        "payload": fingerprint_payload,
        "signature": fingerprint_signature,
    }
    fingerprint_path.write_text(
        json.dumps(fingerprint_document, ensure_ascii=False), encoding="utf-8"
    )

    license_keys_path = _write_hmac_keys_file(
        base_dir / "license_keys.json", key_id="lic-1", secret=_LICENSE_KEY
    )
    fingerprint_keys_path = _write_hmac_keys_file(
        base_dir / "fingerprint_keys.json", key_id="fp-1", secret=_FINGERPRINT_KEY
    )

    return {
        "license": license_path,
        "fingerprint": fingerprint_path,
        "license_keys": license_keys_path,
        "fingerprint_keys": fingerprint_keys_path,
    }


_RISK_TLS_CERT = """-----BEGIN CERTIFICATE-----
MIIDCzCCAfOgAwIBAgIUXg2adJP0b1IzyHUIygGCz5e4CZswDQYJKoZIhvcNAQEL
BQAwFTETMBEGA1UEAwwKdGVzdC5sb2NhbDAeFw0yNTEwMTIxMzUzMThaFw0yNjEw
MTIxMzUzMThaMBUxEzARBgNVBAMMCnRlc3QubG9jYWwwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCoTFYSrtuGtDlfV6zjE1YQZz8rDoW58bid8CcMXAA/
6sXvCDYFepgjJpD1dDaXLIORNaoteLJLDwd/GbWhK589n/+KzLfnrS+vC+Y9/Zwr
9mCJbQ6liPYleqidG98cY50nrru8IiLORWDDWPLMcNJbGuTkg+JUkmVhaLSmWF1u
GLN+5IFJ+NXo3fUW5B1swcGYFlta0KelpNaatyMKQZZ7wO4QXp8H/ajBbXpWnE/n
JAYOqdBS03+AVpy2Qr0HnCw8NxSur9pqLY4EDepPgowBaMuVv8XgG8+XYJm6y9HH
r1sDd7u1Nq+EnqDfuwZ7HRe5sNQXCW8MwOp+kWqVgrGVAgMBAAGjUzBRMB0GA1Ud
DgQWBBQgFv08KgCA0frYHzc8GrR6RFyi+jAfBgNVHSMEGDAWgBQgFv08KgCA0frY
Hzc8GrR6RFyi+jAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQAP
LvBg5dsZoS2IZXZMv89QTJsI2pndypdL8iQND7StJstgp57f18rOIqn6M3Hw+RwN
NnFs84Jm9qrPRto8jd/sxFFEKYwraLOYadTdIorRYFNdoSYA8dQwh4vUbWLGL5pV
EYrSDEItLWPznRLYx2O5NIo1sM6/LWDDSYbwE9EUCxrCFzF9TQT8lVLQDE2KTCbB
EiSKc+jb0Y2FlBCTDbfpK3CrvS2nhP+Vh+M9iyacJNHEhbiEkam0wje5/6gDsBoi
ouji8dqM2xUMITUW+BWHn/eDOqLss5ZlRbGTNm0Hq3CP7uf0u15r5QxBJjzZYddr
WTcLAxCHylb4RYSCze+i
-----END CERTIFICATE-----
"""


_RISK_TLS_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCoTFYSrtuGtDlf
V6zjE1YQZz8rDoW58bid8CcMXAA/6sXvCDYFepgjJpD1dDaXLIORNaoteLJLDwd/
GbWhK589n/+KzLfnrS+vC+Y9/Zwr9mCJbQ6liPYleqidG98cY50nrru8IiLORWDD
WPLMcNJbGuTkg+JUkmVhaLSmWF1uGLN+5IFJ+NXo3fUW5B1swcGYFlta0KelpNaa
tyMKQZZ7wO4QXp8H/ajBbXpWnE/nJAYOqdBS03+AVpy2Qr0HnCw8NxSur9pqLY4E
DepPgowBaMuVv8XgG8+XYJm6y9HHr1sDd7u1Nq+EnqDfuwZ7HRe5sNQXCW8MwOp+
kWqVgrGVAgMBAAECggEATxe89b/OdIJbWibancb3EfNruOD00Lu8XyE/QKw2A9Pi
XKE3viBswkw8INaSVz54wHP/e6o25FZ2V/GtrcZR6oS4dDMclJkMCVBmzqhSzkhV
+w/RK9NvlpKMDnXMR0u7TixslxBV2im5vWSeipzVBzLe8lPWuJcqZPpvt6NcmUHo
dCVW+fqrZeTmYAMi4oCtCr9GqznYjK0Lpp1RaeGTEo9nzdULaR/9KcqjsMSUXrmY
5JgGgA0iFLcA9EYCMC6C6kpLuimW5FYO+pNNhRPPGWA9Lti5UMjLAG9bgr4cnsLK
PXaCYTlL95LGSq4JvMPwsytmSUnzDPejp+sEy8gXawKBgQDPssDA31r8tDpO7MRs
fvmICQ0760EyQ17bpYZ6cKdfHfMfnEqaJU6MBurhkLA9EsajmYgHJQbCyX10Pso9
3fcTlMCGsvFYT8F5iNkEjGg7PAKIEmjgufmOoip20+uKLCm2kFi4km/BY10EQ5sk
znmFV61A6BMLu+cqnNpXYqOvCwKBgQDPb+nTVN8knCXn1CRc/JBvfZpktN0/INZ2
Rlnjn/C1oVVhuXiAuvEfULs6wkViCNd9C6Sle69BCAu3rVX1D2YPAorUx5jp07sE
2HrFIpLl15QqT5YuE6sOP+tXxLSF9J8SiJWYHGJBfHF8HXuci+wRXJb01mXkdAIN
c4lLvMYF3wKBgFkKiRgmqRstKNItLwhUZyWqu8G0WX7y4vfHPp+/LAHbFR+4IUN0
OvhM/uU04llMc1wvteFaPkvDlcUAJjPftMzwOJmGnXD+wDMaN+97QjQixfMP8WZm
VFaRryLCN3hE9p0NxPtbzA1cS8RIN3rQCcjgjaYF2CRvqera08AiyYmBAoGAUlRt
roXB5sretItbP1iyjr2AOLYcFcEXvWugo5pINB5rP9UYAaewqagmF2Uhmo490JB9
cXyMizgBRo5STmglLpHovhjWFQAG+x5cY7+cJAMS+FQMHA+MVaSC6JvWtk/njriM
/wlM6gbVF9ivxes275EbDOPHHwv4AJS5ikjLI2sCgYArrbobSUf8aDrs86EgS4Jb
Nd1bjYx8SWC+uRyZNfsE/TllEtyrP8yWMP6Tq9S+uMXrHfcE1Rj7bj9fJYxeOzQe
geENbbsvj9oU2pNPKWlwk861WMZcppkpmVHrgIIukGC+DSTWFntBGNyDRBdgE+gg
ymp4BN4Riifev8GdFf+lMg==
-----END PRIVATE KEY-----
"""


def _create_signed_document(
    root: Path,
    doc_relative: str,
    signature_relative: str,
    *,
    key_id: str,
    signed_by: Sequence[str],
) -> Mapping[str, str]:
    """Tworzy podpisany dokument i zapisuje plik klucza w secrets/hmac."""

    normalized_key = re.sub(r"[^A-Za-z0-9]+", "_", key_id).strip("_").lower() or "default"
    key_path = root / "secrets" / "hmac" / f"{normalized_key}.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_bytes = os.urandom(48)
    key_path.write_bytes(key_bytes)

    document_path = root / doc_relative
    document_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{doc_relative}:{key_id}".encode("utf-8")
    document_path.write_bytes(content)
    sha_value = hashlib.sha256(content).hexdigest()

    payload = {
        "document": {
            "name": Path(doc_relative).name,
            "path": doc_relative,
            "sha256": sha_value,
            "signed_by": list(signed_by),
            "signed_at": "2024-06-01T10:00:00Z",
        },
        "hashes": {"sha256": sha_value},
        "generated_at": "2024-06-01T10:00:00Z",
    }
    signature = build_hmac_signature(payload, key=key_bytes, key_id=key_id)

    signature_path = root / signature_relative
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(
        json.dumps({"payload": payload, "signature": signature}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"sha256": sha_value, "signature_path": signature_relative}


def _stub_license_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        bootstrap_module,
        "validate_license_from_config",
        lambda _cfg: bootstrap_module.LicenseValidationResult(
            status="ok",
            fingerprint=None,
            license_path=tmp_path / "license.json",
            issued_at=None,
            expires_at=None,
            fingerprint_source=None,
            profile=None,
            issuer=None,
            schema=None,
            schema_version=None,
            license_id=None,
            revocation_list_path=None,
            revocation_status=None,
            revocation_reason=None,
            revocation_revoked_at=None,
            revocation_generated_at=None,
            revocation_checked=True,
            revocation_signature_key=None,
            errors=[],
            warnings=[],
            payload=None,
            license_signature_key=None,
            fingerprint_signature_key=None,
        ),
    )


def _disable_exchange_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bootstrap_module,
        "build_standard_health_checks",
        lambda _adapter: (),
    )


def _apply_license_stub(data: Mapping[str, Any], tmp_path: Path) -> None:
    license_section = {
        "license_path": str(tmp_path / "license.json"),
        "fingerprint_path": str(tmp_path / "fingerprint.json"),
        "license_keys_path": str(tmp_path / "license_keys.json"),
        "fingerprint_keys_path": str(tmp_path / "fingerprint_keys.json"),
    }
    if isinstance(data, dict):
        data["license"] = license_section


def _write_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(_BASE_CONFIG)
    env_section = data.get("environments", {})
    for name, env_cfg in env_section.items():
        cache_dir = tmp_path / "cache" / name
        env_cfg["data_cache_path"] = str(cache_dir)
    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _write_config_custom(
    tmp_path: Path,
    *,
    runtime_metrics: Mapping[str, object] | None = None,
    runtime_risk: Mapping[str, object] | None = None,
    runtime_risk_service: Mapping[str, object] | None = None,
    environment_overrides: Mapping[str, object] | None = None,
) -> Path:
    data = yaml.safe_load(_BASE_CONFIG)
    if environment_overrides:
        data.setdefault("environments", {}).setdefault("binance_paper", {}).update(
            environment_overrides
        )
    env_section = data.get("environments", {})
    for name, env_cfg in env_section.items():
        if "data_cache_path" not in env_cfg or not env_cfg["data_cache_path"]:
            env_cfg["data_cache_path"] = str(tmp_path / "cache" / name)
    if runtime_metrics is not None:
        runtime_section = data.setdefault("runtime", {})
        runtime_section["metrics_service"] = runtime_metrics
    if runtime_risk is not None:
        runtime_section = data.setdefault("runtime", {})
        runtime_section["risk_decision_log"] = runtime_risk
    if runtime_risk_service is not None:
        runtime_section = data.setdefault("runtime", {})
        runtime_section["risk_service"] = runtime_risk_service

    config_path = tmp_path / "core_custom.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _prepare_manager() -> tuple[_MemorySecretStorage, SecretManager]:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")
    credentials_payload = {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.PAPER.value,
    }
    storage.set_secret("tests:binance_paper_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:zonda_paper_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:nowa_gielda_paper_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:coinbase_offline_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:loopback_paper_key:trading", json.dumps(credentials_payload))
    testnet_payload = dict(credentials_payload)
    testnet_payload["environment"] = Environment.TESTNET.value
    storage.set_secret("tests:loopback_testnet_key:trading", json.dumps(testnet_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")
    manager.store_secret_value(
        "smtp_credentials",
        json.dumps({"username": "bot", "password": "secret"}),
        purpose="alerts:email",
    )
    manager.store_secret_value(
        "sms_orange",
        json.dumps({"account_sid": "AC123", "auth_token": "token"}),
        purpose="alerts:sms",
    )
    return storage, manager


def test_catalog_runtime_entrypoints_in_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    core_config = load_core_config(config_path)
    entrypoints = catalog_runtime_entrypoints(core_config)
    assert set(entrypoints) >= {"auto_trader", "trading_gui", "paper_cli"}
    assert entrypoints["auto_trader"].risk_profile == "balanced"
    assert entrypoints["trading_gui"].bootstrap_required is False
    assert entrypoints["paper_cli"].bootstrap_required is False
    assert entrypoints["paper_cli"].description == "Paper CLI"


def test_resolve_runtime_entrypoint_bootstrap(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _storage, manager = _prepare_manager()
    entrypoint, context = resolve_runtime_entrypoint(
        "auto_trader",
        config_path=config_path,
        secret_manager=manager,
    )
    assert entrypoint.environment == "binance_paper"
    assert context is not None
    assert context.environment.exchange == "binance_spot"


def test_resolve_runtime_entrypoint_without_bootstrap(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    entrypoint, context = resolve_runtime_entrypoint(
        "trading_gui",
        config_path=config_path,
        bootstrap=False,
    )
    assert context is None
    assert entrypoint.bootstrap_required is False


def test_resolve_runtime_entrypoint_cli_metadata(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    entrypoint, context = resolve_runtime_entrypoint(
        "paper_cli",
        config_path=config_path,
        bootstrap=False,
    )
    assert context is None
    assert entrypoint.environment == "binance_paper"
    assert entrypoint.description == "Paper CLI"
    assert entrypoint.tags == ("cli", "paper")


def test_bootstrap_environment_initialises_components(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert isinstance(context, BootstrapContext)
    assert context.environment.name == "binance_paper"
    assert context.risk_profile_name == "balanced"
    assert context.risk_profile_config.name == "balanced"
    assert isinstance(context.risk_manager_settings, RiskManagerSettings)
    assert context.risk_manager_settings.profile_name == "balanced"
    assert context.risk_manager_settings.max_daily_loss_pct == pytest.approx(0.015)
    assert context.credentials.key_id == "paper-key"
    assert context.adapter.credentials.key_id == "paper-key"

    assert isinstance(context.risk_engine, ThresholdRiskEngine)
    assert isinstance(context.risk_repository, FileRiskRepository)
    result = context.risk_engine.apply_pre_trade_checks(
        OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.2,
            order_type="limit",
            price=100.0,
            # ATR=1.0, stop_loss_atr_multiple=1.5 => wymagany SL = 98.5
            stop_price=100.0 - 1.0 * 1.5,
            atr=1.0,
        ),
        account=AccountSnapshot(
            balances={"USDT": 1000.0},
            total_equity=1000.0,
            available_margin=1000.0,
            maintenance_margin=0.0,
        ),
        profile_name="balanced",
    )
    assert result.allowed is True

    assert set(context.alert_channels.keys()) == {
        "telegram:primary",
        "email:ops",
        "sms:orange_local",
    }
    assert isinstance(context.alert_channels["telegram:primary"], TelegramChannel)
    assert isinstance(context.alert_channels["email:ops"], EmailChannel)
    assert isinstance(context.alert_channels["sms:orange_local"], SMSChannel)
    assert len(context.alert_router.channels) == 3
    assert context.alert_router.audit_log is context.audit_log
    assert context.alert_router.throttle is not None
    assert context.alert_router.throttle.window.total_seconds() == pytest.approx(60.0)
    assert context.risk_snapshot_publisher is None
    assert "critical" in context.alert_router.throttle.exclude_severities

    # Health check nie powinien zgłaszać wyjątków dla świeżo zainicjalizowanych kanałów.
    snapshot = context.alert_router.health_snapshot()
    assert snapshot["telegram:primary"]["status"] == "ok"

    assert context.risk_engine.should_liquidate(profile_name="balanced") is False
    assert context.adapter_settings == {}
    risk_state_dir = Path(context.environment.data_cache_path) / "risk_state"
    assert risk_state_dir.exists()
    assert getattr(context.risk_repository, "_base_path", None) == risk_state_dir
    assert context.metrics_server is None
    assert context.live_readiness_checklist is None


def test_bootstrap_environment_live_exposes_checklist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_license_validation(monkeypatch, tmp_path)
    _disable_exchange_health(monkeypatch)
    _prepare_signed_license_bundle(tmp_path)
    data = yaml.safe_load(_BASE_CONFIG)
    _apply_license_stub(data, tmp_path)
    live_env = copy.deepcopy(data["environments"]["binance_paper"])
    live_env.update(
        {
            "environment": "live",
            "keychain_key": "binance_live_key",
            "alert_audit": {"backend": "file", "directory": "./var/live_alerts"},
        }
    )
    compliance_doc = _create_signed_document(
        tmp_path,
        "compliance/live/binance/kyc_packet.pdf",
        "compliance/live/binance/kyc_packet.sig",
        key_id="compliance-key",
        signed_by=("compliance",),
    )
    risk_doc = _create_signed_document(
        tmp_path,
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk/live/binance/risk_profile_alignment.sig",
        key_id="risk-key",
        signed_by=("risk",),
    )
    penetration_doc = _create_signed_document(
        tmp_path,
        "security/live/binance/penetration_report.pdf",
        "security/live/binance/penetration_report.sig",
        key_id="security-key",
        signed_by=("security",),
    )
    alerting_doc = _create_signed_document(
        tmp_path,
        "sre/live/binance/alerting_playbook.pdf",
        "sre/live/binance/alerting_playbook.sig",
        key_id="sre-key",
        signed_by=("sre",),
    )
    live_env["live_readiness"] = {
        "checklist_id": "binance-q3",
        "signed": True,
        "signed_by": ["compliance", "security"],
        "signed_at": "2024-06-01T10:00:00Z",
        "signature_path": "compliance/live/binance/checklist.sig",
        "required_documents": [
            "kyc_packet",
            "risk_profile_alignment",
            "penetration_report",
        ],
        "documents": [
            {
                "name": "kyc_packet",
                "path": "compliance/live/binance/kyc_packet.pdf",
                "sha256": compliance_doc["sha256"],
                "signature_path": compliance_doc["signature_path"],
                "signed": True,
                "signed_by": ["compliance"],
                "signed_at": "2024-05-28T09:45:00Z",
            },
            {
                "name": "risk_profile_alignment",
                "path": "risk/live/binance/risk_profile_alignment.pdf",
                "sha256": risk_doc["sha256"],
                "signature_path": risk_doc["signature_path"],
                "signed": True,
                "signed_by": ["risk"],
                "signed_at": "2024-05-29T15:30:00Z",
            },
            {
                "name": "penetration_report",
                "path": "security/live/binance/penetration_report.pdf",
                "sha256": penetration_doc["sha256"],
                "signature_path": penetration_doc["signature_path"],
                "signed": True,
                "signed_by": ["security"],
                "signed_at": "2024-05-30T07:10:00Z",
            },
            {
                "name": "alerting_playbook",
                "path": "sre/live/binance/alerting_playbook.pdf",
                "sha256": alerting_doc["sha256"],
                "signature_path": alerting_doc["signature_path"],
                "signed": True,
                "signed_by": ["sre"],
                "signed_at": "2024-05-30T07:10:00Z",
            },
        ],
    }
    data["environments"]["binance_live"] = live_env
    auto_trader_entry = data["runtime_entrypoints"]["auto_trader"]
    auto_trader_entry["environment"] = "binance_live"
    auto_trader_entry["trusted_auto_confirm"] = True
    auto_trader_entry["compliance"] = {
        "live_allowed": True,
        "signed": True,
        "require_signoff": True,
        "risk_profiles": ["balanced"],
        "signoffs": ["kyc2024", "risk_limits"],
    }

    config_path = tmp_path / "core_live.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    storage, manager = _prepare_manager()
    credentials_payload = {
        "key_id": "live-key",
        "secret": "live-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret(
        "tests:binance_live_key:trading",
        json.dumps(credentials_payload),
    )

    context = bootstrap_environment(
        "binance_live", config_path=config_path, secret_manager=manager
    )

    checklist = context.live_readiness_checklist
    assert checklist is not None
    entries = {entry["item"]: entry for entry in checklist}
    assert entries["kyc_aml_signoff"]["status"] == "ok"
    compliance_details = entries["kyc_aml_signoff"]["details"]
    assert compliance_details
    primary_entry = compliance_details[0]
    assert primary_entry["entrypoint"] == "auto_trader"
    assert primary_entry["signoffs"] == ("kyc2024", "risk_limits")
    checklist_entry = entries["live_checklist"]
    assert checklist_entry["status"] == "ok"
    checklist_details = checklist_entry["details"]
    checklist_meta = checklist_details["checklist"]
    assert checklist_meta["status"] == "ok"
    assert checklist_meta["signed_by"] == ("compliance", "security")
    documents = {doc["name"]: doc for doc in checklist_details["documents"]}
    assert documents["kyc_packet"]["status"] == "ok"
    assert documents["risk_profile_alignment"]["signed_by"] == ("risk",)
    assert documents["penetration_report"]["status"] == "ok"
    assert documents["alerting_playbook"]["signature_path"].endswith("alerting_playbook.sig")
    verification = context.live_signature_verification
    assert verification is not None
    verified_docs = verification["documents"]
    assert set(verified_docs) == {
        "kyc_packet",
        "risk_profile_alignment",
        "penetration_report",
    }
    assert verification["categories"] == {
        "compliance": True,
        "risk": True,
        "penetration": True,
    }
    assert entries["risk_limits"]["status"] == "ok"
    assert entries["alerting"]["status"] == "ok"
    assert context.metrics_ui_alert_sink_active is True
    default_alert_log = DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    if not default_alert_log.is_absolute():
        default_alert_log = default_alert_log.resolve(strict=False)
    assert Path(context.metrics_ui_alerts_path or "") == default_alert_log
    assert context.metrics_jsonl_path is None
    assert context.metrics_service_enabled is None
    assert context.metrics_ui_alerts_metadata is not None
    metadata_path = context.metrics_ui_alerts_metadata.get("path")
    assert metadata_path is not None
    assert Path(str(metadata_path)).name == DEFAULT_UI_ALERTS_JSONL_PATH.name
    assert Path(
        context.metrics_ui_alerts_metadata.get("absolute_path")
    ).name == DEFAULT_UI_ALERTS_JSONL_PATH.name
    assert context.metrics_jsonl_metadata is None
    if context.metrics_security_warnings is not None:
        assert isinstance(context.metrics_security_warnings, tuple)
    assert context.metrics_ui_alerts_settings is not None
    assert context.metrics_ui_alerts_settings["reduce_mode"] == "enable"
    assert context.metrics_ui_alerts_settings["overlay_mode"] == "enable"
    assert context.metrics_ui_alerts_settings["jank_mode"] == "disable"
    assert context.metrics_ui_alerts_settings["reduce_motion_alerts"] is True
    assert context.metrics_ui_alerts_settings["overlay_alerts"] is True
    assert context.metrics_ui_alerts_settings["jank_alerts"] is False
    assert context.metrics_ui_alerts_settings["reduce_motion_logging"] is True
    assert context.metrics_ui_alerts_settings["overlay_logging"] is True
    assert context.metrics_ui_alerts_settings["jank_logging"] is False
    assert context.metrics_ui_alerts_settings["jank_category"] == "ui.performance"
    assert context.metrics_ui_alerts_settings["jank_severity_spike"] == "warning"
    assert context.metrics_ui_alerts_settings["jank_severity_critical"] is None
    assert context.metrics_ui_alerts_settings["jank_critical_over_ms"] is None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "inherit"
    assert audit_info["backend"] == "memory"
    assert audit_info["note"] == "inherited_environment_router"


def test_live_checklist_blocks_on_missing_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_license_validation(monkeypatch, tmp_path)
    _disable_exchange_health(monkeypatch)
    _prepare_signed_license_bundle(tmp_path)
    data = yaml.safe_load(_BASE_CONFIG)
    _apply_license_stub(data, tmp_path)
    live_env = copy.deepcopy(data["environments"]["binance_paper"])
    live_env.update(
        {
            "environment": "live",
            "keychain_key": "binance_live_key",
            "alert_audit": {"backend": "file", "directory": "./var/live_alerts"},
        }
    )
    compliance_doc = _create_signed_document(
        tmp_path,
        "compliance/live/binance/kyc_packet.pdf",
        "compliance/live/binance/kyc_packet.sig",
        key_id="compliance-key",
        signed_by=("compliance",),
    )
    risk_doc = _create_signed_document(
        tmp_path,
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk/live/binance/risk_profile_alignment.sig",
        key_id="risk-key",
        signed_by=("risk",),
    )
    live_env["live_readiness"] = {
        "checklist_id": "binance-q3",
        "signed": True,
        "signed_by": ["compliance"],
        "signature_path": "compliance/live/binance/checklist.sig",
        "required_documents": [
            "kyc_packet",
            "risk_profile_alignment",
            "penetration_report",
        ],
        "documents": [
            {
                "name": "kyc_packet",
                "path": "compliance/live/binance/kyc_packet.pdf",
                "sha256": compliance_doc["sha256"],
                "signed": True,
                "signed_by": ["compliance"],
                "signature_path": compliance_doc["signature_path"],
            },
            {
                "name": "risk_profile_alignment",
                "path": "risk/live/binance/risk_profile_alignment.pdf",
                "sha256": risk_doc["sha256"],
                "signed": True,
                "signed_by": ["risk"],
                "signature_path": risk_doc["signature_path"],
            },
            {
                "name": "penetration_report",
                "path": "security/live/binance/penetration_report.pdf",
                "signed": True,
                "signed_by": ["security"],
            },
        ],
    }
    data["environments"]["binance_live"] = live_env
    auto_trader_entry = data["runtime_entrypoints"]["auto_trader"]
    auto_trader_entry["environment"] = "binance_live"
    auto_trader_entry["trusted_auto_confirm"] = True
    auto_trader_entry["compliance"] = {
        "live_allowed": True,
        "signed": True,
        "require_signoff": True,
        "risk_profiles": ["balanced"],
        "signoffs": ["kyc2024"],
    }

    config_path = tmp_path / "core_live.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    storage, manager = _prepare_manager()
    credentials_payload = {
        "key_id": "live-key",
        "secret": "live-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret(
        "tests:binance_live_key:trading",
        json.dumps(credentials_payload),
    )

    with pytest.raises(RuntimeError) as exc:
        bootstrap_environment("binance_live", config_path=config_path, secret_manager=manager)

    assert "penetration_report" in str(exc.value)


def test_live_checklist_blocks_on_invalid_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_license_validation(monkeypatch, tmp_path)
    _disable_exchange_health(monkeypatch)
    _prepare_signed_license_bundle(tmp_path)
    data = yaml.safe_load(_BASE_CONFIG)
    _apply_license_stub(data, tmp_path)
    live_env = copy.deepcopy(data["environments"]["binance_paper"])
    live_env.update(
        {
            "environment": "live",
            "keychain_key": "binance_live_key",
            "alert_audit": {"backend": "file", "directory": "./var/live_alerts"},
        }
    )

    compliance_doc = _create_signed_document(
        tmp_path,
        "compliance/live/binance/kyc_packet.pdf",
        "compliance/live/binance/kyc_packet.sig",
        key_id="compliance-key",
        signed_by=("compliance",),
    )
    risk_doc = _create_signed_document(
        tmp_path,
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk/live/binance/risk_profile_alignment.sig",
        key_id="risk-key",
        signed_by=("risk",),
    )
    penetration_doc = _create_signed_document(
        tmp_path,
        "security/live/binance/penetration_report.pdf",
        "security/live/binance/penetration_report.sig",
        key_id="security-key",
        signed_by=("security",),
    )

    tampered_signature_path = tmp_path / risk_doc["signature_path"]
    signature_payload = json.loads(tampered_signature_path.read_text(encoding="utf-8"))
    signature_payload["signature"]["value"] = "A" + signature_payload["signature"]["value"]
    tampered_signature_path.write_text(
        json.dumps(signature_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    live_env["live_readiness"] = {
        "checklist_id": "binance-q3",
        "signed": True,
        "signed_by": ["compliance", "security"],
        "signature_path": "compliance/live/binance/checklist.sig",
        "required_documents": [
            "kyc_packet",
            "risk_profile_alignment",
            "penetration_report",
        ],
        "documents": [
            {
                "name": "kyc_packet",
                "path": "compliance/live/binance/kyc_packet.pdf",
                "sha256": compliance_doc["sha256"],
                "signed": True,
                "signed_by": ["compliance"],
                "signature_path": compliance_doc["signature_path"],
            },
            {
                "name": "risk_profile_alignment",
                "path": "risk/live/binance/risk_profile_alignment.pdf",
                "sha256": risk_doc["sha256"],
                "signed": True,
                "signed_by": ["risk"],
                "signature_path": risk_doc["signature_path"],
            },
            {
                "name": "penetration_report",
                "path": "security/live/binance/penetration_report.pdf",
                "sha256": penetration_doc["sha256"],
                "signed": True,
                "signed_by": ["security"],
                "signature_path": penetration_doc["signature_path"],
            },
        ],
    }
    data["environments"]["binance_live"] = live_env

    auto_trader_entry = data["runtime_entrypoints"]["auto_trader"]
    auto_trader_entry["environment"] = "binance_live"
    auto_trader_entry["trusted_auto_confirm"] = True
    auto_trader_entry["compliance"] = {
        "live_allowed": True,
        "signed": True,
        "require_signoff": True,
        "risk_profiles": ["balanced"],
        "signoffs": ["kyc2024", "risk_limits"],
    }

    config_path = tmp_path / "core_live.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    storage, manager = _prepare_manager()
    credentials_payload = {
        "key_id": "live-key",
        "secret": "live-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret(
        "tests:binance_live_key:trading",
        json.dumps(credentials_payload),
    )

    with pytest.raises(RuntimeError) as exc:
        bootstrap_environment("binance_live", config_path=config_path, secret_manager=manager)

    assert "risk_profile_alignment" in str(exc.value)


def test_sms_provider_fallback_without_providers_module(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bootstrap korzysta ze stubu providers i tworzy lokalną konfigurację SMS."""

    module_name = "bot_core.alerts.channels.providers"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    bootstrap_module._get_alert_components.cache_clear()

    bootstrap_module._install_sms_provider_stub()
    stub_module = sys.modules[module_name]

    settings = SMSProviderSettings(
        name="ops",
        provider_key="fallback",
        api_base_url="https://custom.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="OPS",
        credential_key=None,
        display_name="Fallback Provider",
        iso_country_code="pl",
        max_sender_length=14,
        notes="Lokalne ustawienia",
    )

    caplog.set_level("WARNING")
    config = bootstrap_module._resolve_sms_provider(
        settings,
        stub_module.get_sms_provider,
        channel_key="ops",
    )

    assert is_dataclass(config)
    assert config.provider_id == "fallback"
    assert config.display_name == "Fallback Provider"
    assert config.api_base_url == "https://custom.example.com"
    assert config.iso_country_code == "PL"
    assert config.supports_alphanumeric_sender is True
    assert config.max_sender_length == 14
    assert config.notes == "Lokalne ustawienia"
    assert "Fallback rejestru SMS" in caplog.text

    registered = stub_module.DEFAULT_SMS_PROVIDERS["fallback"]
    assert registered == config

    bootstrap_module._get_alert_components.cache_clear()


def test_resolve_sms_provider_rejects_invalid_iso_code(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "bot_core.alerts.channels.providers"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    bootstrap_module._get_alert_components.cache_clear()

    bootstrap_module._install_sms_provider_stub()
    stub_module = sys.modules[module_name]

    settings = SMSProviderSettings(
        name="ops",
        provider_key="fallback",
        api_base_url="https://custom.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=False,
        sender_id=None,
        credential_key=None,
        display_name="Fallback Provider",
        iso_country_code="POL",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._resolve_sms_provider(
            settings,
            stub_module.get_sms_provider,
            channel_key="ops",
        )

    assert "ISO 3166-1" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_resolve_sms_provider_normalizes_invalid_registry_iso(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "broken",
        providers_module.SmsProviderConfig(
            provider_id="broken",
            display_name="Broken",
            api_base_url="https://broken.example.com",
            iso_country_code="POL",
            supports_alphanumeric_sender=False,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="broken",
        api_base_url="https://broken.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=False,
        sender_id=None,
        credential_key="ops_sms_secret",
    )

    caplog.set_level("WARNING")
    config = bootstrap_module._resolve_sms_provider(
        settings,
        providers_module.get_sms_provider,
        channel_key="ops",
    )

    assert config.iso_country_code == "ZZ"
    assert "nieprawidłowy kod kraju" in caplog.text

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_rejects_unsupported_alphanumeric_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=False,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="OPS_TEAM",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "operator go nie obsługuje" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_validates_sender_id_length(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=6,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="TOO_LONG_ID",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "przekracza dopuszczalny limit" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_enforces_sender_id_min_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="OK",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "co najmniej trzy znaki" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_requires_sender_id_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="123456",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "co najmniej jedną literę" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_rejects_duplicate_recipients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=False,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="  +48123456789  ",
        recipients=("+48123456789", " +48123456789"),
        allow_alphanumeric_sender=False,
        sender_id=None,
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "zduplikowane numery odbiorców" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_requires_sender_id_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id=None,
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "nie dostarczono 'sender_id'" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_rejects_sender_id_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=False,
        sender_id="OPS",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "zawiera 'sender_id', lecz 'allow_alphanumeric_sender' pozostaje wyłączone" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_validates_sender_id_charset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=True,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="\u002b48123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=True,
        sender_id="OPS!",
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "może zawierać wyłącznie litery A-Z, cyfry 0-9" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_validates_numeric_sender_format(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=False,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="0048123456789",
        recipients=("+48123456789",),
        allow_alphanumeric_sender=False,
        sender_id=None,
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "formacie E.164" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_build_sms_channel_validates_recipient_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "bot_core.alerts.channels.providers"
    providers_module = sys.modules.get(module_name)
    if providers_module is None:
        providers_module = __import__(module_name, fromlist=["DEFAULT_SMS_PROVIDERS"])

    monkeypatch.setitem(
        providers_module.DEFAULT_SMS_PROVIDERS,
        "strict",
        providers_module.SmsProviderConfig(
            provider_id="strict",
            display_name="Strict SMS",
            api_base_url="https://strict.example.com",
            iso_country_code="US",
            supports_alphanumeric_sender=False,
            notes=None,
            max_sender_length=11,
        ),
    )

    bootstrap_module._get_alert_components.cache_clear()

    settings = SMSProviderSettings(
        name="ops",
        provider_key="strict",
        api_base_url="https://strict.example.com",
        from_number="+48123456789",
        recipients=("48123456789",),
        allow_alphanumeric_sender=False,
        sender_id=None,
        credential_key="ops_sms_secret",
    )

    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_secret_value(
        "ops_sms_secret",
        json.dumps({"account_sid": "AC", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError) as exc:
        bootstrap_module._build_sms_channel({"ops": settings}, "ops", secret_manager)

    assert "formacie E.164" in str(exc.value)

    bootstrap_module._get_alert_components.cache_clear()


def test_bootstrap_environment_offline_disables_network_channels(tmp_path: Path) -> None:
    runtime_metrics = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "log_sink": False,
    }
    runtime_risk_service = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics=runtime_metrics,
        runtime_risk_service=runtime_risk_service,
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "coinbase_offline", config_path=config_path, secret_manager=manager
    )

    assert context.environment.offline_mode is True
    assert context.alert_channels == {}
    assert len(context.alert_router.channels) == 0
    assert context.alert_router.throttle is None
    assert context.alert_router.audit_log is context.audit_log
    assert context.metrics_server is None
    assert context.metrics_service_enabled is False
    assert context.risk_server is None
    assert context.risk_service_enabled is False
    assert context.risk_snapshot_publisher is None


def test_bootstrap_environment_creates_signed_risk_decision_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_risk = {
        "path": "audit/risk_decisions.jsonl",
        "max_entries": 16,
        "signing_key_env": "BOT_CORE_RISK_KEY",
        "signing_key_id": "ci-risk",
    }
    config_path = _write_config_custom(tmp_path, runtime_risk=runtime_risk)
    monkeypatch.setenv("BOT_CORE_RISK_KEY", "risk-secret")
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert context.risk_decision_log is not None
    log_path = context.risk_decision_log.path
    assert log_path is not None

    context.risk_engine.apply_pre_trade_checks(
        OrderRequest(
            symbol="ETHUSDT",
            side="buy",
            quantity=0.5,
            order_type="market",
            price=1_500.0,
        ),
        account=AccountSnapshot(
            balances={"USDT": 50_000.0},
            total_equity=50_000.0,
            available_margin=50_000.0,
            maintenance_margin=0.0,
        ),
        profile_name=context.risk_profile_name,
    )

    assert log_path.exists()
    serialized = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert serialized
    payload = json.loads(serialized[0])
    assert payload["profile"] == context.risk_profile_name
    assert payload["signature"]["key_id"] == "ci-risk"


def test_bootstrap_environment_allows_risk_profile_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    storage, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper",
        config_path=config_path,
        secret_manager=manager,
        risk_profile_name="aggressive",
    )

    assert context.risk_profile_name == "aggressive"
    assert context.environment.risk_profile == "aggressive"
    assert isinstance(context.risk_manager_settings, RiskManagerSettings)
    assert context.risk_manager_settings.profile_name == "aggressive"
    assert context.risk_engine.should_liquidate(profile_name="aggressive") is False


def test_bootstrap_environment_detects_missing_permissions(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_path = _write_config(tmp_path)
    credentials_payload = {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "passphrase": None,
        "permissions": ["read"],
        "environment": Environment.PAPER.value,
    }
    storage.set_secret("tests:binance_paper_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")
    manager.store_secret_value(
        "smtp_credentials",
        json.dumps({"username": "bot", "password": "secret"}),
        purpose="alerts:email",
    )
    manager.store_secret_value(
        "sms_orange",
        json.dumps({"account_sid": "AC123", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError):
        bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)


def test_bootstrap_environment_supports_zonda(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_content = """
    risk_profiles:
      conservative:
        max_daily_loss_pct: 0.01
        max_position_pct: 0.03
        target_volatility: 0.07
        max_leverage: 2.0
        stop_loss_atr_multiple: 1.0
        max_open_positions: 3
        hard_drawdown_pct: 0.05
    environments:
      zonda_live:
        exchange: zonda_spot
        environment: live
        keychain_key: zonda_live_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_live
        risk_profile: conservative
        alert_channels: ["telegram:primary"]
        ip_allowlist: []
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123"
          token_secret: telegram_token
      email_channels: {}
      sms_providers: {}
      signal_channels: {}
      whatsapp_channels: {}
      messenger_channels: {}
    """

    license_bundle = _prepare_signed_license_bundle(tmp_path / "license")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["license"] = {
        "license_path": str(license_bundle["license"]),
        "fingerprint_path": str(license_bundle["fingerprint"]),
        "license_keys_path": str(license_bundle["license_keys"]),
        "fingerprint_keys_path": str(license_bundle["fingerprint_keys"]),
    }
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )

    credentials_payload = {
        "key_id": "zonda-key",
        "secret": "zonda-secret",
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret("tests:zonda_live_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")

    context = bootstrap_environment("zonda_live", config_path=config_path, secret_manager=manager)

    assert context.adapter.name == "zonda_spot"
    assert context.credentials.key_id == "zonda-key"
    assert context.environment.exchange == "zonda_spot"
    assert context.adapter_settings == {}


@pytest.mark.parametrize(
    ("exchange_name", "expected_class"),
    (
        ("coinbase_spot", CoinbaseSpotAdapter),
        ("okx_spot", OKXSpotAdapter),
        ("kucoin_spot", KuCoinSpotAdapter),
        ("bybit_spot", BybitSpotAdapter),
    ),
)
def test_instantiate_adapter_supports_new_ccxt_spot_exchanges(
    exchange_name: str, expected_class: type[ExchangeAdapter]
) -> None:
    credentials = ExchangeCredentials(
        key_id="demo",
        secret="demo-secret",
        environment=Environment.PAPER,
    )

    adapter = _instantiate_adapter(
        exchange_name,
        credentials,
        dict(_DEFAULT_ADAPTERS),
        Environment.PAPER,
        offline_mode=True,
    )

    assert isinstance(adapter, expected_class)
    assert adapter.name == exchange_name


def test_get_registered_adapter_factories_includes_new_ccxt_entries() -> None:
    factories = get_registered_adapter_factories()

    assert {
        "coinbase_spot",
        "okx_spot",
        "kucoin_spot",
        "bybit_spot",
    }.issubset(factories)


def test_register_adapter_factory_requires_override_flag() -> None:
    with pytest.raises(ValueError):
        register_adapter_factory("bybit_spot", KuCoinSpotAdapter)


def test_register_adapter_factory_override_and_cleanup() -> None:
    original = get_registered_adapter_factories()["bybit_spot"]

    try:
        register_adapter_factory("bybit_spot", KuCoinSpotAdapter, override=True)
        factories = get_registered_adapter_factories()
        assert factories["bybit_spot"] is KuCoinSpotAdapter
    finally:
        register_adapter_factory("bybit_spot", original, override=True)


def test_unregister_adapter_factory_removes_entry() -> None:
    register_adapter_factory("custom_exchange", KuCoinSpotAdapter, override=True)

    removed = unregister_adapter_factory("custom_exchange")

    assert removed is True
    assert "custom_exchange" not in get_registered_adapter_factories()


def test_register_adapter_factory_from_path_adds_callable() -> None:
    assert "custom_exchange" not in get_registered_adapter_factories()

    try:
        factory = register_adapter_factory_from_path(
            "custom_exchange",
            "bot_core.exchanges.bybit:BybitSpotAdapter",
        )
        assert factory is BybitSpotAdapter
        assert get_registered_adapter_factories()["custom_exchange"] is BybitSpotAdapter
    finally:
        unregister_adapter_factory("custom_exchange")


def test_register_adapter_factory_from_path_requires_override_for_existing() -> None:
    with pytest.raises(ValueError):
        register_adapter_factory_from_path(
            "bybit_spot",
            "bot_core.exchanges.kucoin:KuCoinSpotAdapter",
        )


def test_register_adapter_factory_from_path_override_existing_and_restore() -> None:
    original = get_registered_adapter_factories()["bybit_spot"]

    try:
        replacement = register_adapter_factory_from_path(
            "bybit_spot",
            "bot_core.exchanges.kucoin:KuCoinSpotAdapter",
            override=True,
        )
        assert replacement is KuCoinSpotAdapter
        assert get_registered_adapter_factories()["bybit_spot"] is KuCoinSpotAdapter
    finally:
        register_adapter_factory("bybit_spot", original, override=True)


def test_register_adapter_factory_from_path_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        register_adapter_factory_from_path(
            "custom_exchange",
            "bot_core.runtime.bootstrap:_DEFAULT_ADAPTERS",
        )


def test_temporary_adapter_factories_adds_and_restores() -> None:
    assert "custom_exchange" not in get_registered_adapter_factories()

    with temporary_adapter_factories(add={"custom_exchange": KuCoinSpotAdapter}) as factories:
        assert factories["custom_exchange"] is KuCoinSpotAdapter
        assert get_registered_adapter_factories()["custom_exchange"] is KuCoinSpotAdapter

    assert "custom_exchange" not in get_registered_adapter_factories()


def test_temporary_adapter_factories_requires_override_for_existing() -> None:
    with pytest.raises(ValueError):
        with temporary_adapter_factories(add={"bybit_spot": KuCoinSpotAdapter}):
            pass


def test_temporary_adapter_factories_override_existing_and_restore() -> None:
    original = get_registered_adapter_factories()["bybit_spot"]

    with temporary_adapter_factories(
        add={"bybit_spot": KuCoinSpotAdapter}, override=True
    ) as factories:
        assert factories["bybit_spot"] is KuCoinSpotAdapter
        assert get_registered_adapter_factories()["bybit_spot"] is KuCoinSpotAdapter

    assert get_registered_adapter_factories()["bybit_spot"] is original


def test_temporary_adapter_factories_can_remove_entries_temporarily() -> None:
    assert "bybit_spot" in get_registered_adapter_factories()

    with temporary_adapter_factories(remove=["bybit_spot"]):
        assert "bybit_spot" not in get_registered_adapter_factories()

    assert "bybit_spot" in get_registered_adapter_factories()


def test_environment_adapter_factories_specs_applied_to_factories(tmp_path: Path) -> None:
    config_path = _write_config_custom(
        tmp_path,
        environment_overrides={
            "exchange": "custom_exchange",
            "adapter_factories": {
                "custom_exchange": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter",
                "binance_spot": {
                    "path": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter",
                    "override": True,
                },
                "bybit_spot": None,
            },
        },
    )
    core_config = load_core_config(config_path)
    environment = core_config.environments["binance_paper"]

    factories = dict(_DEFAULT_ADAPTERS)
    _apply_adapter_factory_specs(
        factories,
        environment.adapter_factories,
        source="konfiguracji testowego środowiska",
        require_override=True,
    )

    credentials = ExchangeCredentials(
        key_id="demo",
        secret="demo",
        environment=Environment.PAPER,
    )
    adapter = _instantiate_adapter(
        "custom_exchange",
        credentials,
        factories,
        Environment.PAPER,
        offline_mode=True,
    )

    assert isinstance(adapter, NowaGieldaSpotAdapter)
    assert "bybit_spot" not in factories
    assert _DEFAULT_ADAPTERS["binance_spot"] is BinanceSpotAdapter


def test_environment_adapter_factories_require_override(tmp_path: Path) -> None:
    config_path = _write_config_custom(
        tmp_path,
        environment_overrides={
            "adapter_factories": {
                "binance_spot": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter",
            }
        },
    )
    core_config = load_core_config(config_path)
    environment = core_config.environments["binance_paper"]

    with pytest.raises(ValueError):
        _apply_adapter_factory_specs(
            dict(_DEFAULT_ADAPTERS),
            environment.adapter_factories,
            source="konfiguracji testowego środowiska",
            require_override=True,
        )


def test_adapter_factories_argument_accepts_string_paths() -> None:
    factories = dict(_DEFAULT_ADAPTERS)

    _apply_adapter_factory_specs(
        factories,
        {"binance_spot": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter"},
        source="parametrze 'adapter_factories'",
        require_override=False,
    )

    credentials = ExchangeCredentials(
        key_id="demo",
        secret="demo",
        environment=Environment.PAPER,
    )
    adapter = _instantiate_adapter(
        "binance_spot",
        credentials,
        factories,
        Environment.PAPER,
        offline_mode=True,
    )

    assert isinstance(adapter, NowaGieldaSpotAdapter)


def test_parse_adapter_factory_cli_specs_accepts_simple_paths() -> None:
    specs = parse_adapter_factory_cli_specs(
        ["binance_spot=bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter"]
    )

    assert specs == {
        "binance_spot": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter"
    }


def test_parse_adapter_factory_cli_specs_supports_remove_flag() -> None:
    specs = parse_adapter_factory_cli_specs(["kucoin_spot=!remove"])

    assert specs == {"kucoin_spot": {"remove": True}}


def test_parse_adapter_factory_cli_specs_supports_inline_json() -> None:
    specs = parse_adapter_factory_cli_specs(
        ['custom={"path": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter", "override": true}']
    )

    assert specs == {
        "custom": {
            "path": "bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter",
            "override": True,
        }
    }


def test_parse_adapter_factory_cli_specs_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        parse_adapter_factory_cli_specs(["binance_spot"])


def test_parse_adapter_factory_cli_specs_rejects_duplicate_entries() -> None:
    with pytest.raises(ValueError):
        parse_adapter_factory_cli_specs([
            "binance_spot=bot_core.exchanges.nowa_gielda:NowaGieldaSpotAdapter",
            "binance_spot=bot_core.exchanges.bybit:BybitSpotAdapter",
        ])


def test_bootstrap_environment_applies_permission_profile_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "nowa_gielda_paper", config_path=config_path, secret_manager=manager
    )

    assert context.environment.permission_profile == "trading_default"
    assert tuple(context.environment.required_permissions) == ("read", "trade")
    assert tuple(context.environment.forbidden_permissions) == ("withdraw",)


def test_bootstrap_metrics_ui_alerts_audit_requested_file_without_backend(tmp_path: Path) -> None:
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={"enabled": True, "ui_alerts_audit_backend": "file"},
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "file"
    assert audit_info["backend"] == "memory"
    assert audit_info["note"] == "file_backend_unavailable"


def test_bootstrap_metrics_ui_alerts_audit_inherits_file_backend(tmp_path: Path) -> None:
    alerts_dir = tmp_path / "ui-alert-audit"
    runtime_metrics = {
        "enabled": True,
        "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
    }
    environment_overrides = {
        "alert_audit": {
            "backend": "file",
            "directory": str(alerts_dir),
            "filename_pattern": "alerts-%Y%m%d.jsonl",
            "retention_days": 14,
            "fsync": True,
        }
    }
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics=runtime_metrics,
        environment_overrides=environment_overrides,
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "inherit"
    assert audit_info["backend"] == "file"
    assert audit_info["note"] == "inherited_environment_router"
    assert audit_info["directory"] == str(alerts_dir)
    assert audit_info["pattern"] == "alerts-%Y%m%d.jsonl"
    assert audit_info["retention_days"] == 14
    assert audit_info["fsync"] is True


def test_bootstrap_metrics_ui_alerts_audit_memory_request_on_file_backend(tmp_path: Path) -> None:
    alerts_dir = tmp_path / "ui-alert-audit"
    runtime_metrics = {
        "enabled": True,
        "ui_alerts_audit_backend": "memory",
        "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
    }
    environment_overrides = {
        "alert_audit": {
            "backend": "file",
            "directory": str(alerts_dir),
            "filename_pattern": "alerts-%Y%m%d.jsonl",
            "retention_days": 7,
            "fsync": False,
        }
    }
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics=runtime_metrics,
        environment_overrides=environment_overrides,
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "memory"
    assert audit_info["backend"] == "file"
    assert audit_info["note"] == "memory_backend_not_selected"


def test_bootstrap_metrics_risk_profiles_file_metadata(tmp_path: Path) -> None:
    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "balanced": {
                        "metrics_service_overrides": {
                            "ui_alerts_overlay_critical_threshold": 7,
                        },
                        "severity_min": "notice",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "jsonl_path": str((tmp_path / "metrics.jsonl").resolve()),
            "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
            "reduce_motion_alerts": True,
            "overlay_alerts": True,
            "jank_alerts": False,
            "ui_alerts_risk_profile": "balanced",
            "ui_alerts_risk_profiles_file": str(profiles_path),
        },
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert context.metrics_ui_alerts_settings is not None
    settings = context.metrics_ui_alerts_settings
    risk_meta = settings.get("risk_profile")
    assert isinstance(risk_meta, dict)
    assert risk_meta.get("name") == "balanced"
    summary_meta = settings.get("risk_profile_summary")
    assert isinstance(summary_meta, dict)
    assert summary_meta.get("name") == "balanced"
    file_meta = settings.get("risk_profiles_file")
    assert isinstance(file_meta, dict)
    assert Path(str(file_meta.get("path"))).resolve() == profiles_path.resolve()
    assert "balanced" in file_meta.get("registered_profiles", [])


def test_bootstrap_metrics_risk_profiles_directory_metadata(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "ops.json").write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "ops_dir": {
                        "metrics_service_overrides": {
                            "ui_alerts_overlay_critical_threshold": 5
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (profiles_dir / "lab.yaml").write_text(
        "risk_profiles:\n  lab_dir:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "ui_alerts_risk_profile": "balanced",
            "ui_alerts_risk_profiles_file": str(profiles_dir),
        },
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    settings = context.metrics_ui_alerts_settings
    assert isinstance(settings, dict)
    summary_meta = settings.get("risk_profile_summary")
    assert isinstance(summary_meta, dict)
    assert summary_meta.get("name") == "balanced"
    file_meta = settings.get("risk_profiles_file")
    assert isinstance(file_meta, dict)
    assert file_meta["type"] == "directory"
    assert file_meta["path"] == str(profiles_dir)
    assert "ops_dir" in file_meta.get("registered_profiles", [])
    assert any(entry["path"].endswith("lab.yaml") for entry in file_meta.get("files", []))


def test_bootstrap_collects_risk_security_metadata(tmp_path: Path) -> None:
    tls_dir = tmp_path / "tls"
    tls_dir.mkdir()
    cert_path = tls_dir / "risk_cert.pem"
    key_path = tls_dir / "risk_key.pem"
    cert_path.write_text(_RISK_TLS_CERT, encoding="utf-8")
    key_path.write_text(_RISK_TLS_KEY, encoding="utf-8")

    config_path = _write_config_custom(
        tmp_path,
        runtime_risk_service={
            "enabled": False,
            "tls": {
                "certificate_path": str(cert_path),
                "private_key_path": str(key_path),
            },
        },
    )

    _, manager = _prepare_manager()
    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert context.risk_security_metadata is not None
    tls_meta = context.risk_security_metadata.get("tls")
    assert isinstance(tls_meta, Mapping)
    certificate_meta = tls_meta.get("certificate")
    assert isinstance(certificate_meta, Mapping)
    certificates = certificate_meta.get("certificates")
    assert isinstance(certificates, list) and certificates
    fingerprint = certificates[0]["fingerprint_sha256"]
    assert fingerprint == "a49ede6616a62c76e2affdf692aa103cfeb89ddbd1f0f03b13a8a3166aa63079"

    auth_meta = context.risk_security_metadata.get("auth")
    assert isinstance(auth_meta, Mapping)
    assert auth_meta.get("token_configured") is False
    assert context.risk_token_validator is None

    # Ostrzeżenia wynikają z luźnych uprawnień plików w katalogu tymczasowym
    warnings = context.risk_security_warnings
    assert warnings is not None
    assert any("Klucz prywatny TLS" in warning for warning in warnings)


def test_bootstrap_collects_metrics_security_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tls_dir = tmp_path / "metrics_tls"
    tls_dir.mkdir()
    cert_path = tls_dir / "metrics_cert.pem"
    key_path = tls_dir / "metrics_key.pem"
    cert_path.write_text(_RISK_TLS_CERT, encoding="utf-8")
    key_path.write_text(_RISK_TLS_KEY, encoding="utf-8")

    if "bot_core.runtime.bootstrap" in sys.modules:
        monkeypatch.setattr(
            sys.modules["bot_core.runtime.bootstrap"],
            "build_metrics_server_from_config",
            lambda *_, **__: None,
        )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "auth_token": "",
            "tls": {
                "enabled": True,
                "certificate_path": str(cert_path),
                "private_key_path": str(key_path),
            },
        },
    )

    _, manager = _prepare_manager()
    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    metadata = context.metrics_security_metadata
    assert isinstance(metadata, Mapping)
    tls_meta = metadata.get("tls")
    assert isinstance(tls_meta, Mapping)
    certificates = tls_meta.get("certificate", {}).get("certificates", [])
    assert certificates
    assert certificates[0]["fingerprint_sha256"] == "a49ede6616a62c76e2affdf692aa103cfeb89ddbd1f0f03b13a8a3166aa63079"

    auth_meta = metadata.get("auth")
    assert isinstance(auth_meta, Mapping)
    assert auth_meta.get("token_configured") is False
    assert context.metrics_token_validator is None

    warnings = context.metrics_security_warnings
    assert warnings is not None
    assert any("MetricsService ma włączone API bez tokenu" in warning for warning in warnings)


def test_bootstrap_warns_on_over_permissive_metrics_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "metrics.token"
    token_file.write_text("secret", encoding="utf-8")
    token_file.chmod(0o644)

    if "bot_core.runtime.bootstrap" in sys.modules:
        monkeypatch.setattr(
            sys.modules["bot_core.runtime.bootstrap"],
            "build_metrics_server_from_config",
            lambda *_, **__: None,
        )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "auth_token_file": str(token_file),
        },
    )

    _, manager = _prepare_manager()
    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    metadata = context.metrics_security_metadata
    assert isinstance(metadata, Mapping)
    auth_meta = metadata.get("auth")
    assert isinstance(auth_meta, Mapping)
    assert auth_meta.get("token_file") == str(token_file)
    assert auth_meta.get("token_file_exists") is True
    assert auth_meta.get("token_file_permissions") == "0o644"

    warnings = context.metrics_security_warnings
    assert warnings is not None
    assert any("zbyt szerokie uprawnienia" in warning for warning in warnings)


def test_bootstrap_loads_tco_report_for_decision_engine(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config_data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tco_report_path = reports_dir / "stage5_tco.json"
    tco_report_payload = {
        "total": {"cost_bps": 9.5},
        "strategies": {
            "mean_reversion": {
                "total": {"cost_bps": 14.0},
                "profiles": {"balanced": {"cost_bps": 11.0}},
            }
        },
    }
    tco_report_path.write_text(json.dumps(tco_report_payload), encoding="utf-8")

    config_data["decision_engine"] = {
        "orchestrator": {
            "max_cost_bps": 20.0,
            "min_net_edge_bps": 5.0,
            "max_daily_loss_pct": 0.2,
            "max_drawdown_pct": 0.3,
            "max_position_ratio": 3.0,
            "max_open_positions": 10,
            "max_latency_ms": 200.0,
            "max_trade_notional": 20000.0,
        },
        "min_probability": 0.4,
        "require_cost_data": True,
        "tco": {
            "reports": ["reports/stage5_tco.json"],
            "require_at_startup": True,
        },
    }
    Path(config_path).write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )

    _, manager = _prepare_manager()
    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert context.decision_orchestrator is not None
    assert context.decision_tco_report_path == str(tco_report_path.resolve())
    assert context.decision_tco_warnings is None

    candidate = DecisionCandidate(
        strategy="mean_reversion",
        action="enter",
        risk_profile="balanced",
        symbol="BTCUSDT",
        notional=2000.0,
        expected_return_bps=25.0,
        expected_probability=0.7,
        latency_ms=90.0,
    )
    snapshot = RiskSnapshot(
        profile="balanced",
        start_of_day_equity=100000.0,
        daily_realized_pnl=0.0,
        peak_equity=100000.0,
        last_equity=100000.0,
        gross_notional=1000.0,
        active_positions=1,
        symbols=("BTCUSDT",),
        force_liquidation=False,
    )

    evaluation = context.decision_orchestrator.evaluate_candidate(candidate, snapshot)
    assert evaluation.accepted is True
    assert evaluation.cost_bps == pytest.approx(11.0)


def test_bootstrap_runtime_tco_reporter_clears_after_export(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config_data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tco_report_path = reports_dir / "stage5_tco.json"
    tco_report_path.write_text(
        json.dumps(
            {
                "generated_at": "2024-04-01T00:00:00Z",
                "total": {"cost_bps": 5.0},
                "strategies": {},
            }
        ),
        encoding="utf-8",
    )

    runtime_dir = tmp_path / "runtime" / "tco"
    config_data["decision_engine"] = {
        "orchestrator": {
            "max_cost_bps": 12.0,
            "min_net_edge_bps": 3.0,
            "max_daily_loss_pct": 0.02,
            "max_drawdown_pct": 0.05,
            "max_position_ratio": 0.25,
            "max_open_positions": 5,
            "max_latency_ms": 200.0,
        },
        "min_probability": 0.5,
        "require_cost_data": True,
        "tco": {
            "reports": [f"reports/{tco_report_path.name}"],
            "runtime_enabled": True,
            "runtime_report_directory": str(runtime_dir),
            "runtime_clear_after_export": True,
        },
    }
    Path(config_path).write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )

    _, manager = _prepare_manager()
    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    reporter = context.tco_reporter
    assert reporter is not None

    reporter.record_execution(
        strategy="trend_follow",
        risk_profile="balanced",
        instrument="BTCUSDT",
        exchange="binance",
        side="buy",
        quantity=1.0,
        executed_price=20000.0,
        reference_price=19995.0,
        commission=2.0,
    )
    assert reporter.events()

    artifacts = reporter.export()

    assert artifacts is not None
    assert reporter.events() == ()
    assert runtime_dir.exists()
    assert any(runtime_dir.glob("*.json"))


class _StubOrchestrator:
    def __init__(self) -> None:
        self.reports: list[object] = []

    def update_costs_from_report(self, payload: object) -> None:
        self.reports.append(payload)


def _tco_config_with_report(
    report_path: Path,
    *,
    warn_hours: float | None,
    max_hours: float | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tco=DecisionEngineTCOConfig(
            report_paths=(str(report_path),),
            warn_report_age_hours=warn_hours,
            max_report_age_hours=max_hours,
        )
    )


def _write_document(root: Path, relative_path: str, *, content: str) -> str:
    data = content.encode("utf-8")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_tco_report(path: Path) -> None:
    payload = {"generated_at": "2025-01-01T00:00:00Z", "total_cost_bps": 12.5}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_initial_tco_costs_warns_about_stale_report(tmp_path: Path) -> None:
    report_path = tmp_path / "tco.json"
    _write_tco_report(report_path)
    two_hours_ago = time.time() - 2 * 3600
    os.utime(report_path, (two_hours_ago, two_hours_ago))

    orchestrator = _StubOrchestrator()
    config = _tco_config_with_report(
        report_path,
        warn_hours=1.0,
        max_hours=5.0,
    )

    loaded_path, warnings = _load_initial_tco_costs(config, orchestrator, None)

    assert loaded_path == str(report_path)
    assert orchestrator.reports, "stale raport powinien zostać załadowany mimo ostrzeżenia"
    assert any(
        entry.startswith(f"stale_warning:{report_path}") for entry in warnings
    )


def test_load_initial_tco_costs_skips_report_past_max_age(tmp_path: Path) -> None:
    report_path = tmp_path / "expired_tco.json"
    _write_tco_report(report_path)
    ten_hours_ago = time.time() - 10 * 3600
    os.utime(report_path, (ten_hours_ago, ten_hours_ago))

    orchestrator = _StubOrchestrator()
    config = _tco_config_with_report(
        report_path,
        warn_hours=1.0,
        max_hours=2.0,
    )

    loaded_path, warnings = _load_initial_tco_costs(config, orchestrator, None)

    assert loaded_path is None
    assert not orchestrator.reports
    assert any(
        entry.startswith(f"stale_critical:{report_path}") for entry in warnings
    )
