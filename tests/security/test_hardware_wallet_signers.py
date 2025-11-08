import pytest

from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import bot_core.security.signing as signing
from bot_core.security.hardware_wallets import LedgerSigner, TrezorSigner
from bot_core.security.signing import (
    HmacTransactionSigner,
    TransactionSignerSelector,
    build_transaction_signer_selector,
)


@pytest.fixture(autouse=True)
def enable_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_HW_SIMULATOR", "1")


def test_ledger_signer_produces_deterministic_signature() -> None:
    payload = {"operation": "withdrawal", "amount": 10.5, "currency": "USDT"}
    signer = LedgerSigner(seed=b"ledger-test", key_id="ledger-main")
    signature = signer.sign(payload)
    assert signature["algorithm"] == "LEDGER-ECDSA"
    assert signature["key_id"] == "ledger-main"
    assert signer.verify(payload, signature)
    assert signer.requires_hardware is True


def test_trezor_signer_uses_ed25519_scheme() -> None:
    payload = {"operation": "withdrawal", "address": "0xabc", "amount": 2.0}
    signer = TrezorSigner(seed=b"trezor-test", key_id="trezor-main")
    signature = signer.sign(payload)
    assert signature["algorithm"] == "TREZOR-EDDSA"
    assert signature["key_id"] == "trezor-main"
    assert signer.verify(payload, signature)


def test_transaction_signer_selector_prefers_override() -> None:
    default = HmacTransactionSigner(key=b"default-secret", key_id="default")
    selector = TransactionSignerSelector(default=default)
    selector.register("ledger", LedgerSigner(seed=b"ledger-selector"))

    assert selector.resolve("ledger").algorithm == "LEDGER-ECDSA"
    assert selector.resolve("unknown").algorithm == "HMAC-SHA256"


def test_transaction_signer_selector_iterates_all_signers() -> None:
    default = HmacTransactionSigner(key=b"default-secret", key_id="default")
    ledger = LedgerSigner(seed=b"ledger-selector")
    selector = TransactionSignerSelector(default=default, overrides={"ledger": ledger})

    entries = list(selector.iter_signers())
    assert (None, default) in entries
    assert ("ledger", ledger) in entries


def test_hardware_signers_close_idempotent() -> None:
    ledger = LedgerSigner(seed=b"ledger-close")
    trezor = TrezorSigner(seed=b"trezor-close")

    ledger.close()
    ledger.close()
    trezor.close()
    trezor.close()


def test_signers_support_context_manager() -> None:
    payload = {"operation": "withdrawal", "amount": 1}
    with LedgerSigner(seed=b"ledger-cm") as ledger:
        assert ledger.sign(payload)["algorithm"] == "LEDGER-ECDSA"
    with TrezorSigner(seed=b"trezor-cm") as trezor:
        assert trezor.sign(payload)["algorithm"] == "TREZOR-EDDSA"


def test_transaction_signer_selector_close_closes_all_unique_signers() -> None:
    class DummySigner(HmacTransactionSigner):
        def __init__(self) -> None:
            super().__init__(key=b"dummy")
            self.closed = 0

        def close(self) -> None:  # type: ignore[override]
            self.closed += 1

    dummy = DummySigner()
    selector = TransactionSignerSelector(default=dummy, overrides={"another": dummy})
    selector.close()

    assert dummy.closed == 1


def test_build_transaction_signer_selector_closes_partial_signers(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    class TrackingSigner:
        algorithm = "TRACK"
        key_id = None

        def __init__(self) -> None:
            self.closed = False

        def sign(self, payload: signing.JsonPayload) -> Mapping[str, str]:
            return {"algorithm": self.algorithm, "value": "dummy"}

        def close(self) -> None:
            self.closed = True

    def fake_builder(config: Mapping[str, Any]) -> TrackingSigner:
        signer = TrackingSigner()
        built.append(signer)
        if config.get("type") == "fail":
            raise ValueError("boom")
        return signer

    monkeypatch.setattr(signing, "build_transaction_signer_from_config", fake_builder)

    with pytest.raises(ValueError):
        build_transaction_signer_selector(
            {
                "default": {"type": "ok"},
                "accounts": {
                    "bad": {"type": "fail"},
                },
            }
        )

    assert built, "powinien zostać zbudowany przynajmniej jeden signer"
    assert getattr(built[0], "closed", False) is True


def test_signers_expose_description_metadata() -> None:
    ledger = LedgerSigner(seed=b"ledger-describe", key_id="ledger-main")
    trezor = TrezorSigner(seed=b"trezor-describe", key_id="trezor-main")
    hmac_signer = HmacTransactionSigner(key=b"secret", key_id="hmac")

    ledger_info = dict(ledger.describe())
    trezor_info = dict(trezor.describe())
    hmac_info = dict(hmac_signer.describe())

    assert ledger_info["algorithm"] == "LEDGER-ECDSA"
    assert ledger_info["requires_hardware"] is True
    assert ledger_info["device_curve"] == "secp256k1"
    assert ledger_info["derivation_path"].startswith("m/")

    assert trezor_info["algorithm"] == "TREZOR-EDDSA"
    assert trezor_info["requires_hardware"] is True
    assert trezor_info["device_curve"] == "ed25519"

    assert hmac_info["algorithm"] == "HMAC-SHA256"
    assert hmac_info["requires_hardware"] is False
    assert hmac_info["key_id"] == "hmac"


def test_selector_describe_signers_deduplicates_instances() -> None:
    default = HmacTransactionSigner(key=b"default-secret", key_id="default")
    ledger = LedgerSigner(seed=b"ledger-selector")
    selector = TransactionSignerSelector(default=default, overrides={"ledger": ledger, "alias": ledger})

    descriptions = selector.describe_signers()
    assert descriptions[None]["algorithm"] == "HMAC-SHA256"
    assert descriptions[None]["requires_hardware"] is False

    ledger_description = descriptions["ledger"]
    alias_description = descriptions["alias"]
    assert ledger_description["algorithm"] == "LEDGER-ECDSA"
    assert ledger_description["requires_hardware"] is True
    assert ledger_description is alias_description


def test_selector_describe_signers_handles_errors(caplog: pytest.LogCaptureFixture) -> None:
    class FailingSigner(HmacTransactionSigner):
        def describe(self) -> Mapping[str, Any]:  # type: ignore[override]
            raise RuntimeError("boom")

    signer = FailingSigner(key=b"x")
    selector = TransactionSignerSelector(default=signer)

    with caplog.at_level("DEBUG"):
        info = selector.describe_signers()

    assert info[None]["algorithm"] == "HMAC-SHA256"
    assert info[None]["requires_hardware"] is False
    assert any("Nie udało się pobrać opisu podpisującego" in record.message for record in caplog.records)


def test_selector_verify_returns_true_for_matching_signature() -> None:
    selector = TransactionSignerSelector(overrides={"ledger": LedgerSigner(seed=b"verify-ledger")})
    signer = selector.resolve("ledger")
    assert signer is not None
    payload = {
        "operation": "withdrawal",
        "amount": 10,
    }
    signature = signer.sign(payload)

    assert selector.verify("ledger", payload, signature) is True


def test_selector_describe_audit_bundle_highlights_issues() -> None:
    selector = TransactionSignerSelector(
        default=LedgerSigner(seed=b"bundle-ledger", key_id="ledger"),
        overrides={
            "software": HmacTransactionSigner(key=b"secret"),
            "missing": LedgerSigner(seed=b"bundle-missing"),
        },
    )

    bundle = selector.describe_audit_bundle()

    assert "signers" in bundle
    assert "key_index" in bundle
    assert "hardware_requirements" in bundle
    issues = bundle["issues"]
    assert isinstance(issues, tuple)
    issue_types = {issue["type"] for issue in issues}
    assert "software_signer" in issue_types
    assert "missing_key_id" in issue_types
    severities = {issue["type"]: issue["severity"] for issue in issues}
    assert severities["software_signer"] == "warning"
    assert severities["missing_key_id"] == "warning"


def test_selector_describe_audit_bundle_reports_conflicting_key_id() -> None:
    ledger = LedgerSigner(seed=b"conflict-ledger", key_id="shared")
    software = HmacTransactionSigner(key=b"conflict", key_id="shared")
    selector = TransactionSignerSelector(default=ledger, overrides={"software": software})

    bundle = selector.describe_audit_bundle()
    issues = bundle["issues"]

    conflict_types = [issue["type"] for issue in issues]
    assert "key_id_algorithm_conflict" in conflict_types
    assert "key_id_hardware_mismatch" in conflict_types

    conflict_issue = next(issue for issue in issues if issue["type"] == "key_id_algorithm_conflict")
    assert conflict_issue["key_id"] == "shared"
    assert conflict_issue["severity"] == "critical"
    assert set(conflict_issue["algorithms"]) == {"LEDGER-ECDSA", "HMAC-SHA256"}

    mismatch_issue = next(issue for issue in issues if issue["type"] == "key_id_hardware_mismatch")
    assert mismatch_issue["key_id"] == "shared"
    assert mismatch_issue["severity"] == "warning"
    assert set(mismatch_issue["hardware_modes"]) == {True, False}

def test_selector_verify_logs_and_returns_false_on_error(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingSigner(HmacTransactionSigner):
        def verify(self, payload: Any, signature: Mapping[str, Any]) -> bool:  # type: ignore[override]
            raise RuntimeError("verification boom")

    signer = ExplodingSigner(key=b"boom")
    selector = TransactionSignerSelector(default=signer)

    with caplog.at_level("DEBUG"):
        assert selector.verify(None, {"operation": "withdrawal"}, {"algorithm": "HMAC-SHA256"}) is False

    assert any("Nie udało się zweryfikować podpisu" in record.message for record in caplog.records)


def test_ledger_verify_uses_metadata_public_key() -> None:
    payload = {"operation": "withdrawal", "amount": 7.0, "currency": "USDC"}
    producer = LedgerSigner(seed=b"ledger-metadata-producer")
    signature = producer.sign(payload)

    verifier = LedgerSigner(seed=b"ledger-metadata-verifier")
    assert verifier.verify(payload, signature) is True

    signature_without_xy = dict(signature)
    x_value = signature_without_xy.pop("device_public_x", None)
    y_value = signature_without_xy.pop("device_public_y", None)
    assert x_value is not None and y_value is not None
    numbers = ec.EllipticCurvePublicNumbers(
        int(str(x_value), 16),
        int(str(y_value), 16),
        ec.SECP256K1(),
    )
    signature_without_xy["device_public_key"] = numbers.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    ).hex()

    assert verifier.verify(payload, signature_without_xy) is True

    missing_metadata_signature = {"algorithm": "LEDGER-ECDSA", "value": signature["value"]}
    assert verifier.verify(payload, missing_metadata_signature) is False


def test_trezor_verify_requires_public_key_metadata() -> None:
    payload = {"operation": "withdrawal", "address": "0x123", "amount": 1.5}
    producer = TrezorSigner(seed=b"trezor-metadata-producer")
    signature = producer.sign(payload)

    verifier = TrezorSigner(seed=b"trezor-metadata-verifier")
    assert verifier.verify(payload, signature) is True

    missing_metadata_signature = {"algorithm": "TREZOR-EDDSA", "value": signature["value"]}
    assert verifier.verify(payload, missing_metadata_signature) is False


def test_selector_verify_falls_back_to_key_id_when_account_unknown() -> None:
    payload = {
        "operation": "withdrawal",
        "account": "primary",
        "exchange": "demo",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    signer = LedgerSigner(seed=b"ledger-key-fallback", key_id="ledger-fallback")
    selector = TransactionSignerSelector(overrides={"primary": signer})

    signature = signer.sign(payload)

    assert selector.verify("secondary", payload, signature) is True


def test_selector_resolve_by_key_id_returns_all_candidates() -> None:
    first = HmacTransactionSigner(key=b"first", key_id="dup")
    second = HmacTransactionSigner(key=b"second", key_id="dup")
    selector = TransactionSignerSelector(default=first, overrides={"alt": second})

    resolved = selector.resolve_by_key_id("dup")
    assert resolved == (first, second)

    # ensure mapping is cached and reused when requesting the same key again
    assert selector.resolve_by_key_id("dup ") == (first, second)


def test_selector_resolve_by_key_id_updates_after_register() -> None:
    default = HmacTransactionSigner(key=b"default", key_id="default")
    selector = TransactionSignerSelector(default=default)

    assert selector.resolve_by_key_id("new") == ()

    second = HmacTransactionSigner(key=b"second", key_id="new")
    selector.register("alt", second)

    assert selector.resolve_by_key_id("new") == (second,)

    # verify() should leverage the key index for fallback lookup
    payload = {"operation": "withdrawal", "id": 1}
    signature = second.sign(payload)
    assert selector.verify("unknown", payload, signature) is True


def test_selector_describe_key_index_groups_accounts_and_metadata() -> None:
    default = HmacTransactionSigner(key=b"default", key_id="shared")
    backup = HmacTransactionSigner(key=b"backup", key_id="shared")
    ledger = LedgerSigner(seed=b"ledger-key-index", key_id="ledger-main")

    selector = TransactionSignerSelector(
        default=default,
        overrides={
            "backup": backup,
            "ledger": ledger,
            "alias": ledger,
        },
    )

    summary = selector.describe_key_index()

    shared = summary["shared"]
    assert shared["accounts"] == (None, "backup")
    assert shared["account_count"] == 2
    assert shared["signer_count"] == 2
    assert shared["algorithms"] == ("HMAC-SHA256",)
    assert shared["requires_hardware"] is False
    assert shared["hardware_modes"] == (False,)
    assert shared["mixed_hardware"] is False

    ledger_info = summary["ledger-main"]
    assert ledger_info["accounts"] == ("ledger", "alias")
    assert ledger_info["signer_count"] == 1
    assert ledger_info["requires_hardware"] is True
    assert ledger_info["hardware_modes"] == (True,)
    assert ledger_info["mixed_hardware"] is False


def test_selector_describe_hardware_requirements_categorizes_accounts() -> None:
    default = LedgerSigner(seed=b"ledger-hw-summary", key_id="ledger-default")
    backup = HmacTransactionSigner(key=b"backup")
    ledger = LedgerSigner(seed=b"ledger-hw-shared", key_id="ledger-shared")

    selector = TransactionSignerSelector(
        default=default,
        overrides={
            "backup": backup,
            "ledger": ledger,
            "alias": ledger,
        },
    )

    summary = selector.describe_hardware_requirements()

    assert summary["accounts"] == (None, "backup", "ledger", "alias")
    assert summary["total_accounts"] == 4
    assert summary["hardware_accounts"] == (None, "ledger", "alias")
    assert summary["hardware_account_count"] == 3
    assert summary["software_accounts"] == ("backup",)
    assert summary["software_account_count"] == 1
    assert summary["missing_key_id_accounts"] == ("backup",)
    assert summary["missing_key_id_count"] == 1
    assert summary["all_require_hardware"] is False
