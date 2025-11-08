"""Obsługa licencji offline podpisanych kluczem Ed25519."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from bot_core.security.capabilities import LicenseCapabilities, build_capabilities_from_payload
from bot_core.security.clock import ClockService
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.fingerprint import (
    FingerprintError,
    sign_license_payload,
    verify_license_payload_signature,
)


LOGGER = logging.getLogger(__name__)


class LicenseServiceError(RuntimeError):
    """Błąd ogólny związany z obsługą licencji offline."""


class LicenseSignatureError(LicenseServiceError):
    """Podpis licencji jest nieprawidłowy."""


class LicenseBundleError(LicenseServiceError):
    """Plik licencji ma niepoprawny format."""


class LicenseHardwareMismatchError(LicenseServiceError):
    """Fingerprint urządzenia nie zgadza się z licencją."""


class LicenseStateTamperedError(LicenseServiceError):
    """Plik stanu licencji został zmodyfikowany lub podpis nie jest poprawny."""


class LicenseRollbackDetectedError(LicenseServiceError):
    """Wykryto próbę wgrania starszej licencji niż ostatnio zaakceptowana."""


@dataclass(slots=True, init=False)
class LicenseSnapshot:
    """Wyliczone możliwości i surowy payload licencji."""

    bundle_path: Path
    payload: Mapping[str, Any]
    payload_bytes: bytes
    payload_sha256: str
    signature_bytes: bytes
    capabilities: LicenseCapabilities
    effective_date: date
    local_hwid: str | None
    requires_hardware_wallet: bool

    def __init__(
        self,
        *,
        bundle_path: Path,
        payload: Mapping[str, Any],
        payload_bytes: bytes,
        signature_bytes: bytes,
        capabilities: LicenseCapabilities,
        effective_date: date,
        local_hwid: str | None,
        payload_sha256: str | None = None,
        requires_hardware_wallet: bool | None = None,
    ) -> None:
        self.bundle_path = bundle_path
        self.payload = payload
        self.payload_bytes = payload_bytes
        self.signature_bytes = signature_bytes
        self.capabilities = capabilities
        self.effective_date = effective_date
        self.local_hwid = local_hwid
        self.requires_hardware_wallet = bool(
            requires_hardware_wallet if requires_hardware_wallet is not None else capabilities.require_hardware_wallet_for_outgoing
        )
        digest = payload_sha256 or hashlib.sha256(payload_bytes).hexdigest()
        self.payload_sha256 = digest.lower()


@dataclass(slots=True)
class _MonotonicState:
    license_id: str | None
    sequence: int | None
    issued_at: datetime | None
    effective_date: date | None
    payload_sha256: str | None
    requires_hardware_wallet: bool | None


class LicenseService:
    """Ładuje licencje offline i buduje warstwę capabilities."""

    DEFAULT_STATE_PATH = Path("var/security/license_state.json")
    DEFAULT_STATUS_PATH = Path("var/security/license_status.json")
    DEFAULT_AUDIT_LOG_PATH = Path("logs/security_admin.log")
    PUBLIC_KEY_ENV = "BOT_CORE_LICENSE_PUBLIC_KEY"

    def __init__(
        self,
        *,
        verify_key_hex: str | None = None,
        state_path: str | Path | None = None,
        status_path: str | Path | None = None,
        audit_log_path: str | Path | None = None,
        today_provider: Callable[[], date] | None = None,
        clock_service: ClockService | None = None,
        hwid_provider: HwIdProvider | None = None,
        binding_secret_path: str | Path | None = None,
    ) -> None:
        if verify_key_hex is None:
            verify_key_hex = os.environ.get(self.PUBLIC_KEY_ENV)
        if not verify_key_hex:
            raise LicenseServiceError(
                "Brak publicznego klucza Ed25519 do weryfikacji licencji (BOT_CORE_LICENSE_PUBLIC_KEY)."
            )
        try:
            self._verify_key = VerifyKey(bytes.fromhex(verify_key_hex))
        except ValueError as exc:  # pragma: no cover - walidacja konfiguracji
            raise LicenseServiceError("Niepoprawny publiczny klucz Ed25519 (hex).") from exc
        if clock_service is not None:
            self._clock = clock_service
        else:
            self._clock = ClockService(
                state_path=state_path or self.DEFAULT_STATE_PATH,
                today_provider=today_provider,
            )
        self._hwid_provider = hwid_provider or HwIdProvider()
        self._status_path = Path(status_path).expanduser() if status_path else self.DEFAULT_STATUS_PATH
        self._audit_log_path = (
            Path(audit_log_path).expanduser() if audit_log_path else self.DEFAULT_AUDIT_LOG_PATH
        )
        self._binding_secret_path = (
            Path(binding_secret_path).expanduser()
            if binding_secret_path is not None
            else None
        )

    # --- API publiczne ---------------------------------------------------------
    def load_from_file(
        self,
        path: str | Path,
        *,
        expected_hwid: str | None = None,
    ) -> LicenseSnapshot:
        bundle_path = Path(path).expanduser()
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)

        bundle = self._read_bundle(bundle_path)
        payload_bytes = bundle["payload_bytes"]
        signature_bytes = bundle["signature_bytes"]

        try:
            self._verify_key.verify(payload_bytes, signature_bytes)
        except BadSignatureError as exc:
            raise LicenseSignatureError("Niepoprawny podpis licencji.") from exc

        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LicenseBundleError("Payload licencji nie zawiera poprawnego JSON.") from exc

        hwid = str(payload.get("hwid") or "").strip() or None
        local_hwid: str | None = None
        hwid_error: HwIdProviderError | None = None
        try:
            local_hwid = self._hwid_provider.read()
        except HwIdProviderError as exc:
            hwid_error = exc
            log_method = LOGGER.warning if (hwid or expected_hwid) else LOGGER.debug
            log_method("Nie udało się odczytać fingerprintu urządzenia: %s", exc)

        reference_hwid = expected_hwid or local_hwid
        if hwid:
            if reference_hwid is None:
                message = "Nie udało się zweryfikować fingerprintu urządzenia względem licencji."
                raise LicenseHardwareMismatchError(message) from hwid_error
            if reference_hwid != hwid:
                raise LicenseHardwareMismatchError(
                    f"Fingerprint urządzenia ({reference_hwid}) nie pasuje do licencji ({hwid})."
                )

        license_id = str(payload.get("license_id") or payload.get("licenseId") or "").strip() or None
        effective_date = self._compute_effective_date(license_id)
        capabilities = build_capabilities_from_payload(payload, effective_date=effective_date)
        self._store_effective_date(license_id, effective_date)

        snapshot = LicenseSnapshot(
            bundle_path=bundle_path,
            payload=payload,
            payload_bytes=payload_bytes,
            signature_bytes=signature_bytes,
            capabilities=capabilities,
            effective_date=effective_date,
            local_hwid=local_hwid,
            requires_hardware_wallet=capabilities.require_hardware_wallet_for_outgoing,
        )

        self._enforce_rollback_protection(snapshot, fingerprint=reference_hwid)
        self._write_status_snapshot(snapshot, fingerprint=reference_hwid)
        self._append_audit_log(snapshot)

        return snapshot


    # --- obsługa plików --------------------------------------------------------
    def _read_bundle(self, path: Path) -> dict[str, Any]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LicenseBundleError("Plik licencji zawiera niepoprawny JSON.") from exc
        if not isinstance(document, Mapping):
            raise LicenseBundleError("Plik licencji musi zawierać obiekt JSON.")

        payload_b64 = document.get("payload_b64")
        signature_b64 = document.get("signature_b64")
        if not isinstance(payload_b64, str) or not isinstance(signature_b64, str):
            raise LicenseBundleError("Licencja musi zawierać pola payload_b64 oraz signature_b64.")

        try:
            payload_bytes = base64.b64decode(payload_b64.encode("ascii"))
        except Exception as exc:
            raise LicenseBundleError("Nie można zdekodować payload_b64 (base64).") from exc

        try:
            signature_bytes = base64.b64decode(signature_b64.encode("ascii"))
        except Exception as exc:
            raise LicenseBundleError("Nie można zdekodować signature_b64 (base64).") from exc

        return {
            "payload_bytes": payload_bytes,
            "signature_bytes": signature_bytes,
        }

    # --- monotoniczny zegar ----------------------------------------------------
    def _compute_effective_date(self, license_id: str | None) -> date:
        return self._clock.effective_today(license_id)

    def _store_effective_date(self, license_id: str | None, value: date) -> None:
        self._clock.remember(license_id, value)

    # --- zapisywanie stanu -----------------------------------------------------
    def _write_status_snapshot(self, snapshot: LicenseSnapshot, *, fingerprint: str | None) -> None:
        document = {
            "license_id": snapshot.capabilities.license_id,
            "edition": snapshot.capabilities.edition,
            "environments": sorted(snapshot.capabilities.environments),
            "modules": sorted(
                name for name, enabled in snapshot.capabilities.modules.items() if enabled
            ),
            "runtime": sorted(
                name for name, enabled in snapshot.capabilities.runtime.items() if enabled
            ),
            "strategies": sorted(
                name for name, enabled in snapshot.capabilities.strategies.items() if enabled
            ),
            "exchanges": sorted(
                name for name, enabled in snapshot.capabilities.exchanges.items() if enabled
            ),
            "limits": {
                "max_paper_controllers": snapshot.capabilities.limits.max_paper_controllers,
                "max_live_controllers": snapshot.capabilities.limits.max_live_controllers,
                "max_concurrent_bots": snapshot.capabilities.limits.max_concurrent_bots,
                "max_alert_channels": snapshot.capabilities.limits.max_alert_channels,
            },
            "trial": {
                "enabled": snapshot.capabilities.trial.enabled,
                "expires_at": (
                    snapshot.capabilities.trial.expires_at.isoformat()
                    if snapshot.capabilities.trial.expires_at
                    else None
                ),
                "active": snapshot.capabilities.is_trial_active(snapshot.effective_date),
            },
            "maintenance": {
                "until": (
                    snapshot.capabilities.maintenance_until.isoformat()
                    if snapshot.capabilities.maintenance_until
                    else None
                ),
                "active": snapshot.capabilities.is_maintenance_active(snapshot.effective_date),
            },
            "issued_at": (
                snapshot.capabilities.issued_at.isoformat()
                if snapshot.capabilities.issued_at
                else None
            ),
            "effective_date": snapshot.effective_date.isoformat(),
            "seats": snapshot.capabilities.seats,
            "holder": dict(snapshot.capabilities.holder),
            "metadata": dict(snapshot.capabilities.metadata),
            "local_hwid": snapshot.local_hwid,
            "payload_sha256": snapshot.payload_sha256,
            "bundle_path": str(snapshot.bundle_path),
            "security": {
                "requires_hardware_wallet_for_outgoing": snapshot.requires_hardware_wallet,
            },
        }

        monotonic_payload = self._build_monotonic_payload(snapshot)
        document["monotonic"] = monotonic_payload

        if fingerprint:
            try:
                signature = sign_license_payload(
                    monotonic_payload,
                    fingerprint=fingerprint,
                    secret_path=self._binding_secret_path,
                )
            except FingerprintError as exc:
                raise LicenseServiceError(
                    "Nie udało się podpisać pliku statusu licencji lokalnym fingerprintem."
                ) from exc
            document["signature"] = dict(signature)
        else:
            document.pop("signature", None)

        self._status_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        tmp_path = self._status_path.with_suffix(self._status_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except (OSError, AttributeError):  # pragma: no cover - platform dependent behaviour
                    LOGGER.debug("Nie udało się wykonać fsync na pliku statusu.", exc_info=True)
            tmp_path.replace(self._status_path)
            dir_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                dir_flags |= os.O_DIRECTORY
            try:
                dir_fd = os.open(str(self._status_path.parent), dir_flags)
            except OSError:  # pragma: no cover - zależne od platformy i uprawnień
                LOGGER.debug("Nie udało się zsynchronizować katalogu statusu.", exc_info=True)
            else:
                try:
                    os.fsync(dir_fd)
                except OSError:  # pragma: no cover - zależne od systemu plików
                    LOGGER.debug("fsync katalogu statusu zakończone ostrzeżeniem.", exc_info=True)
                finally:
                    os.close(dir_fd)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise

    def _build_monotonic_payload(self, snapshot: LicenseSnapshot) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "license_id": snapshot.capabilities.license_id,
            "effective_date": snapshot.effective_date.isoformat(),
            "payload_sha256": snapshot.payload_sha256,
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "requires_hardware_wallet_for_outgoing": snapshot.requires_hardware_wallet,
        }

        issued_at = snapshot.capabilities.issued_at
        if issued_at is None:
            raw = snapshot.payload.get("issued_at") or snapshot.payload.get("issuedAt")
            if isinstance(raw, str) and raw.strip():
                try:
                    issued_at = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
                except ValueError:
                    issued_at = None
        if issued_at:
            payload["issued_at"] = issued_at.isoformat().replace("+00:00", "Z")

        sequence_value = snapshot.payload.get("sequence") or snapshot.payload.get("serial")
        if isinstance(sequence_value, (int, float)):
            payload["sequence"] = int(sequence_value)
        elif isinstance(sequence_value, str) and sequence_value.strip():
            try:
                payload["sequence"] = int(sequence_value.strip())
            except ValueError:
                pass

        return payload

    def _load_status_document(self) -> Mapping[str, Any] | None:
        if not self._status_path.exists():
            return None
        try:
            return json.loads(self._status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.warning("Plik statusu licencji ma niepoprawny format: %s", exc)
        except OSError as exc:
            LOGGER.warning("Nie udało się odczytać statusu licencji: %s", exc)
        return None

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_iso_date(value: str | None) -> date | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError:
                return None

    def _parse_monotonic_payload(self, payload: Mapping[str, Any]) -> _MonotonicState:
        license_id = str(payload.get("license_id") or "").strip() or None

        sequence_value = payload.get("sequence")
        sequence: int | None = None
        if isinstance(sequence_value, int):
            sequence = sequence_value
        elif isinstance(sequence_value, str) and sequence_value.strip():
            try:
                sequence = int(sequence_value.strip())
            except ValueError:
                sequence = None

        issued_at = self._parse_iso_datetime(payload.get("issued_at"))
        effective_date = self._parse_iso_date(payload.get("effective_date"))
        raw_digest = str(payload.get("payload_sha256") or "").strip()
        payload_sha = raw_digest.lower() or None
        requires_value = payload.get("requires_hardware_wallet_for_outgoing")
        requires_hw: bool | None
        if isinstance(requires_value, bool):
            requires_hw = requires_value
        elif isinstance(requires_value, str) and requires_value.strip():
            requires_hw = requires_value.strip().lower() in {"1", "true", "yes", "on"}
        else:
            requires_hw = None

        return _MonotonicState(
            license_id=license_id,
            sequence=sequence,
            issued_at=issued_at,
            effective_date=effective_date,
            payload_sha256=payload_sha,
            requires_hardware_wallet=requires_hw,
        )

    @staticmethod
    def _is_rollback(previous: _MonotonicState, current: _MonotonicState) -> bool:
        if previous.sequence is not None and current.sequence is not None:
            if current.sequence < previous.sequence:
                return True
        if previous.issued_at and current.issued_at:
            if current.issued_at < previous.issued_at:
                return True
        if previous.effective_date and current.effective_date:
            if current.effective_date < previous.effective_date:
                return True
        return False

    def _enforce_rollback_protection(
        self,
        snapshot: LicenseSnapshot,
        *,
        fingerprint: str | None,
    ) -> None:
        if not fingerprint:
            LOGGER.warning(
                "Brak fingerprintu urządzenia – ochrona przed rollbackiem licencji została pominięta."
            )
            return

        document = self._load_status_document()
        if not document:
            return

        monotonic_section = document.get("monotonic")
        signature_section = document.get("signature")
        if not isinstance(monotonic_section, Mapping) or not isinstance(signature_section, Mapping):
            return

        try:
            valid = verify_license_payload_signature(
                monotonic_section,
                signature_section,
                fingerprint=fingerprint,
                secret_path=self._binding_secret_path,
            )
        except FingerprintError as exc:
            raise LicenseStateTamperedError(
                "Nie udało się zweryfikować podpisu pliku statusu licencji."
            ) from exc
        if not valid:
            raise LicenseStateTamperedError(
                "Podpis pliku statusu licencji jest nieprawidłowy lub sekret został usunięty."
            )

        previous_state = self._parse_monotonic_payload(monotonic_section)
        current_payload = self._build_monotonic_payload(snapshot)
        current_state = self._parse_monotonic_payload(current_payload)

        if (
            previous_state.license_id
            and current_state.license_id
            and previous_state.license_id != current_state.license_id
        ):
            return

        if previous_state.payload_sha256 and current_state.payload_sha256:
            if previous_state.payload_sha256 == current_state.payload_sha256:
                return

        if self._is_rollback(previous_state, current_state):
            raise LicenseRollbackDetectedError(
                "Wykryto próbę załadowania starszej wersji licencji (rollback)."
            )

    def _append_audit_log(self, snapshot: LicenseSnapshot) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        local_hwid_hash: str | None = None
        if snapshot.local_hwid:
            digest = hashlib.sha256(snapshot.local_hwid.encode("utf-8")).hexdigest()
            # Skrócenie hash do czytelnej długości – pełna wartość dostępna do korelacji.
            local_hwid_hash = digest[:24]

        previous_activations = 0
        if local_hwid_hash and self._audit_log_path.exists():
            try:
                with self._audit_log_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            LOGGER.warning("Nie udało się sparsować wpisu audytowego licencji: %s", line)
                            continue
                        if record.get("local_hwid_hash") == local_hwid_hash:
                            previous_activations += 1
            except OSError as exc:
                LOGGER.warning("Nie udało się odczytać logu audytowego licencji: %s", exc)

        entry = {
            "timestamp": timestamp,
            "event": "license_snapshot",
            "license_id": snapshot.capabilities.license_id,
            "edition": snapshot.capabilities.edition,
            "trial_active": snapshot.capabilities.is_trial_active(snapshot.effective_date),
            "maintenance_active": snapshot.capabilities.is_maintenance_active(snapshot.effective_date),
            "bundle_path": str(snapshot.bundle_path),
            "payload_sha256": snapshot.payload_sha256,
            "local_hwid_hash": local_hwid_hash,
            "activation_count": (previous_activations + 1) if local_hwid_hash else 1,
            "repeat_activation": previous_activations > 0,
            "requires_hardware_wallet": snapshot.requires_hardware_wallet,
        }

        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = [
    "LicenseBundleError",
    "LicenseHardwareMismatchError",
    "LicenseRollbackDetectedError",
    "LicenseService",
    "LicenseServiceError",
    "LicenseSignatureError",
    "LicenseStateTamperedError",
    "LicenseSnapshot",
]
