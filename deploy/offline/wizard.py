from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from bot_core.security import (
    FingerprintValidationResult,
    validate_fingerprint_document,
    validate_license_bundle,
    verify_update_bundle,
)
from bot_core.security.license import LicenseValidationResult, load_hmac_keys_file
from bot_core.security.messages import ValidationMessage

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LicenseBundle:
    license_path: Path
    license_keys_path: Path | None = None
    fingerprint_path: Path | None = None
    fingerprint_keys_path: Path | None = None


@dataclass(slots=True)
class InstallerSession:
    root_dir: Path
    config_dir: Path
    logs_dir: Path
    updates_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "InstallerSession":
        root = root.expanduser().resolve()
        return cls(
            root_dir=root,
            config_dir=root / "config",
            logs_dir=root / "logs",
            updates_dir=root / "updates",
        )


class InstallerWizard:
    """High level helper for offline installation/updates."""

    def __init__(self, session: InstallerSession):
        self.session = session
        self.session.config_dir.mkdir(parents=True, exist_ok=True)
        self.session.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session.updates_dir.mkdir(parents=True, exist_ok=True)
        self._last_license: LicenseValidationResult | None = None
        self._last_fingerprint: FingerprintValidationResult | None = None
        self._last_update: Mapping[str, object] | None = None

    def validate_fingerprint(self, bundle: LicenseBundle) -> FingerprintValidationResult:
        if not bundle.fingerprint_path:
            raise ValueError("Brak ścieżki fingerprintu do walidacji.")
        result = validate_fingerprint_document(
            document_path=bundle.fingerprint_path,
            keys_path=bundle.fingerprint_keys_path,
        )
        self._last_fingerprint = result
        if result.errors:
            LOGGER.error("Walidacja fingerprintu zakończona błędami: %s", "; ".join(result.errors))
        else:
            LOGGER.info("Fingerprint %s zweryfikowany (klucz=%s).", result.fingerprint, result.key_id or "n/d")
        return result

    def validate_license(self, bundle: LicenseBundle) -> LicenseValidationResult:
        result = validate_license_bundle(
            license_path=bundle.license_path,
            license_keys_path=bundle.license_keys_path,
            fingerprint_path=bundle.fingerprint_path,
            fingerprint_keys_path=bundle.fingerprint_keys_path,
        )
        self._last_license = result
        if result.errors:
            for error in result.errors:
                LOGGER.error("Błąd walidacji licencji: %s", error.message if hasattr(error, "message") else str(error))
        else:
            LOGGER.info("Licencja %s (%s) zweryfikowana.", result.license_path, result.profile or "n/d")
        return result

    def apply_offline_update(
        self,
        *,
        manifest_path: Path,
        payload_dir: Path | None = None,
        payload_archive: Path | None = None,
        signature_path: Path | None = None,
        hmac_keys: Mapping[str, bytes] | None = None,
    ) -> Path:
        manifest_path = manifest_path.expanduser()
        signature_path = signature_path.expanduser() if signature_path else None

        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        if payload_dir and payload_archive:
            raise ValueError("Nie można jednocześnie wskazać katalogu i archiwum aktualizacji.")

        payload_dir = payload_dir.expanduser() if payload_dir else None
        payload_archive = payload_archive.expanduser() if payload_archive else None

        if payload_archive:
            if not payload_archive.exists():
                raise FileNotFoundError(payload_archive)
            with tempfile.TemporaryDirectory(prefix="offline_update_", dir=str(self.session.root_dir)) as extracted:
                extracted_path = Path(extracted)
                shutil.unpack_archive(str(payload_archive), extracted_path)
                payload_source = _resolve_payload_root(extracted_path)
                return self._process_offline_update(
                    manifest_path=manifest_path,
                    payload_source=payload_source,
                    signature_path=signature_path,
                    hmac_keys=hmac_keys,
                    payload_archive=payload_archive,
                )

        if not payload_dir:
            raise ValueError("Wymagane jest wskazanie katalogu lub archiwum aktualizacji.")
        if not payload_dir.exists():
            raise FileNotFoundError(payload_dir)

        return self._process_offline_update(
            manifest_path=manifest_path,
            payload_source=payload_dir,
            signature_path=signature_path,
            hmac_keys=hmac_keys,
            payload_archive=None,
        )

    def _process_offline_update(
        self,
        *,
        manifest_path: Path,
        payload_source: Path,
        signature_path: Path | None,
        hmac_keys: Mapping[str, bytes] | None,
        payload_archive: Path | None,
    ) -> Path:

        verification_warnings: list[str] = []
        selected_key: str | None = None

        def _verify_with_key(hmac_key: bytes | None) -> object:
            result = verify_update_bundle(
                manifest_path=manifest_path,
                base_dir=payload_source,
                signature_path=signature_path,
                hmac_key=hmac_key,
                license_result=self._last_license,
            )
            verification_warnings.extend(_stringify_messages(result.warnings))
            return result

        result = None
        if hmac_keys:
            errors_by_key: list[str] = []
            for key_id, key_bytes in hmac_keys.items():
                attempt = _verify_with_key(key_bytes)
                if getattr(attempt, "is_successful", False):
                    result = attempt
                    selected_key = key_id
                    break
                errors = ", ".join(_stringify_messages(getattr(attempt, "errors", []))) or "nieznany błąd"
                errors_by_key.append(f"{key_id}: {errors}")
            if result is None:
                self._last_update = {
                    "status": "invalid",
                    "manifest": str(manifest_path),
                    "payload": str(payload_source),
                    "signature": str(signature_path) if signature_path else None,
                    "payload_archive": str(payload_archive) if payload_archive else None,
                    "errors": list(errors_by_key),
                    "warnings": list(verification_warnings),
                }
                raise RuntimeError(
                    "Pakiet aktualizacji nie przeszedł walidacji: " + "; ".join(errors_by_key)
                )
        else:
            attempt = _verify_with_key(None)
            if not getattr(attempt, "is_successful", False):
                errors = _stringify_messages(getattr(attempt, "errors", []))
                self._last_update = {
                    "status": "invalid",
                    "manifest": str(manifest_path),
                    "payload": str(payload_source),
                    "signature": str(signature_path) if signature_path else None,
                    "payload_archive": str(payload_archive) if payload_archive else None,
                    "errors": list(errors),
                    "warnings": list(verification_warnings),
                }
                raise RuntimeError(
                    "Pakiet aktualizacji nie przeszedł walidacji: " + "; ".join(errors)
                )
            result = attempt

        for warning in verification_warnings:
            LOGGER.warning("Walidacja aktualizacji: %s", warning)

        target_dir = self.session.updates_dir / manifest_path.stem
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(payload_source, target_dir)
        shutil.copy2(manifest_path, target_dir / manifest_path.name)
        if signature_path:
            shutil.copy2(signature_path, target_dir / signature_path.name)
        LOGGER.info("Skopiowano paczkę aktualizacji do %s", target_dir)

        self._last_update = {
            "status": "ok",
            "manifest": str(manifest_path),
            "payload": str(payload_source),
            "signature": str(signature_path) if signature_path else None,
            "target": str(target_dir),
            "warnings": list(verification_warnings),
            "key_id": selected_key,
            "payload_archive": str(payload_archive) if payload_archive else None,
        }
        return target_dir

    def summary(self, *, include_timestamp: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "license": self._serialize_license(self._last_license),
            "fingerprint": self._serialize_fingerprint(self._last_fingerprint),
            "update": self._last_update,
            "paths": {
                "root": str(self.session.root_dir),
                "config": str(self.session.config_dir),
                "logs": str(self.session.logs_dir),
                "updates": str(self.session.updates_dir),
            },
        }
        if include_timestamp:
            payload["generated_at"] = _utcnow_iso()
        return payload

    def persist_summary(
        self,
        path: str | Path | None = None,
        *,
        data: Mapping[str, object] | None = None,
    ) -> Path:
        target = Path(path).expanduser() if path else self.session.logs_dir / "offline_installer_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data) if data is not None else self.summary(include_timestamp=True)
        if "generated_at" not in payload:
            payload = dict(payload)
            payload["generated_at"] = _utcnow_iso()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @staticmethod
    def _serialize_license(result: LicenseValidationResult | None) -> Mapping[str, object] | None:
        if result is None:
            return None
        return {
            "status": result.status,
            "profile": result.profile,
            "issuer": result.issuer,
            "license_path": str(result.license_path),
            "errors": [msg.message if isinstance(msg, ValidationMessage) else str(msg) for msg in result.errors],
            "warnings": [msg.message if isinstance(msg, ValidationMessage) else str(msg) for msg in result.warnings],
        }

    @staticmethod
    def _serialize_fingerprint(result: FingerprintValidationResult | None) -> Mapping[str, object] | None:
        if result is None:
            return None
        return {
            "status": result.status,
            "fingerprint": result.fingerprint,
            "key_id": result.key_id,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }


def _load_hmac_keys(paths: Iterable[str | Path]) -> Mapping[str, bytes]:
    decoded: MutableMapping[str, bytes] = {}
    for path in paths:
        if not path:
            continue
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        loaded = load_hmac_keys_file(candidate)
        decoded.update(loaded)
    return decoded


def _stringify_messages(messages: Iterable[object]) -> list[str]:
    return [msg.message if isinstance(msg, ValidationMessage) else str(msg) for msg in messages]


def _resolve_payload_root(extracted_dir: Path) -> Path:
    try:
        entries = [entry for entry in extracted_dir.iterdir() if entry.name != "__MACOSX"]
    except FileNotFoundError:
        return extracted_dir
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_dir


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline installer and update helper")
    parser.add_argument("--root", default=".", help="Katalog instalacji (domyślnie bieżący).")
    parser.add_argument("--license", dest="license_path", help="Ścieżka pliku licencji OEM.")
    parser.add_argument("--license-keys", dest="license_keys", help="Plik z kluczami HMAC licencji.")
    parser.add_argument("--fingerprint", dest="fingerprint_path", help="Dokument fingerprintu hosta.")
    parser.add_argument("--fingerprint-keys", dest="fingerprint_keys", help="Plik z kluczami HMAC fingerprintu.")
    parser.add_argument("--update-manifest", dest="update_manifest", help="Manifest aktualizacji offline (opcjonalnie).")
    parser.add_argument("--update-payload", dest="update_payload", help="Katalog z plikami aktualizacji (opcjonalnie).")
    parser.add_argument(
        "--update-archive",
        dest="update_archive",
        help="Archiwum aktualizacji (zip/tar) rozpakowywane przed walidacją (opcjonalnie).",
    )
    parser.add_argument("--update-signature", dest="update_signature", help="Podpis manifestu aktualizacji (opcjonalnie).")
    parser.add_argument(
        "--update-keys",
        dest="update_keys",
        action="append",
        help="Plik z kluczami HMAC podpisu manifestu (można podać wielokrotnie).",
    )
    parser.add_argument("--apply-update", action="store_true", help="Po poprawnej walidacji spróbuj skopiować aktualizację.")
    parser.add_argument(
        "--summary-path",
        dest="summary_path",
        help="Ścieżka do pliku JSON z podsumowaniem (domyślnie logs/offline_installer_summary.json).",
    )
    parser.add_argument(
        "--no-summary-file",
        action="store_true",
        help="Nie zapisuj podsumowania do pliku (zawsze wypisywane na stdout).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_summary_file and args.summary_path:
        parser.error("Opcji --summary-path nie można łączyć z --no-summary-file.")

    session = InstallerSession.from_root(Path(args.root))
    wizard = InstallerWizard(session)

    license_bundle = None
    if args.license_path:
        license_bundle = LicenseBundle(
            license_path=Path(args.license_path),
            license_keys_path=Path(args.license_keys).expanduser() if args.license_keys else None,
            fingerprint_path=Path(args.fingerprint_path).expanduser() if args.fingerprint_path else None,
            fingerprint_keys_path=Path(args.fingerprint_keys).expanduser() if args.fingerprint_keys else None,
        )
        if license_bundle.fingerprint_path:
            wizard.validate_fingerprint(license_bundle)
        wizard.validate_license(license_bundle)

    if args.apply_update:
        if not args.update_manifest:
            parser.error("Opcja --apply-update wymaga podania --update-manifest.")
        if not args.update_payload and not args.update_archive:
            parser.error(
                "Opcja --apply-update wymaga wskazania katalogu (--update-payload) lub archiwum (--update-archive)."
            )
        if args.update_payload and args.update_archive:
            parser.error("Nie można jednocześnie korzystać z --update-payload i --update-archive.")
        hmac_keys = {}
        if args.update_keys:
            hmac_keys = _load_hmac_keys(args.update_keys)
        wizard.apply_offline_update(
            manifest_path=Path(args.update_manifest),
            payload_dir=Path(args.update_payload) if args.update_payload else None,
            payload_archive=Path(args.update_archive) if args.update_archive else None,
            signature_path=Path(args.update_signature) if args.update_signature else None,
            hmac_keys=hmac_keys,
        )

    include_timestamp = not args.no_summary_file or bool(args.summary_path)
    summary = wizard.summary(include_timestamp=include_timestamp)
    if not args.no_summary_file:
        summary_path = wizard.persist_summary(
            Path(args.summary_path).expanduser() if args.summary_path else None,
            data=summary,
        )
        LOGGER.info("Zapisano podsumowanie instalacji do %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
