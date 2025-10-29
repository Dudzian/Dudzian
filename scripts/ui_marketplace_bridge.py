#!/usr/bin/env python3
"""Mostek CLI udostępniający presety strategii i zarządzanie licencjami dla UI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from bot_core.strategies.catalog import (
    StrategyCatalog,
    StrategyPresetProfile,
)
from bot_core.security.hwid import HwIdProvider


def _load_json(path: Path | None, *, stdin_fallback: bool = False) -> Any:
    if path is None and stdin_fallback:
        data = sys.stdin.read()
        if not data.strip():
            raise SystemExit("Brak danych na standardowym wejściu – oczekiwano JSON.")
        return json.loads(data)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywne logowanie
        raise SystemExit(f"Plik {path} zawiera niepoprawny JSON: {exc}") from exc


def _parse_signing_key(value: str) -> tuple[str, bytes]:
    if "=" not in value:
        raise SystemExit("--signing-key wymaga formatu KEY_ID=SECRET")
    key_id, secret = value.split("=", 1)
    key_id = key_id.strip()
    secret = secret.strip()
    if not key_id or not secret:
        raise SystemExit("--signing-key wymaga niepustych wartości KEY_ID i SECRET")
    if all(ch in "0123456789abcdefABCDEF" for ch in secret) and len(secret) % 2 == 0:
        return key_id, bytes.fromhex(secret)
    return key_id, secret.encode("utf-8")


def _load_signing_keys(values: list[str], files: list[str]) -> dict[str, bytes]:
    keys: dict[str, bytes] = {}
    for item in values:
        key_id, payload = _parse_signing_key(item)
        keys[key_id] = payload
    for file_path in files:
        path = Path(file_path)
        payload = _load_json(path)
        if isinstance(payload, Mapping):
            for key_id, secret in payload.items():
                if not isinstance(key_id, str):
                    continue
                if not isinstance(secret, str):
                    continue
                keys[key_id] = _parse_signing_key(f"{key_id}={secret}")[1]
    return keys


def _read_license_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"licenses": {}}
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        licenses = payload.get("licenses")
        if isinstance(licenses, Mapping):
            return {"licenses": dict(licenses)}
    return {"licenses": {}}


def _write_license_store(path: Path, store: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _build_hwid_provider(override: str | None) -> HwIdProvider | None:
    if override is None:
        return HwIdProvider()
    if not override:
        return None
    return HwIdProvider(fingerprint_reader=lambda: override)


def _load_catalog(
    presets_dir: Path,
    *,
    signing_keys: Mapping[str, bytes],
    fingerprint_override: str | None,
    license_store: Mapping[str, Any],
) -> tuple[StrategyCatalog, HwIdProvider | None]:
    provider = _build_hwid_provider(fingerprint_override)
    catalog = StrategyCatalog(hwid_provider=provider)
    catalog.load_presets_from_directory(
        presets_dir,
        signing_keys=signing_keys,
        hwid_provider=provider,
    )
    overrides = license_store.get("licenses")
    if isinstance(overrides, Mapping):
        for preset_id, payload in overrides.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                catalog.install_license_override(preset_id, payload, hwid_provider=provider)
            except KeyError:
                continue
    return catalog, provider


def _descriptor_payload(descriptor, include_strategies: bool) -> Mapping[str, Any]:
    payload = descriptor.as_dict(include_strategies=include_strategies)
    if descriptor.source_path is not None:
        payload["source_path"] = str(descriptor.source_path)
    return payload


def _command_list(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    licenses_path = Path(args.licenses_path)
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    store = _read_license_store(licenses_path)
    catalog, provider = _load_catalog(
        presets_dir,
        signing_keys=signing_keys,
        fingerprint_override=args.fingerprint,
        license_store=store,
    )
    profile = args.profile
    include_strategies = bool(args.include_strategies)
    summaries = catalog.describe_presets(
        profile=profile,
        include_strategies=include_strategies,
        hwid_provider=provider,
    )
    document = {"presets": summaries, "licenses": store.get("licenses", {})}
    json.dump(document, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _read_license_payload(args: argparse.Namespace) -> Mapping[str, Any]:
    license_path = Path(args.license_json) if args.license_json else None
    payload = _load_json(license_path, stdin_fallback=True)
    if not isinstance(payload, Mapping):
        raise SystemExit("Payload licencji musi być obiektem JSON.")
    return dict(payload)


def _command_activate(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    licenses_path = Path(args.licenses_path)
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    store = _read_license_store(licenses_path)
    payload = _read_license_payload(args)
    catalog, provider = _load_catalog(
        presets_dir,
        signing_keys=signing_keys,
        fingerprint_override=args.fingerprint,
        license_store=store,
    )
    store.setdefault("licenses", {})[args.preset_id] = payload
    descriptor = catalog.install_license_override(
        args.preset_id,
        payload,
        hwid_provider=provider,
    )
    _write_license_store(licenses_path, store)
    result = {"preset": _descriptor_payload(descriptor, include_strategies=True)}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _command_deactivate(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    licenses_path = Path(args.licenses_path)
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    store = _read_license_store(licenses_path)
    catalog, provider = _load_catalog(
        presets_dir,
        signing_keys=signing_keys,
        fingerprint_override=args.fingerprint,
        license_store=store,
    )
    store.get("licenses", {}).pop(args.preset_id, None)
    descriptor = catalog.clear_license_override(
        args.preset_id,
        hwid_provider=provider,
    )
    _write_license_store(licenses_path, store)
    result = {"preset": _descriptor_payload(descriptor, include_strategies=True)}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets-dir", default="data/strategies", help="Katalog z plikami presetów JSON")
    parser.add_argument(
        "--licenses-path",
        default="var/marketplace_licenses.json",
        help="Ścieżka do pliku przechowującego stan aktywacji presetów",
    )
    parser.add_argument("--fingerprint", help="Nadpisanie fingerprintu sprzętowego (dla testów/diagnostyki)")
    parser.add_argument("--signing-key", action="append", help="Klucz HMAC w formacie KEY_ID=SECRET")
    parser.add_argument(
        "--signing-key-file",
        action="append",
        help="Plik JSON ze słownikiem kluczy podpisów (key_id -> secret)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Lista dostępnych presetów")
    list_parser.add_argument("--profile", choices=[p.value for p in StrategyPresetProfile], help="Filtr profilu strategii")
    list_parser.add_argument(
        "--include-strategies",
        action="store_true",
        help="Dołącz pełne definicje strategii w wyniku",
    )

    activate_parser = subparsers.add_parser("activate", help="Aktywacja licencji presetu")
    activate_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    activate_parser.add_argument("--license-json", help="Ścieżka do pliku JSON z licencją (domyślnie stdin)")

    deactivate_parser = subparsers.add_parser("deactivate", help="Dezaktywacja licencji presetu")
    deactivate_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        _command_list(args)
    elif args.command == "activate":
        _command_activate(args)
    elif args.command == "deactivate":
        _command_deactivate(args)
    else:  # pragma: no cover - argparse zapewnia poprawność
        parser.error(f"Nieznane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    main()
