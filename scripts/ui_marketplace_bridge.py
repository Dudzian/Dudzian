#!/usr/bin/env python3
"""Mostek CLI udostępniający presety strategii i zarządzanie licencjami dla UI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import InvalidVersion, Version

from bot_core.marketplace import (
    MarketplaceIndex,
    PresetRepository,
    build_marketplace_preset,
    parse_preset_document,
)
from bot_core.marketplace.assignments import PresetAssignmentStore
from bot_core.strategies.catalog import (
    StrategyCatalog,
    StrategyPresetProfile,
)
from bot_core.security.hwid import HwIdProvider
from bot_core.security.license import summarize_license_payload


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


def _normalize_sequence_of_messages(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    normalized: list[str] = []
    for item in values:
        if isinstance(item, str):
            text = item.strip()
            if text and text not in normalized:
                normalized.append(text)
    return normalized


def _normalize_portfolio_ids(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in normalized:
                normalized.append(text)
    normalized.sort()
    return normalized


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _merge_unique(base: Sequence[str], extra: Sequence[str]) -> list[str]:
    merged: list[str] = list(base)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def _build_assignment_summary(
    preset_id: str,
    assigned: Sequence[str],
    license_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "presetId": preset_id,
        "assignedPortfolios": list(assigned),
        "assignedCount": len(assigned),
    }

    seat_summary = None
    if isinstance(license_entry, Mapping):
        candidate = license_entry.get("seatSummary")
        if isinstance(candidate, Mapping):
            seat_summary = candidate

    licensed_assignments = _normalize_portfolio_ids(
        seat_summary.get("assignments") if seat_summary else None
    )
    if licensed_assignments:
        summary["licensedAssignments"] = licensed_assignments

    assigned_set = set(assigned)
    licensed_set = set(licensed_assignments)

    unlicensed = sorted(assigned_set - licensed_set)
    if unlicensed:
        summary["unlicensedAssignments"] = unlicensed

    orphaned = sorted(licensed_set - assigned_set)
    if orphaned:
        summary["orphanedAssignments"] = orphaned

    pending_assignments = _normalize_portfolio_ids(
        seat_summary.get("pending") if seat_summary else None
    )
    if pending_assignments:
        summary["pendingAssignments"] = pending_assignments

    seat_limit = _coerce_int(seat_summary.get("total") if seat_summary else None)
    in_use = _coerce_int(seat_summary.get("inUse") if seat_summary else None)
    available = _coerce_int(seat_summary.get("available") if seat_summary else None)

    if seat_limit is not None:
        summary["seatLimit"] = seat_limit
    if in_use is not None:
        summary["inUseSeats"] = in_use
    if available is not None:
        summary["availableSeats"] = available

    projected_candidates = [len(assigned)]
    if in_use is not None:
        projected_candidates.append(in_use)
    if licensed_assignments:
        projected_candidates.append(len(licensed_assignments))
    projected_in_use = max(projected_candidates) if projected_candidates else None
    if projected_in_use is not None:
        summary["projectedInUseSeats"] = projected_in_use

    seat_shortfall: int | None = None
    if seat_limit is not None and projected_in_use is not None:
        remaining = seat_limit - projected_in_use
        summary["projectedRemainingSeats"] = max(remaining, 0)
        if remaining < 0:
            seat_shortfall = abs(remaining)
            summary["seatShortfall"] = seat_shortfall
    elif seat_limit is not None:
        remaining = seat_limit - len(assigned)
        summary["projectedRemainingSeats"] = max(remaining, 0)
        if remaining < 0:
            seat_shortfall = abs(remaining)
            summary["seatShortfall"] = seat_shortfall

    warning_codes: list[str] = []
    warning_messages: list[str] = []
    if unlicensed:
        warning_codes.append("assignment-unlicensed")
        warning_messages.append("Brak licencji dla portfeli: " + ", ".join(unlicensed))
    if pending_assignments:
        warning_codes.append("assignment-pending")
        warning_messages.append(
            "Portfele oczekują na zatwierdzenie w licencji: "
            + ", ".join(pending_assignments)
        )
    if seat_shortfall:
        warning_codes.append("assignment-seat-shortfall")
        warning_messages.append(
            f"Brakuje {seat_shortfall} miejsc licencyjnych dla przypisanych portfeli."
        )

    if warning_codes:
        summary["warningCodes"] = warning_codes
    if warning_messages:
        summary["warningMessages"] = warning_messages

    return summary


def _build_portfolio_summaries(
    assignment_summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    def _append(entry: dict[str, Any], key: str, preset_id: str) -> None:
        bucket = entry.setdefault(key, [])
        if preset_id not in bucket:
            bucket.append(preset_id)

    def _append_warning(entry: dict[str, Any], code: str, message: str) -> None:
        if code:
            entry.setdefault("warningCodes", [])
            if code not in entry["warningCodes"]:
                entry["warningCodes"].append(code)
        if message:
            entry.setdefault("warningMessages", [])
            if message not in entry["warningMessages"]:
                entry["warningMessages"].append(message)

    summaries: dict[str, dict[str, Any]] = {}

    for preset_id, summary in assignment_summaries.items():
        assigned = _normalize_portfolio_ids(summary.get("assignedPortfolios"))
        licensed = _normalize_portfolio_ids(summary.get("licensedAssignments"))
        unlicensed = set(_normalize_portfolio_ids(summary.get("unlicensedAssignments")))
        orphaned = set(_normalize_portfolio_ids(summary.get("orphanedAssignments")))
        pending = set(_normalize_portfolio_ids(summary.get("pendingAssignments")))

        participants = set(assigned) | set(licensed) | orphaned | pending
        if not participants:
            continue

        seat_shortfall = _coerce_int(summary.get("seatShortfall"))

        for portfolio_id in sorted(participants):
            entry = summaries.setdefault(portfolio_id, {"portfolioId": portfolio_id})

            if portfolio_id in assigned:
                _append(entry, "assignedPresets", preset_id)
            if portfolio_id in licensed:
                _append(entry, "licensedPresets", preset_id)
            if portfolio_id in unlicensed:
                _append(entry, "unlicensedPresets", preset_id)
                _append_warning(
                    entry,
                    "portfolio-assignment-unlicensed",
                    f"Portfel {portfolio_id} nie ma licencji na preset {preset_id}.",
                )
            if portfolio_id in orphaned:
                _append(entry, "orphanedPresets", preset_id)
                _append_warning(
                    entry,
                    "portfolio-license-orphaned",
                    (
                        f"Portfel {portfolio_id} jest przypisany w licencji {preset_id}, "
                        "ale nie ma lokalnego przydziału."
                    ),
                )
            if portfolio_id in pending:
                _append(entry, "pendingPresets", preset_id)
                _append_warning(
                    entry,
                    "portfolio-assignment-pending",
                    f"Portfel {portfolio_id} oczekuje na zatwierdzenie w licencji {preset_id}.",
                )
            if seat_shortfall and (portfolio_id in assigned or portfolio_id in licensed):
                _append(entry, "seatShortfallPresets", preset_id)
                _append_warning(
                    entry,
                    "portfolio-seat-shortfall",
                    (
                        f"Preset {preset_id} wymaga dodatkowych {seat_shortfall} miejsc licencyjnych "
                        "dla przypisanych portfeli."
                    ),
                )

    for entry in summaries.values():
        for key in (
            "assignedPresets",
            "licensedPresets",
            "unlicensedPresets",
            "orphanedPresets",
            "pendingPresets",
            "seatShortfallPresets",
        ):
            if key in entry:
                entry[key] = sorted(entry[key])
        if "warningCodes" in entry:
            entry["warningCodes"] = _merge_unique([], entry["warningCodes"])
        if "warningMessages" in entry:
            entry["warningMessages"] = _merge_unique([], entry["warningMessages"])

    return summaries


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


def _assignment_store(presets_dir: Path) -> PresetAssignmentStore:
    repository = PresetRepository(presets_dir)
    return PresetAssignmentStore(repository.root / ".meta" / "assignments.json")


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
    summaries = [
        dict(entry)
        for entry in catalog.describe_presets(
            profile=profile,
            include_strategies=include_strategies,
            hwid_provider=provider,
        )
    ]

    repository = PresetRepository(presets_dir)
    documents = repository.load_all(signing_keys=signing_keys)
    index = MarketplaceIndex.from_documents(documents)
    assignments_store = PresetAssignmentStore(repository.root / ".meta" / "assignments.json")

    for summary in summaries:
        preset_id = summary.get("preset_id") or summary.get("presetId")
        if not isinstance(preset_id, str):
            continue
        package = index.get(preset_id)
        if package is None:
            preset_entry = catalog.find(preset_id)
            if preset_entry is not None:
                try:
                    document_payload = parse_preset_document(
                        preset_entry.artifact_path.read_bytes(),
                        source=preset_entry.artifact_path,
                    )
                except Exception:  # pragma: no cover - defensywne logowanie
                    package = None
                else:
                    package = build_marketplace_preset(document_payload)
        if package is None:
            continue
        summary["dependencies"] = [dep.to_payload() for dep in package.dependencies]
        summary["update_channels"] = [channel.to_payload() for channel in package.update_channels]
        summary["preferred_channel"] = package.preferred_channel
        installed_doc = next((doc for doc in documents if doc.preset_id == preset_id), None)
        installed_version = installed_doc.version if installed_doc else None
        available_version = package.version
        upgrade_available = False
        upgrade_version = None
        if installed_version and available_version:
            try:
                if Version(installed_version) < Version(available_version):
                    upgrade_available = True
                    upgrade_version = available_version
            except InvalidVersion:
                upgrade_available = False
        summary["installed_version"] = installed_version
        summary["available_version"] = available_version
        summary["upgrade_available"] = upgrade_available
        summary["upgrade_version"] = upgrade_version
        summary["assigned_portfolios"] = list(assignments_store.assigned_portfolios(preset_id))
        license_info = summary.get("license")
        if isinstance(license_info, Mapping):
            metadata = license_info.get("metadata")
            if isinstance(metadata, Mapping):
                validation = metadata.get("validation")
                if isinstance(validation, Mapping):
                    warning_messages = validation.get("warning_messages")
                    if not isinstance(warning_messages, Sequence) or isinstance(warning_messages, (str, bytes)):
                        warning_messages = validation.get("warningMessages")
                    if isinstance(warning_messages, Sequence) and not isinstance(warning_messages, (str, bytes)):
                        summary["warning_messages"] = [
                            str(item)
                            for item in warning_messages
                            if isinstance(item, str) and item.strip()
                        ]
                    warning_codes = validation.get("warning_codes")
                    if not isinstance(warning_codes, Sequence) or isinstance(warning_codes, (str, bytes)):
                        warning_codes = validation.get("warningCodes")
                    if isinstance(warning_codes, Sequence) and not isinstance(warning_codes, (str, bytes)):
                        summary["warnings"] = [
                            str(item)
                            for item in warning_codes
                            if isinstance(item, str) and item.strip()
                        ]
        summary.setdefault("warnings", [])
        summary.setdefault("warning_messages", [])
        summary["warningMessages"] = list(summary["warning_messages"])

    document = {
        "presets": summaries,
        "licenses": store.get("licenses", {}),
        "assignments": assignments_store.all_assignments(),
    }
    json.dump(document, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _command_plan(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    repository = PresetRepository(presets_dir)
    documents = repository.load_all(signing_keys=signing_keys)

    catalog_docs: dict[str, Any] = {}
    for document in documents:
        if document.preset_id:
            catalog_docs[document.preset_id] = document

    index = MarketplaceIndex.from_documents(list(catalog_docs.values()))
    installed_versions = {
        doc.preset_id: doc.version
        for doc in documents
        if doc.preset_id and doc.version
    }
    selection = [value.strip() for value in (args.preset_id or []) if value and value.strip()]
    plan = index.plan_installation(selection, installed_versions=installed_versions)
    payload = plan.to_payload()
    payload["selection"] = selection
    store = _read_license_store(Path(args.licenses_path))
    licenses = store.get("licenses") if isinstance(store, Mapping) else {}
    license_summaries: dict[str, Any] = {}
    assignments_store = PresetAssignmentStore(repository.root / ".meta" / "assignments.json")
    for preset_id in payload.get("installOrder", []):
        entry: dict[str, Any] = {
            "presetId": preset_id,
            "issues": [],
            "warnings": [],
            "warningMessages": [],
        }
        license_payload = None
        if isinstance(licenses, Mapping):
            candidate = licenses.get(preset_id)
            if isinstance(candidate, Mapping):
                license_payload = candidate
        if license_payload is None:
            entry["licenseMissing"] = True
        else:
            entry["license"] = license_payload
            summary = summarize_license_payload(license_payload)
            for key, value in summary.items():
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    entry[key] = list(value)
                else:
                    entry[key] = value
            entry["warningMessages"] = _normalize_sequence_of_messages(entry.get("warningMessages"))
        license_summaries[preset_id] = entry

    for preset_id in selection:
        if preset_id not in license_summaries:
            license_summaries[preset_id] = {
                "presetId": preset_id,
                "issues": [],
                "warnings": [],
                "warningMessages": [],
                "licenseMissing": True,
            }

    assignment_summaries: dict[str, Any] = {}
    for preset_id, license_entry in license_summaries.items():
        assigned = assignments_store.assigned_portfolios(preset_id)
        assignment_summary = _build_assignment_summary(preset_id, assigned, license_entry)
        assignment_summaries[preset_id] = assignment_summary

        warning_messages = assignment_summary.get("warningMessages", [])
        if warning_messages:
            merged_messages = _merge_unique(
                _normalize_sequence_of_messages(license_entry.get("warningMessages")),
                warning_messages,
            )
            license_entry["warningMessages"] = merged_messages

        warning_codes = assignment_summary.get("warningCodes", [])
        if warning_codes:
            merged_codes = _merge_unique(
                _normalize_sequence_of_messages(license_entry.get("warningCodes")),
                warning_codes,
            )
            license_entry["warningCodes"] = merged_codes

    if license_summaries:
        payload["licenseSummaries"] = license_summaries
    if assignment_summaries:
        payload["assignmentSummaries"] = assignment_summaries
        portfolio_summaries = _build_portfolio_summaries(assignment_summaries)
        if portfolio_summaries:
            payload["portfolioSummaries"] = portfolio_summaries
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
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


def _command_assign(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    store = _assignment_store(presets_dir)
    assigned = store.assign(args.preset_id, args.portfolio_id)
    result = {
        "preset_id": args.preset_id,
        "assigned_portfolios": list(assigned),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _command_unassign(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    store = _assignment_store(presets_dir)
    assigned = store.unassign(args.preset_id, args.portfolio_id)
    result = {
        "preset_id": args.preset_id,
        "assigned_portfolios": list(assigned),
    }
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

    assign_parser = subparsers.add_parser("assign", help="Przydziela preset do portfela")
    assign_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    assign_parser.add_argument("--portfolio-id", required=True, help="Identyfikator portfela")

    unassign_parser = subparsers.add_parser("unassign", help="Usuwa powiązanie presetu z portfelem")
    unassign_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    unassign_parser.add_argument("--portfolio-id", required=True, help="Identyfikator portfela")

    plan_parser = subparsers.add_parser(
        "plan",
        help="Oblicza plan instalacji presetów wraz z zależnościami i aktualizacjami",
    )
    plan_parser.add_argument(
        "--preset-id",
        action="append",
        required=True,
        help="Identyfikator presetu (można podać wiele razy)",
    )

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
    elif args.command == "assign":
        _command_assign(args)
    elif args.command == "unassign":
        _command_unassign(args)
    elif args.command == "plan":
        _command_plan(args)
    else:  # pragma: no cover - argparse zapewnia poprawność
        parser.error(f"Nieznane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    main()
