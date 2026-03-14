#!/usr/bin/env python3
"""Mostek CLI udostępniający presety strategii i zarządzanie licencjami dla UI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sitecustomize  # noqa: F401  # zapewnia stabilne importy packaging/numpy

from packaging.version import InvalidVersion, Version

from bot_core.marketplace import (
    MarketplaceIndex,
    PresetRepository,
    build_marketplace_preset,
    parse_preset_document,
)
from bot_core.marketplace.assignments import PresetAssignmentStore
from bot_core.marketplace.preferences import PresetPreferenceStore
from bot_core.strategies.catalog import (
    StrategyCatalog,
    StrategyPresetProfile,
)
from bot_core.strategies.installer import MarketplacePresetInstaller
from bot_core.strategies.personalization.preferences import (
    PresetPreferencePersonalizer,
    UserPreferenceConfig,
)
from bot_core.security.hwid import HwIdProvider
from bot_core.security.license import summarize_license_payload
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reviews_meta_path(presets_dir: Path) -> Path:
    repository = PresetRepository(presets_dir)
    return repository.root / ".meta" / "reviews.json"


def _read_reviews_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"presets": {}}
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        return {"presets": {}}
    presets_payload = payload.get("presets")
    presets: dict[str, Any] = {}
    if isinstance(presets_payload, Mapping):
        for preset_id, entry in presets_payload.items():
            if isinstance(entry, Mapping):
                presets[str(preset_id)] = dict(entry)
    state: dict[str, Any] = {"presets": presets}
    source = payload.get("source")
    if isinstance(source, str) and source.strip():
        state["source"] = source
    updated = payload.get("updated_at") or payload.get("updatedAt")
    if isinstance(updated, str) and updated.strip():
        state["updated_at"] = updated
    return state


def _write_reviews_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == serialized:
        return
    path.write_text(serialized, encoding="utf-8")


def _extract_review_id(entry: Mapping[str, Any]) -> str:
    for key in ("review_id", "reviewId", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"review-{uuid4().hex}"


def _normalize_reports(value: Any) -> int:
    if isinstance(value, Mapping):
        return max(_coerce_int(value.get("count")) or 0, 0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        total = 0
        for item in value:
            total += _normalize_reports(item)
        return total
    numeric = _coerce_int(value)
    return max(numeric or 0, 0)


def _collect_review_documents(source_dir: Path) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    if not source_dir.exists():
        return documents
    for path in sorted(source_dir.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            continue
        preset_id = payload.get("preset_id") or payload.get("presetId") or path.stem
        if not isinstance(preset_id, str):
            preset_id = path.stem
        normalized = dict(payload)
        normalized.setdefault("preset_id", preset_id)
        documents[preset_id] = normalized
    return documents


def _verify_review_signature(
    entry: Mapping[str, Any], signing_keys: Mapping[str, bytes]
) -> tuple[bool, str | None]:
    signature = entry.get("signature")
    if not isinstance(signature, Mapping):
        return False, "Brak podpisu recenzji"
    key_id = signature.get("key_id") or signature.get("keyId")
    if not isinstance(key_id, str) or not key_id.strip():
        return False, "Recenzja nie zawiera identyfikatora klucza"
    key = signing_keys.get(key_id)
    if key is None:
        return False, f"Brak klucza '{key_id}' do weryfikacji recenzji"
    payload = {k: v for k, v in entry.items() if k != "signature"}
    if verify_hmac_signature(payload, signature, key=key):
        return True, None
    return False, "Niepoprawny podpis recenzji"


def _aggregate_review_documents(
    documents: Mapping[str, Mapping[str, Any]],
    signing_keys: Mapping[str, bytes],
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for preset_id, document in documents.items():
        reviews_payload = document.get("reviews")
        if not isinstance(reviews_payload, Sequence) or isinstance(reviews_payload, (str, bytes)):
            continue
        normalized_reviews: list[Mapping[str, Any]] = []
        warnings: list[str] = []
        rating_total = 0
        review_count = 0
        report_count = _normalize_reports(document.get("reports"))
        for raw_review in reviews_payload:
            if not isinstance(raw_review, Mapping):
                continue
            review_id = _extract_review_id(raw_review)
            ok, error = _verify_review_signature(raw_review, signing_keys)
            if not ok:
                warnings.append(f"{preset_id}: {review_id}: {error}")
                continue
            payload = {k: v for k, v in raw_review.items() if k != "signature"}
            rating = _coerce_int(payload.get("rating"))
            if rating is None or rating <= 0:
                warnings.append(f"{preset_id}: {review_id}: brak oceny")
                continue
            rating_total += rating
            review_count += 1
            comments = str(payload.get("comment") or "").strip()
            author = str(payload.get("author") or "community").strip() or "community"
            submitted_at = payload.get("submitted_at") or payload.get("submittedAt")
            review_warnings = _normalize_sequence_of_messages(payload.get("warnings"))
            warnings.extend(f"{preset_id}: {review_id}: {msg}" for msg in review_warnings)
            report_count += _normalize_reports(payload.get("reports"))
            normalized_reviews.append(
                {
                    "reviewId": review_id,
                    "rating": rating,
                    "comment": comments,
                    "author": author,
                    "submittedAt": submitted_at,
                }
            )
        warnings.extend(
            f"{preset_id}: {msg}"
            for msg in _normalize_sequence_of_messages(document.get("warnings"))
        )
        deduped_warnings = []
        for msg in warnings:
            if msg and msg not in deduped_warnings:
                deduped_warnings.append(msg)
        average_rating: float | None = None
        if review_count:
            average_rating = round(rating_total / review_count, 2)
        aggregate_entry = {
            "averageRating": average_rating,
            "reviewCount": review_count,
            "userReports": report_count,
            "warnings": deduped_warnings,
            "reviews": normalized_reviews,
        }
        last_synced_at = document.get("updated_at") or document.get("updatedAt")
        if isinstance(last_synced_at, str) and last_synced_at.strip():
            aggregate_entry["lastSyncedAt"] = last_synced_at
        aggregates[preset_id] = aggregate_entry
    return aggregates


def _sync_reviews_state(
    presets_dir: Path,
    source_dir: Path,
    signing_keys: Mapping[str, bytes],
) -> dict[str, Any]:
    documents = _collect_review_documents(source_dir)
    aggregates = _aggregate_review_documents(documents, signing_keys)
    previous_state = _read_reviews_state(_reviews_meta_path(presets_dir))
    previous_presets_raw = (
        previous_state.get("presets") if isinstance(previous_state, Mapping) else None
    )
    previous_presets = previous_presets_raw if isinstance(previous_presets_raw, Mapping) else {}
    for preset_id, aggregate in aggregates.items():
        if not isinstance(aggregate, dict) or aggregate.get("lastSyncedAt"):
            continue
        previous_entry = previous_presets.get(preset_id)
        if isinstance(previous_entry, Mapping):
            inherited = previous_entry.get("lastSyncedAt")
            if isinstance(inherited, str) and inherited.strip():
                aggregate["lastSyncedAt"] = inherited
    previous_aggregates = (
        previous_state.get("presets") if isinstance(previous_state, Mapping) else None
    )
    if not isinstance(previous_aggregates, Mapping):
        previous_aggregates = {}
    updated_at = (
        _utcnow_iso()
        if dict(previous_aggregates) != aggregates
        else previous_state.get("updated_at")
    )
    if not isinstance(updated_at, str) or not updated_at.strip():
        updated_at = _utcnow_iso()
    state = {
        "presets": aggregates,
        "source": str(source_dir),
        "updated_at": updated_at,
    }
    _write_reviews_state(_reviews_meta_path(presets_dir), state)
    return state


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


def _community_payload(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "averageRating": None,
        "reviewCount": 0,
        "userReports": 0,
        "warnings": [],
        "reviews": [],
    }
    if not isinstance(entry, Mapping):
        return payload
    average = entry.get("averageRating")
    if isinstance(average, (int, float)):
        payload["averageRating"] = round(float(average), 2)
    review_count = _coerce_int(entry.get("reviewCount"))
    if review_count is not None:
        payload["reviewCount"] = review_count
    user_reports = _coerce_int(entry.get("userReports"))
    if user_reports is not None:
        payload["userReports"] = user_reports
    payload["warnings"] = _normalize_sequence_of_messages(entry.get("warnings"))
    reviews_payload = entry.get("reviews")
    if isinstance(reviews_payload, Sequence) and not isinstance(reviews_payload, (str, bytes)):
        normalized: list[dict[str, Any]] = []
        for review in reviews_payload:
            if not isinstance(review, Mapping):
                continue
            normalized.append(
                {
                    "reviewId": review.get("reviewId") or review.get("review_id"),
                    "rating": review.get("rating"),
                    "comment": review.get("comment"),
                    "author": review.get("author"),
                    "submittedAt": review.get("submittedAt") or review.get("submitted_at"),
                }
            )
        payload["reviews"] = normalized
    synced_at = entry.get("lastSyncedAt") or entry.get("last_synced_at")
    if isinstance(synced_at, str) and synced_at.strip():
        payload["lastSyncedAt"] = synced_at
    return payload


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
            "Portfele oczekują na zatwierdzenie w licencji: " + ", ".join(pending_assignments)
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
    assignment_summaries: Mapping[str, Mapping[str, Any]],
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


def _sanitize_identifier(value: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "")]
    normalized = "".join(cleaned).strip("-")
    return normalized or "preset"


def _write_license_payload(directory: Path, preset_id: str, payload: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_sanitize_identifier(preset_id)}.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(serialized, encoding="utf-8")
    return target


def _assignment_store(presets_dir: Path) -> PresetAssignmentStore:
    repository = PresetRepository(presets_dir)
    return PresetAssignmentStore(repository.root / ".meta" / "assignments.json")


def _preference_store(presets_dir: Path) -> PresetPreferenceStore:
    repository = PresetRepository(presets_dir)
    return PresetPreferenceStore(repository.root / ".meta" / "preferences.json")


def _build_hwid_provider(override: str | None) -> HwIdProvider | None:
    if override is None:
        return HwIdProvider()
    if not override:
        return None
    return HwIdProvider(fingerprint_reader=lambda: override)


def _build_installer(
    repository: PresetRepository,
    *,
    catalog_path: Path | None,
    licenses_dir: Path | None,
    signing_keys: Mapping[str, bytes] | None,
    hwid_provider: HwIdProvider | None,
) -> MarketplacePresetInstaller:
    kwargs: dict[str, object] = {}
    if catalog_path is not None:
        kwargs["catalog_path"] = catalog_path
    if licenses_dir is not None:
        kwargs["licenses_dir"] = licenses_dir
    if signing_keys:
        kwargs["signing_keys"] = signing_keys
    if hwid_provider is not None:
        kwargs["hwid_provider"] = hwid_provider
    return MarketplacePresetInstaller(repository, **kwargs)


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
    reviews_state = _read_reviews_state(_reviews_meta_path(presets_dir))
    community_entries = (
        reviews_state.get("presets", {}) if isinstance(reviews_state, Mapping) else {}
    )

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
                    if not isinstance(warning_messages, Sequence) or isinstance(
                        warning_messages, (str, bytes)
                    ):
                        warning_messages = validation.get("warningMessages")
                    if isinstance(warning_messages, Sequence) and not isinstance(
                        warning_messages, (str, bytes)
                    ):
                        summary["warning_messages"] = [
                            str(item)
                            for item in warning_messages
                            if isinstance(item, str) and item.strip()
                        ]
                    warning_codes = validation.get("warning_codes")
                    if not isinstance(warning_codes, Sequence) or isinstance(
                        warning_codes, (str, bytes)
                    ):
                        warning_codes = validation.get("warningCodes")
                    if isinstance(warning_codes, Sequence) and not isinstance(
                        warning_codes, (str, bytes)
                    ):
                        summary["warnings"] = [
                            str(item)
                            for item in warning_codes
                            if isinstance(item, str) and item.strip()
                        ]
        summary.setdefault("warnings", [])
        summary.setdefault("warning_messages", [])
        summary["warningMessages"] = list(summary["warning_messages"])
        entry = None
        if isinstance(community_entries, Mapping):
            entry = community_entries.get(preset_id)
        community_payload = _community_payload(entry if isinstance(entry, Mapping) else None)
        summary["community"] = community_payload
        summary["community_rating"] = community_payload["averageRating"]
        summary["community_review_count"] = community_payload["reviewCount"]
        summary["community_user_reports"] = community_payload["userReports"]
        summary["communityWarnings"] = list(community_payload["warnings"])

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
        doc.preset_id: doc.version for doc in documents if doc.preset_id and doc.version
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


def _command_install_workflow(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    licenses_path = Path(args.licenses_path)
    license_dir = Path(args.licenses_dir).expanduser()
    repository = PresetRepository(presets_dir)
    assignments_store = PresetAssignmentStore(repository.root / ".meta" / "assignments.json")
    preferences_store = PresetPreferenceStore(repository.root / ".meta" / "preferences.json")

    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    hwid_provider = _build_hwid_provider(args.fingerprint)
    catalog_path = Path(args.catalog_path).expanduser() if args.catalog_path else None
    installer = _build_installer(
        repository,
        catalog_path=catalog_path,
        licenses_dir=license_dir,
        signing_keys=signing_keys or None,
        hwid_provider=hwid_provider,
    )

    license_store = _read_license_store(licenses_path)
    license_payload: Mapping[str, Any] | None = None
    license_file_path: Path | None = None
    if args.license_json:
        license_payload = _read_license_payload(args)
        license_store.setdefault("licenses", {})[args.preset_id] = dict(license_payload)
        _write_license_store(licenses_path, license_store)
        license_file_path = _write_license_payload(license_dir, args.preset_id, license_payload)
    else:
        existing = license_store.get("licenses") if isinstance(license_store, Mapping) else None
        candidate = existing.get(args.preset_id) if isinstance(existing, Mapping) else None
        if isinstance(candidate, Mapping):
            license_payload = dict(candidate)
        candidate_path = license_dir / f"{_sanitize_identifier(args.preset_id)}.json"
        if candidate_path.exists():
            license_file_path = candidate_path

    result = installer.install_from_catalog(args.preset_id)

    portfolio_ids = [
        value.strip()
        for value in (args.portfolio_id or [])
        if isinstance(value, str) and value.strip()
    ]
    assignments: dict[str, list[str]] = {}
    if result.success:
        for portfolio_id in portfolio_ids:
            assigned = assignments_store.assign(args.preset_id, portfolio_id)
            assignments[portfolio_id] = list(assigned)
    else:
        for portfolio_id in portfolio_ids:
            assignments[portfolio_id] = list(assignments_store.assigned_portfolios(args.preset_id))

    preference_entries: dict[str, Mapping[str, Any]] = {}
    if args.preferences_json and result.success:
        raw_preferences = _load_json(Path(args.preferences_json), stdin_fallback=True)
        if not isinstance(raw_preferences, Mapping):
            raise SystemExit("Payload preferencji musi być obiektem JSON.")
        preference_config = UserPreferenceConfig.from_mapping(raw_preferences)
        personalizer = PresetPreferencePersonalizer()
        overrides: Mapping[str, Mapping[str, Any]] = {}
        try:
            document, _ = repository.export_preset(args.preset_id)
        except FileNotFoundError:
            document = None
        if document is not None:
            overrides = personalizer.build_overrides(document, preference_config)
        for portfolio_id in portfolio_ids:
            entry = preferences_store.set_entry(
                args.preset_id,
                portfolio_id,
                preferences=preference_config.as_payload(),
                overrides=overrides,
            )
            preference_entries[portfolio_id] = entry

    install_payload = {
        "success": result.success,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "signatureVerified": result.signature_verified,
        "fingerprintVerified": result.fingerprint_verified,
    }

    output: dict[str, Any] = {
        "presetId": args.preset_id,
        "install": install_payload,
        "assignments": assignments,
        "stores": {
            "assignments": str(assignments_store.path),
            "preferences": str(preferences_store.path),
            "licenseIndex": str(licenses_path),
        },
    }
    if license_payload is not None:
        output["license"] = license_payload
    if license_file_path is not None:
        output["licenseFile"] = str(license_file_path)
    if result.installed_path is not None:
        output["installedPath"] = str(result.installed_path)
    if preference_entries:
        output["preferences"] = preference_entries

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
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


def _command_sync_reviews(args: argparse.Namespace) -> None:
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    if not signing_keys:
        raise SystemExit(
            "sync-reviews wymaga przekazania klucza HMAC (--signing-key lub --signing-key-file)"
        )
    presets_dir = Path(args.presets_dir)
    source_dir = Path(args.source_dir)
    state = _sync_reviews_state(presets_dir, source_dir, signing_keys)
    result = {
        "updated_at": state.get("updated_at"),
        "source": state.get("source"),
        "preset_count": len(state.get("presets", {})),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _command_submit_review(args: argparse.Namespace) -> None:
    presets_dir = Path(args.presets_dir)
    licenses_path = Path(args.licenses_path)
    reviews_dir = Path(args.reviews_dir)
    signing_keys = _load_signing_keys(args.signing_key or [], args.signing_key_file or [])
    if not signing_keys:
        raise SystemExit(
            "submit-review wymaga przekazania kluczy podpisów (--signing-key/--signing-key-file)"
        )
    review_key_id = args.review_key_id
    if not review_key_id:
        raise SystemExit(
            "submit-review wymaga parametru --review-key-id wskazującego klucz podpisu"
        )
    review_key = signing_keys.get(review_key_id)
    if review_key is None:
        raise SystemExit(
            f"Brak klucza '{review_key_id}' w mapie --signing-key; nie można podpisać recenzji"
        )
    rating = args.rating
    if rating < 1 or rating > 5:
        raise SystemExit("Ocena musi być w zakresie 1-5 gwiazdek")
    comment = args.comment.strip()
    if not comment:
        raise SystemExit("Komentarz recenzji nie może być pusty")
    store = _read_license_store(licenses_path)
    licenses = store.get("licenses") if isinstance(store, Mapping) else None
    license_entry = None
    if isinstance(licenses, Mapping):
        license_entry = licenses.get(args.preset_id)
    if not isinstance(license_entry, Mapping):
        raise SystemExit(
            "Brak aktywnej licencji na preset – recenzja wymaga posiadania ważnej licencji"
        )
    review_id = args.review_id or f"rvw-{uuid4().hex[:12]}"
    payload: dict[str, Any] = {
        "review_id": review_id,
        "preset_id": args.preset_id,
        "rating": rating,
        "comment": comment,
        "author": (args.author or "community").strip() or "community",
        "submitted_at": _utcnow_iso(),
    }
    warnings_payload = _normalize_sequence_of_messages(args.warning or [])
    if warnings_payload:
        payload["warnings"] = warnings_payload
    signature = build_hmac_signature(payload, key=review_key, key_id=review_key_id)
    document_path = reviews_dir / f"{_sanitize_identifier(args.preset_id)}.json"
    existing = _load_json(document_path) if document_path.exists() else None
    if isinstance(existing, Mapping):
        reviews = existing.get("reviews")
        if isinstance(reviews, list):
            reviews.append({**payload, "signature": signature})
        else:
            existing["reviews"] = [{**payload, "signature": signature}]
        existing["preset_id"] = args.preset_id
        existing["updated_at"] = _utcnow_iso()
        document = existing
    else:
        document = {
            "preset_id": args.preset_id,
            "updated_at": _utcnow_iso(),
            "reviews": [{**payload, "signature": signature}],
        }
    document_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    document_path.write_text(serialized, encoding="utf-8")
    state = _sync_reviews_state(presets_dir, reviews_dir, signing_keys)
    result = {
        "preset_id": args.preset_id,
        "review_id": review_id,
        "reviews_path": str(document_path),
        "community": state.get("presets", {}).get(args.preset_id),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets-dir", default="data/strategies", help="Katalog z plikami presetów JSON"
    )
    parser.add_argument(
        "--licenses-path",
        default="var/marketplace_licenses.json",
        help="Ścieżka do pliku przechowującego stan aktywacji presetów",
    )
    parser.add_argument(
        "--fingerprint", help="Nadpisanie fingerprintu sprzętowego (dla testów/diagnostyki)"
    )
    parser.add_argument(
        "--signing-key", action="append", help="Klucz HMAC w formacie KEY_ID=SECRET"
    )
    parser.add_argument(
        "--signing-key-file",
        action="append",
        help="Plik JSON ze słownikiem kluczy podpisów (key_id -> secret)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Lista dostępnych presetów")
    list_parser.add_argument(
        "--profile",
        choices=[p.value for p in StrategyPresetProfile],
        help="Filtr profilu strategii",
    )
    list_parser.add_argument(
        "--include-strategies",
        action="store_true",
        help="Dołącz pełne definicje strategii w wyniku",
    )

    activate_parser = subparsers.add_parser("activate", help="Aktywacja licencji presetu")
    activate_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    activate_parser.add_argument(
        "--license-json", help="Ścieżka do pliku JSON z licencją (domyślnie stdin)"
    )

    install_parser = subparsers.add_parser(
        "install",
        help="Instaluje preset z katalogu i przypisuje go do portfeli.",
    )
    install_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    install_parser.add_argument(
        "--portfolio-id",
        action="append",
        required=True,
        help="Id portfela docelowego (można podać wielokrotnie)",
    )
    install_parser.add_argument(
        "--license-json",
        help="Ścieżka do pliku licencji Marketplace (opcjonalnie)",
    )
    install_parser.add_argument(
        "--licenses-dir",
        default="var/licenses/presets",
        help="Katalog z indywidualnymi plikami licencji",
    )
    install_parser.add_argument(
        "--catalog-path",
        help="Katalog manifestu Marketplace (domyślnie wbudowany)",
    )
    install_parser.add_argument(
        "--preferences-json",
        help="Plik JSON z preferencjami użytkownika (budżet, target ryzyka)",
    )

    deactivate_parser = subparsers.add_parser("deactivate", help="Dezaktywacja licencji presetu")
    deactivate_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")

    assign_parser = subparsers.add_parser("assign", help="Przydziela preset do portfela")
    assign_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    assign_parser.add_argument("--portfolio-id", required=True, help="Identyfikator portfela")

    unassign_parser = subparsers.add_parser("unassign", help="Usuwa powiązanie presetu z portfelem")
    unassign_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    unassign_parser.add_argument("--portfolio-id", required=True, help="Identyfikator portfela")

    sync_reviews_parser = subparsers.add_parser(
        "sync-reviews",
        help="Synchronizuje recenzje community z repozytorium centralnym",
    )
    sync_reviews_parser.add_argument(
        "--source-dir",
        default="config/marketplace/reviews",
        help="Źródłowy katalog recenzji (domyślnie config/marketplace/reviews)",
    )

    submit_review_parser = subparsers.add_parser(
        "submit-review",
        help="Publikuje recenzję presetu i aktualizuje lokalne metryki community",
    )
    submit_review_parser.add_argument("--preset-id", required=True, help="Identyfikator presetu")
    submit_review_parser.add_argument(
        "--rating", type=int, required=True, help="Ocena 1-5 gwiazdek"
    )
    submit_review_parser.add_argument("--comment", required=True, help="Treść recenzji")
    submit_review_parser.add_argument("--author", help="Opcjonalny podpis recenzji")
    submit_review_parser.add_argument(
        "--reviews-dir",
        default="config/marketplace/reviews",
        help="Repozytorium recenzji community (domyślnie config/marketplace/reviews)",
    )
    submit_review_parser.add_argument(
        "--review-key-id",
        required=True,
        help="Identyfikator klucza HMAC używanego do podpisu recenzji",
    )
    submit_review_parser.add_argument("--review-id", help="Opcjonalne ręczne ID recenzji")
    submit_review_parser.add_argument(
        "--warning",
        action="append",
        help="Dodatkowe ostrzeżenia dołączane do recenzji (można podać wielokrotnie)",
    )

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
    elif args.command == "install":
        _command_install_workflow(args)
    elif args.command == "deactivate":
        _command_deactivate(args)
    elif args.command == "assign":
        _command_assign(args)
    elif args.command == "unassign":
        _command_unassign(args)
    elif args.command == "sync-reviews":
        _command_sync_reviews(args)
    elif args.command == "submit-review":
        _command_submit_review(args)
    elif args.command == "plan":
        _command_plan(args)
    else:  # pragma: no cover - argparse zapewnia poprawność
        parser.error(f"Nieznane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    main()
