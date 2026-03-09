#!/usr/bin/env python3
"""Waliduje i normalizuje snapshoty long-polla dla raportu adapterów."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bot_core.config.long_poll import (
    LONG_POLL_FUTURE_GRACE_MINUTES,
    LONG_POLL_MAX_AGE_MINUTES,
)

DEFAULT_INPUT = Path("var/metrics/long_poll_snapshots.json")
DEFAULT_OUTPUT = Path("var/metrics/long_poll_snapshots.json")
DEFAULT_MAX_AGE_MINUTES = LONG_POLL_MAX_AGE_MINUTES
DEFAULT_FUTURE_GRACE_MINUTES = LONG_POLL_FUTURE_GRACE_MINUTES
_REQUIRED_ADAPTERS = ("deribit_futures", "bitmex_futures")
_REQUIRED_ENVIRONMENTS = ("paper", "live")


class SnapshotValidationError(RuntimeError):
    """Błąd walidacji wejściowych snapshotów long-polla."""


def _coerce_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, str) and value.strip():
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    return None


def _load_snapshots(path: Path) -> tuple[list[dict[str, Any]], dt.datetime | None]:
    if not path.exists():
        raise SnapshotValidationError(f"Brak pliku źródłowego snapshotów: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotValidationError(f"Niepoprawny JSON w {path}: {exc}") from exc

    default_collected_at: dt.datetime | None = None
    entries: Sequence[Any]
    if isinstance(payload, Mapping):
        default_collected_at = _coerce_datetime(payload.get("collected_at"))
        candidate = payload.get("snapshots") or payload.get("entries") or payload.get("data")
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            entries = candidate
        else:
            entries = (payload,)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        entries = payload
    else:
        entries = ()

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        labels = entry.get("labels")
        if not isinstance(labels, Mapping):
            continue

        adapter = str(labels.get("adapter") or "").strip()
        scope = str(labels.get("scope") or "").strip()
        environment = str(labels.get("environment") or "").strip()
        if not adapter or not scope or not environment:
            continue

        snapshot = dict(entry)
        snapshot["labels"] = {
            "adapter": adapter,
            "scope": scope,
            "environment": environment,
        }
        collected_at = _coerce_datetime(snapshot.get("collected_at")) or default_collected_at
        if collected_at is not None:
            snapshot["collected_at"] = collected_at.isoformat().replace("+00:00", "Z")
            snapshot["_collected_at"] = collected_at
        normalized.append(snapshot)

    if not normalized:
        raise SnapshotValidationError(f"Brak poprawnych snapshotów long-polla w {path}")

    return normalized, default_collected_at


def _validate_required_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for snapshot in snapshots:
        labels = snapshot.get("labels")
        if not isinstance(labels, Mapping):
            continue
        adapter = str(labels.get("adapter") or "").strip()
        environment = str(labels.get("environment") or "").strip()
        scope = str(labels.get("scope") or "").strip()
        if not scope:
            continue
        seen.add((adapter, environment))

    missing = [
        f"{adapter}:{environment}"
        for adapter in _REQUIRED_ADAPTERS
        for environment in _REQUIRED_ENVIRONMENTS
        if (adapter, environment) not in seen
    ]
    if missing:
        raise SnapshotValidationError(f"Brak wymaganych snapshotów long-polla: {', '.join(missing)}")


def _validate_freshness(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    max_age_minutes: float,
    future_grace_minutes: float,
    fallback_collected_at: dt.datetime | None,
) -> None:
    now = dt.datetime.now(tz=dt.timezone.utc)
    future_limit = now + dt.timedelta(minutes=max(0.0, future_grace_minutes))
    issues: list[str] = []

    for snapshot in snapshots:
        labels = snapshot.get("labels")
        if not isinstance(labels, Mapping):
            continue
        adapter = str(labels.get("adapter") or "").strip()
        environment = str(labels.get("environment") or "").strip()
        if adapter not in _REQUIRED_ADAPTERS or environment not in _REQUIRED_ENVIRONMENTS:
            continue

        collected_at = _coerce_datetime(snapshot.get("collected_at")) or fallback_collected_at
        if collected_at is None:
            issues.append(f"{adapter}:{environment} (brak collected_at)")
            continue
        if collected_at > future_limit:
            issues.append(
                f"{adapter}:{environment} (collected_at={collected_at.isoformat()} > now+grace)"
            )
            continue

        age_minutes = max(0.0, (now - collected_at).total_seconds() / 60.0)
        if age_minutes > max_age_minutes:
            issues.append(f"{adapter}:{environment} ({age_minutes:.2f} min > {max_age_minutes:.2f} min)")

    if issues:
        raise SnapshotValidationError(
            "Snapshoty long-polla nie przechodzą walidacji świeżości: " + ", ".join(issues)
        )


def prepare_snapshots(
    *,
    source_path: Path,
    output_path: Path,
    max_age_minutes: float,
    future_grace_minutes: float,
    skip_freshness: bool = False,
) -> Path:
    snapshots, default_collected_at = _load_snapshots(source_path)
    _validate_required_snapshots(snapshots)
    if not skip_freshness:
        _validate_freshness(
            snapshots,
            max_age_minutes=max_age_minutes,
            future_grace_minutes=future_grace_minutes,
            fallback_collected_at=default_collected_at,
        )

    output_payload = {
        "collected_at": (
            default_collected_at.isoformat().replace("+00:00", "Z") if default_collected_at else None
        ),
        "snapshots": [{key: value for key, value in snapshot.items() if key != "_collected_at"} for snapshot in snapshots],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Źródłowy plik JSON snapshotów")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Docelowy plik JSON snapshotów")
    parser.add_argument("--max-age-minutes", type=float, default=DEFAULT_MAX_AGE_MINUTES)
    parser.add_argument(
        "--future-grace-minutes",
        type=float,
        default=DEFAULT_FUTURE_GRACE_MINUTES,
        help="Dozwolony skew zegara: collected_at może być maksymalnie tyle minut w przyszłości.",
    )
    parser.add_argument(
        "--skip-freshness",
        action="store_true",
        help=(
            "Pomija walidację świeżości (TTL/clock-skew) i sprawdza tylko strukturę oraz wymagane profile. "
            "Tryb do workflow opartych o fixture'y."
        ),
    )
    args = parser.parse_args(argv)

    try:
        path = prepare_snapshots(
            source_path=args.input,
            output_path=args.output,
            max_age_minutes=float(args.max_age_minutes),
            future_grace_minutes=float(args.future_grace_minutes),
            skip_freshness=bool(args.skip_freshness),
        )
    except SnapshotValidationError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Zweryfikowano snapshoty long-polla: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
