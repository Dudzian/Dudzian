"""Obsługa raportów jakości modeli AI zapisywanych lokalnie."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

DEFAULT_QUALITY_DIR = Path("var/models/quality")
CHAMPION_FILENAME = "champion.json"
CHALLENGERS_FILENAME = "challengers.json"
MAX_CHALLENGERS = 10


@dataclass(slots=True)
class ChampionDecision:
    """Decyzja o statusie champion/challenger dla raportu jakości."""

    model_name: str
    decision: str
    reason: str
    decided_at: datetime
    report_path: Path | None
    candidate: Mapping[str, object]
    champion: Mapping[str, object]
    challengers: tuple[Mapping[str, object], ...]
    previous_champion: Mapping[str, object] | None = None

    @property
    def champion_changed(self) -> bool:
        return self.decision == "champion" and (
            self.previous_champion is None or self.previous_champion != self.champion
        )


def persist_quality_report(
    payload: Mapping[str, object],
    *,
    model_name: str,
    version: str,
    evaluated_at: datetime,
    base_dir: Path | str | None = None,
) -> Path:
    """Zapisuje raport jakości do archiwum i aktualizuje wskaźnik latest."""

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = evaluated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{version}-{timestamp}.json"
    target = model_dir / filename
    _write_json(target, payload)

    latest_path = model_dir / "latest.json"
    temp_path = latest_path.with_suffix(".tmp")
    _write_json(temp_path, payload)
    temp_path.replace(latest_path)
    return target


def load_latest_quality_payload(
    model_name: str,
    *,
    base_dir: Path | str | None = None,
) -> Mapping[str, object] | None:
    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    path = root / model_name / "latest.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - I/O zależne od środowiska
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> Mapping[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):  # pragma: no cover - obrona przed uszkodzonymi plikami
        return None
    return payload if isinstance(payload, Mapping) else None


def _score_payload(payload: Mapping[str, object]) -> tuple[float, float, float]:
    metrics_raw = payload.get("metrics")
    metrics: Mapping[str, object]
    if isinstance(metrics_raw, Mapping):
        summary_raw = metrics_raw.get("summary")
        metrics = summary_raw if isinstance(summary_raw, Mapping) else metrics_raw
    else:
        metrics = {}

    directional = 0.0
    mae = math.inf
    expected_pnl = 0.0

    def _float(value: object, default: float) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    candidates: Sequence[str] = (
        "validation_directional_accuracy",
        "test_directional_accuracy",
        "directional_accuracy",
    )
    for key in candidates:
        if key in metrics:
            directional = max(directional, _float(metrics[key], directional))

    mae_candidates: Sequence[str] = ("validation_mae", "test_mae", "mae")
    for key in mae_candidates:
        if key in metrics:
            mae = min(mae, _float(metrics[key], mae))

    pnl_candidates: Sequence[str] = (
        "validation_expected_pnl",
        "test_expected_pnl",
        "expected_pnl",
    )
    for key in pnl_candidates:
        if key in metrics:
            expected_pnl = max(expected_pnl, _float(metrics[key], expected_pnl))

    if not math.isfinite(mae):
        mae = math.inf

    return (directional, -mae, expected_pnl)


def _normalize_entries(entries: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    normalized: list[Mapping[str, object]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            normalized.append(entry)
    return normalized


def update_champion_registry(
    payload: Mapping[str, object],
    *,
    model_name: str,
    base_dir: Path | str | None = None,
    report_path: Path | None = None,
    challenger_limit: int = MAX_CHALLENGERS,
) -> ChampionDecision:
    """Aktualizuje rejestr champion/challenger dla wskazanego modelu."""

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    champion_path = model_dir / CHAMPION_FILENAME
    challengers_path = model_dir / CHALLENGERS_FILENAME

    champion_data = _load_json(champion_path)
    previous_champion = (
        champion_data.get("report") if isinstance(champion_data, Mapping) else None
    )
    champion_score = _score_payload(previous_champion) if previous_champion else None

    candidate_score = _score_payload(payload)
    status = str(payload.get("status", "")).strip().lower()
    now = datetime.now(timezone.utc)
    decision = "challenger"
    reason = ""
    champion_payload = previous_champion

    def _append_challenger(
        entries: list[Mapping[str, object]],
        report: Mapping[str, object],
        *,
        note: str,
    ) -> None:
        metadata: MutableMapping[str, object] = {
            "decided_at": now.isoformat(),
            "reason": note,
            "report": dict(report),
        }
        entries.insert(0, metadata)

    challengers_data = _load_json(challengers_path) or {}
    existing_entries_raw = challengers_data.get("entries")
    challengers: list[Mapping[str, object]]
    if isinstance(existing_entries_raw, Sequence):
        challengers = _normalize_entries(existing_entries_raw)
    else:
        challengers = []

    promote_candidate = False
    if status == "degraded":
        reason = "Model oznaczony jako 'degraded' został zapisany jako challenger"
    elif champion_score is None:
        promote_candidate = True
        reason = "Brak aktywnego champion – awansowano kandydata"
    elif candidate_score > champion_score:
        promote_candidate = True
        if candidate_score[0] > champion_score[0]:
            reason = "Wyższa dokładność kierunkowa"
        elif candidate_score[1] > champion_score[1]:
            reason = "Niższy błąd MAE"
        else:
            reason = "Lepsza oczekiwana rentowność"
    else:
        reason = "Model nie poprawił metryk champion – zapisano jako challenger"

    if promote_candidate:
        champion_payload = payload
        decision = "champion"
        _write_json(
            champion_path,
            {
                "model_name": model_name,
                "decided_at": now.isoformat(),
                "reason": reason,
                "report": dict(payload),
            },
        )
        if previous_champion is not None:
            _append_challenger(
                challengers,
                previous_champion,
                note="Zastąpiony przez nowszy model",
            )
    else:
        _append_challenger(challengers, payload, note=reason)

    if challenger_limit > 0:
        challengers = challengers[:challenger_limit]

    _write_json(
        challengers_path,
        {
            "model_name": model_name,
            "updated_at": now.isoformat(),
            "entries": challengers,
        },
    )

    champion_payload = champion_payload or {}
    challengers_reports: list[Mapping[str, object]] = []
    for entry in challengers:
        report = entry.get("report") if isinstance(entry, Mapping) else None
        if isinstance(report, Mapping):
            challengers_reports.append(report)

    return ChampionDecision(
        model_name=model_name,
        decision=decision,
        reason=reason,
        decided_at=now,
        report_path=report_path,
        candidate=payload,
        champion=champion_payload,
        challengers=tuple(challengers_reports),
        previous_champion=previous_champion if isinstance(previous_champion, Mapping) else None,
    )


def promote_challenger(
    model_name: str,
    version: str,
    *,
    base_dir: Path | str | None = None,
    reason: str | None = None,
    decided_at: datetime | None = None,
) -> ChampionDecision:
    """Promuje istniejącego challengera do roli championa."""

    normalized_model = str(model_name).strip()
    normalized_version = str(version).strip()
    if not normalized_model:
        raise ValueError("model_name must be a non-empty string")
    if not normalized_version:
        raise ValueError("version must be a non-empty string")

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    model_dir = root / normalized_model
    champion_path = model_dir / CHAMPION_FILENAME
    challengers_path = model_dir / CHALLENGERS_FILENAME

    champion_data = _load_json(champion_path) or {}
    challengers_data = _load_json(challengers_path) or {}
    entries_raw = challengers_data.get("entries") if isinstance(challengers_data, Mapping) else None
    entries = _normalize_entries(entries_raw) if isinstance(entries_raw, Sequence) else []

    candidate_entry: Mapping[str, object] | None = None
    remaining_entries: list[Mapping[str, object]] = []
    for entry in entries:
        report = entry.get("report") if isinstance(entry, Mapping) else None
        if not isinstance(report, Mapping):
            continue
        candidate_version_raw = report.get("version")
        candidate_version = str(candidate_version_raw).strip() if candidate_version_raw else ""
        if candidate_version == normalized_version and candidate_entry is None:
            candidate_entry = entry
            continue
        remaining_entries.append(entry)

    if candidate_entry is None:
        raise KeyError(
            f"Brak challengera wersji '{normalized_version}' dla modelu '{normalized_model}'"
        )

    candidate_report_raw = candidate_entry.get("report") if isinstance(candidate_entry, Mapping) else None
    if not isinstance(candidate_report_raw, Mapping):
        raise ValueError("Wybrany challenger nie zawiera raportu jakości")

    now = decided_at.astimezone(timezone.utc) if decided_at is not None else datetime.now(timezone.utc)
    reason_text = (reason or str(candidate_entry.get("reason", "")).strip()) or (
        f"Ręczna promocja challengera {normalized_version}"
    )

    previous_champion_raw = champion_data.get("report") if isinstance(champion_data, Mapping) else None
    previous_champion = (
        dict(previous_champion_raw)
        if isinstance(previous_champion_raw, Mapping)
        else None
    )

    champion_payload = dict(candidate_report_raw)
    champion_record = {
        "model_name": normalized_model,
        "decided_at": now.isoformat(),
        "reason": reason_text,
        "report": champion_payload,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(champion_path, champion_record)

    updated_entries: list[Mapping[str, object]] = []
    if previous_champion:
        updated_entries.append(
            {
                "decided_at": now.isoformat(),
                "reason": f"Zastąpiony przez {normalized_version} (promocja ręczna)",
                "report": previous_champion,
            }
        )

    for entry in remaining_entries:
        if not isinstance(entry, Mapping):
            continue
        report = entry.get("report")
        if not isinstance(report, Mapping):
            continue
        normalized_entry = {str(key): value for key, value in entry.items()}
        normalized_entry["report"] = dict(report)
        updated_entries.append(normalized_entry)

    if MAX_CHALLENGERS > 0:
        updated_entries = updated_entries[:MAX_CHALLENGERS]

    _write_json(
        challengers_path,
        {
            "model_name": normalized_model,
            "updated_at": now.isoformat(),
            "entries": updated_entries,
        },
    )

    challengers_reports: list[Mapping[str, object]] = []
    for entry in updated_entries:
        report = entry.get("report") if isinstance(entry, Mapping) else None
        if isinstance(report, Mapping):
            challengers_reports.append(report)

    return ChampionDecision(
        model_name=normalized_model,
        decision="champion",
        reason=reason_text,
        decided_at=now,
        report_path=None,
        candidate=champion_payload,
        champion=champion_payload,
        challengers=tuple(challengers_reports),
        previous_champion=previous_champion,
    )


def load_champion_overview(
    model_name: str,
    *,
    base_dir: Path | str | None = None,
) -> Mapping[str, object] | None:
    """Zwraca podsumowanie champion/challenger dla danego modelu."""

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    champion_path = root / model_name / CHAMPION_FILENAME
    champion_data = _load_json(champion_path)
    if not champion_data:
        return None

    challengers_path = root / model_name / CHALLENGERS_FILENAME
    challengers_data = _load_json(challengers_path) or {}
    entries_raw = challengers_data.get("entries")
    entries = (
        _normalize_entries(entries_raw) if isinstance(entries_raw, Sequence) else []
    )

    def _extract_entry(entry: Mapping[str, object]) -> Mapping[str, object]:
        report = entry.get("report") if isinstance(entry, Mapping) else None
        payload = report if isinstance(report, Mapping) else {}
        metadata = {
            "decided_at": entry.get("decided_at"),
            "reason": entry.get("reason"),
        }
        return {
            "report": payload,
            "metadata": metadata,
        }

    challengers_payload = [_extract_entry(entry) for entry in entries]

    champion_report = champion_data.get("report")
    if not isinstance(champion_report, Mapping):
        champion_report = {}

    champion_metadata = {
        "decided_at": champion_data.get("decided_at"),
        "reason": champion_data.get("reason"),
    }

    return {
        "model_name": model_name,
        "champion": champion_report,
        "champion_metadata": champion_metadata,
        "challengers": challengers_payload,
        "updated_at": champion_metadata.get("decided_at"),
        "base_directory": str(root),
    }


def list_tracked_models(*, base_dir: Path | str | None = None) -> Sequence[str]:
    """Zwraca listę modeli posiadających historię jakości."""

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    if not root.exists():
        return ()
    return tuple(sorted(entry.name for entry in root.iterdir() if entry.is_dir()))


__all__ = [
    "CHAMPION_FILENAME",
    "CHALLENGERS_FILENAME",
    "ChampionDecision",
    "DEFAULT_QUALITY_DIR",
    "load_champion_overview",
    "load_latest_quality_payload",
    "list_tracked_models",
    "persist_quality_report",
    "update_champion_registry",
]
