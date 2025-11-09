"""Uruchamia pojedynczy cykl retreningu i generuje raport."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import yaml

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.reporting.model_quality import (
    DEFAULT_QUALITY_DIR,
    list_tracked_models,
    load_champion_overview,
    promote_challenger,
)
from core.ml.training_pipeline import TrainingPipeline, TrainingPipelineResult
from core.monitoring.events import MonitoringEvent
from core.reporting import RetrainingReport
from core.runtime.retraining_scheduler import ChaosSettings, RetrainingRunOutcome, RetrainingScheduler

LOGGER = logging.getLogger("run_retraining_cycle")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uruchamia pojedynczy cykl retreningu Decision Engine i zapisuje raport",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/runtime_retraining.yml"),
        help="Ścieżka do konfiguracji retrainingu (interval, chaos)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Opcjonalny dataset treningowy w formacie JSON (features/targets)",
    )
    parser.add_argument(
        "--preferred-backend",
        action="append",
        dest="preferred_backends",
        default=[],
        help="Preferowane backendy ML (można podać wielokrotnie w kolejności priorytetu)",
    )
    parser.add_argument(
        "--backends-config",
        type=Path,
        help="Plik YAML z konfiguracją backendów ML (backends.yml)",
    )
    parser.add_argument(
        "--fallback-log-dir",
        type=Path,
        default=Path("logs/ml/fallback"),
        help="Katalog do zapisu logów fallbacku backendów",
    )
    parser.add_argument(
        "--validation-log-dir",
        type=Path,
        default=Path("logs/data/validation"),
        help="Katalog do zapisu raportów walidacji datasetów",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/retraining"),
        help="Katalog, w którym zostanie zapisany raport retreningu",
    )
    parser.add_argument(
        "--kpi-snapshot-dir",
        type=Path,
        default=Path("reports/e2e/retraining"),
        help="Katalog docelowy dla snapshotów KPI wykorzystywanych w testach E2E",
    )
    parser.add_argument(
        "--e2e-log-dir",
        type=Path,
        default=Path("logs/e2e/retraining"),
        help="Katalog do zapisu logów scenariusza E2E retreningu",
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=DEFAULT_QUALITY_DIR,
        help="Katalog champion/challenger używany do auto-promocji modeli",
    )
    parser.add_argument(
        "--auto-promote-model",
        action="append",
        dest="auto_promote_models",
        default=None,
        help="Nazwa modelu przeznaczonego do auto-promocji (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _load_retraining_config(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}
    if not path.exists():
        LOGGER.warning("Plik konfiguracji retrainingu %s nie istnieje – używam wartości domyślnych", path)
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - zabezpieczenie przed błędnym formatem
        raise SystemExit("Konfiguracja retrainingu musi być mapą klucz→wartość")
    return payload


def _load_dataset(path: Path | None) -> FeatureDataset:
    if path is None:
        return FeatureDataset(vectors=tuple(_synthetic_vectors()), metadata={"source": "synthetic"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    targets = payload.get("targets")
    if not isinstance(features, Sequence) or not isinstance(targets, Sequence):
        raise SystemExit("Dataset musi zawierać listy 'features' i 'targets'")
    if len(features) != len(targets) or not features:
        raise SystemExit("Dataset musi zawierać co najmniej jedną próbkę z targetem")
    symbol = str(payload.get("symbol", "SYNTH"))
    timestamp = float(payload.get("start_timestamp", 1_700_000_000.0))
    vectors: list[FeatureVector] = []
    for idx, (feature_map, target) in enumerate(zip(features, targets)):
        if not isinstance(feature_map, Mapping):
            raise SystemExit("Każdy element 'features' musi być słownikiem")
        vectors.append(
            FeatureVector(
                timestamp=timestamp + idx * 60.0,
                symbol=symbol,
                features={str(key): float(value) for key, value in feature_map.items()},
                target_bps=float(target),
            )
        )
    metadata = {
        "source": str(path),
        "row_count": len(vectors),
        "feature_names": sorted({key for vector in vectors for key in vector.features}),
    }
    return FeatureDataset(vectors=tuple(vectors), metadata=metadata)


def _synthetic_vectors() -> Iterable[FeatureVector]:
    base_timestamp = 1_700_000_000.0
    for idx in range(5):
        yield FeatureVector(
            timestamp=base_timestamp + idx * 60.0,
            symbol="SYNTH",
            features={
                "momentum": float(idx) * 0.2,
                "volatility": 0.3 + float(idx) * 0.1,
            },
            target_bps=0.01 * (idx - 2),
        )


def _write_kpi_snapshot(report: RetrainingReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = report.generated_at.strftime("%Y%m%dT%H%M%S")
    payload = {
        "generated_at": report.generated_at.isoformat(),
        "status": report.status,
        "backend": report.backend,
        "kpi": dict(report.kpi),
        "alerts": list(report.alerts),
        "errors": list(report.errors),
        "dataset_metadata": dict(report.dataset_metadata),
        "fallback_chain": [dict(item) for item in report.fallback_chain],
    }
    path = directory / f"kpi_{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_execution_log(
    *,
    directory: Path,
    started_at: datetime,
    finished_at: datetime,
    outcome: RetrainingRunOutcome,
    report_json: Path,
    report_markdown: Path,
    kpi_snapshot: Path,
    training_result: TrainingPipelineResult | None,
    promotion_summary: Mapping[str, object] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = finished_at.strftime("%Y%m%dT%H%M%S")
    payload: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "status": outcome.status,
        "reason": outcome.reason,
        "delay_seconds": outcome.delay_seconds,
        "drift_score": outcome.drift_score,
        "report_json": str(report_json),
        "report_markdown": str(report_markdown),
        "kpi_snapshot": str(kpi_snapshot),
    }
    if training_result and training_result.log_path:
        payload["fallback_log"] = str(training_result.log_path)
    if training_result and training_result.validation_log_path:
        payload["validation_log"] = str(training_result.validation_log_path)
    if promotion_summary is not None:
        payload["promotion"] = promotion_summary
    path = directory / f"retraining_run_{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _score_report_payload(payload: Mapping[str, object]) -> tuple[float, float, float]:
    metrics_raw = payload.get("metrics") if isinstance(payload, Mapping) else None
    metrics: Mapping[str, object]
    if isinstance(metrics_raw, Mapping):
        summary_raw = metrics_raw.get("summary")
        metrics = summary_raw if isinstance(summary_raw, Mapping) else metrics_raw
    else:
        metrics = {}

    def _float(value: object, default: float) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    directional = 0.0
    mae = math.inf
    expected_pnl = 0.0

    for key in ("validation_directional_accuracy", "test_directional_accuracy", "directional_accuracy"):
        if key in metrics:
            directional = max(directional, _float(metrics[key], directional))

    for key in ("validation_mae", "test_mae", "mae"):
        if key in metrics:
            mae = min(mae, _float(metrics[key], mae))

    for key in ("validation_expected_pnl", "test_expected_pnl", "expected_pnl"):
        if key in metrics:
            expected_pnl = max(expected_pnl, _float(metrics[key], expected_pnl))

    if not math.isfinite(mae):
        mae = math.inf

    return (directional, -mae, expected_pnl)


def _select_auto_promotion_candidate(overview: Mapping[str, object]) -> Mapping[str, object] | None:
    champion_payload_raw = overview.get("champion") if isinstance(overview, Mapping) else None
    champion_payload = champion_payload_raw if isinstance(champion_payload_raw, Mapping) else {}
    champion_version = str(champion_payload.get("version", "")).strip()
    champion_score = _score_report_payload(champion_payload) if champion_payload else None

    challengers = overview.get("challengers") if isinstance(overview, Mapping) else None
    if not isinstance(challengers, Sequence):
        return None

    best_report: Mapping[str, object] | None = None
    best_score: tuple[float, float, float] | None = None
    for entry in challengers:
        if not isinstance(entry, Mapping):
            continue
        report = entry.get("report")
        if not isinstance(report, Mapping):
            continue
        status = str(report.get("status", "")).strip().lower()
        if status == "degraded":
            continue
        version = str(report.get("version", "")).strip()
        if not version or version == champion_version:
            continue
        score = _score_report_payload(report)
        if champion_score is not None and score <= champion_score:
            continue
        if best_score is None or score > best_score:
            best_report = report
            best_score = score

    return best_report


def _promotion_prerequisites(report: RetrainingReport) -> tuple[bool, MutableMapping[str, object]]:
    details: MutableMapping[str, object] = {
        "status": report.status,
        "alerts": list(report.alerts),
        "errors": list(report.errors),
    }
    if report.status != "completed":
        details["failed_condition"] = "status"
        return False, details
    if report.errors:
        details["failed_condition"] = "errors"
        return False, details
    if report.alerts:
        details["failed_condition"] = "alerts"
        return False, details

    fallback_raw = report.kpi.get("fallback_count")
    try:
        fallback_count = int(fallback_raw)
    except (TypeError, ValueError):
        fallback_count = 0
    details["fallback_count"] = fallback_count
    if fallback_count:
        details["failed_condition"] = "fallback"
        return False, details

    return True, details


def _auto_promote_models(
    report: RetrainingReport,
    *,
    quality_dir: Path,
    model_filters: Sequence[str] | None,
) -> Mapping[str, object]:
    summary: MutableMapping[str, object] = {
        "status": "skipped",
        "reason": None,
        "decisions": [],
        "timestamp": report.generated_at.isoformat(),
        "quality_dir": str(quality_dir),
    }

    should_run, details = _promotion_prerequisites(report)
    summary["details"] = details
    if not should_run:
        summary["reason"] = details.get("failed_condition", "conditions_not_met")
        return summary

    quality_root = quality_dir.expanduser()
    if not quality_root.exists():
        summary["reason"] = "quality_dir_missing"
        return summary

    if model_filters:
        models = [str(model).strip() for model in model_filters if str(model).strip()]
    else:
        models = list(list_tracked_models(base_dir=quality_root))
    if not models:
        summary["reason"] = "no_models"
        return summary

    ordered_models = list(dict.fromkeys(models))
    summary["status"] = "executed"
    summary["reason"] = None
    summary["models_considered"] = ordered_models

    decisions: list[MutableMapping[str, object]] = []
    for model_name in ordered_models:
        overview = load_champion_overview(model_name, base_dir=quality_root)
        if not overview:
            decisions.append(
                {
                    "model": model_name,
                    "decision": "missing_overview",
                    "reason": "Brak champion/challenger w katalogu jakości",
                }
            )
            continue

        champion_payload = overview.get("champion") if isinstance(overview, Mapping) else {}
        champion_version = None
        if isinstance(champion_payload, Mapping):
            champion_version = str(champion_payload.get("version", "")).strip() or None

        candidate = _select_auto_promotion_candidate(overview)
        if candidate is None:
            decisions.append(
                {
                    "model": model_name,
                    "decision": "noop",
                    "reason": "Brak challengera z lepszymi metrykami",
                    "champion_version": champion_version,
                }
            )
            continue

        candidate_version = str(candidate.get("version", "")).strip()
        if not candidate_version:
            decisions.append(
                {
                    "model": model_name,
                    "decision": "invalid_candidate",
                    "reason": "Challenger bez wersji",
                }
            )
            continue

        decision = promote_challenger(
            model_name,
            candidate_version,
            base_dir=quality_root,
            reason=f"Auto-promocja po retreningu ({report.generated_at.isoformat()})",
        )

        champion_payload_final = (
            dict(decision.champion) if isinstance(decision.champion, Mapping) else {}
        )
        previous_payload = (
            dict(decision.previous_champion)
            if isinstance(decision.previous_champion, Mapping)
            else None
        )
        decisions.append(
            {
                "model": model_name,
                "decision": decision.decision,
                "reason": decision.reason,
                "candidate_version": candidate_version,
                "champion_version": champion_payload_final.get("version"),
                "previous_champion": previous_payload,
            }
        )

    summary["decisions"] = decisions
    return summary


@dataclass(slots=True)
class RetrainingRunResult:
    outcome: RetrainingRunOutcome
    training_result: TrainingPipelineResult | None


async def _execute_cycle(
    scheduler: RetrainingScheduler,
    pipeline: TrainingPipeline,
    dataset: FeatureDataset,
) -> RetrainingRunResult:
    async def _train() -> TrainingPipelineResult:
        return await asyncio.to_thread(pipeline.train, dataset)

    outcome = await scheduler.run_once(_train)
    training_result: TrainingPipelineResult | None
    if isinstance(outcome.result, TrainingPipelineResult):
        training_result = outcome.result
    else:
        training_result = None
    return RetrainingRunResult(outcome=outcome, training_result=training_result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    dataset = _load_dataset(args.dataset)
    preferred_backends = tuple(args.preferred_backends or ("lightgbm", "reference"))

    pipeline = TrainingPipeline(
        preferred_backends=preferred_backends,
        config_path=args.backends_config,
        fallback_log_dir=args.fallback_log_dir,
        validation_log_dir=args.validation_log_dir,
    )

    captured_events: list[MonitoringEvent] = []
    config = _load_retraining_config(args.config)
    interval_minutes = float(config.get("interval_minutes", 180))
    chaos = ChaosSettings.from_mapping(config.get("chaos")) if hasattr(ChaosSettings, "from_mapping") else ChaosSettings()
    if not hasattr(ChaosSettings, "from_mapping"):
        LOGGER.warning("Używana wersja ChaosSettings nie wspiera from_mapping – stosuję konstruktor domyślny")
        chaos = ChaosSettings(**(config.get("chaos", {}) or {}))
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=interval_minutes),
        chaos=chaos,
        event_publisher=captured_events.append,
    )

    started_at = datetime.now(timezone.utc)
    result = asyncio.run(_execute_cycle(scheduler, pipeline, dataset))
    finished_at = datetime.now(timezone.utc)

    report = RetrainingReport.from_execution(
        started_at=started_at,
        finished_at=finished_at,
        outcome=result.outcome,
        training_result=result.training_result,
        events=tuple(captured_events),
        dataset_metadata=dict(dataset.metadata),
    )

    promotion_summary = _auto_promote_models(
        report,
        quality_dir=args.quality_dir,
        model_filters=args.auto_promote_models,
    )
    if promotion_summary.get("status") == "executed":
        promoted_entries = [
            f"{entry['model']}→{entry.get('champion_version', 'n/d')}"
            for entry in promotion_summary.get("decisions", [])
            if entry.get("decision") == "champion"
        ]
        if promoted_entries:
            LOGGER.info("Auto-promocja championów: %s", ", ".join(promoted_entries))
        else:
            LOGGER.info("Auto-promocja wykonana bez zmiany championa")
    else:
        LOGGER.info(
            "Auto-promocja pominięta (powód: %s)",
            promotion_summary.get("reason", "n/d"),
        )

    markdown_path = report.write_markdown(args.report_dir)
    report_payload = dict(report.to_dict())
    report_payload["promotion"] = promotion_summary

    report_dir = Path(args.report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"retraining_{report.generated_at.strftime('%Y%m%dT%H%M%S')}.json"
    json_path = report_dir / filename
    json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Raport retreningu zapisany w %s oraz %s", markdown_path, json_path)

    kpi_snapshot_path = _write_kpi_snapshot(report, args.kpi_snapshot_dir)
    run_log_path = _write_execution_log(
        directory=args.e2e_log_dir,
        started_at=started_at,
        finished_at=finished_at,
        outcome=result.outcome,
        report_json=json_path,
        report_markdown=markdown_path,
        kpi_snapshot=kpi_snapshot_path,
        training_result=result.training_result,
        promotion_summary=promotion_summary,
    )
    LOGGER.info("Snapshot KPI zapisany w %s, log scenariusza w %s", kpi_snapshot_path, run_log_path)

    if result.training_result and result.training_result.fallback_chain:
        LOGGER.warning(
            "Aktywowano fallback backendów: %s",
            ", ".join(entry.get("backend", "n/d") for entry in result.training_result.fallback_chain),
        )
    if result.training_result and result.training_result.validation_log_path:
        LOGGER.info(
            "Raport walidacji datasetu zapisany w %s",
            result.training_result.validation_log_path,
        )

    print(json.dumps(report_payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - wywołanie z CLI
    raise SystemExit(main())
