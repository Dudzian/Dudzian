from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.ai.inference import ModelRepository
from bot_core.reporting.model_quality import DEFAULT_QUALITY_DIR, load_champion_overview


DEFAULT_PERFORMANCE_GUARD: MutableMapping[str, object] = {
    "fps_target": 60,
    "reduce_motion_after_seconds": 1.0,
    "jank_threshold_ms": 18.0,
    "max_overlay_count": 3,
    "disable_secondary_when_fps_below": 0,
}


def _normalize_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_value(item) for item in value]
    return value


def _ensure_dict(value: object) -> MutableMapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return {}


def _default_repository(model: str | None) -> Path:
    model_name = model or "decision_engine"
    return Path("var") / "models" / model_name


def _default_quality_dir(repository: Path) -> Path:
    if repository.parent.name == "models":
        return repository.parent / "quality"
    return DEFAULT_QUALITY_DIR


def build_auto_mode_snapshot(
    *,
    model_name: str | None = None,
    repository: str | Path | None = None,
    quality_dir: str | Path | None = None,
) -> MutableMapping[str, object]:
    env_model = os.environ.get("BOT_CORE_UI_MODEL_NAME")
    model = (model_name or env_model or "decision_engine").strip() or "decision_engine"

    repo_override = os.environ.get("BOT_CORE_UI_MODEL_REPOSITORY")
    repo_path = Path(repository) if repository is not None else None
    if repo_override:
        repo_path = Path(repo_override)
    if repo_path is None:
        repo_path = _default_repository(model)
    repo_path = repo_path.expanduser().resolve()

    repository_obj = ModelRepository(repo_path)
    active_version = repository_obj.get_active_version()

    quality_override = os.environ.get("BOT_CORE_UI_MODEL_QUALITY_DIR")
    quality_path = Path(quality_dir) if quality_dir is not None else None
    if quality_override:
        quality_path = Path(quality_override)
    if quality_path is None:
        quality_path = _default_quality_dir(repo_path)
    quality_path = quality_path.expanduser().resolve()

    overview = load_champion_overview(model, base_dir=quality_path)
    if not overview:
        return {
            "model": model,
            "repository": str(repository_obj.base_path),
            "quality_dir": str(quality_path),
            "active_version": active_version,
            "decision_summary": {},
            "guardrail_summary": {},
            "controller_history": [],
            "recommendations": [],
            "performance_guard": dict(DEFAULT_PERFORMANCE_GUARD),
            "guardrail_state": {},
            "guardrail_trace": [],
            "risk_alerts": [],
            "decision_history": [],
            "model_events": [],
            "signal_quality": {},
            "failover": {},
            "retraining_cycles": [],
            "journal_performance": {"mode": "baseline"},
            "exchange_allocation": {
                "selected": None,
                "allocations": [],
                "history": [],
            },
            "performance_indicators": {
                "rolling_pnl": None,
                "max_drawdown_pct": None,
                "win_rate": None,
                "journal": {"mode": "baseline"},
                "strategy": {
                    "current": "neutral",
                    "state": "baseline",
                    "leverage": 1.0,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.04,
                    "history": [],
                },
                "exchange": {
                    "selected": None,
                    "allocations": [],
                    "history": [],
                },
            },
        }

    champion_report = _ensure_dict(overview.get("champion"))
    metadata = _ensure_dict(overview.get("champion_metadata"))
    challengers_raw = overview.get("challengers")
    challengers: list[MutableMapping[str, object]] = []
    if isinstance(challengers_raw, Sequence):
        for entry in challengers_raw:
            if not isinstance(entry, Mapping):
                continue
            challengers.append(
                {
                    "report": _ensure_dict(entry.get("report")),
                    "metadata": _ensure_dict(entry.get("metadata")),
                }
            )

    metrics_payload = _ensure_dict(champion_report.get("metrics"))
    summary_section = champion_report.get("metrics")
    summary_snapshot: MutableMapping[str, object] = {}
    if isinstance(summary_section, Mapping):
        summary_snapshot = _ensure_dict(summary_section.get("summary"))

    decision_summary: MutableMapping[str, object] = {
        "model": champion_report.get("model_name") or model,
        "version": champion_report.get("version") or active_version,
        "status": champion_report.get("status"),
        "evaluated_at": champion_report.get("evaluated_at"),
        "baseline_version": champion_report.get("baseline_version"),
        "trained_at": champion_report.get("trained_at"),
        "dataset_rows": champion_report.get("dataset_rows"),
        "metrics": metrics_payload,
        "summary": summary_snapshot,
        "delta": _ensure_dict(champion_report.get("delta")),
        "validation": _ensure_dict(champion_report.get("validation")),
        "decided_at": metadata.get("decided_at"),
        "reason": metadata.get("reason"),
        "active_version": active_version,
    }

    guardrail_summary: MutableMapping[str, object] = {
        "status": champion_report.get("status"),
        "decided_at": metadata.get("decided_at"),
        "reason": metadata.get("reason"),
        "delta": decision_summary.get("delta", {}),
        "validation": decision_summary.get("validation", {}),
    }

    controller_history: list[MutableMapping[str, object]] = []
    if metadata.get("decided_at") or metadata.get("reason"):
        controller_history.append(
            {
                "event": "champion",
                "timestamp": metadata.get("decided_at"),
                "reason": metadata.get("reason"),
                "model": decision_summary.get("version"),
                "status": champion_report.get("status"),
            }
        )
    for entry in challengers:
        report = entry.get("report", {})
        meta = entry.get("metadata", {})
        if not isinstance(report, Mapping):
            continue
        controller_history.append(
            {
                "event": "challenger",
                "timestamp": meta.get("decided_at") if isinstance(meta, Mapping) else None,
                "reason": meta.get("reason") if isinstance(meta, Mapping) else None,
                "model": report.get("version"),
                "status": report.get("status"),
            }
        )

    recommendations: list[MutableMapping[str, object]] = []
    directional = summary_snapshot.get("directional_accuracy")
    mae_value = summary_snapshot.get("mae")
    try:
        confidence = float(directional)
    except (TypeError, ValueError):
        confidence = 0.5
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    if isinstance(mae_value, (int, float)):
        reason_text = f"Champion {decision_summary['version']} MAE={float(mae_value):.4f}"
    else:
        reason_text = metadata.get("reason") or f"Champion {decision_summary['version']} aktywny"
    recommendations.append(
        {
            "mode": "champion",
            "confidence": confidence,
            "reason": reason_text,
            "model": decision_summary.get("version"),
            "suggested_actions": [
                f"Monitoruj champion {decision_summary.get('version')}",
            ],
        }
    )

    return {
        "model": model,
        "repository": str(repository_obj.base_path),
        "quality_dir": str(quality_path),
        "active_version": active_version,
        "decision_summary": decision_summary,
        "guardrail_summary": guardrail_summary,
        "controller_history": controller_history,
        "recommendations": recommendations,
        "champion": champion_report,
        "challengers": challengers,
        "performance_guard": dict(DEFAULT_PERFORMANCE_GUARD),
        "guardrail_state": {},
        "guardrail_trace": [],
        "risk_alerts": [],
        "decision_history": [],
        "model_events": [],
        "signal_quality": {},
        "failover": {},
        "retraining_cycles": [],
        "journal_performance": {"mode": "baseline"},
        "exchange_allocation": {
            "selected": None,
            "allocations": [],
            "history": [],
        },
        "performance_indicators": {
            "rolling_pnl": None,
            "max_drawdown_pct": None,
            "win_rate": None,
            "journal": {"mode": "baseline"},
            "strategy": {
                "current": "neutral",
                "state": "baseline",
                "leverage": 1.0,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "history": [],
            },
            "exchange": {
                "selected": None,
                "allocations": [],
                "history": [],
            },
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot_core.runtime.ui_bridge",
        description="Mostek danych runtime do aplikacji desktopowej",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("auto-mode-snapshot", help="Buduje snapshot automatyzacji na podstawie champion registry")
    snap.add_argument("--model", dest="model", help="Nazwa modelu decision engine")
    snap.add_argument("--repository", dest="repository", help="Ścieżka repozytorium modeli")
    snap.add_argument("--quality-dir", dest="quality_dir", help="Katalog historii jakości")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "auto-mode-snapshot":
        snapshot = build_auto_mode_snapshot(
            model_name=args.model,
            repository=args.repository,
            quality_dir=args.quality_dir,
        )
        json.dump(snapshot, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    parser.print_help()
    return 1


__all__ = ["build_auto_mode_snapshot", "main"]


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
