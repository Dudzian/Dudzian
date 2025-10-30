#!/usr/bin/env python3
"""Migracja do zunifikowanej konfiguracji ``config/runtime.yaml``.

Skrypt odczytuje istniejący `config/core.yaml`, wykorzystuje dostępne
entrypointy runtime (jeżeli są zdefiniowane), profile ryzyka i konfigurację
licencji, a następnie generuje zunifikowany plik `runtime.yaml`.  Wartości
domyślne zostały zachowane tak, aby odpowiadały wcześniejszemu układowi
konfiguracji (AI, risk, licensing, UI).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    RuntimeAISettings,
    RuntimeAppConfig,
    RuntimeCoreReference,
    RuntimeLicensingSettings,
    RuntimeRiskSettings,
    RuntimeTradingSettings,
    RuntimeUISettings,
    RuntimeEntrypointConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = REPO_ROOT / "config/core.yaml"
DEFAULT_RUNTIME = REPO_ROOT / "config/runtime.yaml"


def _strip_nulls(value: Any) -> Any:
    """Usuń wartości puste tak, aby YAML pozostał czytelny."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _strip_nulls(item)
            if normalized in (None, {}, [], ()):  # pomijamy puste struktury
                continue
            cleaned[key] = normalized
        return cleaned
    if isinstance(value, list):
        return [item for item in (_strip_nulls(elem) for elem in value) if item not in (None, {}, [], ())]
    return value


def _as_dict(dataclass_obj: Any) -> Any:
    if is_dataclass(dataclass_obj):
        return {key: _as_dict(val) for key, val in asdict(dataclass_obj).items()}
    if isinstance(dataclass_obj, Mapping):
        return {key: _as_dict(val) for key, val in dataclass_obj.items()}
    if isinstance(dataclass_obj, (list, tuple)):
        return type(dataclass_obj)(_as_dict(val) for val in dataclass_obj)
    return dataclass_obj


def build_runtime_config(core_path: Path, *, model_registry: str = "models") -> RuntimeAppConfig:
    core_config = load_core_config(core_path)

    if core_config.runtime_entrypoints:
        entrypoints = dict(core_config.runtime_entrypoints)
        default_entrypoint = next(iter(entrypoints))
    else:
        first_environment = next(iter(core_config.environments.values()))
        generated_name = f"auto_{first_environment.name}"
        entrypoints = {
            generated_name: RuntimeEntrypointConfig(
                environment=first_environment.name,
                description=f"Auto-generated entrypoint for {first_environment.exchange}",
                controller=None,
                strategy=getattr(first_environment, "default_strategy", None),
                risk_profile=first_environment.risk_profile,
                tags=("auto-generated",),
                bootstrap=True,
                trusted_auto_confirm=False,
            )
        }
        default_entrypoint = generated_name

    runtime_core = RuntimeCoreReference(path=str(core_path.name))
    ai_settings = RuntimeAISettings(model_registry_path=model_registry)
    trading_settings = RuntimeTradingSettings(
        default_entrypoint=default_entrypoint,
        entrypoints=entrypoints,
        auto_start=False,
        enable_paper_mode=True,
        enable_live_mode=False,
        strategy_overrides={},
    )
    risk_settings = RuntimeRiskSettings(
        service=core_config.risk_service,
        decision_log=core_config.risk_decision_log,
        portfolio_log=core_config.portfolio_decision_log,
        max_drawdown_alert_pct=None,
    )
    licensing_settings = RuntimeLicensingSettings(
        enforcement=True,
        grace_period_hours=24.0,
        offline_activation_required=True,
        license=core_config.license,
    )
    ui_settings = RuntimeUISettings(
        theme="dark",
        workspace_root="ui/workspaces/default",
        enable_advanced_mode=True,
        auto_connect_runtime=True,
        restore_layout_on_start=True,
        telemetry_sink="logs/ui_metrics.jsonl",
    )

    return RuntimeAppConfig(
        core=runtime_core,
        ai=ai_settings,
        trading=trading_settings,
        risk=risk_settings,
        licensing=licensing_settings,
        ui=ui_settings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generuj zunifikowaną konfigurację runtime")
    parser.add_argument("--core", default=str(DEFAULT_CORE), help="Ścieżka do config/core.yaml")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RUNTIME),
        help="Docelowy plik runtime.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz istniejący plik runtime.yaml",
    )
    parser.add_argument(
        "--model-registry",
        default="models",
        help="Ścieżka do katalogu modeli AI (domyślnie 'models')",
    )
    args = parser.parse_args()

    core_path = Path(args.core).expanduser()
    if not core_path.exists():
        parser.error(f"Nie znaleziono pliku konfiguracyjnego: {core_path}")

    runtime_path = Path(args.output).expanduser()
    if runtime_path.exists() and not args.force:
        parser.error(f"Plik {runtime_path} już istnieje (użyj --force aby nadpisać)")

    runtime_config = build_runtime_config(core_path, model_registry=args.model_registry)
    payload = _strip_nulls(_as_dict(runtime_config))

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    print(f"Zapisano konfigurację runtime do {runtime_path}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
