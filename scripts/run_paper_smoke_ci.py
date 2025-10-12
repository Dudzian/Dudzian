"""Orkiestracja smoke testu paper trading na potrzeby pipeline'u CI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.render_paper_smoke_summary import DEFAULT_MAX_JSON_CHARS, render_summary_markdown


_LOGGER = logging.getLogger(__name__)
_RAW_OUTPUT_LIMIT = 2000


def _normalize_sha256_fingerprint(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    if ":" in candidate:
        candidate = candidate.split(":", 1)[-1]
    candidate = candidate.replace(":", "")
    if not candidate:
        return None
    if any(char not in "0123456789abcdef" for char in candidate):
        return None
    return candidate


try:  # pragma: no cover - zależne od środowiska CI
    import yaml  # type: ignore
except Exception:  # pragma: no cover - instalacja PyYAML może być opcjonalna
    yaml = None  # type: ignore[assignment]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia smoke test strategii Daily Trend w trybie paper trading "
            "z pełną automatyzacją publikacji artefaktów."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska papierowego z pliku konfiguracyjnego",
    )
    parser.add_argument(
        "--output-dir",
        default="var/paper_smoke_ci",
        help="Katalog roboczy na raporty, logi audytowe i podsumowanie smoke",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Nazwa operatora zapisywana w logach audytowych (domyślnie PAPER_SMOKE_OPERATOR)",
    )
    parser.add_argument(
        "--allow-auto-publish-failure",
        action="store_true",
        help="Nie wymagaj sukcesu auto-publikacji artefaktów",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wyświetl polecenie bez jego wykonywania",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania skryptu",
    )
    parser.add_argument(
        "--run-daily-trend-arg",
        action="append",
        default=[],
        help=(
            "Dodatkowe argumenty przekazywane do run_daily_trend.py. "
            "Wartość może zawierać wiele parametrów rozdzielonych spacjami; "
            "w razie potrzeby użyj cudzysłowów zgodnych z powłoką. "
            "Opcję można powtarzać."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help=(
            "Opcjonalna ścieżka pliku środowiskowego (KEY=VALUE), do którego zostaną dopisane "
            "kluczowe informacje o wyniku smoke testu (ścieżki raportów, statusy)."
        ),
    )
    parser.add_argument(
        "--render-summary-markdown",
        default=None,
        help=(
            "Jeśli podano, zapisze raport Markdown wygenerowany na bazie summary.json w tej ścieżce."
        ),
    )
    parser.add_argument(
        "--render-summary-title",
        default=None,
        help="Opcjonalny tytuł raportu Markdown (przekazywany do renderera).",
    )
    parser.add_argument(
        "--render-summary-max-json-chars",
        type=int,
        default=None,
        help=(
            "Maksymalna liczba znaków JSON w sekcjach 'details' raportu Markdown (domyślnie 2000)."
        ),
    )
    parser.add_argument(
        "--skip-summary-validation",
        action="store_true",
        help="Pomiń automatyczną walidację summary.json po zakończeniu smoke testu.",
    )
    parser.add_argument(
        "--summary-validator-arg",
        action="append",
        default=[],
        help=(
            "Dodatkowe argumenty przekazywane do validate_paper_smoke_summary.py. "
            "Wartość może zawierać kilka parametrów rozdzielonych spacjami i może być powtarzana."
        ),
    )
    parser.add_argument(
        "--risk-profile-bundle-dir",
        default=None,
        help=(
            "Katalog, do którego zostanie zapisany pakiet presetów telemetrycznych. "
            "Domyślnie <output-dir>/telemetry/bundle."
        ),
    )
    parser.add_argument(
        "--risk-profile-stage",
        dest="risk_profile_stage",
        action="append",
        default=[],
        metavar="NAME=PROFILE",
        help=(
            "Nadpisz mapowanie etapów bundla (np. --risk-profile-stage live=manual). "
            "Opcję można powtarzać; domyślnie generowane są etapy demo/paper/live."
        ),
    )
    parser.add_argument(
        "--risk-profile-bundle-env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help="Styl formatowania plików środowiskowych generowanych przez bundler (dotenv/export).",
    )
    parser.add_argument(
        "--risk-profile-bundle-config-format",
        choices=("yaml", "json"),
        default="yaml",
        help="Format pliku konfiguracyjnego MetricsService generowanego przez bundler (yaml/json).",
    )
    return parser.parse_args(argv)


def _build_command(
    *,
    config_path: Path,
    environment: str,
    output_dir: Path,
    operator: str,
    auto_publish_required: bool,
    extra_run_daily_trend_args: Sequence[str],
) -> tuple[list[str], dict[str, Path]]:
    script_path = Path(__file__).with_name("run_daily_trend.py")
    if not script_path.exists():  # pragma: no cover - brak pliku to błąd środowiska
        raise FileNotFoundError("run_daily_trend.py not found next to run_paper_smoke_ci.py")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "paper_smoke_summary.json"
    json_log_path = output_dir / "paper_trading_log.jsonl"
    audit_log_path = output_dir / "paper_trading_log.md"
    precheck_dir = output_dir / "paper_precheck_reports"
    precheck_dir.mkdir(parents=True, exist_ok=True)
    smoke_runs_dir = output_dir / "runs"
    smoke_runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--environment",
        environment,
        "--paper-smoke",
        "--paper-smoke-json-log",
        str(json_log_path),
        "--paper-smoke-audit-log",
        str(audit_log_path),
        "--paper-precheck-audit-dir",
        str(precheck_dir),
        "--paper-smoke-summary-json",
        str(summary_path),
        "--paper-smoke-operator",
        operator,
        "--smoke-output",
        str(smoke_runs_dir),
        "--archive-smoke",
    ]

    cmd.append("--paper-smoke-auto-publish")
    if auto_publish_required:
        cmd.append("--paper-smoke-auto-publish-required")

    for raw in extra_run_daily_trend_args:
        if not raw:
            continue
        cmd.extend(shlex.split(raw))

    return cmd, {
        "summary": summary_path,
        "json_log": json_log_path,
        "audit_log": audit_log_path,
    }


def _load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise RuntimeError(
            "Brak pliku podsumowania smoke. Upewnij się, że run_daily_trend zakończył się poprawnie."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _truncate_text(text: str, *, limit: int = _RAW_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = max(limit - 3, 0)
    return text[:half] + "..."


def _resolve_config_path(base_dir: Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).expanduser()
    return path.resolve(strict=False)


def _load_core_yaml(config_path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "Automatyczne artefakty telemetryjne wymagają biblioteki PyYAML (pip install pyyaml)"
        )

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Plik konfiguracji {config_path} nie istnieje") from exc
    except OSError as exc:
        raise RuntimeError(f"Nie udało się odczytać konfiguracji {config_path}: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise RuntimeError("Plik core.yaml powinien zawierać mapę konfiguracji (YAML mapping)")

    return loaded


def _load_telemetry_requirements(config_path: Path, environment: str) -> dict[str, Any]:
    loaded = _load_core_yaml(config_path)

    runtime_cfg = loaded.get("runtime") if isinstance(loaded.get("runtime"), Mapping) else {}
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    metrics_cfg = runtime_cfg.get("metrics_service")
    if not isinstance(metrics_cfg, Mapping):
        metrics_cfg = {}

    metrics_path_raw = (
        metrics_cfg.get("ui_alerts_jsonl_path")
        or metrics_cfg.get("jsonl_path")
        or metrics_cfg.get("ui_alerts_log_path")
    )
    risk_profile = metrics_cfg.get("ui_alerts_risk_profile") or metrics_cfg.get("risk_profile")
    profiles_file_raw = (
        metrics_cfg.get("ui_alerts_risk_profiles_file")
        or metrics_cfg.get("risk_profiles_file")
    )

    env_cfg: Mapping[str, Any] | None = None
    environments_cfg = loaded.get("environments")
    if isinstance(environments_cfg, Mapping):
        raw_env = environments_cfg.get(environment)
        if isinstance(raw_env, Mapping):
            env_cfg = raw_env

    environment_profile = None
    if env_cfg is not None:
        raw_profile = env_cfg.get("risk_profile")
        if isinstance(raw_profile, str) and raw_profile.strip():
            environment_profile = raw_profile.strip()

    profile_source = "metrics_service" if risk_profile else None
    if not risk_profile and environment_profile:
        risk_profile = environment_profile
        profile_source = "environment"

    return {
        "metrics_path": _resolve_config_path(config_path.parent, metrics_path_raw),
        "risk_profile": risk_profile.strip().lower() if isinstance(risk_profile, str) else None,
        "risk_profile_source": profile_source,
        "risk_profiles_file": _resolve_config_path(config_path.parent, profiles_file_raw),
        "environment_profile": environment_profile.strip().lower() if isinstance(environment_profile, str) else None,
    }


def _load_security_baseline_settings(config_path: Path) -> dict[str, Any]:
    loaded = _load_core_yaml(config_path)

    runtime_cfg = loaded.get("runtime") if isinstance(loaded.get("runtime"), Mapping) else {}
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}

    baseline_cfg = runtime_cfg.get("security_baseline")
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}

    signing_cfg = baseline_cfg.get("signing")
    if not isinstance(signing_cfg, Mapping):
        signing_cfg = {}

    def _clean_text(value: Any) -> str | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        return text or None

    key_env = _clean_text(signing_cfg.get("signing_key_env"))
    key_value = _clean_text(signing_cfg.get("signing_key_value"))
    key_id = _clean_text(signing_cfg.get("signing_key_id"))
    key_path_raw = signing_cfg.get("signing_key_path")
    key_path = _resolve_config_path(config_path.parent, key_path_raw) if key_path_raw else None

    return {
        "signing_key_env": key_env,
        "signing_key_value": key_value,
        "signing_key_path": key_path,
        "signing_key_id": key_id,
        "require_signature": bool(signing_cfg.get("require_signature", False)),
    }


def _load_manifest_requirements(config_path: Path, environment: str) -> dict[str, Any]:
    loaded = _load_core_yaml(config_path)

    environments_cfg = loaded.get("environments")
    env_cfg = environments_cfg.get(environment) if isinstance(environments_cfg, Mapping) else None
    if not isinstance(env_cfg, Mapping):
        raise RuntimeError(f"Środowisko '{environment}' nie istnieje w konfiguracji core.yaml")

    data_cache_raw = env_cfg.get("data_cache_path")
    if not isinstance(data_cache_raw, str) or not data_cache_raw.strip():
        raise RuntimeError(
            "Środowisko nie definiuje data_cache_path – wymagane do oceny manifestu danych"
        )

    data_cache_path = _resolve_config_path(config_path.parent, data_cache_raw)
    if data_cache_path is None:
        raise RuntimeError("Nie udało się rozwiązać ścieżki data_cache_path dla środowiska")

    manifest_path = (data_cache_path / "ohlcv_manifest.sqlite").expanduser()

    stage_raw = env_cfg.get("environment")
    stage = stage_raw.strip().lower() if isinstance(stage_raw, str) else None
    risk_profile_raw = env_cfg.get("risk_profile")
    risk_profile = risk_profile_raw.strip().lower() if isinstance(risk_profile_raw, str) else None

    reporting_cfg = loaded.get("reporting") if isinstance(loaded.get("reporting"), Mapping) else {}
    manifest_cfg = (
        reporting_cfg.get("manifest_metrics")
        if isinstance(reporting_cfg.get("manifest_metrics"), Mapping)
        else {}
    )
    deny_status_raw = manifest_cfg.get("deny_status") if isinstance(manifest_cfg, Mapping) else None

    deny_status: list[str] = []
    if isinstance(deny_status_raw, (list, tuple, set)):
        for item in deny_status_raw:
            if isinstance(item, str) and item.strip():
                deny_status.append(item.strip().lower())
    elif isinstance(deny_status_raw, str) and deny_status_raw.strip():
        deny_status.append(deny_status_raw.strip().lower())

    signing_cfg_raw = manifest_cfg.get("signing") if isinstance(manifest_cfg, Mapping) else None
    signing_cfg: dict[str, str] = {}
    if isinstance(signing_cfg_raw, Mapping):
        key_env = signing_cfg_raw.get("key_env")
        if isinstance(key_env, str) and key_env.strip():
            signing_cfg["key_env"] = key_env.strip()
        key_file = signing_cfg_raw.get("key_file")
        if isinstance(key_file, str) and key_file.strip():
            signing_cfg["key_file"] = key_file.strip()
        key_value = signing_cfg_raw.get("key")
        if isinstance(key_value, str) and key_value.strip():
            signing_cfg["key"] = key_value
        key_id = signing_cfg_raw.get("key_id")
        if isinstance(key_id, str) and key_id.strip():
            signing_cfg["key_id"] = key_id.strip()
        require = signing_cfg_raw.get("require")
        if isinstance(require, bool):
            signing_cfg["require"] = require

    return {
        "manifest_path": manifest_path,
        "stage": stage,
        "risk_profile": risk_profile,
        "deny_status": deny_status,
        "signing": signing_cfg,
    }


def _ensure_script_exists(name: str) -> Path:
    script_path = Path(__file__).with_name(name)
    if not script_path.exists():  # pragma: no cover - brak pliku to błąd środowiska
        raise RuntimeError(f"Nie znaleziono skryptu pomocniczego {name} w katalogu scripts/")
    return script_path


def _run_watch_metrics_summary(
    *,
    metrics_path: Path,
    risk_profile: str,
    risk_profiles_file: Path | None,
    core_config: Path,
    output_dir: Path,
    risk_profile_source: str | None,
) -> tuple[Path, Path, Mapping[str, Any]]:
    if not metrics_path.exists():
        raise RuntimeError(
            f"Brak artefaktu telemetryjnego {metrics_path} – upewnij się, że smoke test wygenerował snapshoty UI"
        )

    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    summary_path = telemetry_dir / "ui_alerts_summary.json"
    decision_log_path = telemetry_dir / "ui_alerts_decision_log.jsonl"

    script_path = _ensure_script_exists("watch_metrics_stream.py")

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--from-jsonl",
        str(metrics_path),
        "--summary",
        "--summary-output",
        str(summary_path),
        "--decision-log",
        str(decision_log_path),
        "--format",
        "json",
        "--core-config",
        str(core_config),
        "--risk-profile",
        risk_profile,
    ]

    if risk_profiles_file is not None:
        cmd.extend(["--risk-profiles-file", str(risk_profiles_file)])

    if risk_profile_source:
        _LOGGER.info(
            "Generuję podpisane podsumowanie telemetryczne UI (%s) przy pomocy watch_metrics_stream.py",
            risk_profile,
        )
    else:
        _LOGGER.info(
            "Generuję podpisane podsumowanie telemetryczne UI przy pomocy watch_metrics_stream.py"
        )

    completed = subprocess.run(cmd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "watch_metrics_stream.py zakończył się niepowodzeniem – weryfikacja telemetryczna nie może zostać dokończona"
        )

    if not summary_path.exists():
        raise RuntimeError("watch_metrics_stream.py nie wygenerował pliku podsumowania telemetrycznego")

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Nieprawidłowy JSON podsumowania telemetrycznego: {exc}") from exc

    return summary_path, decision_log_path, payload


def _render_risk_profile_snippets(
    *,
    risk_profile: str,
    risk_profiles_file: Path | None,
    core_config: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    script_path = _ensure_script_exists("telemetry_risk_profiles.py")
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    slug = risk_profile.replace("/", "_")
    env_path = telemetry_dir / f"{slug}_metrics.env"
    yaml_path = telemetry_dir / f"{slug}_metrics.yaml"

    base_args: list[str] = [sys.executable, str(script_path)]
    if risk_profiles_file is not None:
        base_args.extend(["--risk-profiles-file", str(risk_profiles_file)])
    base_args.extend(["--core-config", str(core_config)])

    env_cmd = [*base_args, "render", risk_profile, "--format", "env", "--output", str(env_path)]
    yaml_cmd = [*base_args, "render", risk_profile, "--format", "yaml", "--output", str(yaml_path)]

    _LOGGER.info(
        "Generuję snippety konfiguracji telemetrycznej dla profilu %s przy pomocy telemetry_risk_profiles.py",
        risk_profile,
    )

    env_completed = subprocess.run(env_cmd, text=True, check=False)
    if env_completed.returncode != 0:
        raise RuntimeError("Nie udało się wygenerować pliku .env z nadpisaniami profilu ryzyka")

    yaml_completed = subprocess.run(yaml_cmd, text=True, check=False)
    if yaml_completed.returncode != 0:
        raise RuntimeError("Nie udało się wygenerować pliku YAML z nadpisaniami profilu ryzyka")

    return env_path, yaml_path


def _prepare_bundle_stage_args(
    stage_args: Sequence[str] | None,
    *,
    default_paper_profile: str,
) -> list[str]:
    prepared: list[str] = []
    seen: set[str] = set()
    for raw in stage_args or []:
        if "=" not in raw:
            raise ValueError(
                "Opcja --risk-profile-stage wymaga formatu etap=profil (np. --risk-profile-stage demo=conservative)"
            )
        stage, profile = raw.split("=", 1)
        normalized_stage = stage.strip().lower()
        normalized_profile = profile.strip().lower()
        if not normalized_stage or not normalized_profile:
            raise ValueError("Opcja --risk-profile-stage wymaga niepustych nazw etapu oraz profilu")
        prepared.append(f"{normalized_stage}={normalized_profile}")
        seen.add(normalized_stage)
    if default_paper_profile and "paper" not in seen:
        prepared.append(f"paper={default_paper_profile.strip().lower()}")
    return prepared


def _run_risk_profile_bundle(
    *,
    risk_profile: str,
    risk_profiles_file: Path | None,
    core_config: Path,
    output_dir: Path,
    env_style: str,
    config_format: str,
    stage_args: Sequence[str],
) -> tuple[Path, Mapping[str, Any], list[str]]:
    script_path = _ensure_script_exists("telemetry_risk_profiles.py")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args: list[str] = [sys.executable, str(script_path)]
    if risk_profiles_file is not None:
        base_args.extend(["--risk-profiles-file", str(risk_profiles_file)])
    base_args.extend(["--core-config", str(core_config)])

    cmd: list[str] = [
        *base_args,
        "bundle",
        "--output-dir",
        str(output_dir),
        "--env-style",
        env_style,
        "--config-format",
        config_format,
    ]

    for entry in stage_args:
        cmd.extend(["--stage", entry])

    _LOGGER.info(
        "Generuję pakiet presetów telemetrycznych (%s) przy pomocy telemetry_risk_profiles.py",
        risk_profile,
    )

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            "telemetry_risk_profiles.py bundle zakończył się niepowodzeniem: "
            + (stderr or f"exit_code={completed.returncode}")
        )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise RuntimeError("telemetry_risk_profiles.py bundle nie zwrócił manifestu JSON")

    try:
        manifest = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Nieprawidłowy JSON manifestu bundla: {exc}") from exc

    manifest_path_raw = manifest.get("manifest_path") if isinstance(manifest, Mapping) else None
    if isinstance(manifest_path_raw, str) and manifest_path_raw.strip():
        manifest_path = Path(manifest_path_raw).expanduser()
    else:
        manifest_path = output_dir / "manifest.json"

    return manifest_path, manifest, cmd


def _run_decision_log_verifier(
    *,
    decision_log_path: Path,
    summary_path: Path,
    risk_profile: str,
    risk_profiles_file: Path | None,
    env_snippet: Path,
    yaml_snippet: Path,
    core_config: Path,
    output_dir: Path,
    required_auth_scopes: Sequence[str] | None = None,
    risk_cli_args: Sequence[str] | None = None,
) -> tuple[int, Path, Mapping[str, Any] | None, Sequence[str]]:
    script_path = _ensure_script_exists("verify_decision_log.py")
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    report_path = telemetry_dir / "decision_log_report.json"

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        str(decision_log_path),
        "--summary-json",
        str(summary_path),
        "--report-output",
        str(report_path),
        "--risk-profile",
        risk_profile,
        "--risk-profile-env-snippet",
        str(env_snippet),
        "--risk-profile-yaml-snippet",
        str(yaml_snippet),
        "--core-config",
        str(core_config),
    ]

    if risk_profiles_file is not None:
        cmd.extend(["--risk-profiles-file", str(risk_profiles_file)])

    scopes_argument: list[str] = []
    if required_auth_scopes:
        for scope in sorted({scope.strip().lower() for scope in required_auth_scopes if scope.strip()}):
            scopes_argument.extend(["--require-auth-scope", scope])

    if scopes_argument:
        cmd.extend(scopes_argument)

    if risk_cli_args:
        cmd.extend(risk_cli_args)

    _LOGGER.info(
        "Waliduję decision log telemetryczny (%s) przy pomocy verify_decision_log.py",
        decision_log_path,
    )

    completed = subprocess.run(cmd, text=True, check=False)

    report_payload: Mapping[str, Any] | None = None
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report_payload = None

    return completed.returncode, report_path, report_payload, tuple(cmd)


def _collect_telemetry_artifacts(
    *,
    config_path: Path,
    environment: str,
    output_dir: Path,
    bundle_dir: Path | None,
    bundle_env_style: str,
    bundle_config_format: str,
    bundle_stage_args: Sequence[str] | None,
) -> dict[str, Any]:
    requirements = _load_telemetry_requirements(config_path, environment)
    metrics_path = requirements.get("metrics_path")
    risk_profile = requirements.get("risk_profile")

    if metrics_path is None:
        raise RuntimeError(
            "Konfiguracja core.yaml nie definiuje ścieżki runtime.metrics_service.ui_alerts_jsonl_path"
        )
    if not isinstance(metrics_path, Path):
        raise RuntimeError("Ścieżka do artefaktu telemetrycznego powinna być typu Path")
    if not risk_profile:
        raise RuntimeError(
            "Brak przypisanego profilu ryzyka telemetryjnego (runtime.metrics_service.ui_alerts_risk_profile)"
        )

    env_path, yaml_path = _render_risk_profile_snippets(
        risk_profile=risk_profile,
        risk_profiles_file=requirements.get("risk_profiles_file"),
        core_config=config_path,
        output_dir=output_dir,
    )

    summary_path, decision_log_path, summary_payload = _run_watch_metrics_summary(
        metrics_path=metrics_path,
        risk_profile=risk_profile,
        risk_profiles_file=requirements.get("risk_profiles_file"),
        core_config=config_path,
        output_dir=output_dir,
        risk_profile_source=requirements.get("risk_profile_source"),
    )

    required_auth_scope_set: set[str] = set()
    auth_scope_details: dict[str, Any] = {}

    def _normalize_scope(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip().lower()
        return text or None

    def _record_auth_scopes(service: str, meta: Mapping[str, Any], source: str) -> None:
        required_scopes: set[str] = set()
        primary_scope = _normalize_scope(meta.get("auth_token_scope_required"))
        if primary_scope:
            required_scopes.add(primary_scope)
        scopes_field = meta.get("auth_token_scopes")
        if isinstance(scopes_field, (list, tuple, set)):
            for candidate in scopes_field:
                normalized = _normalize_scope(candidate)
                if normalized:
                    required_scopes.add(normalized)
        required_map = meta.get("required_scopes")
        if isinstance(required_map, Mapping):
            for maybe_scope in required_map.keys():
                normalized = _normalize_scope(maybe_scope)
                if normalized:
                    required_scopes.add(normalized)
        if not required_scopes:
            return
        required_auth_scope_set.update(required_scopes)
        details = auth_scope_details.setdefault(
            service,
            {"required_scopes": [], "sources": []},
        )
        merged = set(details["required_scopes"]) | required_scopes
        details["required_scopes"] = sorted(merged)
        details["sources"].append({"source": source, "metadata": meta})

    risk_metadata_sources: list[tuple[str, Mapping[str, Any]]] = []

    if isinstance(summary_payload, Mapping):
        metadata = summary_payload.get("metadata")
        if isinstance(metadata, Mapping):
            core_config_section = metadata.get("core_config")
            if isinstance(core_config_section, Mapping):
                risk_core_section = core_config_section.get("risk_service")
                if isinstance(risk_core_section, Mapping):
                    risk_metadata_sources.append(("core_config.risk_service", risk_core_section))
            risk_summary_section = metadata.get("risk_service")
            if isinstance(risk_summary_section, Mapping):
                risk_metadata_sources.append(("summary", risk_summary_section))

            for key, value in metadata.items():
                if key == "core_config" and isinstance(value, Mapping):
                    for service_key, service_meta in value.items():
                        if isinstance(service_meta, Mapping):
                            _record_auth_scopes(service_key, service_meta, f"core_config.{service_key}")
                    continue
                if key in {"metrics_service", "risk_service"} and isinstance(value, Mapping):
                    _record_auth_scopes(key, value, "summary")

    required_auth_scopes = sorted(required_auth_scope_set)

    risk_cli_args: list[str] = []
    risk_requirements_details: dict[str, Any] = {}
    combined_risk_meta: dict[str, Any] = {}
    if risk_metadata_sources:
        for source, meta in risk_metadata_sources:
            for key, value in meta.items():
                if value is not None:
                    combined_risk_meta[key] = value
        if combined_risk_meta.get("tls_enabled"):
            risk_cli_args.append("--require-risk-service-tls")
            risk_requirements_details["require_tls"] = True
        material_map = {
            "root_cert": "root_cert_configured",
            "client_cert": "client_cert_configured",
            "client_key": "client_key_configured",
            "client_auth": "client_auth",
        }
        risk_materials: list[str] = []
        for cli_label, field_name in material_map.items():
            if combined_risk_meta.get(field_name):
                risk_cli_args.extend(["--require-risk-service-tls-material", cli_label])
                risk_materials.append(cli_label)
        if risk_materials:
            risk_requirements_details["tls_materials"] = risk_materials

        pinned = combined_risk_meta.get("pinned_fingerprints")
        normalized_pins: list[str] = []
        if isinstance(pinned, (list, tuple, set)):
            for entry in pinned:
                normalized = _normalize_sha256_fingerprint(entry)
                if normalized:
                    normalized_pins.append(normalized)
        if normalized_pins:
            risk_requirements_details["expected_server_sha256"] = normalized_pins
            for fingerprint in normalized_pins:
                risk_cli_args.extend(
                    ["--expect-risk-service-server-sha256", fingerprint]
                )

        risk_required_scopes: set[str] = set()
        primary_scope = combined_risk_meta.get("auth_token_scope_required")
        if isinstance(primary_scope, str):
            candidate = primary_scope.strip().lower()
            if candidate:
                risk_required_scopes.add(candidate)
        scope_map = combined_risk_meta.get("required_scopes")
        if isinstance(scope_map, Mapping):
            for scope_name in scope_map.keys():
                if isinstance(scope_name, str):
                    candidate = scope_name.strip().lower()
                    if candidate:
                        risk_required_scopes.add(candidate)
        scopes_field = combined_risk_meta.get("auth_token_scopes")
        if isinstance(scopes_field, (list, tuple, set)):
            for entry in scopes_field:
                if isinstance(entry, str):
                    candidate = entry.strip().lower()
                    if candidate:
                        risk_required_scopes.add(candidate)
        if risk_required_scopes:
            sorted_scopes = sorted(risk_required_scopes)
            risk_requirements_details["required_scopes"] = sorted_scopes
            for scope in sorted_scopes:
                risk_cli_args.extend(["--require-risk-service-scope", scope])

        if combined_risk_meta.get("auth_token_scope_checked") is True:
            risk_cli_args.append("--require-risk-service-auth-token")
            risk_requirements_details["require_auth_token"] = True

        # Wymagane identyfikatory tokenów (jeśli dostępne w metadanych)
        token_ids: list[str] = []
        token_id_value = combined_risk_meta.get("auth_token_token_id")
        if isinstance(token_id_value, str):
            candidate = token_id_value.strip()
            if candidate:
                token_ids.append(candidate)
        tokens_list = combined_risk_meta.get("auth_token_tokens")
        if isinstance(tokens_list, (list, tuple, set)):
            for entry in tokens_list:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        token_ids.append(candidate)
        unique_token_ids = sorted({token for token in token_ids if token})
        if unique_token_ids:
            risk_requirements_details["required_token_ids"] = unique_token_ids
            for token_id in unique_token_ids:
                risk_cli_args.extend(["--require-risk-service-token-id", token_id])

    (
        verify_exit_code,
        report_path,
        report_payload,
        verify_command,
    ) = _run_decision_log_verifier(
        decision_log_path=decision_log_path,
        summary_path=summary_path,
        risk_profile=risk_profile,
        risk_profiles_file=requirements.get("risk_profiles_file"),
        env_snippet=env_path,
        yaml_snippet=yaml_path,
        core_config=config_path,
        output_dir=output_dir,
        required_auth_scopes=required_auth_scopes,
        risk_cli_args=risk_cli_args,
    )

    bundle_output_dir = bundle_dir or (output_dir / "telemetry" / "bundle")
    try:
        stage_args = _prepare_bundle_stage_args(
            bundle_stage_args,
            default_paper_profile=risk_profile,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    bundle_manifest_path, bundle_manifest, bundle_command = _run_risk_profile_bundle(
        risk_profile=risk_profile,
        risk_profiles_file=requirements.get("risk_profiles_file"),
        core_config=config_path,
        output_dir=bundle_output_dir,
        env_style=bundle_env_style,
        config_format=bundle_config_format,
        stage_args=stage_args,
    )

    risk_profile_summary = None
    metadata = summary_payload.get("metadata") if isinstance(summary_payload, Mapping) else None
    if isinstance(metadata, Mapping):
        rps = metadata.get("risk_profile_summary")
        if isinstance(rps, Mapping):
            risk_profile_summary = rps

    return {
        "metrics_path": metrics_path,
        "risk_profile": risk_profile,
        "risk_profile_source": requirements.get("risk_profile_source"),
        "risk_profiles_file": requirements.get("risk_profiles_file"),
        "env_snippet_path": env_path,
        "yaml_snippet_path": yaml_path,
        "summary_path": summary_path,
        "decision_log_path": decision_log_path,
        "summary_payload": summary_payload,
        "verify_exit_code": verify_exit_code,
        "report_path": report_path,
        "report_payload": report_payload,
        "risk_profile_summary": risk_profile_summary,
        "environment_profile": requirements.get("environment_profile"),
        "bundle_dir": bundle_output_dir,
        "bundle_manifest_path": bundle_manifest_path,
        "bundle_manifest": bundle_manifest,
        "bundle_command": bundle_command,
        "required_auth_scopes": required_auth_scopes,
        "auth_scope_details": auth_scope_details if auth_scope_details else None,
        "verify_command": list(verify_command),
        "risk_service_requirements": {
            "cli_args": list(risk_cli_args) if risk_cli_args else None,
            "metadata": [
                {"source": source, "metadata": dict(meta)}
                for source, meta in risk_metadata_sources
            ]
            if risk_metadata_sources
            else None,
            "combined_metadata": combined_risk_meta or None,
            "details": risk_requirements_details or None,
        },
    }


def _run_manifest_metrics_export(
    *,
    config_path: Path,
    environment: str,
    output_dir: Path,
) -> dict[str, Any]:
    requirements = _load_manifest_requirements(config_path, environment)
    manifest_path = requirements.get("manifest_path")
    if not isinstance(manifest_path, Path):
        raise RuntimeError("Nie udało się ustalić ścieżki manifestu OHLCV dla środowiska")
    if not manifest_path.exists():
        raise RuntimeError(f"Plik manifestu danych OHLCV {manifest_path} nie istnieje")

    export_dir = output_dir / "manifest"
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = export_dir / "manifest_metrics.prom"
    summary_path = export_dir / "manifest_summary.json"

    script_path = _ensure_script_exists("export_manifest_metrics.py")

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--environment",
        environment,
        "--manifest-path",
        str(manifest_path),
        "--output",
        str(metrics_path),
        "--summary-output",
        str(summary_path),
    ]

    stage = requirements.get("stage")
    if isinstance(stage, str) and stage:
        cmd.extend(["--stage", stage])

    deny_status = requirements.get("deny_status")
    if not deny_status:
        deny_status = ["warning", "missing_metadata", "invalid_metadata"]
    for status in deny_status:
        cmd.extend(["--deny-status", str(status)])

    signing_cfg = requirements.get("signing")
    if isinstance(signing_cfg, Mapping) and signing_cfg:
        key_value = signing_cfg.get("key")
        key_file = signing_cfg.get("key_file")
        key_env = signing_cfg.get("key_env")
        key_id = signing_cfg.get("key_id")
        require_signature = signing_cfg.get("require")

        provided_sources = [bool(key_value), bool(key_file), bool(key_env)]
        if sum(provided_sources) > 1:
            raise RuntimeError(
                "Konfiguracja manifest_metrics.signing może wskazywać tylko jedno źródło klucza (key/key_file/key_env)."
            )

        if key_value:
            cmd.extend(["--summary-hmac-key", str(key_value)])
        elif key_file:
            resolved = _resolve_config_path(config_path.parent, str(key_file))
            if resolved is None:
                raise RuntimeError(
                    f"Nie udało się rozwiązać ścieżki klucza HMAC manifestu: {key_file}"
                )
            cmd.extend(["--summary-hmac-key-file", str(resolved)])
        elif key_env:
            cmd.extend(["--summary-hmac-key-env", str(key_env)])

        if key_id:
            cmd.extend(["--summary-hmac-key-id", str(key_id)])
        if require_signature:
            cmd.append("--require-summary-signature")

    _LOGGER.info(
        "Eksportuję metryki manifestu OHLCV przy pomocy export_manifest_metrics.py (%s)",
        manifest_path,
    )

    completed = subprocess.run(cmd, text=True, check=False)

    if not summary_path.exists():
        raise RuntimeError("export_manifest_metrics.py nie wygenerował pliku podsumowania manifestu")
    if not metrics_path.exists():
        raise RuntimeError("export_manifest_metrics.py nie wygenerował pliku metryk manifestu")

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Nieprawidłowy JSON podsumowania manifestu: {exc}") from exc

    return {
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "summary_payload": summary_payload,
        "exit_code": completed.returncode,
        "command": cmd,
        "deny_status": list(deny_status),
        "stage": stage,
        "risk_profile": requirements.get("risk_profile"),
        "signing": signing_cfg,
    }


def _run_tls_audit(
    *,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit_dir = output_dir / "tls_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / "tls_audit_report.json"

    script_path = _ensure_script_exists("audit_tls_assets.py")

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--json-output",
        str(report_path),
        "--pretty",
        "--print",
        "--fail-on-error",
    ]

    _LOGGER.info(
        "Uruchamiam audyt TLS usług runtime przy pomocy audit_tls_assets.py (%s)",
        config_path,
    )

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)

    if not report_path.exists():
        raise RuntimeError("audit_tls_assets.py nie wygenerował raportu JSON z audytu TLS")

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Nieprawidłowy JSON raportu audytu TLS: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    parsed_stdout: Any | None = None
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_stdout = None

    status = "ok"
    warnings = report_payload.get("warnings") if isinstance(report_payload, Mapping) else None
    errors = report_payload.get("errors") if isinstance(report_payload, Mapping) else None
    has_warnings = bool(warnings)
    has_errors = bool(errors)
    if completed.returncode != 0 or has_errors:
        status = "failed"
    elif has_warnings:
        status = "warning"

    return {
        "report_path": report_path,
        "report": report_payload,
        "exit_code": completed.returncode,
        "command": cmd,
        "stdout": _truncate_text(stdout) if stdout else "",
        "stderr": _truncate_text(stderr) if stderr else "",
        "stdout_parsed": parsed_stdout,
        "status": status,
        "warnings": warnings if isinstance(warnings, list) else (list(warnings) if isinstance(warnings, (tuple, set)) else warnings),
        "errors": errors if isinstance(errors, list) else (list(errors) if isinstance(errors, (tuple, set)) else errors),
    }


def _run_token_audit(
    *,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit_dir = output_dir / "token_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / "service_token_audit.json"

    script_path = _ensure_script_exists("audit_service_tokens.py")
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--json-output",
        str(report_path),
        "--pretty",
        "--print",
        "--fail-on-warning",
        "--fail-on-error",
    ]

    _LOGGER.info(
        "Uruchamiam audyt tokenów usługowych przy pomocy audit_service_tokens.py (%s)",
        config_path,
    )

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)

    if not report_path.exists():
        raise RuntimeError("audit_service_tokens.py nie wygenerował raportu JSON z audytu tokenów")

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - diagnostyka outputu
        raise RuntimeError(f"Nieprawidłowy JSON raportu audytu tokenów: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    parsed_stdout: Any | None = None
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_stdout = None

    warnings_list = report_payload.get("warnings") if isinstance(report_payload, Mapping) else None
    errors_list = report_payload.get("errors") if isinstance(report_payload, Mapping) else None
    has_warnings = bool(warnings_list)
    has_errors = bool(errors_list)
    if completed.returncode != 0 or has_errors:
        status = "failed"
    elif has_warnings:
        status = "warning"
    else:
        status = "ok"

    return {
        "report_path": report_path,
        "report": report_payload,
        "exit_code": completed.returncode,
        "command": cmd,
        "stdout": _truncate_text(stdout) if stdout else "",
        "stderr": _truncate_text(stderr) if stderr else "",
        "stdout_parsed": parsed_stdout,
        "status": status,
        "warnings": warnings_list if isinstance(warnings_list, list) else (list(warnings_list) if isinstance(warnings_list, (tuple, set)) else warnings_list),
        "errors": errors_list if isinstance(errors_list, list) else (list(errors_list) if isinstance(errors_list, (tuple, set)) else errors_list),
    }


def _run_security_baseline(
    *,
    config_path: Path,
    output_dir: Path,
    baseline_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit_dir = output_dir / "security_baseline_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / "security_baseline_report.json"

    script_path = _ensure_script_exists("audit_security_baseline.py")

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--json-output",
        str(report_path),
        "--pretty",
        "--print",
        "--fail-on-error",
    ]

    if baseline_settings:
        key_value = baseline_settings.get("signing_key_value")
        key_path = baseline_settings.get("signing_key_path")
        key_env = baseline_settings.get("signing_key_env")
        key_id = baseline_settings.get("signing_key_id")
        require_signature = bool(baseline_settings.get("require_signature"))

        provided_sources = [
            bool(key_value),
            bool(key_path),
            bool(key_env),
        ]
        if sum(provided_sources) > 1:
            raise RuntimeError(
                "Konfiguracja security_baseline.signing może wskazywać tylko jedno źródło klucza (value/path/env)."
            )

        if key_value:
            cmd.extend(["--summary-hmac-key", str(key_value)])
        elif key_path:
            cmd.extend(["--summary-hmac-key-file", str(key_path)])
        elif key_env:
            cmd.extend(["--summary-hmac-key-env", str(key_env)])

        if key_id:
            cmd.extend(["--summary-hmac-key-id", str(key_id)])
        if require_signature:
            cmd.append("--require-summary-signature")

    _LOGGER.info(
        "Uruchamiam zbiorczy audyt bezpieczeństwa przy pomocy audit_security_baseline.py (%s)",
        config_path,
    )

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)

    if not report_path.exists():
        raise RuntimeError("audit_security_baseline.py nie wygenerował raportu JSON z audytu bezpieczeństwa")

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Nieprawidłowy JSON raportu audytu bezpieczeństwa: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    parsed_stdout: Any | None = None
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_stdout = None

    warnings_payload = ()
    errors_payload = ()
    signature_payload: Any | None = None
    if isinstance(report_payload, Mapping):
        raw_warnings = report_payload.get("warnings")
        raw_errors = report_payload.get("errors")
        if isinstance(raw_warnings, (list, tuple, set)):
            warnings_payload = tuple(str(item) for item in raw_warnings)
        elif isinstance(raw_warnings, str) and raw_warnings:
            warnings_payload = (raw_warnings,)
        if isinstance(raw_errors, (list, tuple, set)):
            errors_payload = tuple(str(item) for item in raw_errors)
        elif isinstance(raw_errors, str) and raw_errors:
            errors_payload = (raw_errors,)
        signature_payload = report_payload.get("summary_signature")

    baseline_status = "unknown"
    if isinstance(report_payload, Mapping) and report_payload.get("status"):
        baseline_status = str(report_payload.get("status"))

    status = baseline_status
    if completed.returncode != 0:
        status = "failed"
    elif baseline_status.lower() == "error":
        status = "failed"
    elif not baseline_status or baseline_status == "unknown":
        if errors_payload:
            status = "failed"
        elif warnings_payload:
            status = "warning"
        else:
            status = "ok"

    return {
        "report_path": report_path,
        "report": report_payload,
        "exit_code": completed.returncode,
        "command": cmd,
        "stdout": _truncate_text(stdout) if stdout else "",
        "stderr": _truncate_text(stderr) if stderr else "",
        "stdout_parsed": parsed_stdout,
        "status": status,
        "baseline_status": baseline_status,
        "warnings": list(warnings_payload),
        "errors": list(errors_payload),
        "summary_signature": signature_payload,
    }


def _run_summary_validation(
    *,
    summary_path: Path,
    environment: str,
    operator: str,
    publish_required: bool,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    script_path = Path(__file__).with_name("validate_paper_smoke_summary.py")
    if not script_path.exists():
        return {
            "status": "failed",
            "reason": "validator_missing",
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "required_publish_success": publish_required,
        }

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--summary",
        str(summary_path),
        "--require-environment",
        environment,
    ]
    if operator:
        cmd.extend(["--require-operator", operator])
    if publish_required:
        cmd.extend(
            [
                "--require-publish-success",
                "--require-publish-required",
                "--require-publish-exit-zero",
            ]
        )
    for raw in extra_args:
        if not raw:
            continue
        cmd.extend(shlex.split(raw))

    _LOGGER.info("Waliduję podsumowanie smoke przy pomocy validate_paper_smoke_summary.py")

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - uruchomienie interpretatora może zawieść
        return {
            "status": "failed",
            "reason": "validator_exec_failed",
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "required_publish_success": publish_required,
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    trimmed_stdout = _truncate_text(stdout) if stdout else ""
    trimmed_stderr = _truncate_text(stderr) if stderr else ""

    parsed: Any | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None

    status = "ok" if completed.returncode == 0 else "failed"

    return {
        "status": status,
        "exit_code": completed.returncode,
        "stdout": trimmed_stdout,
        "stderr": trimmed_stderr,
        "raw_stdout_present": bool(stdout),
        "raw_stderr_present": bool(stderr),
        "result": parsed,
        "required_publish_success": publish_required,
        "command": cmd,
    }


def _write_env_file(
    env_file: Path,
    *,
    operator: str,
    paths: dict[str, Path],
    summary: dict,
    markdown_path: Path | None,
    validation: dict[str, Any],
) -> None:
    env_file = env_file.expanduser()
    env_file.parent.mkdir(parents=True, exist_ok=True)

    def _normalise(value: str) -> str:
        # GitHub Actions obsługuje format KEY=VALUE, dlatego zabezpieczamy znaki nowej linii.
        return value.replace("\n", "\\n")

    publish_status = summary.get("publish", {}).get("status", "unknown")
    validation_status = validation.get("status", "unknown")

    data = {
        "PAPER_SMOKE_OPERATOR": operator,
        "PAPER_SMOKE_SUMMARY_PATH": str(paths["summary"].resolve()),
        "PAPER_SMOKE_JSON_LOG_PATH": str(paths["json_log"].resolve()),
        "PAPER_SMOKE_AUDIT_LOG_PATH": str(paths["audit_log"].resolve()),
        "PAPER_SMOKE_STATUS": summary.get("status", "unknown"),
        "PAPER_SMOKE_PUBLISH_STATUS": publish_status,
        "PAPER_SMOKE_VALIDATION_STATUS": validation_status,
    }

    if markdown_path is not None:
        data["PAPER_SMOKE_MARKDOWN_PATH"] = str(markdown_path.resolve())

    telemetry_section = summary.get("telemetry", {}) if isinstance(summary, Mapping) else {}
    if isinstance(telemetry_section, Mapping):
        summary_path = telemetry_section.get("summary_path")
        decision_log_path = telemetry_section.get("decision_log_path")
        metrics_source = telemetry_section.get("metrics_source_path")
        if summary_path:
            data["PAPER_SMOKE_TELEMETRY_SUMMARY_PATH"] = str(summary_path)
        if decision_log_path:
            data["PAPER_SMOKE_DECISION_LOG_PATH"] = str(decision_log_path)
        if metrics_source:
            data["PAPER_SMOKE_TELEMETRY_SOURCE_PATH"] = str(metrics_source)

        snippets = telemetry_section.get("snippets")
        if isinstance(snippets, Mapping):
            env_path = snippets.get("env_path")
            yaml_path = snippets.get("yaml_path")
            if env_path:
                data["PAPER_SMOKE_RISK_PROFILE_ENV_PATH"] = str(env_path)
            if yaml_path:
                data["PAPER_SMOKE_RISK_PROFILE_YAML_PATH"] = str(yaml_path)

        decision_log_report = telemetry_section.get("decision_log_report")
        if isinstance(decision_log_report, Mapping):
            report_path = decision_log_report.get("path")
            report_status = decision_log_report.get("status")
            if report_path:
                data["PAPER_SMOKE_DECISION_LOG_REPORT_PATH"] = str(report_path)
            if report_status:
                data["PAPER_SMOKE_DECISION_LOG_STATUS"] = str(report_status)

        bundle_section = telemetry_section.get("bundle")
        if isinstance(bundle_section, Mapping):
            bundle_dir = bundle_section.get("output_dir")
            bundle_manifest_path = bundle_section.get("manifest_path")
            if bundle_dir:
                data["PAPER_SMOKE_RISK_BUNDLE_DIR"] = str(bundle_dir)
            if bundle_manifest_path:
                data["PAPER_SMOKE_RISK_BUNDLE_MANIFEST_PATH"] = str(bundle_manifest_path)

        risk_profile_info = telemetry_section.get("risk_profile")
        if isinstance(risk_profile_info, Mapping):
            profile_name = risk_profile_info.get("name")
            profile_source = risk_profile_info.get("source")
            profiles_file = risk_profile_info.get("profiles_file")
            if profile_name:
                data["PAPER_SMOKE_RISK_PROFILE"] = str(profile_name)
            if profile_source:
                data["PAPER_SMOKE_RISK_PROFILE_SOURCE"] = str(profile_source)
            if profiles_file:
                data["PAPER_SMOKE_RISK_PROFILE_FILE"] = str(profiles_file)

    manifest_section = summary.get("manifest", {}) if isinstance(summary, Mapping) else {}
    if isinstance(manifest_section, Mapping):
        manifest_path = manifest_section.get("manifest_path")
        manifest_metrics = manifest_section.get("metrics_path")
        manifest_summary_path = manifest_section.get("summary_path")
        manifest_status = manifest_section.get("worst_status")
        manifest_exit_code = manifest_section.get("exit_code")
        manifest_stage = manifest_section.get("stage")
        manifest_profile = manifest_section.get("risk_profile")
        signature_payload = manifest_section.get("summary_signature")

        if manifest_path:
            data["PAPER_SMOKE_MANIFEST_PATH"] = str(manifest_path)
        if manifest_metrics:
            data["PAPER_SMOKE_MANIFEST_METRICS_PATH"] = str(manifest_metrics)
        if manifest_summary_path:
            data["PAPER_SMOKE_MANIFEST_SUMMARY_PATH"] = str(manifest_summary_path)
        if manifest_status:
            data["PAPER_SMOKE_MANIFEST_STATUS"] = str(manifest_status)
        if manifest_exit_code is not None:
            data["PAPER_SMOKE_MANIFEST_EXIT_CODE"] = str(manifest_exit_code)
        if manifest_stage:
            data["PAPER_SMOKE_MANIFEST_STAGE"] = str(manifest_stage)
        if manifest_profile:
            data["PAPER_SMOKE_MANIFEST_RISK_PROFILE"] = str(manifest_profile)
        if isinstance(signature_payload, Mapping):
            signature_value = signature_payload.get("value")
            if signature_value:
                data["PAPER_SMOKE_MANIFEST_SIGNATURE"] = str(signature_value)
            signature_algorithm = signature_payload.get("algorithm")
            if signature_algorithm:
                data["PAPER_SMOKE_MANIFEST_SIGNATURE_ALGORITHM"] = str(signature_algorithm)
            signature_key_id = signature_payload.get("key_id")
            if signature_key_id:
                data["PAPER_SMOKE_MANIFEST_SIGNATURE_KEY_ID"] = str(signature_key_id)

    tls_section = summary.get("tls_audit", {}) if isinstance(summary, Mapping) else {}
    if isinstance(tls_section, Mapping):
        report_path = tls_section.get("report_path")
        exit_code = tls_section.get("exit_code")
        status_value = tls_section.get("status")
        warnings_list = tls_section.get("warnings")
        errors_list = tls_section.get("errors")

        if report_path:
            data["PAPER_SMOKE_TLS_AUDIT_PATH"] = str(report_path)
        if exit_code is not None:
            data["PAPER_SMOKE_TLS_AUDIT_EXIT_CODE"] = str(exit_code)
        if status_value:
            data["PAPER_SMOKE_TLS_AUDIT_STATUS"] = str(status_value)
        if isinstance(warnings_list, (list, tuple, set)) and warnings_list:
            data["PAPER_SMOKE_TLS_AUDIT_WARNINGS"] = "|".join(str(item) for item in warnings_list)
        elif isinstance(warnings_list, str) and warnings_list:
            data["PAPER_SMOKE_TLS_AUDIT_WARNINGS"] = warnings_list
        if isinstance(errors_list, (list, tuple, set)) and errors_list:
            data["PAPER_SMOKE_TLS_AUDIT_ERRORS"] = "|".join(str(item) for item in errors_list)
        elif isinstance(errors_list, str) and errors_list:
            data["PAPER_SMOKE_TLS_AUDIT_ERRORS"] = errors_list

    token_section = summary.get("token_audit", {}) if isinstance(summary, Mapping) else {}
    if isinstance(token_section, Mapping):
        report_path = token_section.get("report_path")
        exit_code = token_section.get("exit_code")
        status_value = token_section.get("status")
        warnings_list = token_section.get("warnings")
        errors_list = token_section.get("errors")

        if report_path:
            data["PAPER_SMOKE_TOKEN_AUDIT_PATH"] = str(report_path)
        if exit_code is not None:
            data["PAPER_SMOKE_TOKEN_AUDIT_EXIT_CODE"] = str(exit_code)
        if status_value:
            data["PAPER_SMOKE_TOKEN_AUDIT_STATUS"] = str(status_value)
        if isinstance(warnings_list, (list, tuple, set)) and warnings_list:
            data["PAPER_SMOKE_TOKEN_AUDIT_WARNINGS"] = "|".join(str(item) for item in warnings_list)
        elif isinstance(warnings_list, str) and warnings_list:
            data["PAPER_SMOKE_TOKEN_AUDIT_WARNINGS"] = warnings_list
        if isinstance(errors_list, (list, tuple, set)) and errors_list:
            data["PAPER_SMOKE_TOKEN_AUDIT_ERRORS"] = "|".join(str(item) for item in errors_list)
        elif isinstance(errors_list, str) and errors_list:
            data["PAPER_SMOKE_TOKEN_AUDIT_ERRORS"] = errors_list

    security_section = summary.get("security_baseline", {}) if isinstance(summary, Mapping) else {}
    if isinstance(security_section, Mapping):
        report_path = security_section.get("report_path")
        exit_code = security_section.get("exit_code")
        status_value = security_section.get("status")
        warnings_list = security_section.get("warnings")
        errors_list = security_section.get("errors")
        signature_payload = security_section.get("summary_signature")

        if report_path:
            data["PAPER_SMOKE_SECURITY_BASELINE_PATH"] = str(report_path)
        if exit_code is not None:
            data["PAPER_SMOKE_SECURITY_BASELINE_EXIT_CODE"] = str(exit_code)
        if status_value:
            data["PAPER_SMOKE_SECURITY_BASELINE_STATUS"] = str(status_value)
        if isinstance(warnings_list, (list, tuple, set)) and warnings_list:
            data["PAPER_SMOKE_SECURITY_BASELINE_WARNINGS"] = "|".join(
                str(item) for item in warnings_list
            )
        elif isinstance(warnings_list, str) and warnings_list:
            data["PAPER_SMOKE_SECURITY_BASELINE_WARNINGS"] = warnings_list
        if isinstance(errors_list, (list, tuple, set)) and errors_list:
            data["PAPER_SMOKE_SECURITY_BASELINE_ERRORS"] = "|".join(
                str(item) for item in errors_list
            )
        elif isinstance(errors_list, str) and errors_list:
            data["PAPER_SMOKE_SECURITY_BASELINE_ERRORS"] = errors_list
        if isinstance(signature_payload, Mapping):
            signature_value = signature_payload.get("value")
            if signature_value:
                data["PAPER_SMOKE_SECURITY_BASELINE_SIGNATURE"] = str(signature_value)
            signature_algorithm = signature_payload.get("algorithm")
            if signature_algorithm:
                data["PAPER_SMOKE_SECURITY_BASELINE_SIGNATURE_ALGORITHM"] = str(signature_algorithm)
            signature_key_id = signature_payload.get("key_id")
            if signature_key_id:
                data["PAPER_SMOKE_SECURITY_BASELINE_SIGNATURE_KEY_ID"] = str(signature_key_id)

    with env_file.open("a", encoding="utf-8") as fp:
        for key, value in data.items():
            fp.write(f"{key}={_normalise(value)}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    operator = (
        args.operator
        or os.environ.get("PAPER_SMOKE_OPERATOR")
        or os.environ.get("CI_OPERATOR")
        or "CI Agent"
    )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    output_dir = Path(args.output_dir)
    command, paths = _build_command(
        config_path=config_path,
        environment=args.environment,
        output_dir=output_dir,
        operator=operator,
        auto_publish_required=not args.allow_auto_publish_failure,
        extra_run_daily_trend_args=args.run_daily_trend_arg,
    )

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command}, indent=2))
        return 0

    _LOGGER.info(
        "Uruchamiam smoke test paper trading w trybie CI: %s",
        " ".join(map(shlex.quote, command)),
    )

    completed = subprocess.run(command, text=True, check=False)

    if completed.returncode != 0:
        _LOGGER.error("Smoke test zakończył się kodem %s", completed.returncode)
        return completed.returncode

    summary = _load_summary(paths["summary"])

    bundle_dir = Path(args.risk_profile_bundle_dir).expanduser() if args.risk_profile_bundle_dir else None

    try:
        telemetry_artifacts = _collect_telemetry_artifacts(
            config_path=config_path,
            environment=args.environment,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            bundle_env_style=args.risk_profile_bundle_env_style,
            bundle_config_format=args.risk_profile_bundle_config_format,
            bundle_stage_args=args.risk_profile_stage,
        )
    except RuntimeError as exc:
        _LOGGER.error("Automatyzacja telemetryjna nie powiodła się: %s", exc)
        return 1

    telemetry_summary_payload = telemetry_artifacts.get("summary_payload")
    telemetry_summary: Mapping[str, Any] | None = None
    if isinstance(telemetry_summary_payload, Mapping):
        telemetry_summary = telemetry_summary_payload

    risk_profile_info: dict[str, Any] = {
        "name": telemetry_artifacts.get("risk_profile"),
    }
    if telemetry_artifacts.get("risk_profile_source"):
        risk_profile_info["source"] = telemetry_artifacts["risk_profile_source"]
    if telemetry_artifacts.get("environment_profile"):
        risk_profile_info["environment_fallback"] = telemetry_artifacts["environment_profile"]
    if telemetry_artifacts.get("risk_profiles_file"):
        risk_profile_info["profiles_file"] = str(telemetry_artifacts["risk_profiles_file"])
    if telemetry_artifacts.get("risk_profile_summary") is not None:
        risk_profile_info["summary"] = telemetry_artifacts["risk_profile_summary"]
        summary["risk_profile_summary"] = telemetry_artifacts["risk_profile_summary"]

    snippets_info = {
        "env_path": str(telemetry_artifacts["env_snippet_path"]),
        "yaml_path": str(telemetry_artifacts["yaml_snippet_path"]),
    }

    decision_log_report_payload = telemetry_artifacts.get("report_payload")
    decision_log_report = {
        "path": str(telemetry_artifacts["report_path"]),
        "exit_code": int(telemetry_artifacts.get("verify_exit_code", 0)),
        "status": "ok" if int(telemetry_artifacts.get("verify_exit_code", 0)) == 0 else "failed",
    }
    if decision_log_report_payload is not None:
        decision_log_report["payload"] = decision_log_report_payload
    if telemetry_artifacts.get("verify_command"):
        decision_log_report["command"] = telemetry_artifacts["verify_command"]

    bundle_manifest = telemetry_artifacts.get("bundle_manifest")
    bundle_section: dict[str, Any] | None = None
    if isinstance(bundle_manifest, Mapping):
        bundle_section = {
            "output_dir": str(telemetry_artifacts["bundle_dir"]),
            "manifest_path": str(telemetry_artifacts["bundle_manifest_path"]),
            "manifest": bundle_manifest,
            "command": telemetry_artifacts.get("bundle_command"),
        }
    elif telemetry_artifacts.get("bundle_manifest_path"):
        bundle_section = {
            "output_dir": str(telemetry_artifacts["bundle_dir"]),
            "manifest_path": str(telemetry_artifacts["bundle_manifest_path"]),
            "command": telemetry_artifacts.get("bundle_command"),
        }

    summary["telemetry"] = {
        "summary_path": str(telemetry_artifacts["summary_path"]),
        "decision_log_path": str(telemetry_artifacts["decision_log_path"]),
        "metrics_source_path": str(telemetry_artifacts["metrics_path"]),
        "summary": telemetry_summary,
        "risk_profile": risk_profile_info,
        "snippets": snippets_info,
        "decision_log_report": decision_log_report,
    }
    risk_requirements = telemetry_artifacts.get("risk_service_requirements")
    if isinstance(risk_requirements, Mapping):
        filtered_requirements = {
            key: value for key, value in risk_requirements.items() if value
        }
        if filtered_requirements:
            summary["telemetry"]["risk_service_requirements"] = filtered_requirements
    required_auth_scopes = telemetry_artifacts.get("required_auth_scopes") or []
    if required_auth_scopes:
        summary["telemetry"]["required_auth_scopes"] = list(required_auth_scopes)
    auth_scope_details = telemetry_artifacts.get("auth_scope_details")
    if isinstance(auth_scope_details, Mapping) and auth_scope_details:
        summary["telemetry"]["auth_scope_requirements"] = auth_scope_details
    if bundle_section is not None:
        summary["telemetry"]["bundle"] = bundle_section

    try:
        manifest_artifacts = _run_manifest_metrics_export(
            config_path=config_path,
            environment=args.environment,
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        _LOGGER.error("Eksport metryk manifestu nie powiódł się: %s", exc)
        return 1

    manifest_summary_payload = manifest_artifacts.get("summary_payload")
    manifest_summary: Mapping[str, Any] | None = None
    if isinstance(manifest_summary_payload, Mapping):
        manifest_summary = manifest_summary_payload

    manifest_signature: Mapping[str, Any] | None = None
    if isinstance(manifest_summary, Mapping):
        signature_candidate = manifest_summary.get("summary_signature")
        if isinstance(signature_candidate, Mapping):
            manifest_signature = signature_candidate

    summary["manifest"] = {
        "manifest_path": str(manifest_artifacts["manifest_path"]),
        "metrics_path": str(manifest_artifacts["metrics_path"]),
        "summary_path": str(manifest_artifacts["summary_path"]),
        "summary": manifest_summary,
        "status_counts": manifest_summary.get("status_counts") if isinstance(manifest_summary, Mapping) else None,
        "total_entries": manifest_summary.get("total_entries") if isinstance(manifest_summary, Mapping) else None,
        "worst_status": manifest_summary.get("worst_status") if isinstance(manifest_summary, Mapping) else None,
        "summary_signature": manifest_signature,
        "exit_code": int(manifest_artifacts.get("exit_code", 0)),
        "deny_status": manifest_artifacts.get("deny_status"),
        "stage": manifest_artifacts.get("stage"),
        "risk_profile": manifest_artifacts.get("risk_profile"),
        "command": manifest_artifacts.get("command"),
        "signing": manifest_artifacts.get("signing"),
    }

    try:
        tls_audit_artifacts = _run_tls_audit(
            config_path=config_path,
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        _LOGGER.error("Audyt TLS nie powiódł się: %s", exc)
        # Kontynuujemy mimo błędu TLS, ale zarejestrujemy status jako failed

    tls_report = tls_audit_artifacts.get("report") if 'tls_audit_artifacts' in locals() else None
    warnings = []
    errors = []
    if isinstance(tls_report, Mapping):
        warn_payload = tls_report.get("warnings")
        err_payload = tls_report.get("errors")
        if isinstance(warn_payload, (list, tuple, set)):
            warnings = [str(item) for item in warn_payload]
        elif isinstance(warn_payload, str):
            warnings = [warn_payload]
        if isinstance(err_payload, (list, tuple, set)):
            errors = [str(item) for item in err_payload]
        elif isinstance(err_payload, str):
            errors = [err_payload]

    if 'tls_audit_artifacts' in locals():
        summary["tls_audit"] = {
            "report_path": str(tls_audit_artifacts["report_path"]),
            "report": tls_report,
            "exit_code": int(tls_audit_artifacts.get("exit_code", 0)),
            "stdout": tls_audit_artifacts.get("stdout"),
            "stderr": tls_audit_artifacts.get("stderr"),
            "status": tls_audit_artifacts.get("status"),
            "warnings": warnings,
            "errors": errors,
            "command": tls_audit_artifacts.get("command"),
        }
    else:
        summary["tls_audit"] = {
            "status": "failed",
            "errors": ["TLS audit did not run"],
        }

    try:
        token_audit_artifacts = _run_token_audit(
            config_path=config_path,
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        _LOGGER.error("Audyt tokenów RBAC nie powiódł się: %s", exc)
        return 1

    token_report = token_audit_artifacts.get("report")
    token_warnings: list[str] = []
    token_errors: list[str] = []
    if isinstance(token_report, Mapping):
        warn_payload = token_report.get("warnings")
        err_payload = token_report.get("errors")
        if isinstance(warn_payload, (list, tuple, set)):
            token_warnings = [str(item) for item in warn_payload]
        elif isinstance(warn_payload, str):
            token_warnings = [warn_payload]
        if isinstance(err_payload, (list, tuple, set)):
            token_errors = [str(item) for item in err_payload]
        elif isinstance(err_payload, str):
            token_errors = [err_payload]

    summary["token_audit"] = {
        "report_path": str(token_audit_artifacts["report_path"]),
        "report": token_report,
        "exit_code": int(token_audit_artifacts.get("exit_code", 0)),
        "stdout": token_audit_artifacts.get("stdout"),
        "stderr": token_audit_artifacts.get("stderr"),
        "status": token_audit_artifacts.get("status"),
        "warnings": token_warnings,
        "errors": token_errors,
        "command": token_audit_artifacts.get("command"),
    }

    security_baseline_settings = _load_security_baseline_settings(config_path)

    try:
        security_baseline_artifacts = _run_security_baseline(
            config_path=config_path,
            output_dir=output_dir,
            baseline_settings=security_baseline_settings,
        )
    except RuntimeError as exc:
        _LOGGER.error("Audyt bezpieczeństwa nie powiódł się: %s", exc)
        return 1

    baseline_report = security_baseline_artifacts.get("report")
    baseline_warnings = []
    baseline_errors: list[str] = []
    if isinstance(baseline_report, Mapping):
        warn_payload = baseline_report.get("warnings")
        err_payload = baseline_report.get("errors")
        if isinstance(warn_payload, (list, tuple, set)):
            baseline_warnings = [str(item) for item in warn_payload]
        elif isinstance(warn_payload, str):
            baseline_warnings = [warn_payload]
        if isinstance(err_payload, (list, tuple, set)):
            baseline_errors = [str(item) for item in err_payload]
        elif isinstance(err_payload, str):
            baseline_errors = [err_payload]

    summary["security_baseline"] = {
        "report_path": str(security_baseline_artifacts["report_path"]),
        "report": baseline_report,
        "exit_code": int(security_baseline_artifacts.get("exit_code", 0)),
        "stdout": security_baseline_artifacts.get("stdout"),
        "stderr": security_baseline_artifacts.get("stderr"),
        "status": security_baseline_artifacts.get("status"),
        "baseline_status": security_baseline_artifacts.get("baseline_status"),
        "warnings": baseline_warnings,
        "errors": baseline_errors,
        "command": security_baseline_artifacts.get("command"),
        "require_signature": bool(security_baseline_settings.get("require_signature")),
        "summary_signature": security_baseline_artifacts.get("summary_signature"),
    }

    markdown_path: Path | None = None
    if args.render_summary_markdown:
        markdown_path = Path(args.render_summary_markdown).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        limit = (
            args.render_summary_max_json_chars
            if args.render_summary_max_json_chars is not None
            else DEFAULT_MAX_JSON_CHARS
        )
        markdown = render_summary_markdown(
            summary,
            title_override=args.render_summary_title,
            max_json_chars=max(limit, 0),
        )
        markdown_path.write_text(markdown, encoding="utf-8")

    if args.skip_summary_validation:
        validation_result = {
            "status": "skipped",
            "reason": "validation_disabled",
            "exit_code": 0,
            "required_publish_success": not args.allow_auto_publish_failure,
        }
    else:
        validation_result = _run_summary_validation(
            summary_path=paths["summary"],
            environment=args.environment,
            operator=operator,
            publish_required=not args.allow_auto_publish_failure,
            extra_args=args.summary_validator_arg,
        )

    summary["validation"] = validation_result
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.env_file:
        _write_env_file(
            Path(args.env_file),
            operator=operator,
            paths=paths,
            summary=summary,
            markdown_path=markdown_path,
            validation=validation_result,
        )

    payload = {
        "summary_path": str(paths["summary"]),
        "summary": summary,
        "json_log_path": str(paths["json_log"]),
        "audit_log_path": str(paths["audit_log"]),
        "validation": validation_result,
    }
    if markdown_path is not None:
        payload["markdown_report_path"] = str(markdown_path)

    telemetry_section = summary.get("telemetry", {}) if isinstance(summary, Mapping) else {}
    if isinstance(telemetry_section, Mapping):
        payload["telemetry_summary_path"] = telemetry_section.get("summary_path")
        payload["telemetry_decision_log_path"] = telemetry_section.get("decision_log_path")
        payload["telemetry_decision_log_report"] = telemetry_section.get("decision_log_report")
        payload["telemetry_snippets"] = telemetry_section.get("snippets")
        payload["telemetry_bundle"] = telemetry_section.get("bundle")

    manifest_section = summary.get("manifest", {}) if isinstance(summary, Mapping) else {}
    if isinstance(manifest_section, Mapping):
        payload["manifest_summary_path"] = manifest_section.get("summary_path")
        payload["manifest_metrics_path"] = manifest_section.get("metrics_path")
        payload["manifest_status"] = manifest_section.get("worst_status")

    token_section = summary.get("token_audit", {}) if isinstance(summary, Mapping) else {}
    if isinstance(token_section, Mapping):
        payload["token_audit_report_path"] = token_section.get("report_path")
        payload["token_audit_status"] = token_section.get("status")

    security_section = summary.get("security_baseline", {}) if isinstance(summary, Mapping) else {}
    if isinstance(security_section, Mapping):
        payload["security_baseline_report_path"] = security_section.get("report_path")
        payload["security_baseline_status"] = security_section.get("status")

    tls_section = summary.get("tls_audit", {}) if isinstance(summary, Mapping) else {}
    if isinstance(tls_section, Mapping):
        payload["tls_audit_report_path"] = tls_section.get("report_path")
        payload["tls_audit_status"] = tls_section.get("status")

    validation_exit = int(validation_result.get("exit_code", 0))
    verify_exit = int(decision_log_report.get("exit_code", 0))
    manifest_exit = int(manifest_section.get("exit_code", 0)) if isinstance(manifest_section, Mapping) else 0
    token_exit = int(token_section.get("exit_code", 0)) if isinstance(token_section, Mapping) else 0
    tls_exit = int(tls_section.get("exit_code", 0)) if isinstance(tls_section, Mapping) else 0
    baseline_exit = int(security_section.get("exit_code", 0)) if isinstance(security_section, Mapping) else 0
    baseline_require_signature = bool(security_section.get("require_signature")) if isinstance(security_section, Mapping) else False
    baseline_status_text = (
        str(security_section.get("status") or "").lower()
        if isinstance(security_section, Mapping)
        else ""
    )
    baseline_effective_exit = baseline_exit
    if baseline_require_signature:
        if baseline_effective_exit == 0 and baseline_status_text in {"warning", "failed", "error"}:
            baseline_effective_exit = 2
    tls_effective_exit = tls_exit
    if baseline_require_signature and baseline_effective_exit != 0:
        tls_effective_exit = max(tls_effective_exit, baseline_effective_exit)

    payload["manifest_exit_code"] = manifest_exit
    payload["token_audit_exit_code"] = token_exit
    payload["tls_audit_exit_code"] = tls_effective_exit
    payload["security_baseline_exit_code"] = baseline_effective_exit
    exit_reasons: list[str] = []
    fatal_reasons: list[tuple[str, int]] = []

    def _classify(
        reason: str,
        exit_value: int,
        status_value: Any,
        *,
        treat_warning_as_warning: bool = True,
    ) -> None:
        status_text = str(status_value or "").lower()
        is_warning = status_text == "warning"
        is_skipped = status_text == "skipped"

        if exit_value == 0 and not is_warning:
            return

        exit_reasons.append(reason)

        if exit_value == 0:
            if not treat_warning_as_warning or not is_warning:
                fatal_reasons.append((reason, 1))
            return

        if is_skipped and treat_warning_as_warning:
            return

        if is_warning and treat_warning_as_warning:
            return

        fatal_reasons.append((reason, exit_value))

    _classify("summary_validation", validation_exit, validation_result.get("status"))
    _classify("decision_log", verify_exit, decision_log_report.get("status"))
    manifest_status = manifest_section.get("worst_status") if isinstance(manifest_section, Mapping) else None
    _classify("manifest_metrics", manifest_exit, manifest_status, treat_warning_as_warning=False)
    _classify("token_audit", token_exit, token_section.get("status") if isinstance(token_section, Mapping) else None)
    _classify(
        "tls_audit",
        tls_effective_exit,
        tls_section.get("status") if isinstance(tls_section, Mapping) else None,
    )
    _classify(
        "security_baseline",
        baseline_effective_exit,
        security_section.get("status") if isinstance(security_section, Mapping) else None,
    )

    exit_code = 0
    if fatal_reasons:
        exit_code = max(code for _, code in fatal_reasons)
        payload["status"] = "+".join(exit_reasons) + "_failed"
    elif exit_reasons:
        payload["status"] = "+".join(exit_reasons) + "_warning"
    else:
        payload["status"] = "ok"

    payload["decision_log_exit_code"] = verify_exit

    try:
        import builtins  # noqa: WPS433 (świadomie odwołujemy się do builtins)

        setattr(builtins, "payload", payload)
    except Exception:  # pragma: no cover - best effort dla środowisk ograniczonych
        _LOGGER.debug("Nie udało się opublikować payload w builtins", exc_info=True)

    print(json.dumps(payload, indent=2))

    if validation_exit != 0:
        _LOGGER.error("Walidacja summary.json zakończyła się kodem %s", validation_exit)
    if verify_exit != 0:
        _LOGGER.error("verify_decision_log.py zakończył się kodem %s", verify_exit)
    if manifest_exit != 0:
        _LOGGER.error("export_manifest_metrics.py zakończył się kodem %s", manifest_exit)
    if token_exit != 0:
        _LOGGER.error("audit_service_tokens.py zakończył się kodem %s", token_exit)
    if tls_exit != 0:
        _LOGGER.error("audit_tls_assets.py zakończył się kodem %s", tls_exit)
    if baseline_exit != 0:
        _LOGGER.error("audit_security_baseline.py zakończył się kodem %s", baseline_exit)

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
