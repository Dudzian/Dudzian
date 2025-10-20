#!/usr/bin/env python3
"""Uruchamia kontrolę pokrycia OHLCV i opcjonalnie wysyła alerty."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.coverage import (
    build_environment_coverage_report,
    dispatch_coverage_alert,
)
from bot_core.config import CoreConfig, load_core_config
from bot_core.config.models import EnvironmentConfig
from bot_core.data.ohlcv import CoverageReportPayload
from bot_core.runtime.bootstrap import build_alert_channels
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage


@dataclass(slots=True)
class _RunSpec:
    environment: EnvironmentConfig
    category: str
    severity_override: str | None
    dispatch_enabled: bool
    dispatch_requested: bool


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprawdza pokrycie danych OHLCV i (opcjonalnie) wysyła alerty",
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        dest="environments",
        action="append",
        default=[],
        help=(
            "Nazwa środowiska z sekcji environments. Opcję można powtórzyć, "
            "aby sprawdzić wiele środowisk."
        ),
    )
    parser.add_argument(
        "--all-configured",
        action="store_true",
        help=(
            "Uruchom monitoring dla środowisk z sekcji coverage_monitoring "
            "(wymaga konfiguracji w pliku core.yaml)."
        ),
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Znacznik czasu ISO8601 używany do oceny opóźnień danych (domyślnie teraz, UTC)",
    )
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="Wyślij alert przez skonfigurowany router po wykryciu problemów",
    )
    parser.add_argument(
        "--category",
        default="data.ohlcv",
        help="Kategoria alertu (domyślnie data.ohlcv)",
    )
    parser.add_argument(
        "--severity-override",
        dest="severity_override",
        default=None,
        help="Wymuś poziom ważności alertu (info/warning/critical)",
    )
    parser.add_argument("--json", action="store_true", help="Zwróć wynik w formacie JSON")
    parser.add_argument(
        "--output",
        help=(
            "Ścieżka do pliku, w którym zostanie zapisany wynik w formacie JSON. "
            "Katalogi zostaną utworzone automatycznie."
        ),
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany przy odczycie sekretów (keychain / plik szyfrowany)",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do magazynu sekretów w środowiskach headless (Linux)",
    )
    parser.add_argument(
        "--headless-secrets-path",
        default=None,
        help="Ścieżka do zaszyfrowanego magazynu sekretów w trybie headless",
    )
    return parser.parse_args(argv)


def _parse_as_of(arg: str | None) -> datetime:
    if not arg:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(arg)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_secrets_path,
    )
    return SecretManager(storage, namespace=args.secret_namespace)


def _initialize_router(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
    secret_manager: SecretManager,
) -> DefaultAlertRouter:
    _, router, _ = build_alert_channels(
        core_config=config,
        environment=environment,
        secret_manager=secret_manager,
    )
    return router


def _build_report_without_dispatch(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
    as_of: datetime,
) -> CoverageReportPayload:
    return build_environment_coverage_report(
        config=config,
        environment=environment,
        as_of=as_of,
    )


def _serialize_report(
    report: CoverageReportPayload,
    *,
    dispatch_enabled: bool,
    dispatch_requested: bool,
    dispatched: bool,
) -> dict[str, object]:
    payload = dict(report.payload)
    payload["dispatch_requested"] = bool(dispatch_requested)
    payload["dispatch_enabled"] = bool(dispatch_enabled)
    payload["alert_dispatched"] = bool(dispatched)
    return payload


def _print_text_report(
    report: CoverageReportPayload,
    *,
    dispatched: bool,
    dispatch_enabled: bool,
    dispatch_requested: bool,
) -> None:
    payload = report.payload
    summary = report.summary
    threshold = report.threshold_result.to_mapping() if report.threshold_result else None

    print(f"Manifest: {payload['manifest_path']}")
    print(f"Środowisko: {payload['environment']} ({payload['exchange']})")
    print(f"Ocena na: {payload['as_of']}")
    if dispatch_enabled:
        print(f"Alert wysłany: {'TAK' if dispatched else 'NIE (brak naruszeń)'}")
    elif dispatch_requested:
        print("Alerty: oczekiwane wg konfiguracji, ale wyłączone (--dispatch nie podano)")
    else:
        print("Alerty: wyłączone (tryb tylko-raport)")

    for entry in payload["entries"]:
        print(
            " - {symbol} {interval}: status={status} row_count={row_count} "
            "required={required_rows} gap={gap_minutes}".format(**entry)
        )

    print(
        "Podsumowanie: status={status} ok={ok}/{total} warning={warning} "
        "error={error} stale_entries={stale_entries} ok_ratio={ok_ratio}".format(
            status=summary.get("status"),
            ok=summary.get("ok"),
            total=summary.get("total"),
            warning=summary.get("warning"),
            error=summary.get("error"),
            stale_entries=summary.get("stale_entries"),
            ok_ratio=summary.get("ok_ratio"),
        )
    )

    if threshold:
        thresholds = threshold.get("thresholds") or {}
        observed = threshold.get("observed") or {}
        issues = threshold.get("issues") or []
        print("Progi jakości danych:")
        if thresholds:
            for name, value in thresholds.items():
                print(f" * {name}={value}")
        else:
            print(" * brak skonfigurowanych progów")
        if observed:
            observed_lines = ", ".join(f"{name}={value}" for name, value in observed.items())
            print(f"Wartości obserwowane: {observed_lines}")
        if issues:
            print("Naruszenia progów:")
            for issue in issues:
                print(f" * {issue}")
        else:
            print("Brak naruszeń progów jakości danych")

    issue_counts = summary.get("issue_counts") or {}
    issue_examples = summary.get("issue_examples") or {}
    if issue_counts:
        print("Kody problemów:")
        for code in sorted(issue_counts):
            count = issue_counts[code]
            example = issue_examples.get(code)
            if example:
                print(f" * {code}: count={count} example={example}")
            else:
                print(f" * {code}: count={count}")

    worst_gap = summary.get("worst_gap")
    if isinstance(worst_gap, dict):
        details = {
            "symbol": worst_gap.get("symbol", "?"),
            "interval": worst_gap.get("interval", "?"),
            "gap": worst_gap.get("gap_minutes"),
            "threshold": worst_gap.get("threshold_minutes"),
            "manifest_status": worst_gap.get("manifest_status"),
            "last": worst_gap.get("last_timestamp_iso"),
        }
        print(
            "Największa luka: {symbol} {interval} gap={gap}min threshold={threshold} "
            "manifest_status={manifest_status} last={last}".format(**details)
        )

    if report.issues:
        print("Problemy manifestu:")
        for issue in report.issues:
            print(f" * {issue}")
    else:
        print("Brak problemów z pokryciem danych")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    config_path = Path(args.config)
    try:
        config = load_core_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Błąd ładowania konfiguracji: {exc}", file=sys.stderr)
        return 2

    try:
        as_of = _parse_as_of(args.as_of)
    except ValueError as exc:  # noqa: BLE001
        print(f"Niepoprawny format --as-of: {exc}", file=sys.stderr)
        return 2

    run_specs: list[_RunSpec] = []
    manual_envs = list(args.environments)

    if not manual_envs and not args.all_configured:
        print(
            "Podaj przynajmniej jedno środowisko (--environment) lub użyj flagi --all-configured",
            file=sys.stderr,
        )
        return 2

    for env_name in manual_envs:
        environment = config.environments.get(env_name)
        if environment is None:
            print(f"Nie znaleziono środowiska: {env_name}", file=sys.stderr)
            return 2
        if not environment.instrument_universe:
            print(
                f"Środowisko {environment.name} nie ma przypisanego instrument_universe",
                file=sys.stderr,
            )
            return 2
        run_specs.append(
            _RunSpec(
                environment=environment,
                category=args.category,
                severity_override=args.severity_override,
                dispatch_enabled=bool(args.dispatch),
                dispatch_requested=bool(args.dispatch),
            )
        )

    monitoring = getattr(config, "coverage_monitoring", None)
    if args.all_configured:
        if monitoring is None:
            if not run_specs:
                print(
                    "Konfiguracja nie zawiera sekcji coverage_monitoring — uzupełnij core.yaml",
                    file=sys.stderr,
                )
                return 2
        elif not monitoring.enabled:
            if not run_specs:
                print("Monitoring pokrycia jest wyłączony w konfiguracji", file=sys.stderr)
                return 0
        else:
            if not getattr(monitoring, "targets", ()):  # pragma: no branch - defensywnie
                if not run_specs:
                    print(
                        "Sekcja coverage_monitoring nie zawiera żadnych środowisk",
                        file=sys.stderr,
                    )
                    return 2
            for target in monitoring.targets:
                env = config.environments.get(target.environment)
                if env is None:
                    print(
                        f"Nie znaleziono środowiska monitorowanego: {target.environment}",
                        file=sys.stderr,
                    )
                    return 2
                if not env.instrument_universe:
                    print(
                        f"Środowisko {env.name} nie ma przypisanego instrument_universe",
                        file=sys.stderr,
                    )
                    return 2
                target_dispatch = (
                    monitoring.default_dispatch if target.dispatch is None else bool(target.dispatch)
                )
                run_specs.append(
                    _RunSpec(
                        environment=env,
                        category=target.category or monitoring.default_category or args.category,
                        severity_override=target.severity_override or args.severity_override,
                        dispatch_enabled=bool(args.dispatch) and target_dispatch,
                        dispatch_requested=bool(target_dispatch),
                    )
                )

    if not run_specs:
        print(
            "Brak środowisk do sprawdzenia — zweryfikuj flagi CLI i konfigurację coverage_monitoring",
            file=sys.stderr,
        )
        return 2

    needs_dispatch = any(spec.dispatch_enabled for spec in run_specs)
    secret_manager: SecretManager | None = None
    if needs_dispatch:
        try:
            secret_manager = _create_secret_manager(args)
        except SecretStorageError as exc:
            print(f"Nie udało się przygotować kanałów alertów: {exc}", file=sys.stderr)
            return 2

    results: list[tuple[_RunSpec, CoverageReportPayload, bool]] = []
    exit_code = 0

    for spec in run_specs:
        try:
            report = _build_report_without_dispatch(
                config=config,
                environment=spec.environment,
                as_of=as_of,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Błąd walidacji pokrycia ({spec.environment.name}): {exc}",
                file=sys.stderr,
            )
            return 2

        dispatched = False
        if spec.dispatch_enabled:
            assert secret_manager is not None  # dla mypy/analizy statycznej
            try:
                router = _initialize_router(
                    config=config,
                    environment=spec.environment,
                    secret_manager=secret_manager,
                )
            except KeyError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            try:
                dispatched = dispatch_coverage_alert(
                    router,
                    payload=report.payload,
                    severity_override=spec.severity_override,
                    category=spec.category,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Nie udało się wysłać alertu dla {spec.environment.name}: {exc}",
                    file=sys.stderr,
                )
                return 2

        if report.issues or report.threshold_issues:
            exit_code = 1

        results.append((spec, report, dispatched))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json:
        if len(results) == 1:
            spec, report, dispatched = results[0]
            payload = _serialize_report(
                report,
                dispatch_enabled=spec.dispatch_enabled,
                dispatch_requested=spec.dispatch_requested,
                dispatched=dispatched,
            )
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            runs_payload: list[dict[str, object]] = []
            failing: list[str] = []
            suppressed: list[str] = []
            for spec, report, dispatched in results:
                payload = _serialize_report(
                    report,
                    dispatch_enabled=spec.dispatch_enabled,
                    dispatch_requested=spec.dispatch_requested,
                    dispatched=dispatched,
                )
                runs_payload.append(payload)
                if report.issues or report.threshold_issues:
                    failing.append(str(payload.get("environment")))
                if spec.dispatch_requested and not spec.dispatch_enabled:
                    suppressed.append(spec.environment.name)
            aggregated = {
                "runs": runs_payload,
                "total_runs": len(runs_payload),
                "failed_runs": len(failing),
                "environments_with_issues": failing,
                "overall_status": "ok" if not failing else "error",
                "dispatch_flag": bool(args.dispatch),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            if suppressed:
                aggregated["suppressed_dispatch_environments"] = suppressed
            serialized = json.dumps(aggregated, ensure_ascii=False, indent=2)

        if args.output:
            output_path.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    else:
        if args.output and len(results) == 1:
            spec, report, dispatched = results[0]
            payload = _serialize_report(
                report,
                dispatch_enabled=spec.dispatch_enabled,
                dispatch_requested=spec.dispatch_requested,
                dispatched=dispatched,
            )
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif args.output:
            runs_payload = [
                _serialize_report(
                    report,
                    dispatch_enabled=spec.dispatch_enabled,
                    dispatch_requested=spec.dispatch_requested,
                    dispatched=dispatched,
                )
                for spec, report, dispatched in results
            ]
            aggregated = {
                "runs": runs_payload,
                "total_runs": len(runs_payload),
                "overall_status": "ok"
                if not any(run.get("issues") or run.get("threshold_issues") for run in runs_payload)
                else "error",
            }
            output_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if len(results) > 1:
            print(f"Kontrola {len(results)} środowisk — dispatch globalny: {'TAK' if args.dispatch else 'NIE'}")
        for idx, (spec, report, dispatched) in enumerate(results, start=1):
            if len(results) > 1:
                print("\n=== Środowisko #{idx}: {name} ===".format(idx=idx, name=spec.environment.name))
            _print_text_report(
                report,
                dispatched=dispatched,
                dispatch_enabled=spec.dispatch_enabled,
                dispatch_requested=spec.dispatch_requested,
            )

    return exit_code


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
