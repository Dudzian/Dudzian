#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage6 Resilience – failover drill runner & evaluator (merged HEAD/main).

Subcommands:
  - run       : execute ResilienceFailoverDrill from core.yaml and write signed report
  - evaluate  : evaluate failover plan against a resilience bundle and (optionally) run self-healing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Common imports available in both branches ---
from bot_core.config import load_core_config

# main-branch API
from bot_core.resilience import ResilienceFailoverDrill

# HEAD-branch APIs
from bot_core.resilience.audit import audit_bundle
from bot_core.resilience.drill import (
    evaluate_failover_drill,
    load_failover_plan,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)
from bot_core.resilience.self_healing import (
    SubprocessSelfHealingExecutor,
    build_self_healing_plan,
    execute_self_healing_plan,
    load_self_healing_rules,
    summarize_self_healing_plan,
    write_self_healing_report,
    write_self_healing_signature,
)

_LOGGER = logging.getLogger("stage6.resilience.drill")


# ----------------------------- helpers -----------------------------
def _read_key_file(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:  # user-friendly error
        raise SystemExit(f"Nie znaleziono pliku z kluczem HMAC: {path}") from exc
    if not raw:
        raise SystemExit(f"Plik klucza HMAC jest pusty: {path}")
    return raw


def _default_signature_path(output_json: Path) -> Path:
    return output_json.with_suffix(output_json.suffix + ".sig")


def _default_self_heal_output(output_json: Path) -> Path:
    stem = output_json.stem + "_self_healing"
    return output_json.with_name(stem + output_json.suffix)


# ----------------------------- subcommand: run (main) -----------------------------
def _configure_run_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracji core (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka pliku raportu JSON (domyślnie wg resilience.report_directory)",
    )
    parser.add_argument(
        "--signing-key-path",
        help="Plik z kluczem HMAC do podpisania raportu",
    )
    parser.add_argument(
        "--signing-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC",
    )
    parser.add_argument(
        "--signing-key-id",
        help="Identyfikator klucza HMAC dołączany do podpisu",
    )
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Zakończ z kodem != 0 jeśli drill zakończy się naruszeniami progów",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    parser.set_defaults(_handler=_handle_run)
    return parser


def _build_parser_run(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "run",
        help="Uruchamia drill wg core.yaml i zapisuje podpisany raport",
        description="Uruchamia Stage6 Resilience Failover Drill i generuje podpisany raport.",
    )
    return _configure_run_parser(p)


def _resolve_signing_key_for_run(
    args: argparse.Namespace, config
) -> Tuple[Optional[bytes], Optional[str]]:
    resilience = getattr(config, "resilience", None)
    key_path = args.signing_key_path or (getattr(resilience, "signing_key_path", None) if resilience else None)
    key_env = args.signing_key_env or (getattr(resilience, "signing_key_env", None) if resilience else None)
    key_id = args.signing_key_id or (getattr(resilience, "signing_key_id", None) if resilience else None)

    key_bytes: Optional[bytes] = None
    if key_path:
        key_bytes = Path(key_path).read_bytes()
    elif key_env:
        value = os.environ.get(key_env)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {key_env} jest pusta")
        key_bytes = value.encode("utf-8")
    return key_bytes, key_id


def _handle_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    config = load_core_config(args.config)
    if config.resilience is None or not config.resilience.enabled:
        _LOGGER.warning("Resilience drills są wyłączone w konfiguracji")
        return 0

    resilience_config = config.resilience
    output_path = Path(
        args.output
        if args.output
        else Path(resilience_config.report_directory) / "resilience_failover_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drill = ResilienceFailoverDrill(resilience_config)
    report = drill.run()
    report.write_json(output_path)
    _LOGGER.info("Raport resilience zapisany do %s", output_path)

    key_bytes, key_id = _resolve_signing_key_for_run(args, config)
    if key_bytes:
        signature_path = output_path.with_suffix(output_path.suffix + ".sig")
        report.write_signature(signature_path, key=key_bytes, key_id=key_id)
        _LOGGER.info("Podpis HMAC zapisany do %s", signature_path)

    if (args.fail_on_breach or resilience_config.require_success) and report.has_failures():
        _LOGGER.error("Drill resilience wykazał naruszenia progów")
        return 3

    report_name = getattr(report, "name", getattr(resilience_config, "name", "resilience_failover"))
    status = "failed" if report.has_failures() else "passed"
    print(f"Zakończono drill '{report_name}' – status: {status}.")
    return 0


# ----------------------------- subcommand: evaluate (HEAD) -----------------------------
def _configure_evaluate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--bundle", required=True, help="Ścieżka do paczki odpornościowej (.zip/.tar itp.)")
    parser.add_argument("--plan", required=True, help="Ścieżka do planu failover (JSON)")
    parser.add_argument("--output-json", required=True, help="Ścieżka do zapisu podsumowania JSON")
    parser.add_argument("--output-csv", help="Opcjonalna ścieżka do raportu CSV z wynikami usług")

    # podpis podsumowania (HEAD naming)
    parser.add_argument("--signing-key", help="Ścieżka do klucza HMAC (podpis podsumowania)")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC dołączany do podpisu")
    parser.add_argument(
        "--signature-path", help="Ręcznie wskazana ścieżka dokumentu podpisu (domyślnie obok JSON)"
    )

    # self-healing (HEAD)
    parser.add_argument("--self-heal-config", help="Konfiguracja self-healing (JSON z regułami)")
    parser.add_argument(
        "--self-heal-mode",
        choices=("plan", "execute"),
        default="plan",
        help="Tryb self-healing: tylko plan lub wykonanie",
    )
    parser.add_argument(
        "--self-heal-output", help="Ścieżka zapisu raportu self-healing (domyślnie obok podsumowania)"
    )
    parser.add_argument("--self-heal-signing-key", help="Klucz HMAC do podpisu raportu self-healing")
    parser.add_argument(
        "--self-heal-signing-key-id", help="Identyfikator klucza HMAC raportu self-healing"
    )
    parser.add_argument(
        "--self-heal-signature-path", help="Ścieżka pliku podpisu self-healing (domyślnie obok raportu)"
    )

    parser.add_argument("--log-level", default="INFO", help="Poziom logowania (domyślnie INFO)")
    parser.set_defaults(_handler=_handle_evaluate)
    return parser


def _build_parser_evaluate(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "evaluate",
        help="Ocena paczki failover + planu oraz (opcjonalnie) self-healing",
        description="Symulacja ćwiczenia failover Stage6 wraz z raportowaniem i podpisem HMAC.",
    )
    return _configure_evaluate_parser(p)


def _handle_evaluate(args: argparse.Namespace) -> int:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    bundle_path = Path(args.bundle).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    output_json = Path(args.output_json).expanduser()
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else None

    plan = load_failover_plan(plan_path)
    bundle_audit = audit_bundle(bundle_path)
    if bundle_audit.manifest is None:
        for error in bundle_audit.errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    summary = evaluate_failover_drill(plan, bundle_audit.manifest, bundle_audit=bundle_audit)
    summary_payload = write_summary_json(summary, output_json)
    if output_csv is not None:
        write_summary_csv(summary, output_csv)

    if args.signing_key:
        key_path = Path(args.signing_key).expanduser().resolve()
        signature_path = (
            Path(args.signature_path).expanduser()
            if args.signature_path
            else _default_signature_path(output_json)
        )
        key = _read_key_file(key_path)
        write_summary_signature(
            summary_payload,
            signature_path,
            key=key,
            key_id=args.signing_key_id,
            target=output_json.name,
        )

    # opcjonalny self-healing
    if args.self_heal_config:
        rules_path = Path(args.self_heal_config).expanduser().resolve()
        self_heal_output = (
            Path(args.self_heal_output).expanduser()
            if args.self_heal_output
            else _default_self_heal_output(output_json)
        )
        rules = load_self_healing_rules(rules_path)
        plan_sh = build_self_healing_plan(summary, rules)
        if args.self_heal_mode == "execute":
            executor = SubprocessSelfHealingExecutor()
            report = execute_self_healing_plan(plan_sh, executor)
        else:
            report = summarize_self_healing_plan(plan_sh)

        report_payload = write_self_healing_report(report, self_heal_output)

        if args.self_heal_signing_key:
            key_path = Path(args.self_heal_signing_key).expanduser().resolve()
            signature_path = (
                Path(args.self_heal_signature_path).expanduser()
                if args.self_heal_signature_path
                else _default_signature_path(self_heal_output)
            )
            key = _read_key_file(key_path)
            write_self_healing_signature(
                report_payload,
                signature_path,
                key=key,
                key_id=args.self_heal_signing_key_id,
                target=self_heal_output.name,
            )

        print(f"Self-healing {report.mode} zakończony statusem: {report.status} (akcje: {len(report.actions)}).")

    print(
        f"Zakończono ćwiczenie failover '{summary.drill_name}' – status: {summary.status} "
        f"(usług: {summary.counts['total']})."
    )
    return 0


# ----------------------------- entrypoint -----------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage6 Resilience Failover Drill – runner & evaluator (merged CLI)"
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{run|evaluate}", required=True)
    _build_parser_run(sub)
    _build_parser_evaluate(sub)
    return parser


def _build_compat_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage6 Resilience Failover Drill – tryb kompatybilności (ustawienia core.yaml)"
        )
    )
    return _configure_run_parser(parser)


def _build_compat_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage6 Resilience Failover Drill – tryb kompatybilności (bundle + plan failover)"
        )
    )
    return _configure_evaluate_parser(parser)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        parser = _build_parser()
        parser.print_help()
        return 2

    if any(arg in {"run", "evaluate"} for arg in args_list):
        parser = _build_parser()
        parsed = parser.parse_args(args_list)
        return parsed._handler(parsed)

    if any(arg == "--bundle" or arg.startswith("--bundle=") for arg in args_list):
        compat_parser = _build_compat_evaluate_parser()
        args_namespace = compat_parser.parse_args(args_list)
        return args_namespace._handler(args_namespace)

    compat_parser = _build_compat_run_parser()
    args_namespace = compat_parser.parse_args(args_list)
    return args_namespace._handler(args_namespace)


def run(argv: Sequence[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
