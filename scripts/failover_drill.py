#!/usr/bin/env python3
"""Symulacja ćwiczenia failover Stage6 wraz z raportowaniem i podpisem HMAC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _read_key(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise SystemExit(f"Nie znaleziono pliku z kluczem HMAC: {path}") from exc
    if not raw:
        raise SystemExit(f"Plik klucza HMAC jest pusty: {path}")
    return raw


def _default_signature_path(output_json: Path) -> Path:
    return output_json.with_suffix(output_json.suffix + ".sig")


def _default_self_heal_output(output_json: Path) -> Path:
    stem = output_json.stem + "_self_healing"
    return output_json.with_name(stem + output_json.suffix)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Ścieżka do paczki odpornościowej (.zip)")
    parser.add_argument("--plan", required=True, help="Ścieżka do planu failover (JSON)")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Ścieżka do zapisu podsumowania JSON",
    )
    parser.add_argument(
        "--output-csv",
        help="Opcjonalna ścieżka do raportu CSV z wynikami usług",
    )
    parser.add_argument(
        "--signing-key",
        help="Ścieżka do klucza HMAC (podpis podsumowania)",
    )
    parser.add_argument(
        "--signing-key-id",
        help="Identyfikator klucza HMAC dołączany do podpisu",
    )
    parser.add_argument(
        "--signature-path",
        help="Ręcznie wskazana ścieżka dla dokumentu podpisu (domyślnie obok JSON)",
    )
    parser.add_argument(
        "--self-heal-config",
        help="Opcjonalna konfiguracja self-healing (JSON z regułami)",
    )
    parser.add_argument(
        "--self-heal-mode",
        choices=("plan", "execute"),
        default="plan",
        help="Tryb self-healing: tylko plan lub wykonanie",
    )
    parser.add_argument(
        "--self-heal-output",
        help="Ścieżka zapisu raportu self-healing (domyślnie obok podsumowania)",
    )
    parser.add_argument(
        "--self-heal-signing-key",
        help="Klucz HMAC do podpisu raportu self-healing",
    )
    parser.add_argument(
        "--self-heal-signing-key-id",
        help="Identyfikator klucza HMAC raportu self-healing",
    )
    parser.add_argument(
        "--self-heal-signature-path",
        help="Ścieżka pliku podpisu self-healing (domyślnie obok raportu)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
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
        key = _read_key(key_path)
        write_summary_signature(
            summary_payload,
            signature_path,
            key=key,
            key_id=args.signing_key_id,
            target=output_json.name,
        )

    if args.self_heal_config:
        rules_path = Path(args.self_heal_config).expanduser().resolve()
        self_heal_output = (
            Path(args.self_heal_output).expanduser()
            if args.self_heal_output
            else _default_self_heal_output(output_json)
        )
        rules = load_self_healing_rules(rules_path)
        plan = build_self_healing_plan(summary, rules)
        if args.self_heal_mode == "execute":
            executor = SubprocessSelfHealingExecutor()
            report = execute_self_healing_plan(plan, executor)
        else:
            report = summarize_self_healing_plan(plan)
        report_payload = write_self_healing_report(report, self_heal_output)

        if args.self_heal_signing_key:
            key_path = Path(args.self_heal_signing_key).expanduser().resolve()
            signature_path = (
                Path(args.self_heal_signature_path).expanduser()
                if args.self_heal_signature_path
                else _default_signature_path(self_heal_output)
            )
            key = _read_key(key_path)
            write_self_healing_signature(
                report_payload,
                signature_path,
                key=key,
                key_id=args.self_heal_signing_key_id,
                target=self_heal_output.name,
            )

    print(
        f"Zakończono ćwiczenie failover '{summary.drill_name}' – status: {summary.status} (usług: {summary.counts['total']})."
    )
    if args.self_heal_config:
        print(
            f"Self-healing {report.mode} zakończony statusem: {report.status} (akcje: {len(report.actions)})."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
