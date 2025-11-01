"""CLI uruchamiające audyt zgodności i generujące raport."""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.compliance import ComplianceAuditor
from core.reporting import ComplianceReport

LOGGER = logging.getLogger("run_compliance_audit")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uruchamia audyt zgodności (KYC/AML) i zapisuje raport.",
    )
    parser.add_argument(
        "--audit-config",
        type=Path,
        default=Path("config/compliance/audit.yml"),
        help="Plik konfiguracji audytu zgodności (YAML)",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        help="Plik JSON/YAML z konfiguracją strategii (name/exchange/tags)",
    )
    parser.add_argument(
        "--data-sources",
        type=Path,
        help="Plik JSON/YAML zawierający listę źródeł danych.",
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        help="Plik JSON/YAML z listą transakcji do audytu.",
    )
    parser.add_argument(
        "--kyc-profile",
        type=Path,
        help="Plik JSON/YAML z profilem KYC (status, kraj, pola obowiązkowe).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/compliance"),
        help="Katalog docelowy dla raportów audytu.",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Zwróć kod wyjścia 2, jeśli audyt wykryje naruszenia.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (np. INFO, DEBUG).",
    )
    return parser


def _read_payload(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Plik {path} nie istnieje")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _load_mapping(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = _read_payload(path)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SystemExit(f"Plik {path} musi zawierać mapę klucz→wartość")
    return payload


def _load_sequence(path: Path | None) -> Sequence[Any] | None:
    if path is None:
        return None
    payload = _read_payload(path)
    if payload is None:
        return None
    if not isinstance(payload, Sequence) or isinstance(payload, (bytes, str)):
        raise SystemExit(f"Plik {path} musi zawierać listę wartości")
    return payload


def _default_strategy() -> Mapping[str, Any]:
    return {
        "name": "synthetic_strategy",
        "exchange": "demo_exchange",
        "tags": ["synthetic"],
    }


def _default_data_sources() -> Sequence[str]:
    return ("historical_prices", "fundamental_feed")


def _default_transactions() -> Sequence[Mapping[str, Any]]:
    return ()


def _default_kyc_profile() -> Mapping[str, Any]:
    return {"status": "verified", "country": "PL", "full_name": "Demo"}


def run_cli(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    auditor = ComplianceAuditor(config_path=args.audit_config)

    strategy_config = _load_mapping(args.strategy) or _default_strategy()
    data_sources = list(_load_sequence(args.data_sources) or _default_data_sources())
    transactions_payload = _load_sequence(args.transactions) or _default_transactions()
    transactions: list[Mapping[str, Any]] = []
    for item in transactions_payload:
        if not isinstance(item, Mapping):
            raise SystemExit("Każda transakcja musi być mapą z polami (wartość, timestamp, kraj)")
        transactions.append(dict(item))
    kyc_profile = _load_mapping(args.kyc_profile) or _default_kyc_profile()

    LOGGER.info("Uruchamiam audyt zgodności dla strategii: %s", strategy_config.get("name"))
    result = auditor.audit(
        strategy_config=strategy_config,
        data_sources=tuple(str(source) for source in data_sources),
        transactions=tuple(transactions),
        kyc_profile=kyc_profile,
        as_of=datetime.now(timezone.utc),
    )

    report = ComplianceReport.from_audit(result)
    json_path = report.write_json(args.report_dir)
    markdown_path = report.write_markdown(args.report_dir)

    LOGGER.info("Raport audytu zapisany w %s oraz %s", json_path, markdown_path)
    if not result.passed:
        LOGGER.warning("Audyt wykrył %s naruszeń", len(result.findings))
        return 2 if args.fail_on_findings else 0
    LOGGER.info("Audyt zakończony bez naruszeń")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
