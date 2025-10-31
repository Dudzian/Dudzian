"""Uruchamia health-check runtime'u korzystając z istniejącego CLI bot_core."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Sprawdza zdrowie runtime'u bota")
    parser.add_argument("--exchange", required=True, help="Identyfikator giełdy do testu health-check")
    parser.add_argument(
        "--credentials-file",
        dest="credentials_file",
        default=str(Path("config/exchanges.json")),
        help="Plik z danymi uwierzytelniającymi (domyślnie config/exchanges.json)",
    )
    parser.add_argument("--mode", dest="mode", default=None, help="Tryb giełdy (spot/margin/futures)")
    parser.add_argument("--testnet", dest="testnet", action="store_true", help="Wymuś tryb testnet")
    parser.add_argument(
        "--environment-config",
        dest="environment_config",
        default=None,
        help="Opcjonalny plik konfiguracji środowiskowej",
    )
    parser.add_argument("--environment", dest="environment", default=None, help="Nazwa środowiska z pliku konfiguracyjnego")
    parser.add_argument(
        "--output",
        dest="output",
        choices=("text", "json"),
        default="text",
        help="Format wyjścia health-check (domyślnie text)",
    )
    parser.add_argument(
        "--list-checks",
        dest="list_checks",
        action="store_true",
        help="Wyświetl dostępne testy health-check zamiast je uruchamiać",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, remainder = _parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "bot_core.cli",
        "health-check",
        "--exchange",
        args.exchange,
        "--credentials-file",
        args.credentials_file,
        "--output-format",
        args.output,
    ]
    if args.mode:
        command.extend(["--mode", args.mode])
    if args.testnet:
        command.append("--testnet")
    if args.environment_config:
        command.extend(["--environment-config", args.environment_config])
    if args.environment:
        command.extend(["--environment", args.environment])
    if args.list_checks:
        command.append("--list-checks")
    command.extend(remainder)

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
