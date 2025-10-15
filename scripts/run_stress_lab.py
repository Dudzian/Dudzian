"""Uruchamia Stress Lab Stage6 i generuje podpisany raport."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from bot_core.config import load_core_config
from bot_core.risk.stress_lab import StressLab

_LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracji core (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka pliku raportu JSON (domyślnie na podstawie konfiguracji)",
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
        help="Identyfikator klucza umieszczany w podpisie",
    )
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Zakończ z kodem != 0 jeśli scenariusze zakończą się niepowodzeniem",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _resolve_signing_key(args: argparse.Namespace, config) -> tuple[Optional[bytes], Optional[str]]:
    key_path = args.signing_key_path or config.stress_lab.signing_key_path if config.stress_lab else None
    key_env = args.signing_key_env or config.stress_lab.signing_key_env if config.stress_lab else None
    key_id = args.signing_key_id or (config.stress_lab.signing_key_id if config.stress_lab else None)

    key_bytes: Optional[bytes] = None
    key_identifier: Optional[str] = key_id

    if key_path:
        key_bytes = Path(key_path).read_bytes()
    elif key_env:
        value = os.environ.get(key_env)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {key_env} jest pusta")
        key_bytes = value.encode("utf-8")

    return key_bytes, key_identifier


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    config = load_core_config(args.config)
    if config.stress_lab is None or not config.stress_lab.enabled:
        _LOGGER.warning("Stress Lab jest wyłączony w konfiguracji")
        return 0

    stress_config = config.stress_lab
    output_path = Path(
        args.output
        if args.output
        else Path(stress_config.report_directory) / "stress_lab_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lab = StressLab(stress_config)
    report = lab.run()
    report.write_json(output_path)
    _LOGGER.info("Raport Stress Lab zapisany do %s", output_path)

    key_bytes, key_id = _resolve_signing_key(args, config)
    if key_bytes:
        signature_path = output_path.with_suffix(output_path.suffix + ".sig")
        report.write_signature(signature_path, key=key_bytes, key_id=key_id)
        _LOGGER.info("Podpis HMAC zapisany do %s", signature_path)

    if (args.fail_on_breach or stress_config.require_success) and report.has_failures():
        _LOGGER.error("Stress Lab wykrył naruszenia progów")
        return 3

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
