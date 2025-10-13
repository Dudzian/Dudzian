"""Narzędzie CLI do walidacji znormalizowanych danych backtestowych."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - środowisko uruchomieniowe
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.data import BacktestDatasetLibrary, DataQualityValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weryfikuje integralność biblioteki znormalizowanych danych backtestowych.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/backtests/normalized/manifest.yaml"),
        help="Ścieżka do manifestu biblioteki danych.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    library = BacktestDatasetLibrary(args.manifest)
    validator = DataQualityValidator(library)
    reports: Dict[str, Dict[str, Any]] = {}
    for dataset_name in library.list_dataset_names():
        report = validator.validate(dataset_name)
        reports[dataset_name] = {
            "is_passing": report.is_passing,
            "issues": [issue.__dict__ for issue in report.issues],
            "row_count": report.row_count,
            "start_timestamp": report.start_timestamp,
            "end_timestamp": report.end_timestamp,
            "strategies": list(report.descriptor.strategies),
            "risk_profiles": list(report.descriptor.risk_profiles),
        }
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
