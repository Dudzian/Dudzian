"""Pomocniczy skrypt do kopiowania lub dekompresji metryk Stage6."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.observability.io import load_slo_measurements


DEFAULT_OUTPUT = Path("var/metrics/stage6_measurements.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kopiuje lub dekompresuje plik stage6_measurements.json do lokalizacji "
            "używanej przez hypercare"
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Ścieżka źródłowa do pliku z metrykami (JSON lub JSON.gz)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Ścieżka docelowa (domyślnie var/metrics/stage6_measurements.json). "
            "Ścieżka może być względna względem bieżącego katalogu."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Zastąp istniejący plik wyjściowy, jeśli już istnieje",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Nie weryfikuj struktury JSON po skopiowaniu",
    )
    return parser.parse_args(argv)


def _validate_metrics(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywnie
        raise ValueError(
            f"Plik {path} nie zawiera prawidłowego JSON z metrykami Stage6"
        ) from exc

    try:
        measurements = load_slo_measurements(path)
    except Exception as exc:  # pragma: no cover - defensywnie
        raise ValueError(
            "Nie udało się zweryfikować struktury metryk Stage6 (load_slo_measurements)."
        ) from exc

    if not measurements:
        raise ValueError(
            f"Plik {path} nie zawiera żadnych pomiarów SLO Stage6"
        )

    return len(measurements)


def _format_measurement_phrase(count: int) -> str:
    if count == 1:
        return "1 pomiar Stage6"
    return f"{count} pomiarów Stage6"


def _copy_metrics(source: Path, destination: Path, *, validate: bool) -> int | None:
    if not source.is_file():
        raise FileNotFoundError(f"Nie znaleziono źródłowego pliku metryk: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix == ".gz":
        with gzip.open(source, "rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source, destination)

    if validate:
        return _validate_metrics(destination)

    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()

    if output.exists() and not args.force:
        print(
            f"[sync_stage6_metrics] Plik docelowy {output} już istnieje. "
            "Użyj --force aby go nadpisać.",
            file=sys.stderr,
        )
        return 1

    try:
        measurement_count = _copy_metrics(source, output, validate=not args.skip_validate)
    except Exception as exc:  # pragma: no cover - komunikat CLI
        print(f"[sync_stage6_metrics] Błąd: {exc}", file=sys.stderr)
        return 1

    message = [
        f"[sync_stage6_metrics] Zapisano metryki Stage6 do {output} (źródło: {source})."
    ]
    if measurement_count is not None:
        message.append(
            f"Zweryfikowano {_format_measurement_phrase(measurement_count)}."
        )
    else:
        message.append("Walidacja struktury pominięta (--skip-validate).")

    print(" ".join(message))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

