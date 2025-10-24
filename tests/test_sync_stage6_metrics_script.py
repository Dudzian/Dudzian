"""Testy skryptu synchronizacji metryk Stage6."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import sync_stage6_metrics


def _write_metrics(path: Path) -> None:
    """Zapisuje minimalny zestaw metryk Stage6 do pliku JSON."""

    payload = [
        {
            "indicator": "router_availability_pct",
            "value": 99.5,
            "window_start": datetime(2024, 5, 1, tzinfo=timezone.utc).isoformat(),
            "window_end": datetime(2024, 5, 2, tzinfo=timezone.utc).isoformat(),
            "sample_size": 1440,
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sync_stage6_metrics_copies_and_validates(tmp_path, capsys):
    """Skrypt powinien kopiować plik JSON i raportować liczbę pomiarów."""

    source = tmp_path / "stage6.json"
    destination = tmp_path / "metrics" / "stage6_measurements.json"
    _write_metrics(source)

    exit_code = sync_stage6_metrics.main(
        ["--source", str(source), "--output", str(destination)]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Zweryfikowano 1 pomiar Stage6" in captured
    assert destination.is_file()


def test_sync_stage6_metrics_decompresses_gzip(tmp_path):
    """Plik wejściowy .gz powinien zostać zdekompresowany i zweryfikowany."""

    source = tmp_path / "stage6.json.gz"
    destination = tmp_path / "out.json"

    buffer = Path(tmp_path / "buffer.json")
    _write_metrics(buffer)
    with gzip.open(source, "wb") as stream:
        stream.write(buffer.read_bytes())

    exit_code = sync_stage6_metrics.main(
        ["--source", str(source), "--output", str(destination)]
    )

    assert exit_code == 0
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["indicator"] == "router_availability_pct"


def test_sync_stage6_metrics_rejects_empty_payload(tmp_path, capsys):
    """Pusta lista pomiarów powinna zostać odrzucona jako błąd walidacji."""

    source = tmp_path / "empty.json"
    destination = tmp_path / "metrics.json"
    source.write_text("[]", encoding="utf-8")

    exit_code = sync_stage6_metrics.main(
        ["--source", str(source), "--output", str(destination)]
    )

    assert exit_code == 1
    captured = capsys.readouterr().err
    assert "nie zawiera żadnych pomiarów" in captured


def test_sync_stage6_metrics_skip_validation_allows_empty(tmp_path, capsys):
    """Opcja --skip-validate powinna umożliwić kopiowanie bez sprawdzania danych."""

    source = tmp_path / "empty.json"
    destination = tmp_path / "metrics.json"
    source.write_text("[]", encoding="utf-8")

    exit_code = sync_stage6_metrics.main(
        [
            "--source",
            str(source),
            "--output",
            str(destination),
            "--skip-validate",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Walidacja struktury pominięta" in captured
