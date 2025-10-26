from __future__ import annotations

import csv
import builtins
import gzip
import gc
import json
import math
import subprocess
import sys
import weakref
from array import array
from weakref import ReferenceType
from collections.abc import Callable, Iterable, Mapping
from typing import TextIO
from types import GeneratorType
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

import pytest

import scripts.calibrate_autotrade_thresholds as calibrate_autotrade_thresholds

from bot_core.runtime.journal import TradingDecisionEvent

from scripts.calibrate_autotrade_thresholds import (
    _AMBIGUOUS_SYMBOL_MAPPING,
    _canonicalize_symbol_key,
    _build_symbol_map,
    _MetricSeries,
    _generate_report,
    _load_autotrade_entries,
    _load_current_signal_thresholds,
    _load_journal_events,
    _parse_percentiles,
    _parse_threshold_mapping,
    _resolve_freeze_event_limit,
)


class _TrackingReadHandle:
    def __init__(
        self,
        path: Path,
        opener: Callable[[Path], TextIO] | None = None,
    ) -> None:
        if opener is None:
            self._handle = path.open("r", encoding="utf-8")
        else:
            self._handle = opener(path)
        self.read_requests: list[int] = []
        self.read_results: list[int] = []
        self.readline_requests: list[int] = []
        self.readline_results: list[int] = []

    def read(self, size: int = -1) -> str:
        self.read_requests.append(size)
        chunk = self._handle.read(size)
        if chunk:
            self.read_results.append(len(chunk))
        return chunk

    def readline(self, size: int = -1) -> str:
        self.readline_requests.append(size)
        chunk = self._handle.readline(size)
        if chunk:
            self.readline_results.append(len(chunk))
        return chunk

    def __iter__(self) -> "_TrackingReadHandle":
        return self

    def __next__(self) -> str:
        line = self.readline()
        if line == "":
            raise StopIteration
        return line

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_TrackingReadHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._handle.close()

    def __getattr__(self, item: str):  # pragma: no cover - passthrough helper
        return getattr(self._handle, item)


def _write_journal(path: Path) -> None:
    events = [
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2023, 12, 31, 23, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.90",
                "signal_after_clamp": "0.88",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.62",
                "signal_after_clamp": "0.55",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 1, 15, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "NaN",
                "signal_after_clamp": "Infinity",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.48",
                "signal_after_clamp": "0.52",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="ETHUSDT",
            primary_exchange="kraken",
            strategy="mean_reversion",
            metadata={
                "signal_after_adjustment": "-0.30",
                "signal_after_clamp": "-0.28",
            },
        ),
        TradingDecisionEvent(
            event_type="risk_freeze",
            timestamp=datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            status="risk_freeze",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "reason": "manual_override",
                "frozen_for": "120",
            },
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.as_dict()))
            handle.write("\n")


def _write_autotrade_export(path: Path) -> None:
    payload = {
        "version": 1,
        "entries": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.72},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T01:30:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.65},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T01:45:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": "NaN"},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T02:00:00Z",
                "detail": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.41},
                },
            },
            {
                "timestamp": "2024-01-01T03:30:00Z",
                "status": "auto_risk_freeze",
                "detail": {
                    "symbol": "BTCUSDT",
                    "reason": "risk_score_threshold",
                    "frozen_for": 180,
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {
                        "risk_score": 0.83,
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                    },
                },
            },
            {
                "timestamp": "2024-01-01T03:45:00Z",
                "status": "auto_risk_freeze",
                "detail": {
                    "symbol": "BTCUSDT",
                    "reason": "risk_score_threshold",
                    "frozen_for": "NaN",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.9},
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_autotrade_export_is_streamed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "huge_autotrade.json"
    entries: list[dict[str, object]] = []
    padding = "X" * 8192
    for index in range(512):
        entry = {
            "timestamp": f"2024-02-01T00:{index:02d}:00Z",
            "decision": {
                "details": {
                    "symbol": f"ASSET{index:03d}",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": float(index % 10) / 10.0},
                    "note": padding,
                }
            },
        }
        entries.append(entry)
    payload = {"entries": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    total_size = path.stat().st_size

    read_sizes: list[int] = []
    entry_positions: list[int] = []
    stream_holder: dict[str, object] = {}

    class InstrumentedHandle:
        def __init__(self, real_path: Path):
            self._handle = real_path.open("r", encoding="utf-8")
            self.bytes_consumed = 0

        def read(self, size: int = -1) -> str:
            chunk = self._handle.read(size)
            if chunk:
                self.bytes_consumed += len(chunk)
                read_sizes.append(len(chunk))
            return chunk

        def __enter__(self) -> "InstrumentedHandle":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self._handle.close()

        def close(self) -> None:
            self._handle.close()

    def fake_open_text_file(candidate: Path) -> InstrumentedHandle:
        handle = InstrumentedHandle(candidate)
        stream_holder["stream"] = handle
        return handle

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds._open_text_file",
        fake_open_text_file,
    )

    from scripts import calibrate_autotrade_thresholds as module

    original_extract = module._extract_entry_timestamp

    def patched_extract_entry_timestamp(entry: Mapping[str, object]) -> datetime | None:
        stream = stream_holder.get("stream")
        if stream is not None:
            entry_positions.append(stream.bytes_consumed)
        return original_extract(entry)

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds._extract_entry_timestamp",
        patched_extract_entry_timestamp,
    )

    collected = list(module._load_autotrade_entries([path]))
    assert len(collected) == len(entries)
    assert entry_positions[0] < total_size
    assert entry_positions == sorted(entry_positions)
    non_empty_reads = [size for size in read_sizes if size]
    assert non_empty_reads
    assert non_empty_reads[0] < total_size
    assert stream_holder["stream"].bytes_consumed == total_size


def test_autotrade_array_is_streamed_incrementally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "array_autotrade.json"
    entries = [
        {"timestamp": "2024-02-01T00:00:00Z", "value": 1},
        {"timestamp": "2024-02-01T00:01:00Z", "value": 2},
        {"timestamp": "2024-02-01T00:02:00Z", "value": 3},
    ]
    export_path.write_text(json.dumps(entries), encoding="utf-8")
    total_size = export_path.stat().st_size

    from scripts import calibrate_autotrade_thresholds as module

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 8)

    class CountingHandle:
        def __init__(self, path: Path):
            self._handle = path.open("r", encoding="utf-8")
            self.read_calls = 0
            self.bytes_read = 0

        def read(self, size: int = -1) -> str:
            chunk = self._handle.read(size)
            if chunk:
                self.read_calls += 1
                self.bytes_read += len(chunk)
            return chunk

        def readline(self, size: int = -1) -> str:
            return self._handle.readline(size)

        def tell(self) -> int:
            return self._handle.tell()

        def close(self) -> None:
            self._handle.close()

        def __iter__(self):
            return iter(self._handle)

        def __enter__(self) -> "CountingHandle":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self._handle.close()

    holder: dict[str, CountingHandle] = {}

    def fake_open_text_file(path: Path) -> CountingHandle:
        handle = CountingHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    since = datetime(2024, 2, 1, 0, 1, tzinfo=timezone.utc)
    until = datetime(2024, 2, 1, 0, 2, tzinfo=timezone.utc)

    iterator = module._load_autotrade_entries([export_path], since=since, until=until)

    first_entry = next(iterator)
    handle = holder["handle"]
    assert first_entry["timestamp"] == "2024-02-01T00:01:00Z"
    assert handle.bytes_read < total_size
    first_calls = handle.read_calls

    second_entry = next(iterator)
    assert second_entry["timestamp"] == "2024-02-01T00:02:00Z"
    assert handle.read_calls >= first_calls

    with pytest.raises(StopIteration):
        next(iterator)

    assert handle.bytes_read == total_size


def test_autotrade_nested_entries_are_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "nested_autotrade.json"
    nested_entries = [
        {"timestamp": "2024-03-01T00:00:00Z", "value": 10},
        {"timestamp": "2024-03-01T00:05:00Z", "value": 11},
    ]
    section_entries = [
        {"timestamp": "2024-03-01T00:10:00Z", "value": 20},
        {"timestamp": "2024-03-01T00:15:00Z", "value": 21},
    ]
    payload = {
        "meta": {"note": "test"},
        "data": {
            "nested": {
                "entries": nested_entries,
                "aux": {"flags": [1, 2, 3]},
            },
            "sections": [
                {"entries": section_entries},
                {"info": "noop"},
            ],
            "summary": {"count": 4},
        },
    }
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    from scripts import calibrate_autotrade_thresholds as module

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 8)

    class CountingHandle:
        def __init__(self, path: Path):
            self._handle = path.open("r", encoding="utf-8")
            self.read_calls = 0
            self.bytes_read = 0

        def read(self, size: int = -1) -> str:
            chunk = self._handle.read(size)
            if chunk:
                self.read_calls += 1
                self.bytes_read += len(chunk)
            return chunk

        def readline(self, size: int = -1) -> str:
            return self._handle.readline(size)

        def __iter__(self):
            return iter(self._handle)

        def __enter__(self) -> "CountingHandle":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self._handle.close()

        def close(self) -> None:
            self._handle.close()

    holder: dict[str, CountingHandle] = {}

    def fake_open_text_file(path: Path) -> CountingHandle:
        handle = CountingHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    collected = list(module._load_autotrade_entries([export_path]))
    assert len(collected) == len(nested_entries) + len(section_entries)
    assert {entry["value"] for entry in collected} == {10, 11, 20, 21}
    assert holder["handle"].read_calls > 1


def test_script_generates_report(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5,0.9",
            "--suggestion-percentile",
            "0.9",
            "--since",
            "2024-01-01T00:30:00Z",
            "--until",
            "2024-01-01T04:00:00Z",
            "--current-threshold",
            "signal_after_adjustment=0.82,signal_after_clamp=0.78",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Zapisano raport JSON" in result.stdout
    assert output_json.exists()
    assert output_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.autotrade.threshold_calibration"
    sources_payload = payload["sources"]
    assert sources_payload["current_thresholds"]["files"] == []
    inline_sources = sources_payload["current_thresholds"]["inline"]
    assert inline_sources["signal_after_adjustment"] == pytest.approx(0.82)
    assert inline_sources["signal_after_clamp"] == pytest.approx(0.78)
    assert sources_payload["risk_thresholds"]["files"] == []
    assert sources_payload["risk_thresholds"]["inline"] == {}
    groups = {
        (entry["primary_exchange"], entry["strategy"]): entry for entry in payload["groups"]
    }
    trend_group = groups[("binance", "trend_following")]
    signal_stats = trend_group["metrics"]["signal_after_adjustment"]
    assert signal_stats["count"] == 1
    assert signal_stats["percentiles"]["p90"] >= 0.48
    assert signal_stats["suggested_threshold"] >= signal_stats["percentiles"]["p90"]
    assert signal_stats["current_threshold"] == 0.82
    assert all(math.isfinite(value) for value in signal_stats["percentiles"].values())

    signal_clamp_stats = trend_group["metrics"]["signal_after_clamp"]
    assert signal_clamp_stats["current_threshold"] == 0.78
    assert all(math.isfinite(value) for value in signal_clamp_stats["percentiles"].values())

    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["count"] == 1
    assert risk_stats["current_threshold"] >= 0.7
    assert all(math.isfinite(value) for value in risk_stats["percentiles"].values())

    freeze_stats = trend_group["metrics"]["risk_freeze_duration"]
    assert freeze_stats["count"] == 2
    assert freeze_stats["max"] >= 180
    assert all(math.isfinite(value) for value in freeze_stats["percentiles"].values())

    freeze_summary = trend_group["freeze_summary"]
    assert freeze_summary["total"] == 3
    assert freeze_summary["auto"] == 2
    assert freeze_summary["manual"] == 1
    reasons = {item["reason"]: item["count"] for item in freeze_summary["reasons"]}
    assert reasons["manual_override"] == 1
    assert reasons["risk_score_threshold"] == 2
    assert "raw_freeze_events" not in trend_group

    mean_rev_group = groups[("kraken", "mean_reversion")]
    mean_rev_risk = mean_rev_group["metrics"]["risk_score"]
    assert mean_rev_risk["count"] == 1

    global_summary = payload["global_summary"]
    assert global_summary["freeze_summary"]["total"] == 3
    global_signal = global_summary["metrics"]["signal_after_adjustment"]
    assert global_signal["count"] == 2
    assert global_signal["current_threshold"] == 0.82
    assert "raw_values" not in trend_group

    with output_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert {row["metric"] for row in rows} >= {
        "signal_after_adjustment",
        "risk_score",
        "risk_freeze_duration",
    }
    for row in rows:
        if row["metric"] == "risk_score" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) >= 0.7
        if row["metric"] == "signal_after_adjustment" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) == 0.82
        if row["metric"] == "signal_after_clamp" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) == 0.78
    assert any(row["primary_exchange"] == "__all__" for row in rows)

    assert "raw_values" not in payload["global_summary"]


def test_script_accepts_cli_risk_score_threshold(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            "risk_score=0.72",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    signal_sources = payload["sources"]["current_thresholds"]
    assert signal_sources["files"] == []
    assert "risk_score" not in signal_sources["inline"]
    assert signal_sources["risk_score"] == {
        "kind": "inline",
        "source": "risk_score=0.72",
        "value": pytest.approx(0.72),
    }
    risk_sources = payload["sources"]["risk_thresholds"]
    assert risk_sources["files"] == []
    assert risk_sources["inline"]["risk_score"] == pytest.approx(0.72)
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["current_threshold"] == pytest.approx(0.72)


def test_cli_risk_score_source_tracks_specific_pair(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            "signal_after_adjustment=0.82,risk_score=0.73",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    signal_sources = payload["sources"]["current_thresholds"]
    assert signal_sources["inline"]["signal_after_adjustment"] == pytest.approx(0.82)
    assert signal_sources["risk_score"] == {
        "kind": "inline",
        "source": "risk_score=0.73",
        "value": pytest.approx(0.73),
    }
    risk_sources = payload["sources"]["risk_thresholds"]
    assert risk_sources["inline"]["risk_score"] == pytest.approx(0.73)


def test_script_accepts_thresholds_from_json_file(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_payload = {
        "overrides": {
            "signal_after_adjustment": "0.81",
        },
        "risk_score": "0.71",
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--current-threshold",
            str(thresholds_path),
            "--current-threshold",
            "signal_after_clamp=0.77",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    sources_payload = payload["sources"]["current_thresholds"]
    assert sources_payload["files"] == [str(thresholds_path)]
    assert sources_payload["inline"]["signal_after_clamp"] == pytest.approx(0.77)
    assert "risk_score" not in sources_payload["inline"]
    assert sources_payload["risk_score"] == {
        "kind": "file",
        "source": str(thresholds_path),
        "value": pytest.approx(0.71),
    }
    risk_sources = payload["sources"]["risk_thresholds"]
    assert risk_sources["files"] == [str(thresholds_path)]
    assert risk_sources["inline"] == {}
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    metrics = trend_group["metrics"]
    assert metrics["signal_after_adjustment"]["current_threshold"] == 0.81
    assert metrics["signal_after_clamp"]["current_threshold"] == 0.77


def test_cli_risk_score_overrides_file_threshold(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps({"risk_score": 0.71}), encoding="utf-8")

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--current-threshold",
            "risk_score=0.72",
            "--current-threshold",
            str(thresholds_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    sources_payload = payload["sources"]["current_thresholds"]
    assert sources_payload["files"] == [str(thresholds_path)]
    assert "risk_score" not in sources_payload["inline"]
    assert sources_payload["risk_score"] == {
        "kind": "inline",
        "source": "risk_score=0.72",
        "value": pytest.approx(0.72),
    }
    risk_sources = payload["sources"]["risk_thresholds"]
    assert risk_sources["files"] == [str(thresholds_path)]
    assert risk_sources["inline"]["risk_score"] == pytest.approx(0.72)
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["current_threshold"] == pytest.approx(0.72)


def test_current_threshold_cli_rejects_non_finite_values() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["signal_after_adjustment=NaN"])

    message = str(excinfo.value)
    assert "signal_after_adjustment" in message
    assert "CLI 'signal_after_adjustment=NaN'" in message


def test_current_threshold_cli_rejects_non_finite_risk_score() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["risk_score=NaN"])

    message = str(excinfo.value)
    assert "risk_score" in message
    assert "CLI 'risk_score=NaN'" in message


def test_current_threshold_file_rejects_non_finite_values(tmp_path: Path) -> None:
    payload = {
        "signal_after_adjustment": {
            "current_threshold": "Infinity",
        }
    }
    source = tmp_path / "thresholds.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(source)])

    message = str(excinfo.value)
    assert "signal_after_adjustment" in message
    assert str(source) in message


def test_current_threshold_file_rejects_non_finite_risk_score(tmp_path: Path) -> None:
    payload = {
        "risk_score": {
            "current_threshold": "NaN",
        }
    }
    source = tmp_path / "thresholds.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(source)])

    message = str(excinfo.value)
    assert "risk_score" in message
    assert str(source) in message


def test_script_normalizes_cli_threshold_keys(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            "Signal_After_Adjustment=0.8",
            "--current-threshold",
            "signal-after-clamp=0.7",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    signal_stats = trend_group["metrics"]["signal_after_adjustment"]
    assert signal_stats["current_threshold"] == pytest.approx(0.8)
    clamp_stats = trend_group["metrics"]["signal_after_clamp"]
    assert clamp_stats["current_threshold"] == pytest.approx(0.7)
    current_sources = payload["sources"]["current_signal_thresholds"]
    assert current_sources["inline"]["signal_after_adjustment"] == pytest.approx(0.8)
    assert current_sources["inline"]["signal_after_clamp"] == pytest.approx(0.7)


def test_script_rejects_non_finite_current_threshold(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--current-threshold",
            "signal_after_adjustment=NaN",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    stderr = result.stderr
    assert "musi być skończoną liczbą" in stderr
    assert "signal_after_adjustment" in stderr


def test_load_current_thresholds_rejects_directory(tmp_path: Path) -> None:
    thresholds_dir = tmp_path / "thresholds"
    thresholds_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(thresholds_dir)])

    assert "musi wskazywać plik" in str(excinfo.value)


def test_load_current_thresholds_errors_on_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_thresholds.json"

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(missing_path)])

    assert "Ścieżka z progami nie istnieje" in str(excinfo.value)


def test_load_current_thresholds_rejects_nan_inline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["signal_after_adjustment=NaN"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "CLI" in message


def test_load_current_thresholds_rejects_infinite_inline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["signal_after_adjustment=Infinity"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "Infinity" in message
    assert "CLI" in message


def test_load_current_thresholds_rejects_negative_infinite_inline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["signal_after_adjustment=-Infinity"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "-inf" in message.lower()


def test_load_current_thresholds_rejects_nan_inline_risk() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["risk_score=NaN"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message


def test_load_current_thresholds_rejects_infinite_inline_risk() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["risk_score=Infinity"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message


def test_load_current_thresholds_rejects_mixed_inline_with_nan() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([
            "signal_after_adjustment=0.7,signal_after_clamp=NaN",
        ])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "signal_after_clamp" in message


def test_load_current_thresholds_rejects_negative_infinite_inline_risk() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["risk_score=-Infinity"])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "-inf" in message.lower()


def test_parse_threshold_mapping_rejects_non_finite_value() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_threshold_mapping("signal_after_adjustment=NaN")

    message = str(excinfo.value)
    assert "signal_after_adjustment" in message
    assert "musi być skończoną liczbą" in message


def test_load_current_thresholds_rejects_nan_from_file(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"signal_after_adjustment": "NaN"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_infinite_from_file(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"signal_after_clamp": "Infinity"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "inf" in message.lower()
    assert str(path) in message


def test_load_current_thresholds_rejects_negative_infinite_from_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"signal_after_clamp": "-Infinity"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "-inf" in message.lower()
    assert str(path) in message


def test_load_current_thresholds_rejects_nan_risk_from_file(tmp_path: Path) -> None:
    path = tmp_path / "risk_thresholds.json"
    path.write_text(json.dumps({"risk_score": "NaN"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_infinite_risk_from_file(tmp_path: Path) -> None:
    path = tmp_path / "risk_thresholds.json"
    path.write_text(json.dumps({"risk_score": "Infinity"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_nested_non_finite_from_file(tmp_path: Path) -> None:
    path = tmp_path / "thresholds_nested.json"
    payload = [
        {
            "metric": "signal_after_adjustment",
            "value": float("inf"),
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "signal_after_adjustment" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_nested_non_finite_risk_from_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "risk_thresholds_nested.json"
    payload = [
        {
            "metric": "risk_score",
            "value": "NaN",
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_negative_infinite_risk_from_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "risk_thresholds.json"
    path.write_text(json.dumps({"risk_score": "-Infinity"}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "-inf" in message.lower()
    assert str(path) in message


def test_load_current_thresholds_rejects_non_finite_inline_metadata_from_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "threshold_sources.json"
    payload = {"inline": {"signal_after_adjustment": "NaN"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "signal_after_adjustment" in message
    assert str(path) in message


def test_load_current_thresholds_rejects_non_finite_inline_risk_metadata_from_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "risk_threshold_sources.json"
    payload = {"risk_inline": {"risk_score": "Infinity"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(path)])

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "inf" in message.lower()
    assert str(path) in message


def test_load_current_thresholds_supports_nested_structures(tmp_path: Path) -> None:
    payload = {
        "signals": {
            "signal_after_adjustment": {"current_threshold": "0.84"},
            "signal_after_clamp": {"threshold": 0.76},
        },
        "overrides": [
            {"metric": "signal_after_adjustment", "value": 0.85},
            {"name": "signal_after_clamp", "current": "0.74"},
        ],
        "legacy_signal_after_adjustment_threshold": 0.83,
    }

    thresholds_path = tmp_path / "nested_thresholds.json"
    thresholds_path.write_text(json.dumps(payload), encoding="utf-8")

    thresholds, risk_score, sources_payload = _load_current_signal_thresholds(
        [str(thresholds_path)]
    )

    assert thresholds["signal_after_adjustment"] == pytest.approx(0.85)
    assert thresholds["signal_after_clamp"] == pytest.approx(0.74)
    assert risk_score is None
    assert sources_payload["files"] == [str(thresholds_path)]
    assert sources_payload["inline"] == {}
    assert sources_payload["risk_files"] == []
    assert sources_payload["risk_inline"] == {}
    assert sources_payload["risk_score_source"] is None


def test_load_current_thresholds_collects_risk_files(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "risk_score": 0.66,
                "signal_after_clamp": 0.73,
            }
        ),
        encoding="utf-8",
    )

    thresholds, risk_score, sources_payload = _load_current_signal_thresholds([str(thresholds_path)])

    assert thresholds["signal_after_clamp"] == pytest.approx(0.73)
    assert risk_score == pytest.approx(0.66)
    assert sources_payload["files"] == [str(thresholds_path)]
    assert sources_payload["inline"] == {}
    assert sources_payload["risk_files"] == [str(thresholds_path)]
    assert sources_payload["risk_inline"] == {}
    assert sources_payload["risk_score_source"] == {
        "kind": "file",
        "source": str(thresholds_path),
        "value": 0.66,
    }


def test_load_current_thresholds_prefers_inline_risk_score(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps({"risk_score": 0.61}), encoding="utf-8")

    thresholds, risk_score, sources_payload = _load_current_signal_thresholds(
        ["risk_score=0.72", str(thresholds_path)]
    )

    assert thresholds == {}
    assert risk_score == pytest.approx(0.72)
    assert sources_payload["files"] == [str(thresholds_path)]
    assert sources_payload["risk_files"] == [str(thresholds_path)]
    assert sources_payload["risk_inline"] == {"risk_score": pytest.approx(0.72)}
    assert sources_payload["risk_score_source"] == {
        "kind": "inline",
        "source": "risk_score=0.72",
        "value": pytest.approx(0.72),
    }


def test_parse_percentiles_rejects_nan() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_percentiles("0.5,NaN")

    message = str(excinfo.value)
    assert "Percentyl" in message
    assert "skończoną liczbą" in message
    assert "NaN" in message


def test_parse_percentiles_rejects_infinity() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_percentiles("Infinity")

    message = str(excinfo.value)
    assert "Percentyl" in message
    assert "skończoną liczbą" in message
    assert "Infinity" in message


def test_group_resolution_prefers_entry_metadata_over_symbol_map() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.5,
            "signal_after_clamp": 0.4,
        }
    ]

    entry_kraken = {
        "decision": {
            "details": {
                "symbol": "BTCUSDT",
                "primary_exchange": "kraken",
                "strategy": "mean_reversion",
                "summary": {"risk_score": 0.61},
            }
        }
    }
    entry_binance = {
        "decision": {
            "details": {
                "symbol": "BTCUSDT",
                "summary": {"risk_score": 0.73},
            },
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    }

    for entries in ([entry_kraken, entry_binance], [entry_binance, entry_kraken]):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=list(entries),
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

        groups = {
            (group["primary_exchange"], group["strategy"]): group
            for group in report["groups"]
        }

        kraken_group = groups[("kraken", "mean_reversion")]
        binance_group = groups[("binance", "trend_following")]

        assert (
            kraken_group["metrics"]["risk_score"]["count"] == 1
        ), "Entry-specific metadata should override symbol_map fallback"
        assert (
            binance_group["metrics"]["risk_score"]["count"] == 1
        ), "Entries should remain grouped by their own metadata"


def test_group_resolution_merges_entry_and_summary_metadata_before_symbol_map() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "kraken",
                "summary": {
                    "risk_score": 0.45,
                    "strategy": "mean_reversion",
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "kraken"
    assert group["strategy"] == "mean_reversion"


def test_report_groups_routing_identifiers_case_insensitively() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "Binance",
            "strategy": "TREND_FOLLOWING",
            "signal_after_adjustment": 0.61,
            "signal_after_clamp": 0.6,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.59,
            "signal_after_clamp": 0.58,
        },
    ]

    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "BINANCE",
                    "strategy": "Trend_Following",
                    "summary": {"risk_score": 0.67},
                }
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["primary_exchange"] == "Binance"
    assert group["strategy"] == "TREND_FOLLOWING"
    assert group["primary_exchange"].casefold() == "binance"
    assert group["strategy"].casefold() == "trend_following"
    assert group["metrics"]["signal_after_adjustment"]["count"] == 2
    assert group["metrics"]["risk_score"]["count"] == 1


def test_autotrade_entry_keeps_detail_routing_when_symbol_missing() -> None:
    journal_events = [
        {
            "primary_exchange": "kraken",
            "strategy": "mean_reversion",
        }
    ]
    autotrade_entries = [
        {
            "detail": {
                "symbol": "LTCUSDT",
                "primary_exchange": "kraken",
                "strategy": "mean_reversion",
                "summary": {
                    "risk_score": 0.51,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"])
        for group in report["groups"]
    }
    assert ("kraken", "mean_reversion") in groups
    assert ("unknown", "unknown") not in groups


def test_autotrade_entry_reads_decision_level_routing_metadata() -> None:
    journal_events = []
    autotrade_entries = [
        {
            "decision": {
                "primary_exchange": "coinbase",
                "strategy": "momentum",
                "details": {
                    "symbol": "ADAUSD",
                    "summary": {
                        "risk_score": 0.23,
                    },
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "coinbase"
    assert group["strategy"] == "momentum"


def test_autotrade_entry_reads_nested_routing_metadata() -> None:
    journal_events = [
        {
            "symbol": "ADAUSD",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "routing": {
                    "primary_exchange": "ftx",
                    "strategy": "grid_bot",
                },
                "symbol": "ADAUSD",
                "summary": {
                    "risk_score": 0.67,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("ftx", "grid_bot") in groups
    nested_group = groups[("ftx", "grid_bot")]
    assert nested_group["metrics"]["risk_score"]["count"] == 1


def test_generate_report_uses_custom_risk_threshold_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.61},
                }
            }
        }
    ]

    config_path = tmp_path / "risk_thresholds.yaml"
    config_path.write_text("auto_trader: {}\n", encoding="utf-8")

    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None):
        calls.append(config_path)
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": 0.42}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(config_path)],
    )

    assert calls == [config_path]
    sources = report["sources"]["risk_thresholds"]
    assert sources["files"] == [str(config_path)]
    assert sources["inline"] == {}
    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.42)


def test_generate_report_rejects_non_finite_current_inline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            current_threshold_sources={
                "inline": {"signal_after_adjustment": "NaN"},
            },
        )

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "signal_after_adjustment" in message
    assert "current_thresholds.inline" in message


def test_generate_report_rejects_invalid_current_threshold_mapping_value() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            current_signal_thresholds={"signal_after_adjustment": "abc"},
        )

    message = str(excinfo.value)
    assert "Nie udało się zinterpretować" in message
    assert "signal_after_adjustment" in message


def test_generate_report_rejects_non_finite_risk_inline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            current_threshold_sources={
                "risk_inline": {"risk_score": "Infinity"},
            },
        )

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "risk_thresholds.inline" in message


def test_generate_report_rejects_non_finite_current_threshold_mapping() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            current_signal_thresholds={
                "signal_after_adjustment": math.nan,
                "signal_after_clamp": "-inf",
            },
        )

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "signal_after_adjustment" in message or "signal_after_clamp" in message


def test_generate_report_rejects_non_finite_risk_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "risk_thresholds.json"
    config_file.write_text("{}", encoding="utf-8")

    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None) -> Mapping[str, object]:
        calls.append(config_path)
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": math.nan}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            risk_threshold_sources=[str(config_file)],
        )

    assert calls == [config_file]
    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert str(config_file) in message


def test_generate_report_rejects_non_finite_risk_from_default_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None) -> Mapping[str, object]:
        calls.append(config_path)
        assert config_path is None
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": math.inf}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert calls == [None]
    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "load_risk_thresholds()" in message


def test_generate_report_rejects_non_finite_cli_risk_threshold() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            risk_score_override=math.nan,
            risk_score_source={"kind": "inline", "source": "CLI risk_score_threshold"},
        )

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "CLI risk_score_threshold" in message


def test_generate_report_rejects_non_finite_cli_risk_score() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _generate_report(
            journal_events=[],
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            risk_score_override=math.inf,
            risk_score_source={"kind": "inline", "source": "CLI risk_score"},
        )

    message = str(excinfo.value)
    assert "musi być skończoną liczbą" in message
    assert "risk_score" in message
    assert "CLI risk_score" in message


def test_generate_report_uses_normalized_current_threshold_mapping() -> None:
    journal_events = [
        {
            "event_type": "ai_inference",
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend",
            "signal_after_adjustment": 0.6,
        }
    ]

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=[],
        percentiles=[0.5],
        suggestion_percentile=0.5,
        current_signal_thresholds={
            "signal_after_adjustment": "0.51",
            "SIGNAL-AFTER-CLAMP": 0.49,
            "unknown_metric": 1.23,
        },
    )

    groups = report["groups"]
    assert len(groups) == 1
    metrics = groups[0]["metrics"]
    assert metrics["signal_after_adjustment"]["current_threshold"] == pytest.approx(0.51)
    if "signal_after_clamp" in metrics:
        assert metrics["signal_after_clamp"]["current_threshold"] is None
def test_generate_report_merges_multiple_risk_threshold_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.7},
                }
            }
        }
    ]

    first = tmp_path / "risk_a.yaml"
    second = tmp_path / "risk_b.yaml"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    responses = {first: 0.33, second: 0.88}
    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None):
        calls.append(config_path)
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": responses[config_path]}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(first), str(second)],
    )

    assert calls == [first, second]
    sources = report["sources"]["risk_thresholds"]
    assert sources["files"] == [str(first), str(second)]
    assert sources["inline"] == {}
    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.88)


def test_cli_risk_score_override_beats_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.6},
                }
            }
        }
    ]

    override_path = tmp_path / "risk_thresholds.yaml"
    override_path.write_text("auto_trader: {}\n", encoding="utf-8")

    def _fake_loader(*, config_path: Path | None = None):
        assert config_path == override_path
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": 0.9}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(override_path)],
        cli_risk_score=0.42,
    )

    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.42)
    assert report["sources"]["risk_score_override"] == pytest.approx(0.42)


def test_autotrade_entry_reads_routing_sequences() -> None:
    journal_events = [
        {
            "symbol": "ADAUSD",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "ADAUSD",
                "routing": {
                    "legs": [
                        {
                            "route": {
                                "primary_exchange": "kraken",
                                "strategy": "mean_reversion",
                            }
                        }
                    ]
                },
                "summary": {
                    "risk_score": 0.51,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("kraken", "mean_reversion") in groups
    nested_group = groups[("kraken", "mean_reversion")]
    assert nested_group["metrics"]["risk_score"]["count"] == 1


def test_journal_freeze_event_preserves_event_metadata_over_symbol_map() -> None:
    journal_events = [
        {
            "event_type": "ai_inference",
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.73,
        },
        {
            "event_type": "risk_freeze",
            "symbol": "BTCUSDT",
            "primary_exchange": "kraken",
            "strategy": "mean_reversion",
            "status": "risk_freeze",
            "freeze_duration": 300,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "trend_following") in groups
    assert ("kraken", "mean_reversion") in groups

    assert groups[("kraken", "mean_reversion")]["freeze_summary"]["total"] == 1
    assert groups[("binance", "trend_following")]["freeze_summary"]["total"] == 0


def test_autotrade_entry_falls_back_to_summary_metadata() -> None:
    journal_events = []
    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "primary_exchange": "binance",
                    "strategy": "breakout",
                    "risk_score": 0.37,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "binance"
    assert group["strategy"] == "breakout"


def test_build_symbol_map_updates_incomplete_entries() -> None:
    events = [
        {"symbol": "ADAUSDT", "primary_exchange": None, "strategy": None},
        {"symbol": "ADAUSDT", "primary_exchange": "binance", "strategy": "trend"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("ADAUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "trend")


def test_build_symbol_map_combines_partial_details() -> None:
    events = [
        {"symbol": "DOGEUSDT", "primary_exchange": "binance", "strategy": None},
        {"symbol": "DOGEUSDT", "primary_exchange": None, "strategy": "momentum"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("DOGEUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "momentum")


def test_build_symbol_map_marks_ambiguous_when_conflicting_routing() -> None:
    events = [
        {
            "symbol": "XRPUSDT",
            "primary_exchange": "binance",
            "strategy": "scalping",
        },
        {
            "symbol": "XRPUSDT",
            "primary_exchange": "kraken",
            "strategy": "scalping",
        },
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("XRPUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == _AMBIGUOUS_SYMBOL_MAPPING


def test_generate_report_accepts_generators() -> None:
    journal_events = (
        {
            "primary_exchange": "Binance",
            "strategy": "Trend",
            "symbol": "BTCUSDT",
            "signal_after_adjustment": 0.6,
            "signal_after_clamp": 0.58,
        },
        {
            "primary_exchange": "Binance",
            "strategy": "Trend",
            "symbol": "BTCUSDT",
            "signal_after_adjustment": 0.4,
            "signal_after_clamp": 0.38,
        },
    )
    autotrade_entries = (
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "detail": {
                "symbol": "btcusdt",
                "primary_exchange": "binance",
                "strategy": "trend",
                "summary": {"risk_score": 0.42},
            },
        },
    )

    report = _generate_report(
        journal_events=(dict(event) for event in journal_events),
        autotrade_entries=(dict(entry) for entry in autotrade_entries),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    sources = report["sources"]
    assert sources["journal_events"] == 2
    assert sources["autotrade_entries"] == 1
    assert report["groups"]
    trend_group = next(iter(report["groups"]))
    assert trend_group["metrics"]["signal_after_adjustment"]["count"] == 2
def test_build_symbol_map_coalesces_case_variants() -> None:
    events = [
        {"symbol": "BtcUsdt", "primary_exchange": "binance", "strategy": "trend"},
        {"symbol": "BTCUSDT", "primary_exchange": None, "strategy": "TREND"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("BTCUSDT")
    assert canonical_symbol is not None
    assert set(mapping) == {canonical_symbol}
    assert mapping[canonical_symbol] == ("binance", "trend")


def test_symbol_map_fallback_supplements_missing_metadata() -> None:
    journal_events = [
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "binance",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "risk_score": 0.42,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "unknown") in groups
    assert groups[("binance", "unknown")]["metrics"]["risk_score"]["count"] == 1


def test_generate_report_handles_large_inputs_with_low_peak_memory(monkeypatch, tmp_path: Path) -> None:
    event_count = 2000
    journal_path = tmp_path / "large_journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for index in range(event_count):
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"2024-01-01T00:{index % 60:02d}:00Z",
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "signal_after_adjustment": 0.5 + (index % 10) * 0.01,
                        "signal_after_clamp": 0.45 + (index % 10) * 0.01,
                    }
                )
            )
            handle.write("\n")

    autotrade_path = tmp_path / "large_autotrade.json"
    entries = [
        {
            "timestamp": f"2024-01-01T01:{index % 60:02d}:00Z",
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.6 + (index % 10) * 0.01},
                }
            },
        }
        for index in range(event_count)
    ]
    autotrade_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    journal_iter = _load_journal_events([journal_path])
    autotrade_iter = _load_autotrade_entries([str(autotrade_path)])

    assert isinstance(journal_iter, GeneratorType)
    assert isinstance(autotrade_iter, GeneratorType)

    append_calls = 0
    peak_sizes: dict[tuple[tuple[str, str], str], int] = {}

    def _observer(key: tuple[str, str], metric: str, size: int) -> None:
        nonlocal append_calls
        append_calls += 1
        current_peak = peak_sizes.get((key, metric), 0)
        if size > current_peak:
            peak_sizes[(key, metric)] = size

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds._METRIC_APPEND_OBSERVER", _observer
    )

    report = _generate_report(
        journal_events=journal_iter,
        autotrade_entries=autotrade_iter,
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert report["sources"]["journal_events"] == event_count
    assert report["sources"]["autotrade_entries"] == event_count

    assert all("raw_values" not in group for group in report["groups"])
    assert "raw_values" not in report["global_summary"]

    expected_metric_values = event_count * 2 + event_count
    assert append_calls == expected_metric_values

    assert peak_sizes[("binance", "trend_following"), "signal_after_adjustment"] == event_count
    assert peak_sizes[("binance", "trend_following"), "signal_after_clamp"] == event_count
    assert peak_sizes[("binance", "trend_following"), "risk_score"] == event_count


def test_loaders_stream_without_materializing_large_lists(monkeypatch, tmp_path: Path) -> None:
    event_count = 1500

    journal_path = tmp_path / "stream_journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for index in range(event_count):
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"2024-01-01T00:{index % 60:02d}:00Z",
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "signal_after_adjustment": 0.55,
                        "signal_after_clamp": 0.5,
                    }
                )
            )
            handle.write("\n")

    autotrade_path = tmp_path / "stream_autotrade.json"
    entries = [
        {
            "timestamp": f"2024-01-01T01:{index % 60:02d}:00Z",
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.6},
                }
            },
        }
        for index in range(event_count)
    ]
    autotrade_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    del entries

    from scripts import calibrate_autotrade_thresholds as module

    original_journal_loader = module._load_journal_events
    journal_loaded = 0

    def _counting_journal_loader(paths: Iterable[Path], **kwargs):
        nonlocal journal_loaded
        for payload in original_journal_loader(paths, **kwargs):
            journal_loaded += 1
            yield payload

    monkeypatch.setattr(module, "_load_journal_events", _counting_journal_loader)

    original_autotrade_loader = module._load_autotrade_entries
    autotrade_loaded = 0

    def _counting_autotrade_loader(paths: Iterable[str], **kwargs):
        nonlocal autotrade_loaded
        for payload in original_autotrade_loader(paths, **kwargs):
            autotrade_loaded += 1
            yield payload

    monkeypatch.setattr(module, "_load_autotrade_entries", _counting_autotrade_loader)

    journal_iter = module._load_journal_events([journal_path])
    autotrade_iter = module._load_autotrade_entries([str(autotrade_path)])

    report = module._generate_report(
        journal_events=journal_iter,
        autotrade_entries=autotrade_iter,
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert report["sources"]["journal_events"] == event_count
    assert report["sources"]["autotrade_entries"] == event_count

    assert journal_loaded == event_count
    assert autotrade_loaded == event_count

    import gc

    gc.collect()
    large_lists = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, list)
        and len(obj) >= event_count
        and all(isinstance(item, Mapping) for item in obj)
    ]

    assert not large_lists


def test_generate_report_streams_inputs_and_aggregates(monkeypatch) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    event_count = 120
    freeze_count = 15
    journal_processed = 0
    autotrade_processed = 0

    def _journal_stream() -> Iterable[Mapping[str, object]]:
        nonlocal journal_processed
        for index in range(event_count):
            journal_processed += 1
            yield {
                "timestamp": f"2024-01-01T00:{index % 60:02d}:00Z",
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "signal_after_adjustment": 0.5 + (index % 7) * 0.01,
                "signal_after_clamp": 0.45 + (index % 7) * 0.01,
            }

    def _autotrade_stream() -> Iterable[Mapping[str, object]]:
        nonlocal autotrade_processed
        for index in range(event_count):
            autotrade_processed += 1
            payload: dict[str, object] = {
                "timestamp": f"2024-01-01T01:{index % 60:02d}:00Z",
                "detail": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.6 + (index % 5) * 0.01},
                },
            }
            if index < freeze_count:
                detail_payload = dict(payload["detail"])
                detail_payload.setdefault("reason", "risk_score_threshold")
                detail_payload.setdefault("frozen_for", 30 + index)
                payload["detail"] = detail_payload
                payload["status"] = "auto_risk_freeze"
            yield payload

    last_risk_size = 0

    def _observer(key: tuple[str, str], metric: str, size: int) -> None:
        nonlocal last_risk_size
        if key == ("binance", "trend_following") and metric == "risk_score":
            last_risk_size = size

    monkeypatch.setattr(module, "load_risk_thresholds", lambda **_: {})
    monkeypatch.setattr(module, "_METRIC_APPEND_OBSERVER", _observer, raising=False)

    report = module._generate_report(
        journal_events=_journal_stream(),
        autotrade_entries=_autotrade_stream(),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert journal_processed == event_count
    assert autotrade_processed == event_count

    sources = report["sources"]
    assert sources["journal_events"] == event_count
    assert sources["autotrade_entries"] == event_count

    group = report["groups"][0]
    assert group["metrics"]["risk_score"]["count"] == event_count - freeze_count
    assert group["freeze_summary"]["total"] == freeze_count
    assert last_risk_size == event_count - freeze_count

    global_freeze_summary = report["global_summary"]["freeze_summary"]
    assert global_freeze_summary["total"] == freeze_count
    assert global_freeze_summary["auto"] == freeze_count

    gc.collect()
    large_lists = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, list)
        and len(obj) >= event_count
        and all(isinstance(item, Mapping) for item in obj)
    ]

    assert not large_lists


def test_loaders_yield_records_incrementally(monkeypatch, tmp_path: Path) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    journal_path = tmp_path / "incremental_journal.jsonl"
    autotrade_path = tmp_path / "incremental_autotrade.jsonl"
    journal_path.touch()
    autotrade_path.touch()

    journal_lines = [
        json.dumps({"timestamp": "2024-01-01T00:00:00Z", "value": 1}) + "\n",
        json.dumps({"timestamp": "2024-01-01T00:01:00Z", "value": 2}) + "\n",
        json.dumps({"timestamp": "2024-01-01T00:02:00Z", "value": 3}) + "\n",
    ]
    autotrade_lines = [
        json.dumps({"timestamp": "2024-01-01T01:00:00Z", "decision": {"summary": {"risk_score": 1}}})
        + "\n",
        json.dumps({"timestamp": "2024-01-01T01:01:00Z", "decision": {"summary": {"risk_score": 2}}})
        + "\n",
        json.dumps({"timestamp": "2024-01-01T01:02:00Z", "decision": {"summary": {"risk_score": 3}}})
        + "\n",
    ]

    opened: dict[Path, "_FakeFile"] = {}

    class _FakeFile:
        def __init__(self, path: Path, lines: list[str]):
            self._path = path
            self._lines = lines
            self._index = 0
            self.read_calls = 0
            self.closed = False

        def __iter__(self) -> "_FakeFile":
            return self

        def __next__(self) -> str:
            if self._index >= len(self._lines):
                raise StopIteration
            value = self._lines[self._index]
            self._index += 1
            self.read_calls += 1
            return value

        def __enter__(self) -> "_FakeFile":
            opened[self._path] = self
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.closed = True
            return False

    def _fake_open_text_file(path: Path) -> _FakeFile:
        if path == journal_path:
            return _FakeFile(path, journal_lines)
        if path == autotrade_path:
            return _FakeFile(path, autotrade_lines)
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr(module, "_open_text_file", _fake_open_text_file)

    journal_iter = module._load_journal_events([journal_path])
    autotrade_iter = module._load_autotrade_entries([autotrade_path])

    first_event = next(journal_iter)
    first_entry = next(autotrade_iter)

    assert first_event["value"] == 1
    assert first_entry["decision"]["summary"]["risk_score"] == 1

    assert opened[journal_path].read_calls == 1
    assert opened[autotrade_path].read_calls == 1

    journal_iter.close()
    autotrade_iter.close()

    assert opened[journal_path].closed
    assert opened[autotrade_path].closed


def test_generate_report_does_not_build_large_lists(monkeypatch) -> None:
    event_count = 2500

    from scripts import calibrate_autotrade_thresholds as module

    original_list_type = builtins.list

    class _TrackingListMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:  # pragma: no cover - trivial
            return isinstance(instance, original_list_type)

    class TrackingList(original_list_type, metaclass=_TrackingListMeta):
        created_lengths: list[int] = []

        def __new__(cls, iterable=()):  # type: ignore[override]
            instance = super().__new__(cls, iterable)
            cls.created_lengths.append(len(instance))
            return instance

    TrackingList.created_lengths = []

    monkeypatch.setattr(builtins, "list", TrackingList)
    monkeypatch.setattr(module, "load_risk_thresholds", lambda **_: {})

    def _journal_stream() -> Iterable[Mapping[str, object]]:
        for index in range(event_count):
            yield {
                "timestamp": f"2024-01-01T00:{index % 60:02d}:00Z",
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "signal_after_adjustment": 0.5 + (index % 7) * 0.01,
                "signal_after_clamp": 0.45 + (index % 7) * 0.01,
            }

    def _autotrade_stream() -> Iterable[Mapping[str, object]]:
        for index in range(event_count):
            yield {
                "timestamp": f"2024-01-01T01:{index % 60:02d}:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.6 + (index % 7) * 0.01},
                    }
                },
            }

    report = module._generate_report(
        journal_events=_journal_stream(),
        autotrade_entries=_autotrade_stream(),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert report["sources"]["journal_events"] == event_count
    assert report["sources"]["autotrade_entries"] == event_count

    max_created = max(TrackingList.created_lengths or [0])
    assert max_created < event_count


def test_large_volume_inputs_do_not_allocate_large_lists(monkeypatch, tmp_path: Path) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    event_count = 3200

    journal_path = tmp_path / "bulk_journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for index in range(event_count):
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"2024-02-01T00:{index % 60:02d}:00Z",
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "signal_after_adjustment": 0.5 + (index % 11) * 0.01,
                        "signal_after_clamp": 0.45 + (index % 7) * 0.01,
                    }
                )
            )
            handle.write("\n")

    autotrade_path = tmp_path / "bulk_autotrade.jsonl"
    with autotrade_path.open("w", encoding="utf-8") as handle:
        for index in range(event_count):
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"2024-02-01T01:{index % 60:02d}:00Z",
                        "decision": {
                            "details": {
                                "symbol": "BTCUSDT",
                                "primary_exchange": "binance",
                                "strategy": "trend_following",
                                "summary": {"risk_score": 0.6 + (index % 9) * 0.01},
                            }
                        },
                    }
                )
            )
            handle.write("\n")

    original_list_type = builtins.list

    class _TrackingListMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:  # pragma: no cover - trivial
            return isinstance(instance, original_list_type)

    class TrackingList(original_list_type, metaclass=_TrackingListMeta):
        created_lengths: list[int] = []

        def __new__(cls, iterable=()):  # type: ignore[override]
            instance = super().__new__(cls, iterable)
            cls.created_lengths.append(len(instance))
            return instance

    TrackingList.created_lengths = []

    monkeypatch.setattr(builtins, "list", TrackingList)
    monkeypatch.setattr(module, "load_risk_thresholds", lambda **_: {})

    report = module._generate_report(
        journal_events=module._load_journal_events([journal_path]),
        autotrade_entries=module._load_autotrade_entries([autotrade_path]),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    sources = report["sources"]
    assert sources["journal_events"] == event_count
    assert sources["autotrade_entries"] == event_count

    max_created = max(TrackingList.created_lengths or [0])
    assert max_created < event_count


def test_high_volume_monkeypatched_counters_do_not_materialize_lists(monkeypatch) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    event_count = 6000
    journal_processed = 0
    autotrade_processed = 0

    def _journal_stream() -> Iterable[Mapping[str, object]]:
        nonlocal journal_processed
        for index in range(event_count):
            journal_processed += 1
            yield {
                "timestamp": f"2024-03-01T00:{index % 60:02d}:00Z",
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "signal_after_adjustment": 0.6 + (index % 5) * 0.01,
                "signal_after_clamp": 0.55 + (index % 5) * 0.01,
            }

    def _autotrade_stream() -> Iterable[Mapping[str, object]]:
        nonlocal autotrade_processed
        for index in range(event_count):
            autotrade_processed += 1
            yield {
                "timestamp": f"2024-03-01T01:{index % 60:02d}:00Z",
                "detail": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.5 + (index % 5) * 0.01},
                },
            }

    original_list_type = builtins.list

    class _TrackingListMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:  # pragma: no cover - trivial
            return isinstance(instance, original_list_type)

    class TrackingList(original_list_type, metaclass=_TrackingListMeta):
        created_lengths: list[int] = []

        def __new__(cls, iterable=()):  # type: ignore[override]
            instance = super().__new__(cls, iterable)
            length = len(instance)
            cls.created_lengths.append(length)
            if length >= event_count:
                raise AssertionError("unexpected materialization of the entire stream")
            return instance

    TrackingList.created_lengths = []

    observed_sizes: dict[tuple[tuple[str, str], str], int] = {}

    def _observer(key: tuple[str, str], metric: str, size: int) -> None:
        observed_sizes[(key, metric)] = size

    monkeypatch.setattr(builtins, "list", TrackingList)
    monkeypatch.setattr(module, "load_risk_thresholds", lambda **_: {})
    monkeypatch.setattr(module, "_METRIC_APPEND_OBSERVER", _observer, raising=False)

    report = module._generate_report(
        journal_events=_journal_stream(),
        autotrade_entries=_autotrade_stream(),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert journal_processed == event_count
    assert autotrade_processed == event_count

    sources = report["sources"]
    assert sources["journal_events"] == event_count
    assert sources["autotrade_entries"] == event_count

    max_created = max(TrackingList.created_lengths or [0])
    assert max_created < event_count

    key = ("binance", "trend_following")
    assert observed_sizes[(key, "signal_after_adjustment")] == event_count
    assert observed_sizes[(key, "signal_after_clamp")] == event_count
    assert observed_sizes[(key, "risk_score")] == event_count


def test_load_autotrade_entries_supports_jsonl(tmp_path: Path) -> None:
    autotrade_path = tmp_path / "autotrade.jsonl"
    entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.55},
                }
            },
        },
        {
            "timestamp": "2024-01-01T00:15:00Z",
            "decision": {
                "details": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.42},
                }
            },
        },
    ]
    with autotrade_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    loaded = list(_load_autotrade_entries([str(autotrade_path)]))

    assert len(loaded) == len(entries)
    assert loaded[0]["decision"]["details"]["summary"]["risk_score"] == 0.55
    assert loaded[1]["decision"]["details"]["summary"]["risk_score"] == 0.42


def test_load_autotrade_entries_supports_gzipped_jsonl(tmp_path: Path) -> None:
    autotrade_path = tmp_path / "autotrade.jsonl.gz"
    entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.73},
                }
            },
        },
        {
            "timestamp": "2024-01-01T00:15:00Z",
            "decision": {
                "details": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.39},
                }
            },
        },
    ]
    with gzip.open(autotrade_path, "wt", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    loaded = list(_load_autotrade_entries([str(autotrade_path)]))

    assert len(loaded) == len(entries)
    assert loaded[0]["decision"]["details"]["summary"]["risk_score"] == 0.73
    assert loaded[1]["decision"]["details"]["summary"]["risk_score"] == 0.39


def test_load_autotrade_entries_supports_gzipped_json(tmp_path: Path) -> None:
    autotrade_path = tmp_path / "autotrade.json.gz"
    payload = {
        "entries": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.73},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T00:15:00Z",
                "decision": {
                    "details": {
                        "symbol": "ETHUSDT",
                        "primary_exchange": "kraken",
                        "strategy": "mean_reversion",
                        "summary": {"risk_score": 0.39},
                    }
                },
            },
        ]
    }
    with gzip.open(autotrade_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    loaded = list(_load_autotrade_entries([str(autotrade_path)]))

    assert len(loaded) == 2
    assert loaded[0]["decision"]["details"]["summary"]["risk_score"] == 0.73
    assert loaded[1]["decision"]["details"]["summary"]["risk_score"] == 0.39


@pytest.mark.parametrize("root_format", ["object", "array"])
def test_load_autotrade_entries_json_with_bom(tmp_path: Path, root_format: str) -> None:
    autotrade_path = tmp_path / "autotrade.json"
    entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.51},
                }
            },
        },
        {
            "timestamp": "2024-01-01T00:30:00Z",
            "decision": {
                "details": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.44},
                }
            },
        },
    ]
    if root_format == "object":
        payload: object = {"entries": entries, "version": 2}
    else:
        payload = entries

    with autotrade_path.open("w", encoding="utf-8") as handle:
        handle.write("\ufeff")
        json.dump(payload, handle)

    loaded = list(_load_autotrade_entries([str(autotrade_path)]))

    assert loaded == entries


@pytest.mark.parametrize("root_format", ["object", "array"])
def test_load_autotrade_entries_streams_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_format: str
) -> None:
    export_path = tmp_path / "autotrade.json"
    entries = [
        {"timestamp": "2024-02-01T00:00:00Z", "value": 1, "padding": "x" * 64},
        {"timestamp": "2024-02-01T00:10:00Z", "value": 2, "padding": "x" * 64},
        {"timestamp": "2024-02-01T00:20:00Z", "value": 3, "padding": "x" * 64},
    ]
    if root_format == "array":
        export_path.write_text(json.dumps(entries), encoding="utf-8")
    else:
        export_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 16)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    since = datetime(2024, 2, 1, 0, 5, tzinfo=timezone.utc)
    until = datetime(2024, 2, 1, 0, 15, tzinfo=timezone.utc)

    loaded = list(module._load_autotrade_entries([export_path], since=since, until=until))

    assert [entry["timestamp"] for entry in loaded] == ["2024-02-01T00:10:00Z"]

    handle = holder["handle"]
    assert len(handle.read_requests) > 1
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert -1 not in handle.read_requests
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_streams_nested_entries_without_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json"
    entries = [
        {
            "timestamp": "2024-03-01T00:00:00Z",
            "value": 1,
            "padding": "x" * 64,
        },
        {
            "timestamp": "2024-03-01T00:10:00Z",
            "value": 2,
            "padding": "y" * 64,
        },
        {
            "timestamp": "2024-03-01T00:20:00Z",
            "value": 3,
            "padding": "z" * 64,
        },
    ]
    payload = {
        "metadata": {"info": "m" * 128},
        "container": {
            "details": {"notes": ["n" * 32, "o" * 32]},
            "wrapper": {"entries": entries},
        },
        "footer": "p" * 64,
    }
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 32)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == entries

    handle = holder["handle"]
    assert len(handle.read_requests) > 1
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_streams_entries_nested_in_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json"
    group_one = [
        {"timestamp": "2024-04-01T00:00:00Z", "value": 1, "note": "alpha"},
        {"timestamp": "2024-04-01T00:10:00Z", "value": 2, "note": "beta"},
    ]
    group_two = [
        {"timestamp": "2024-04-01T00:20:00Z", "value": 3, "note": "gamma"},
        {"timestamp": "2024-04-01T00:30:00Z", "value": 4, "note": "delta"},
    ]
    group_three = [
        {"timestamp": "2024-04-01T00:40:00Z", "value": 5, "note": "epsilon"},
        {"timestamp": "2024-04-01T00:50:00Z", "value": 6, "note": "zeta"},
    ]
    payload = {
        "version": 3,
        "groups": [
            {"metadata": {"id": "g1", "padding": "x" * 64}, "entries": group_one},
            {
                "wrapper": {
                    "details": {"padding": "y" * 64},
                    "entries": group_two,
                },
                "metadata": {"id": "g2"},
            },
            {
                "nodes": [
                    {"info": "n" * 32},
                    {"entries": group_three, "metadata": {"id": "g3"}},
                ]
            },
        ],
        "footer": {"summary": "done", "extra": "z" * 64},
    }
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 28)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == group_one + group_two + group_three

    handle = holder["handle"]
    assert len(handle.read_requests) > 1
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_streams_entries_nested_in_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json"
    batch_one = [
        {"timestamp": "2024-04-02T00:00:00Z", "value": 1, "note": "alpha"},
        {"timestamp": "2024-04-02T00:05:00Z", "value": 2, "note": "beta"},
    ]
    batch_two = [
        {"timestamp": "2024-04-02T00:10:00Z", "value": 3, "note": "gamma"},
        {"timestamp": "2024-04-02T00:15:00Z", "value": 4, "note": "delta"},
    ]
    batch_three = [
        {"timestamp": "2024-04-02T00:20:00Z", "value": 5, "note": "epsilon"},
    ]
    payload = {
        "header": {"generated_at": "2024-04-02T00:00:00Z", "padding": "x" * 64},
        "entries": {
            "primary": [
                {
                    "batch": {
                        "entries": batch_one,
                        "metadata": {"count": len(batch_one)},
                    },
                    "notes": ["ignored", "values"],
                },
                {
                    "summary": {"info": "wrapped"},
                    "container": {
                        "items": [
                            {"entries": [{"metadata": "skip"}]},
                            {"entries": batch_two},
                        ]
                    },
                },
            ],
            "secondary": {
                "wrapper": {
                    "payload": {"entries": batch_three},
                    "metadata": {"count": len(batch_three)},
                }
            },
        },
        "footer": "done",
    }
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 26)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == batch_one + batch_two + batch_three

    handle = holder["handle"]
    assert len(handle.read_requests) > 1
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_streams_entries_with_complex_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json"
    complex_text = (
        "Line1 with brackets [] and braces {}"
        "\nLine2 with quotes \"' and commas, plus unicode: ąćęłńóśźż"
        " -- repeated -- " * 8
    )
    entries = [
        {
            "timestamp": "2024-05-01T00:00:00Z",
            "value": 1,
            "note": complex_text + " end",
        },
        {
            "timestamp": "2024-05-01T00:10:00Z",
            "value": 2,
            "note": complex_text[::-1],
        },
    ]
    export_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 19)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == entries

    handle = holder["handle"]
    assert len(handle.read_requests) > 1
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_jsonl_with_bom(tmp_path: Path) -> None:
    export_path = tmp_path / "autotrade.jsonl"
    payloads = [
        {"timestamp": "2024-01-01T00:00:00Z", "value": 1},
        {"timestamp": "2024-01-01T01:00:00Z", "value": 2},
    ]
    with export_path.open("w", encoding="utf-8") as handle:
        handle.write("\ufeff")
        for item in payloads:
            handle.write(json.dumps(item))
            handle.write("\n")

    loaded = list(_load_autotrade_entries([export_path]))

    assert loaded == payloads


@pytest.mark.parametrize(
    ("extension", "with_bom"),
    [
        (".jsonl", False),
        (".jsonl", True),
        (".jsonl.gz", False),
        (".jsonl.gz", True),
    ],
)
def test_load_autotrade_entries_jsonl_exports_are_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str, with_bom: bool
) -> None:
    export_path = tmp_path / f"autotrade{extension}"
    entries = [
        {"timestamp": "2024-02-03T00:00:00Z", "value": 1, "padding": "x" * 64},
        {"timestamp": "2024-02-03T00:10:00Z", "value": 2, "padding": "y" * 64},
        {"timestamp": "2024-02-03T00:20:00Z", "value": 3, "padding": "z" * 64},
    ]

    if extension.endswith(".gz"):
        writer = lambda path: gzip.open(path, "wt", encoding="utf-8")
        reader = lambda path: gzip.open(path, "rt", encoding="utf-8")
    else:
        writer = lambda path: path.open("w", encoding="utf-8")
        reader = lambda path: path.open("r", encoding="utf-8")

    with writer(export_path) as handle:
        if with_bom:
            handle.write("\ufeff")
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    with reader(export_path) as handle:
        expected_length = sum(len(line) for line in handle)

    module = calibrate_autotrade_thresholds

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path, opener=reader)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == entries

    handle = holder["handle"]
    assert not handle.read_requests
    assert not handle.read_results
    assert handle.readline_requests
    assert len(handle.readline_results) == len(entries)
    assert sum(handle.readline_results) == expected_length
    assert max(handle.readline_results) < expected_length


def test_load_autotrade_entries_json_with_bom_is_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json"
    padding = "x" * 64
    entries = [
        {"timestamp": "2024-02-01T00:00:00Z", "value": 1, "padding": padding},
        {"timestamp": "2024-02-01T00:10:00Z", "value": 2, "padding": padding},
        {"timestamp": "2024-02-01T00:20:00Z", "value": 3, "padding": padding},
    ]
    with export_path.open("w", encoding="utf-8") as handle:
        handle.write("\ufeff")
        json.dump({"entries": entries}, handle)

    expected_length = len(export_path.read_text(encoding="utf-8"))

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 16)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(path)
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == entries

    handle = holder["handle"]
    assert handle.read_requests
    assert len(handle.read_requests) > 1
    assert -1 not in handle.read_requests
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_autotrade_entries_gzipped_json_with_bom_is_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "autotrade.json.gz"
    padding = "x" * 64
    entries = [
        {"timestamp": "2024-02-02T00:00:00Z", "value": 1, "padding": padding},
        {"timestamp": "2024-02-02T00:10:00Z", "value": 2, "padding": padding},
        {"timestamp": "2024-02-02T00:20:00Z", "value": 3, "padding": padding},
    ]
    with gzip.open(export_path, "wt", encoding="utf-8") as handle:
        handle.write("\ufeff")
        json.dump({"entries": entries}, handle)

    with gzip.open(export_path, "rt", encoding="utf-8") as handle:
        expected_length = len(handle.read())

    module = calibrate_autotrade_thresholds

    monkeypatch.setattr(module, "_JSON_STREAM_CHUNK_SIZE", 24)

    holder: dict[str, _TrackingReadHandle] = {}

    def fake_open_text_file(path: Path) -> _TrackingReadHandle:
        handle = _TrackingReadHandle(
            path,
            opener=lambda current: gzip.open(current, "rt", encoding="utf-8"),
        )
        holder["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_text_file", fake_open_text_file)

    loaded = list(module._load_autotrade_entries([export_path]))

    assert loaded == entries

    handle = holder["handle"]
    assert handle.read_requests
    assert len(handle.read_requests) > 1
    assert -1 not in handle.read_requests
    assert all(size == module._JSON_STREAM_CHUNK_SIZE for size in handle.read_requests)
    assert handle.read_results
    assert max(handle.read_results) <= module._JSON_STREAM_CHUNK_SIZE
    assert max(handle.read_results) < expected_length
    assert sum(handle.read_results) == expected_length


def test_load_journal_events_supports_gzipped_jsonl(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl.gz"
    events = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.5,
            "signal_after_clamp": 0.48,
        },
        {
            "timestamp": "2024-01-01T00:15:00Z",
            "primary_exchange": "kraken",
            "strategy": "mean_reversion",
            "signal_after_adjustment": 0.61,
            "signal_after_clamp": 0.6,
        },
    ]
    with gzip.open(journal_path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event))
            handle.write("\n")

    loaded = list(_load_journal_events([journal_path]))

    assert len(loaded) == len(events)
    assert loaded[0]["signal_after_adjustment"] == 0.5
    assert loaded[1]["signal_after_clamp"] == 0.6


def test_generate_report_releases_streamed_events(monkeypatch) -> None:
    event_count = 1600

    class TrackingMapping(dict):
        __slots__ = ("__weakref__",)

    journal_refs: list[ReferenceType[TrackingMapping]] = []
    autotrade_refs: list[ReferenceType[TrackingMapping]] = []

    def _journal_stream() -> Iterable[Mapping[str, object]]:
        for index in range(event_count):
            payload = TrackingMapping(
                {
                    "timestamp": f"2024-01-01T00:{index % 60:02d}:00Z",
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "signal_after_adjustment": 0.55 + 0.0001 * index,
                    "signal_after_clamp": 0.5 + 0.0001 * index,
                }
            )
            journal_refs.append(weakref.ref(payload))
            yield payload

    def _autotrade_stream() -> Iterable[Mapping[str, object]]:
        for index in range(event_count):
            payload = TrackingMapping(
                {
                    "timestamp": f"2024-01-01T01:{index % 60:02d}:00Z",
                    "decision": {
                        "details": {
                            "symbol": "BTCUSDT",
                            "primary_exchange": "binance",
                            "strategy": "trend_following",
                            "summary": {
                                "risk_score": 0.61 + 0.0002 * index,
                            },
                        }
                    },
                }
            )
            autotrade_refs.append(weakref.ref(payload))
            yield payload

    from scripts import calibrate_autotrade_thresholds as module

    append_calls = 0

    def _observer(key: tuple[str, str], metric: str, size: int) -> None:
        nonlocal append_calls
        append_calls += 1

    monkeypatch.setattr(module, "_METRIC_APPEND_OBSERVER", _observer)

    journal_iter = _journal_stream()
    autotrade_iter = _autotrade_stream()

    report = module._generate_report(
        journal_events=journal_iter,
        autotrade_entries=autotrade_iter,
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    assert report["sources"]["journal_events"] == event_count
    assert report["sources"]["autotrade_entries"] == event_count

    expected_metric_events = event_count * 3
    assert append_calls == expected_metric_events

    del report, journal_iter, autotrade_iter
    gc.collect()

    assert all(ref() is None for ref in journal_refs)
    assert all(ref() is None for ref in autotrade_refs)


def test_autotrade_loader_accepts_directory(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    first = export_dir / "first.json"
    first.write_text(
        json.dumps({"entries": [{"timestamp": "2024-01-01T00:00:00Z"}]}),
        encoding="utf-8",
    )
    second = export_dir / "second.JSON"
    second.write_text(
        json.dumps({"entries": [{"timestamp": "2024-01-01T01:00:00Z"}]}),
        encoding="utf-8",
    )

    entries = list(_load_autotrade_entries([str(export_dir)]))

    assert {entry["timestamp"] for entry in entries} == {
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
    }


def test_autotrade_loader_rejects_empty_directory(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    entries = _load_autotrade_entries([str(export_dir)])

    with pytest.raises(SystemExit) as excinfo:
        next(entries)

    message = str(excinfo.value)
    assert "nie zawiera plików" in message or "Nie znaleziono" in message


def test_autotrade_loader_handles_object_with_entries_key_only(tmp_path: Path) -> None:
    path = tmp_path / "autotrade.json"
    payload = {
        "entries": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.5},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T00:05:00Z",
                "detail": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.42},
                },
            },
        ]
    }
    path.write_text(json.dumps(payload) + "\n\n", encoding="utf-8")

    entries = list(_load_autotrade_entries([str(path)]))

    assert len(entries) == 2
    assert entries[0]["timestamp"] == "2024-01-01T00:00:00Z"
    assert entries[1]["timestamp"] == "2024-01-01T00:05:00Z"


def test_generate_report_can_collect_raw_values() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.51,
            "signal_after_clamp": 0.49,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.62,
            "signal_after_clamp": 0.6,
        },
    ]

    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.73},
                }
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
            include_raw_values=True,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert "raw_values" in group
    assert group["raw_values"]["signal_after_adjustment"] == [0.51, 0.62]
    assert group["raw_values"]["risk_score"] == [0.73]

    global_summary = report["global_summary"]
    assert "raw_values" in global_summary
    assert set(global_summary["raw_values"].keys()) >= {
        "signal_after_adjustment",
        "signal_after_clamp",
        "risk_score",
    }


def test_generate_report_omits_raw_freeze_events_by_default() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert "raw_freeze_events" not in group

    global_summary = report["global_summary"]
    assert "raw_freeze_events" not in global_summary

    sources = report["sources"]
    assert sources["raw_freeze_events"] == {"mode": "omit"}


def test_generate_report_samples_raw_freeze_events() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
            "duration": 180,
            "risk_score": 0.91,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 120,
            "risk_score": 0.87,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 75,
            "risk_score": 0.86,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            raw_freeze_events_mode="sample",
            limit_freeze_events=1,
        )

    assert report["groups"]
    group = report["groups"][0]
    freeze_summary = group["freeze_summary"]
    assert freeze_summary["total"] == 3

    raw_freeze_payload = group["raw_freeze_events"]
    assert raw_freeze_payload["limit"] == 1
    assert len(raw_freeze_payload["events"]) == 1
    sampled_event = raw_freeze_payload["events"][0]
    assert sampled_event["status"] == "risk_freeze"
    assert sampled_event["duration"] == pytest.approx(180)
    overflow_summary = raw_freeze_payload["overflow_summary"]
    assert overflow_summary["total"] == 2
    overflow_reasons = {item["reason"]: item["count"] for item in overflow_summary["reasons"]}
    assert overflow_reasons == {"risk_score_threshold": 2}

    global_raw = report["global_summary"]["raw_freeze_events"]
    assert global_raw["limit"] == 1
    assert len(global_raw["events"]) == 1
    assert global_raw["events"][0]["reason"] == "manual_override"
    assert global_raw["overflow_summary"]["total"] == 2

    sources = report["sources"]["raw_freeze_events"]
    assert sources["mode"] == "sample"
    assert sources["limit"] == 1
    source_overflow = sources["overflow_summary"]
    assert source_overflow["total"] == 2
    assert {item["reason"]: item["count"] for item in source_overflow["reasons"]} == {
        "risk_score_threshold": 2
    }


def test_generate_report_can_omit_raw_freeze_events_when_requested() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            raw_freeze_events_mode="sample",
            limit_freeze_events=5,
            omit_raw_freeze_events=True,
        )

    assert report["groups"]
    group = report["groups"][0]
    freeze_summary = group["freeze_summary"]
    assert freeze_summary["total"] == 2
    assert "raw_freeze_events" not in group

    global_summary = report["global_summary"]
    assert "raw_freeze_events" not in global_summary

    sources = report["sources"]["raw_freeze_events"]
    assert sources == {"mode": "omit"}


def test_generate_report_includes_full_freeze_events_without_limit() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
            "duration": 180,
            "risk_score": 0.91,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 120,
            "risk_score": 0.87,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 75,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    freeze_events = group["freeze_events"]
    assert freeze_events["mode"] == "all"
    assert len(freeze_events["events"]) == 3
    assert freeze_events["events"][0]["status"] == "risk_freeze"
    assert freeze_events["events"][0]["type"] == "manual"
    assert freeze_events["events"][0]["duration"] == pytest.approx(180)
    assert freeze_events["total"] == 3
    assert freeze_events["type_counts"] == {"auto": 2, "manual": 1}
    assert freeze_events["status_counts"]["auto_risk_freeze"] == 2
    assert freeze_events["reason_counts"]["risk_score_threshold"] == 2
    assert "overflow_summary" not in freeze_events

    global_freeze = report["global_summary"]["freeze_events"]
    assert global_freeze["mode"] == "all"
    assert len(global_freeze["events"]) == 3
    assert global_freeze["total"] == 3
    assert "overflow_summary" not in global_freeze

    sources = report["sources"]["freeze_events"]
    assert sources == {"mode": "all"}


def test_generate_report_limits_freeze_events_when_max_set() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
            "duration": 180,
            "risk_score": 0.91,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 120,
            "risk_score": 0.87,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 75,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            max_freeze_events=2,
        )

    assert report["groups"]
    group = report["groups"][0]
    freeze_events = group["freeze_events"]
    assert freeze_events["mode"] == "limit"
    assert freeze_events["limit"] == 2
    assert len(freeze_events["events"]) == 2
    assert freeze_events["total"] == 3
    assert freeze_events["events"][0]["status"] == "risk_freeze"
    assert freeze_events["events"][1]["status"] == "auto_risk_freeze"
    assert freeze_events["type_counts"] == {"auto": 2, "manual": 1}
    overflow = freeze_events["overflow_summary"]
    assert overflow["total"] == 1
    assert {item["status"]: item["count"] for item in overflow["statuses"]} == {
        "auto_risk_freeze": 1
    }
    assert {item["reason"]: item["count"] for item in overflow["reasons"]} == {
        "risk_score_threshold": 1
    }

    global_freeze = report["global_summary"]["freeze_events"]
    assert global_freeze["mode"] == "limit"
    assert global_freeze["limit"] == 2
    assert len(global_freeze["events"]) == 2
    assert global_freeze["total"] == 3
    global_overflow = global_freeze["overflow_summary"]
    assert global_overflow["total"] == 1
    assert {item["status"]: item["count"] for item in global_overflow["statuses"]} == {
        "auto_risk_freeze": 1
    }

    sources = report["sources"]["freeze_events"]
    assert sources == {
        "mode": "limit",
        "limit": 2,
        "overflow_summary": global_overflow,
    }


def test_generate_report_limits_freeze_events_when_max_zero() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "risk_freeze",
            "reason": "manual_override",
            "duration": 180,
            "risk_score": 0.91,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "status": "auto_risk_freeze",
            "reason": "risk_score_threshold",
            "duration": 120,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
            max_freeze_events=0,
        )

    assert report["groups"]
    group = report["groups"][0]
    freeze_events = group["freeze_events"]
    assert freeze_events["mode"] == "limit"
    assert freeze_events["limit"] == 0
    assert freeze_events["events"] == []
    assert freeze_events["total"] == 2
    assert freeze_events["type_counts"] == {"auto": 1, "manual": 1}
    overflow = freeze_events["overflow_summary"]
    assert overflow["total"] == 2
    assert {item["status"]: item["count"] for item in overflow["statuses"]} == {
        "auto_risk_freeze": 1,
        "risk_freeze": 1,
    }

    global_freeze = report["global_summary"]["freeze_events"]
    assert global_freeze["mode"] == "limit"
    assert global_freeze["limit"] == 0
    assert global_freeze["events"] == []
    assert global_freeze["total"] == 2
    global_overflow = global_freeze["overflow_summary"]
    assert global_overflow["total"] == 2
    assert {item["status"]: item["count"] for item in global_overflow["statuses"]} == {
        "auto_risk_freeze": 1,
        "risk_freeze": 1,
    }

    sources = report["sources"]["freeze_events"]
    assert sources == {
        "mode": "limit",
        "limit": 0,
        "overflow_summary": global_overflow,
    }


def test_main_respects_freeze_events_limit_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "_load_journal_events", lambda *_, **__: [])
    monkeypatch.setattr(module, "_load_autotrade_entries", lambda *_, **__: [])
    monkeypatch.setattr(
        module,
        "_load_current_signal_thresholds",
        lambda *_: ({}, None, {}),
    )

    captured: dict[str, object] = {}

    def _fake_generate_report(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "groups": [],
            "global_summary": {"metrics": {}, "freeze_summary": {}},
            "sources": {"journal_events": 0, "autotrade_entries": 0},
            "percentiles": [],
        }

    monkeypatch.setattr(module, "_generate_report", _fake_generate_report)

    exit_code = module.main(
        [
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--freeze-events-limit",
            "3",
        ]
    )

    assert exit_code == 0
    assert captured["max_freeze_events"] == 3
    assert captured["limit_freeze_events"] is None
    assert captured["raw_freeze_events_mode"] == "omit"
    assert captured["omit_raw_freeze_events"] is False

    output = capsys.readouterr().out
    assert "Przetworzono" in output


def test_resolve_freeze_event_limit_prefers_new_flag() -> None:
    result = _resolve_freeze_event_limit(
        limit_freeze_events=5,
        raw_freeze_events_mode="sample",
        raw_freeze_events_limit=99,
    )
    assert result == 5


def test_resolve_freeze_event_limit_legacy_sample_default() -> None:
    result = _resolve_freeze_event_limit(
        limit_freeze_events=None,
        raw_freeze_events_mode="sample",
        raw_freeze_events_limit=7,
    )
    assert result == 7


def test_resolve_freeze_event_limit_legacy_omit() -> None:
    result = _resolve_freeze_event_limit(
        limit_freeze_events=None,
        raw_freeze_events_mode="omit",
        raw_freeze_events_limit=15,
    )
    assert result is None


def test_main_can_omit_raw_freeze_events_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import calibrate_autotrade_thresholds as module

    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "_load_journal_events", lambda *_, **__: [])
    monkeypatch.setattr(module, "_load_autotrade_entries", lambda *_, **__: [])
    monkeypatch.setattr(
        module,
        "_load_current_signal_thresholds",
        lambda *_: ({}, None, {}),
    )

    captured: dict[str, object] = {}

    def _fake_generate_report(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "groups": [],
            "global_summary": {"metrics": {}, "freeze_summary": {}},
            "sources": {"journal_events": 0, "autotrade_entries": 0},
            "percentiles": [],
        }

    monkeypatch.setattr(module, "_generate_report", _fake_generate_report)

    exit_code = module.main(
        [
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--limit-freeze-events",
            "4",
            "--omit-raw-freeze-events",
        ]
    )

    assert exit_code == 0
    assert captured["omit_raw_freeze_events"] is True
    assert captured["raw_freeze_events_mode"] == "sample"
    assert captured["limit_freeze_events"] == 4


def test_metric_series_caches_sorted_sequences() -> None:
    series = _MetricSeries()
    series.append(2.0)
    series.append(-3.0)

    cached_absolute = series.absolute_values()
    assert list(cached_absolute) == [2.0, 3.0]
    assert series.absolute_values() is cached_absolute

    series.extend([4.0, -5.0])

    refreshed_absolute = series.absolute_values()
    assert refreshed_absolute is not cached_absolute
    assert list(refreshed_absolute) == [2.0, 3.0, 4.0, 5.0]
    assert series.absolute_values() is refreshed_absolute

    assert list(series.values()) == [-5.0, -3.0, 2.0, 4.0]
    stats = series.statistics([0.5])
    assert stats["count"] == 4
    assert stats["min"] == -5.0
    assert stats["max"] == 4.0
    assert stats["percentiles"]["p50"] == pytest.approx(-0.5)
    assert series.suggest(0.5, absolute=True) == pytest.approx(3.5)


def test_metric_series_absolute_values_cover_negatives_and_zeros() -> None:
    series = _MetricSeries()
    series.extend([-4.0, -1.0, 0.0, 1.5, 3.5])

    absolute = series.absolute_values()
    assert isinstance(absolute, array)
    assert list(absolute) == [0.0, 1.0, 1.5, 3.5, 4.0]

    # ponowne wywołanie korzysta z cache
    assert series.absolute_values() is absolute

    series.append(-0.25)
    refreshed = series.absolute_values()
    assert list(refreshed) == [0.0, 0.25, 1.0, 1.5, 3.5, 4.0]


def test_metric_series_absolute_values_keep_duplicate_magnitudes_sorted() -> None:
    series = _MetricSeries()
    series.extend([-3.0, -3.0, 2.0, 2.0, 0.0])

    absolute = series.absolute_values()
    assert list(absolute) == [0.0, 2.0, 2.0, 3.0, 3.0]

    series.append(-2.0)
    refreshed = series.absolute_values()
    assert list(refreshed) == [0.0, 2.0, 2.0, 2.0, 3.0, 3.0]


def test_metric_series_absolute_values_reuse_sorted_for_non_negative_values() -> None:
    series = _MetricSeries()
    series.extend([0.0, 1.0, 2.5, 3.25])

    absolute_first = series.absolute_values()
    values_after = series.values()
    assert absolute_first is values_after

    absolute_second = series.absolute_values()
    assert absolute_second is values_after

    series.append(-1.5)
    refreshed_absolute = series.absolute_values()
    assert refreshed_absolute is not values_after
    assert list(refreshed_absolute) == [0.0, 1.0, 1.5, 2.5, 3.25]


def test_symbol_map_matches_symbols_case_insensitively() -> None:
    journal_events = [
        {
            "symbol": "ethusdt",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    mapping = _build_symbol_map(journal_events)
    canonical_symbol = _canonicalize_symbol_key("ETHUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "trend_following")

    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "ETHUSDT",
                    "summary": {"risk_score": 0.42},
                }
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "trend_following") in groups
    matched_group = groups[("binance", "trend_following")]
    assert matched_group["metrics"]["risk_score"]["count"] == 1


def test_unknown_routing_display_is_canonicalized() -> None:
    journal_events = [
        {
            "symbol": "ADAUSDT",
            "primary_exchange": "Unknown",
            "strategy": "UNKNOWN",
            "signal_after_adjustment": 0.51,
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "unknown"
    assert group["strategy"] == "unknown"


def test_unknown_synonyms_are_normalized() -> None:
    journal_events = [
        {
            "symbol": "ADAUSDT",
            "primary_exchange": "N/A",
            "strategy": "None",
            "signal_after_adjustment": 0.48,
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "unknown"
    assert group["strategy"] == "unknown"


def test_symbol_map_ambiguous_entry_keeps_unknown_routing() -> None:
    journal_events = [
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "binance",
            "strategy": "breakout",
        },
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "kraken",
            "strategy": "breakout",
        },
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "risk_score": 0.38,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert set(groups) == {("unknown", "unknown")}
    assert groups[("unknown", "unknown")]["metrics"]["risk_score"]["count"] == 1


def test_streaming_generators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Tracker:
        __slots__ = ("_counter",)

        def __init__(self, counter: dict[str, int]) -> None:
            self._counter = counter
            counter["active"] += 1
            counter["peak"] = max(counter["peak"], counter["active"])

        def __del__(self) -> None:  # pragma: no cover - gc driven
            counter = self._counter
            counter["active"] -= 1

    counters = {"active": 0, "peak": 0}

    count = 512
    journal_path = tmp_path / "stream_journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            payload = {
                "timestamp": f"2024-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "signal_after_adjustment": 0.5 + (index % 10) * 0.01,
                "signal_after_clamp": 0.4 + (index % 10) * 0.01,
            }
            print(json.dumps(payload), file=handle)

    autotrade_path = tmp_path / "stream_autotrade.json"
    entries: list[dict[str, object]] = []
    for index in range(count):
        entries.append(
            {
                "timestamp": f"2024-01-01T01:{index // 60:02d}:{index % 60:02d}Z",
                "decision": {
                    "details": {
                        "symbol": f"SYM{index % 5}",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.6 + (index % 5) * 0.01},
                    }
                },
            }
        )
    autotrade_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    journal_probe = _load_journal_events([journal_path])
    entries_probe = _load_autotrade_entries([str(autotrade_path)])

    assert isinstance(journal_probe, GeneratorType)
    assert isinstance(entries_probe, GeneratorType)

    def _attach_tracker(stream: Iterable[Mapping[str, object]]) -> Iterable[Mapping[str, object]]:
        for item in stream:
            if isinstance(item, Mapping):
                item["__tracker__"] = Tracker(counters)
                yield item
            else:
                mapping = dict(item)
                mapping["__tracker__"] = Tracker(counters)
                yield mapping

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        lambda config_path=None: {},
    )

    report = _generate_report(
        journal_events=_attach_tracker(_load_journal_events([journal_path])),
        autotrade_entries=_attach_tracker(_load_autotrade_entries([str(autotrade_path)])),
        percentiles=[0.5],
        suggestion_percentile=0.5,
        since=None,
        until=None,
        current_signal_thresholds={},
        current_threshold_sources={},
        risk_score_override=None,
        risk_score_source=None,
        risk_threshold_sources=[],
        include_raw_values=False,
    )

    assert report["sources"]["journal_events"] == count
    assert report["sources"]["autotrade_entries"] == count
    assert report["groups"]

    gc.collect()
    assert counters["active"] == 0
    assert counters["peak"] < count // 4
