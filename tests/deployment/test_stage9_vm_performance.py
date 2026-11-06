"""Testy instrumentu D07: dane syntetyczne, bez VM, uprawnień ani zmian hosta."""

from __future__ import annotations

import math
import sys

import pytest

from deployment.stage9_vm_performance import Recorder


def record(recorder, value=10, **changes):
    metadata = {
        "scenario": "poprawny",
        "temperature": "warm",
        "metric": "client_exchange",
        "source": "PROCESS",
        "unit": "ns",
        "outcome": "complete",
    }
    metadata.update(changes)
    recorder.record(value, **metadata)


@pytest.mark.parametrize("value", [True, False, -1, math.nan, math.inf, -math.inf, "1", None])
def test_invalid_observation_is_refused_without_retaining_partial_sample(value):
    recorder = Recorder()
    with pytest.raises((ValueError, TypeError)):
        record(recorder, value)
    assert recorder.summary()["samples"] == []


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "BASELINE_MEASURED"},
        {"source": "POSTGRES"},
        {"temperature": "hot"},
        {"outcome": "PASS"},
        {"metric": "unknown"},
        {"unit": "seconds"},
        {"scenario": ""},
    ],
)
def test_unknown_metadata_cannot_become_measurement_evidence(changes):
    recorder = Recorder()
    with pytest.raises((ValueError, TypeError)):
        record(recorder, **changes)
    assert recorder.summary()["samples"] == []


def test_real_process_and_postgresql_sources_do_not_promote_recorder_to_baseline():
    recorder = Recorder()
    record(recorder)
    record(recorder, metric="postgresql", source="POSTGRESQL")
    report = recorder.summary()
    assert report["classification"] == "MODEL_ONLY"
    assert len(report["samples"]) == 2


def test_model_postgresql_and_cold_warm_results_are_never_merged():
    recorder = Recorder()
    record(recorder, 100, metric="postgresql", source="MODEL", temperature="cold")
    record(recorder, 1_000, metric="postgresql", source="MODEL", temperature="warm")
    record(recorder, 10_000, metric="postgresql", source="POSTGRESQL", temperature="cold")
    record(recorder, 100_000, metric="postgresql", source="POSTGRESQL", temperature="warm")
    report = recorder.summary()
    groups = report["groups"]
    assert len(groups) == 4
    assert {(item["source"], item["temperature"]) for item in groups} == {
        ("MODEL", "cold"),
        ("MODEL", "warm"),
        ("POSTGRESQL", "cold"),
        ("POSTGRESQL", "warm"),
    }
    assert report["classification"] == "MODEL_ONLY"


def test_latency_quantiles_keep_interruptions_outside_completed_distribution():
    recorder = Recorder()
    for value in range(1, 101):
        record(recorder, value)
    record(recorder, 5_000_000_000, outcome="error")
    record(recorder, 3_500_000_000, outcome="timeout")
    group = recorder.summary()["groups"][0]
    assert group["n"] == 102
    assert group["errors"] == group["timeouts"] == 1
    assert group["deadline_exceeded"] == 2
    assert group["complete"] == {"n": 100, "p50": 50, "p95": 95, "p99": 99, "max": 100}
    assert group["interrupted"] == {
        "n": 2,
        "p50": 3_500_000_000,
        "p95": 5_000_000_000,
        "p99": 5_000_000_000,
        "max": 5_000_000_000,
    }
    assert group["uncertainty"]


def test_deadline_overrun_is_strict_and_does_not_reclassify_other_phase_durations():
    recorder = Recorder()
    record(recorder, 3_000_000_000)
    record(recorder, 3_000_000_001)
    record(recorder, 4_000_000_000, metric="process_start")
    report = recorder.summary()
    exchange = next(item for item in report["groups"] if item["metric"] == "client_exchange")
    startup = next(item for item in report["groups"] if item["metric"] == "process_start")
    assert report["client_deadline_ns"] == 3_000_000_000
    assert exchange["deadline_exceeded"] == 1
    assert startup["deadline_exceeded"] == 0
    assert [item["value"] for item in report["samples"]] == [
        3_000_000_000,
        3_000_000_001,
        4_000_000_000,
    ]


def test_all_timeout_series_reports_no_success_quantiles():
    recorder = Recorder()
    record(recorder, 3_000_000_001, outcome="timeout")
    group = recorder.summary()["groups"][0]
    assert group["n"] == group["timeouts"] == 1
    assert group["complete"] == {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    assert group["interrupted"]["n"] == 1


@pytest.mark.parametrize(
    ("exception", "expected_outcome"), [(TimeoutError, "timeout"), (LookupError, "error")]
)
def test_sample_retains_failure_and_resource_observations_then_propagates(
    exception, expected_outcome
):
    recorder = Recorder()
    with pytest.raises(exception):
        with recorder.sample(
            scenario="utrata-odpowiedzi",
            temperature="warm",
            metric="client_exchange",
            source="PROCESS",
        ):
            raise exception("syntetyczna awaria")
    samples = recorder.summary()["samples"]
    wall = [item for item in samples if item["metric"] == "client_exchange"]
    assert len(wall) == 1
    assert wall[0]["outcome"] == expected_outcome
    assert wall[0]["unit"] == "ns"
    resources = [item for item in samples if item["metric"] in {"cpu", "rss"}]
    assert any(item["metric"] == "cpu" for item in resources)
    assert all(item["measured_for"] == "client_exchange" for item in resources)
    assert all(item["source"] == "PROCESS" for item in resources)


def test_resource_observations_do_not_inherit_the_database_model_source():
    recorder = Recorder()
    with recorder.sample(
        scenario="writer-model",
        temperature="unspecified",
        metric="postgresql",
        source="MODEL",
    ):
        pass
    samples = recorder.summary()["samples"]
    database = next(item for item in samples if item["metric"] == "postgresql")
    assert database["source"] == "MODEL"
    resources = [item for item in samples if item["metric"] in {"cpu", "rss"}]
    assert resources
    assert all(item["source"] == "PROCESS" for item in resources)
    assert all(item["measured_for"] == "postgresql" for item in resources)
    assert not any(item["source"] == "POSTGRESQL" for item in samples)


def test_profile_measures_selected_code_and_restores_previous_profile_after_error():
    recorder = Recorder()

    def selected():
        raise LookupError("syntetyczna odmowa")

    previous = sys.getprofile()
    with pytest.raises(LookupError, match="syntetyczna odmowa"):
        with recorder.profile(
            {selected.__code__: ("cryptography", "PROCESS")},
            scenario="bledny-podpis",
            temperature="cold",
        ):
            selected()
    assert sys.getprofile() is previous
    samples = recorder.summary()["samples"]
    crypto = [item for item in samples if item["metric"] == "cryptography"]
    assert len(crypto) == 1
    assert crypto[0]["source"] == "PROCESS"
    assert crypto[0]["temperature"] == "cold"


def test_profile_refuses_to_replace_an_existing_observer():
    recorder = Recorder()

    def observer(frame, event, arg):
        return None

    def selected():
        return 1

    previous = sys.getprofile()
    sys.setprofile(observer)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            with recorder.profile(
                {selected.__code__: ("identity", "PROCESS")},
                scenario="zajety-profiler",
                temperature="unspecified",
            ):
                pytest.fail("Nie wolno zastąpić istniejącego obserwatora")
        assert sys.getprofile() is observer
    finally:
        sys.setprofile(previous)
    assert recorder.summary()["samples"] == []


@pytest.mark.parametrize(
    "mapping",
    [
        {"nie-kod": ("identity", "PROCESS")},
        {record.__code__: ["identity", "PROCESS"]},
        {record.__code__: ("identity", "UNKNOWN")},
    ],
)
def test_invalid_profile_mapping_never_installs_a_partial_observer(mapping):
    recorder = Recorder()
    previous = sys.getprofile()
    with pytest.raises(ValueError):
        with recorder.profile(mapping, scenario="bledny-profil", temperature="unspecified"):
            pytest.fail("Walidacja musi poprzedzać uruchomienie obserwatora")
    assert sys.getprofile() is previous
    assert recorder.summary()["samples"] == []


def test_editing_a_report_does_not_change_retained_evidence():
    recorder = Recorder()
    record(recorder, 123)
    report = recorder.summary()
    report["samples"][0]["source"] = "POSTGRESQL"
    report["samples"][0]["value"] = 0
    report["groups"][0]["complete"]["p99"] = 0
    fresh = recorder.summary()
    assert fresh["samples"][0]["source"] == "PROCESS"
    assert fresh["samples"][0]["value"] == 123
    assert fresh["groups"][0]["complete"]["p99"] == 123
