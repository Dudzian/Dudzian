"""S9-RPI-32 handoff never promotes untrusted observations."""

from __future__ import annotations

import copy
import json
import sys

import pytest

from deployment import stage9_native_linux_diagnostic_handoff as handoff

_ROLES = handoff._ROLES
_REFERENCES = handoff._REFERENCES


def _preflight():
    return {
        "schema": "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1",
        **handoff._COMMON,
        "blockers": ["PHYSICAL_QUALIFICATION_MISSING"],
        "principal_observations": {
            role: {"uid": 4000 + index, "gid": 4000 + index}
            for index, role in enumerate(_ROLES)
        },
        "reference_principal_observations": {
            role: {"uid": 5000 + index, "gid": 5000 + index}
            for index, role in enumerate(_REFERENCES)
        },
        "unobserved_reference_principals": [],
        "systemd_observations": {
            role: {"observation": "PROPOSED_DECLARATIONS_MATCHED_ONLY"}
            for role in _ROLES
        },
        "paths": {
            path: {"observation": "MATCHED_PROPOSED_METADATA_ONLY"}
            for path in handoff._PATHS
        },
        "observed_failures": [],
    }


def _drift():
    return {
        "schema": "CryptoHunter.Stage9NativeLinuxTwoPassDriftObservationV1",
        **handoff._COMMON,
        "blockers": ["CROSS_ROLE_DENIAL_NOT_QUALIFIED"],
        "snapshot": "SYNTHETIC_TEST_INPUT",
        "sweeps": 2,
        "sweeps_attempted": 2,
        "drift_observations": {
            role: {
                "observation": "NO_TWO_PASS_CONTRADICTION_OBSERVED",
                "sources_observed_both": 5,
                "issues": [],
            }
            for role in _ROLES
        },
        "observed_failures": [],
    }


def _run(preflight_report=None, drift_report=None):
    preflight_report = _preflight() if preflight_report is None else preflight_report
    drift_report = _drift() if drift_report is None else drift_report
    return handoff.inspect_host(
        preflight_observer=lambda: preflight_report,
        drift_observer=lambda: drift_report,
    )


def _blocked(report):
    assert report["schema"] == "CryptoHunter.Stage9NativeLinuxDiagnosticHandoffV1"
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert report["blockers"]
    assert "UNTRUSTED_NON_ATOMIC" not in report["observed_failures"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_complete_synthetic_inputs_never_promote_or_copy_identity_metadata():
    calls = []
    source = _preflight()
    source["principal_observations"][_ROLES[0]]["private"] = "PRIVATE_SOURCE_SECRET"
    drift = _drift()

    def first():
        calls.append("preflight")
        return source

    def second():
        calls.append("drift")
        return drift

    output = handoff.inspect_host(
        preflight_observer=first,
        drift_observer=second,
    )
    _blocked(output)
    assert calls == ["preflight", "drift"]
    assert output["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert output["observed_failures"] == []
    assert output["preflight"]["observation"] == "UNTRUSTED_DECLARATIONS_ONLY"
    assert output["preflight"]["proposed_paths_matching"] == 6
    assert output["preflight"]["reference_roles_seen"] == 3
    assert output["two_pass"]["observation"] == "UNTRUSTED_TWO_PASS_METADATA_ONLY"
    assert set(output["two_pass"]["roles"]) == set(_ROLES)
    assert all(
        row == {
            "observation": "UNTRUSTED_NO_CONTRADICTION",
            "sources_observed_both": 5,
            "issue_count": 0,
        }
        for row in output["two_pass"]["roles"].values()
    )
    serialized = json.dumps(output, allow_nan=False)
    assert "PRIVATE_SOURCE_SECRET" not in serialized
    assert "4000" not in serialized
    assert "5000" not in serialized


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("kind", ["preflight", "two_pass"])
def test_malformed_or_promoted_subreport_is_rejected(kind):
    first, second = _preflight(), _drift()
    record = first if kind == "preflight" else second
    record["status"] = "GO"
    record["untrusted_private"] = "PRIVATE_SOURCE_SECRET"
    result = _run(first, second)
    _blocked(result)
    assert result[kind]["observation"] == "NOT_QUALIFIED"
    assert result[kind]["reason"] == (
        "INVALID_PREFLIGHT_REPORT" if kind == "preflight" else "INVALID_DRIFT_REPORT"
    )
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(result)
    other = "two_pass" if kind == "preflight" else "preflight"
    assert result[other]["observation"].startswith("UNTRUSTED_")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "field", ["principal_observations", "systemd_observations", "paths"]
)
def test_missing_preflight_evidence_prevents_clean_summary(field):
    first = _preflight()
    first[field].pop(next(iter(first[field])))
    report = _run(first, _drift())
    _blocked(report)
    assert report["preflight"]["observation"] == "NOT_QUALIFIED"
    assert report["observed_failures"] == ["PREFLIGHT_NOT_QUALIFIED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_missing_reference_and_reported_failure_refuse_without_echoing():
    first = _preflight()
    first["reference_principal_observations"].pop(_REFERENCES[0])
    first["unobserved_reference_principals"] = [_REFERENCES[0]]
    first["observed_failures"] = ["PRIVATE_SOURCE_SECRET"]
    report = _run(first, _drift())
    _blocked(report)
    assert report["preflight"]["observation"] == "NOT_QUALIFIED"
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "bad_value",
    [
        {"arbitrary": "PRIVATE_SOURCE_SECRET"},
        ["PRIVATE_SOURCE_SECRET"],
        17,
        None,
    ],
)
def test_malformed_preflight_shape_is_rejected(bad_value):
    first = _preflight()
    first["paths"] = bad_value
    report = _run(first, _drift())
    assert report["preflight"]["reason"] == "INVALID_PREFLIGHT_REPORT"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["issues", "sources_observed_both", "observation"])
def test_malformed_drift_role_is_not_qualified(field):
    second = _drift()
    role = _ROLES[0]
    second["drift_observations"][role][field] = {
        "issues": "PRIVATE_SOURCE_SECRET",
        "sources_observed_both": True,
        "observation": "READY",
    }[field]
    report = _run(_preflight(), second)
    _blocked(report)
    assert report["two_pass"]["reason"] == "INVALID_DRIFT_REPORT"
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_one_drift_anomaly_stays_in_its_role_and_remains_unqualified():
    second = _drift()
    role = _ROLES[0]
    second["drift_observations"][role] = {
        "observation": "NOT_QUALIFIED",
        "sources_observed_both": 4,
        "issues": ["PRIVATE_SOURCE_SECRET"],
    }
    second["observed_failures"] = ["PRIVATE_SOURCE_SECRET"]
    output = _run(_preflight(), second)
    _blocked(output)
    assert output["two_pass"]["observation"] == "NOT_QUALIFIED"
    assert output["two_pass"]["roles"][role] == {
        "observation": "NOT_QUALIFIED",
        "sources_observed_both": 4,
        "issue_count": 1,
    }
    assert all(
        output["two_pass"]["roles"][other]["observation"]
        == "UNTRUSTED_NO_CONTRADICTION"
        for other in _ROLES[1:]
    )
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(output)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_incomplete_drift_and_partial_sample_are_never_clean():
    second = _drift()
    second["sweeps_attempted"] = 0
    second["drift_observations"] = {}
    report = _run(_preflight(), second)
    _blocked(report)
    assert report["two_pass"]["observation"] == "NOT_QUALIFIED"
    assert report["two_pass"]["sweeps_attempted"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_observer_exceptions_are_independent_and_sanitized():
    called = []

    def forbidden():
        called.append("preflight")
        raise RuntimeError("PRIVATE_SOURCE_SECRET")

    def other():
        called.append("two_pass")
        return _drift()

    report = handoff.inspect_host(preflight_observer=forbidden, drift_observer=other)
    _blocked(report)
    assert called == ["preflight", "two_pass"]
    assert report["preflight"]["reason"] == "OBSERVER_EXCEPTION"
    assert report["two_pass"]["observation"] == "UNTRUSTED_TWO_PASS_METADATA_ONLY"
    assert report["observed_failures"] == ["PREFLIGHT_NOT_QUALIFIED"]
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize("platform", ("win32", "darwin"))
def test_non_linux_refuses_without_invoking_backends(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden():
        pytest.fail("No native observer must be invoked")

    report = handoff.inspect_host(
        preflight_observer=forbidden,
        drift_observer=forbidden,
    )
    _blocked(report)
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic CLI")
def test_cli_always_exits_two_even_for_matching_inputs(monkeypatch, capsys):
    actual = handoff.inspect_host
    monkeypatch.setattr(
        handoff, "inspect_host",
        lambda: actual(
            preflight_observer=_preflight,
            drift_observer=_drift,
        ),
    )
    assert handoff.main() == 2
    result = json.loads(capsys.readouterr().out)
    _blocked(result)
    assert result["preflight"]["observation"] == "UNTRUSTED_DECLARATIONS_ONLY"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic input validation")
def test_nested_mutation_does_not_change_sanitized_snapshot():
    first, second = _preflight(), _drift()
    output = _run(first, second)
    clone = copy.deepcopy(output)
    first["observed_failures"].append("PRIVATE_SOURCE_SECRET")
    second["drift_observations"][_ROLES[0]]["issues"].append("PRIVATE_SOURCE_SECRET")
    assert output == clone
