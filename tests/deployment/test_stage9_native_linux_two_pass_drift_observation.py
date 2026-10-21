"""S9-RPI-30: two untrusted observations must never authorize production."""

from __future__ import annotations

import json
import sys

import pytest

from deployment import stage9_native_linux_two_pass_drift_observation as probe

_ROLES = probe._ROLES
_SOURCES = probe._SOURCES


def _report(
    source, *, pids=(4321, 4322, 4323), ticks=(1001, 1002, 1003), fingerprint="ab"
):
    field, marker = probe.consistency._FIELDS[source]
    rows = {}
    for i, (role, unit) in enumerate(_ROLES):
        pid, tick = pids[i], ticks[i]
        row = {"unit": unit, "observation": marker}
        if source == "identity":
            row["identity"] = {
                "pid": pid,
                "start_time_ticks": tick,
                "uid": 2100 + i,
                "gid": 2100 + i,
                "groups": [2100 + i],
                "namespaces": {
                    name: 10000 + i
                    for name in probe.consistency.identity._NS_TYPES
                },
            }
        elif source == "cgroup":
            row.update(pid=pid, start_time_ticks=tick)
        elif source == "fragment":
            row.update(
                pid_hint=pid,
                fragment_sha256=fingerprint * 32,
                fragment_size=200,
            )
        elif source == "executable":
            row.update(
                pid_hint=pid,
                start_time_ticks=tick,
                executable_sha256=fingerprint * 32,
                executable_size=400,
            )
        else:
            row.update(
                pid_hint=pid,
                start_time_ticks=tick,
                pidfd_poll_alive=True,
            )
        rows[role] = row
    return {
        "schema": probe.consistency._SCHEMAS[source],
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT",
        "blockers": ["UNTRUSTED"],
        "observed_failures": [],
        field: rows,
    }


def _sweep(**overrides):
    return {
        source: overrides[source] if source in overrides else _report(source)
        for source in _SOURCES
    }


def _observers(first, second, *, calls=None):
    calls = calls if calls is not None else []
    hits = {source: 0 for source in _SOURCES}

    def get(source):
        def reader():
            hits[source] += 1
            calls.append(source)
            assert hits[source] <= 2, "Source invoked too many times"
            return first[source] if hits[source] == 1 else second[source]

        return reader

    return {source: get(source) for source in _SOURCES}


def _assert_blocked(report):
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert report["sweeps"] == 2
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(report, allow_nan=False)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Synthetic Linux two-pass experiment"
)
def test_clean_two_pass_observation_is_explicitly_not_production_qualified():
    calls = []
    reports = _sweep()
    observed = probe.inspect_host(observers=_observers(reports, reports, calls=calls))
    _assert_blocked(observed)
    assert observed["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert calls == list(_SOURCES) * 2
    assert not observed["observed_failures"]
    assert len(observed["drift_observations"]) == 3
    assert all(
        item == {
            "observation": "NO_TWO_PASS_CONTRADICTION_OBSERVED",
            "sources_observed_both": 5,
            "issues": [],
        }
        for item in observed["drift_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic two-pass PID drift")
def test_all_sources_can_agree_within_round_but_change_pid_and_ticks_between_rounds():
    first = _sweep()
    second = {
        source: _report(source, pids=(5555, 4322, 4323), ticks=(2001, 1002, 1003))
        for source in _SOURCES
    }
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    left, middle, right = [role for role, _unit in _ROLES]
    assert report["drift_observations"][left]["observation"] == "NOT_QUALIFIED"
    assert (
        sum(
            "PID_CHANGED_BETWEEN_SWEEPS" in issue
            for issue in report["drift_observations"][left]["issues"]
        )
        == 5
    )
    assert (
        sum(
            "STARTTIME_CHANGED_BETWEEN_SWEEPS" in issue
            for issue in report["drift_observations"][left]["issues"]
        )
        == 4
    )
    for role in (middle, right):
        assert report["drift_observations"][role]["issues"] == []


@pytest.mark.skipif(
    sys.platform != "linux", reason="Synthetic two-pass starttime drift"
)
@pytest.mark.parametrize("source", ("identity", "cgroup", "executable", "pidfd"))
def test_starttime_change_with_same_pid_detected(source):
    first, second = _sweep(), _sweep()
    second[source] = _report(source, ticks=(9999, 1002, 1003))
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    role = _ROLES[0][0]
    assert f"{source}: STARTTIME_CHANGED_BETWEEN_SWEEPS" in (
        report["drift_observations"][role]["issues"]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic unpinned digest drift")
@pytest.mark.parametrize("source", ["fragment", "executable"])
def test_unpinned_object_digest_drift_does_not_log_digest(source):
    first, second = _sweep(), _sweep()
    second[source] = _report(source, fingerprint="cd")
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    for role, _unit in _ROLES:
        assert f"{source}: UNPINNED_OBJECT_CHANGED_BETWEEN_SWEEPS" in (
            report["drift_observations"][role]["issues"]
        )
    serialized = json.dumps(report)
    assert "cd" * 32 not in serialized
    assert "ab" * 32 not in serialized


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed observer")
@pytest.mark.parametrize("invalid_round", [0, 1])
def test_malformed_or_promoted_subreport_fails_closed(invalid_round):
    first, second = _sweep(), _sweep()
    target = (first, second)[invalid_round]["pidfd"]
    target["qualification"] = "GO"
    target["secret"] = "PRIVATE_SOURCE_SECRET"
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert all(
        "pidfd: INCOMPLETE_TWO_PASS_OBSERVATION" in item["issues"]
        for item in report["drift_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend exception")
def test_second_sweep_backend_exception_is_sanitized_without_retry():
    first, second = _sweep(), _sweep()
    calls = []
    observers = _observers(first, second, calls=calls)
    real = observers["identity"]

    def fail_second():
        if calls.count("identity"):
            calls.append("identity")
            raise RuntimeError("PRIVATE_SOURCE_SECRET")
        return real()

    observers["identity"] = fail_second
    report = probe.inspect_host(observers=observers)
    _assert_blocked(report)
    assert len(calls) == 10
    assert all(
        "identity: INCOMPLETE_TWO_PASS_OBSERVATION" in item["issues"]
        for item in report["drift_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic role refusal")
def test_role_refusal_scoped_to_one_role_in_first_sweep():
    first, second = _sweep(), _sweep()
    role, unit = _ROLES[0]
    first["pidfd"]["pidfd_observations"][role] = {
        "unit": unit,
        "observation": "NOT_QUALIFIED",
        "issues": ["PIDFD_NOT_LIVE"],
    }
    first["pidfd"]["observed_failures"] = [f"{role}.pidfd: PIDFD_NOT_LIVE"]
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert report["drift_observations"][role]["issues"] == [
        "pidfd: INCOMPLETE_TWO_PASS_OBSERVATION"
    ]
    assert all(
        report["drift_observations"][other]["issues"] == []
        for other, _unit in _ROLES[1:]
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="Synthetic global diagnostic failure"
)
def test_global_untrusted_subobserver_failure_affects_all_roles_without_leaking():
    first, second = _sweep(), _sweep()
    first["cgroup"]["observed_failures"] = ["GLOBAL_PRIVATE_SOURCE_SECRET"]
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert all(
        "cgroup: SUBREPORT_ANOMALY" in row["issues"]
        for row in report["drift_observations"].values()
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="Synthetic cross-source contradictions"
)
def test_first_sweep_cross_source_disagreement_survives_clean_second():
    first, second = _sweep(), _sweep()
    first["fragment"] = _report("fragment", pids=(9999, 4322, 4323))
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    role = _ROLES[0][0]
    row = report["drift_observations"][role]
    assert "CROSS_SOURCE_PID_CONTRADICTION" in row["issues"]
    assert "fragment: PID_CHANGED_BETWEEN_SWEEPS" in row["issues"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic cross-role collisions")
def test_collision_only_in_second_sweep_is_caught_for_both_roles():
    first, second = _sweep(), {
        source: _report(source, pids=(4322, 4322, 4323))
        for source in _SOURCES
    }
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    for role, _unit in _ROLES[:2]:
        assert "CROSS_ROLE_SHARED_PID_HINT" in (
            report["drift_observations"][role]["issues"]
        )
    assert report["drift_observations"][_ROLES[2][0]]["issues"] == []


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_refuses_without_observer_calls(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden():
        pytest.fail("No observer should be invoked")

    report = probe.inspect_host(observers={key: forbidden for key in _SOURCES})
    _assert_blocked(report)
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic configuration test")
def test_invalid_observer_map_refuses_without_calls():
    called = []

    def forbidden():
        called.append(1)
        return None

    invalid_maps = (
        {},
        {"identity": forbidden},
        {**{source: forbidden for source in _SOURCES}, "extra": forbidden},
    )
    for invalid in invalid_maps:
        result = probe.inspect_host(observers=invalid)
        _assert_blocked(result)
        assert result["observed_failures"] == ["INVALID_OBSERVER_CONFIGURATION"]
    assert not called


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic CLI")
def test_cli_always_exits_two_on_clean_synthetic_observation(monkeypatch, capsys):
    original = probe.inspect_host
    reports = _sweep()
    monkeypatch.setattr(
        probe, "inspect_host", lambda: original(observers=_observers(reports, reports))
    )
    assert probe.main() == 2
    _assert_blocked(json.loads(capsys.readouterr().out))


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic credential drift")
@pytest.mark.parametrize(
    "field, replacement",
    [
        ("uid", 8111),
        ("gid", 8222),
        ("groups", [8333]),
    ],
)
def test_credentials_change_with_stable_pid_and_starttime(field, replacement):
    first, second = _sweep(), _sweep()
    role = _ROLES[0][0]
    second["identity"]["process_observations"][role]["identity"][field] = replacement
    observed = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(observed)
    assert observed["sweeps_attempted"] == 2
    assert observed["drift_observations"][role]["issues"] == [
        "identity: CREDENTIAL_HINT_CHANGED_BETWEEN_SWEEPS"
    ]
    assert all(
        observed["drift_observations"][other]["issues"] == []
        for other, _unit in _ROLES[1:]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic namespace drift")
@pytest.mark.parametrize("namespace", probe.consistency.identity._NS_TYPES)
def test_identity_namespace_changes_with_same_pid_and_credentials(namespace):
    first, second = _sweep(), _sweep()
    role = _ROLES[0][0]
    second["identity"]["process_observations"][role]["identity"]["namespaces"][
        namespace
    ] = 991234
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert report["drift_observations"][role]["issues"] == [
        "identity: NAMESPACE_HINT_CHANGED_BETWEEN_SWEEPS"
    ]
    assert all(
        report["drift_observations"][other]["issues"] == []
        for other, _unit in _ROLES[1:]
    )
    assert "991234" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic shared credentials")
@pytest.mark.parametrize("field", ("uid", "gid", "groups"))
def test_shared_credentials_without_backend_failure(field):
    first, second = _sweep(), _sweep()
    left, right, other = [role for role, _unit in _ROLES]
    rows = second["identity"]["process_observations"]
    rows[left]["identity"][field] = rows[right]["identity"][field]
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    for role in (left, right):
        assert "CROSS_ROLE_SHARED_CREDENTIAL_HINT" in (
            report["drift_observations"][role]["issues"]
        )
        assert report["drift_observations"][role]["observation"] == "NOT_QUALIFIED"
    assert report["drift_observations"][other]["issues"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic shared namespaces")
@pytest.mark.parametrize("namespace", probe.consistency.identity._NS_TYPES)
@pytest.mark.parametrize("round_index", [0, 1])
def test_shared_namespace_is_caught_in_either_sweep(namespace, round_index):
    first, second = _sweep(), _sweep()
    left, right, other = [role for role, _unit in _ROLES]
    rows = (first, second)[round_index]["identity"]["process_observations"]
    rows[left]["identity"]["namespaces"][namespace] = rows[right]["identity"][
        "namespaces"
    ][namespace]
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    for role in (left, right):
        assert "CROSS_ROLE_SHARED_NAMESPACE_HINT" in (
            report["drift_observations"][role]["issues"]
        )
    assert report["drift_observations"][other]["issues"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic rejected identity")
def test_refused_identity_does_not_appear_as_trusted_cross_role_observation():
    first, second = _sweep(), _sweep()
    role, unit = _ROLES[0]
    first["identity"]["process_observations"][role] = {
        "unit": unit,
        "observation": "NOT_QUALIFIED",
        "issues": ["PROCESS_IDENTITY_NOT_AVAILABLE"],
    }
    first["identity"]["observed_failures"] = [
        f"{role}.process: PROCESS_IDENTITY_NOT_AVAILABLE"
    ]
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert report["drift_observations"][role]["issues"] == [
        "identity: INCOMPLETE_TWO_PASS_OBSERVATION"
    ]
    for other, _unit in _ROLES[1:]:
        assert report["drift_observations"][other]["issues"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic object size drift")
@pytest.mark.parametrize(
    "source, size_field",
    [("fragment", "fragment_size"), ("executable", "executable_size")],
)
def test_same_digest_but_different_object_size_is_still_diagnosed(source, size_field):
    first, second = _sweep(), _sweep()
    role = _ROLES[0][0]
    field = probe.consistency._FIELDS[source][0]
    second[source][field][role][size_field] += 1
    report = probe.inspect_host(observers=_observers(first, second))
    _assert_blocked(report)
    assert f"{source}: UNPINNED_OBJECT_CHANGED_BETWEEN_SWEEPS" in (
        report["drift_observations"][role]["issues"]
    )
    assert all(
        report["drift_observations"][other]["issues"] == []
        for other, _unit in _ROLES[1:]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic no-observer checks")
def test_attempt_count_zero_before_invalid_configuration():
    result = probe.inspect_host(observers={})
    _assert_blocked(result)
    assert result["sweeps_attempted"] == 0
