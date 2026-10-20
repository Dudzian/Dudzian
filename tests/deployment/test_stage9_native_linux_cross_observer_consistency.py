"""S9-RPI-29: closed-shape, untrusted cross-observer correlation regressions."""

from __future__ import annotations

import json
import sys

import pytest

from deployment import stage9_native_linux_cross_observer_consistency as probe


def _make_report(source, *, pids=(4321, 4322, 4323), ticks=(1001, 1002, 1003)):
    field, marker = probe._FIELDS[source]
    rows = {}
    for index, (role, unit) in enumerate(probe._ROLES):
        pid, tick = pids[index], ticks[index]
        if source == "identity":
            row = {
                "unit": unit,
                "observation": marker,
                "identity": {
                    "pid": pid,
                    "start_time_ticks": tick,
                    "uid": 2200 + index,
                    "gid": 2200 + index,
                    "groups": [2200 + index],
                    "namespaces": {name: 10000 + index for name in probe.identity._NS_TYPES},
                },
            }
        elif source == "cgroup":
            row = {"unit": unit, "observation": marker, "pid": pid, "start_time_ticks": tick}
        elif source == "fragment":
            row = {
                "unit": unit,
                "observation": marker,
                "pid_hint": pid,
                "fragment_sha256": "ab" * 32,
                "fragment_size": 123,
            }
        elif source == "executable":
            row = {
                "unit": unit,
                "observation": marker,
                "pid_hint": pid,
                "start_time_ticks": tick,
                "executable_sha256": "bc" * 32,
                "executable_size": 1000,
            }
        else:
            row = {
                "unit": unit,
                "observation": marker,
                "pid_hint": pid,
                "start_time_ticks": tick,
                "pidfd_poll_alive": True,
            }
        rows[role] = row
    return {
        "schema": probe._SCHEMAS[source],
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT",
        "blockers": ["NEVER_TRUST_UNPINNED_INPUT"],
        "observed_failures": [],
        field: rows,
    }


def _observers(reports):
    return {source: (lambda s=source: reports[s]) for source in probe._SOURCES}


def _reports():
    return {source: _make_report(source) for source in probe._SOURCES}


def _blocked(report):
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert "PRIVATE_UNTRUSTED_INPUT" not in json.dumps(report, allow_nan=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_complete_matching_hints_are_still_untrusted_not_qualified():
    reports = _reports()
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert observed["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert not observed["observed_failures"]
    assert len(observed["correlation"]) == 3
    assert all(
        item == {
            "observation": "NO_CROSS_SOURCE_CONTRADICTION_OBSERVED",
            "issues": [],
            "sources_observed": 5,
        }
        for item in observed["correlation"].values()
    )
    assert all(source_report["status"] == "BLOCKED" for source_report in reports.values())


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
@pytest.mark.parametrize("source", probe._SOURCES)
def test_mismatched_pid_is_detected_for_every_source(source):
    reports = _reports()
    field, _marker = probe._FIELDS[source]
    role = probe._ROLES[1][0]
    row = reports[source][field][role]
    if source == "identity":
        row["identity"]["pid"] = 7777
    elif source == "cgroup":
        row["pid"] = 7777
    else:
        row["pid_hint"] = 7777
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert "CROSS_OBSERVER_MAINPID_MISMATCH" in observed["correlation"][role]["issues"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
@pytest.mark.parametrize("source", ("identity", "cgroup", "executable", "pidfd"))
def test_mismatched_starttime_is_detected_for_every_source_with_ticks(source):
    reports = _reports()
    field, _marker = probe._FIELDS[source]
    role = probe._ROLES[2][0]
    row = reports[source][field][role]
    if source == "identity":
        row["identity"]["start_time_ticks"] += 99
    else:
        row["start_time_ticks"] += 99
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert "CROSS_OBSERVER_STARTTIME_MISMATCH" in observed["correlation"][role]["issues"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
@pytest.mark.parametrize("source", probe._SOURCES)
def test_subobserver_refusals_are_not_forwarded_to_public_report(source):
    reports = _reports()
    field, _marker = probe._FIELDS[source]
    role, unit = probe._ROLES[0]
    reports[source][field][role] = {
        "unit": unit, "observation": "NOT_QUALIFIED", "issues": ["NO_PERMISSION"]
    }
    reports[source]["observed_failures"] = [f"{role}: NO_PERMISSION"]
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert observed["correlation"][role]["observation"] == "NOT_QUALIFIED"
    assert f"{source}: NOT_OBSERVED" in observed["correlation"][role]["issues"]
    assert "NO_PERMISSION" not in json.dumps(observed)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
@pytest.mark.parametrize("source", probe._SOURCES)
def test_untrusted_observer_exceptions_are_sanitized(source):
    reports = _reports()
    observers = _observers(reports)

    def invalid():
        raise RuntimeError("PRIVATE_UNTRUSTED_INPUT")

    observers[source] = invalid
    observed = probe.inspect_host(observers=observers)
    _blocked(observed)
    assert f"{source}: OBSERVER_FAILED_OR_INVALID" in observed["observed_failures"]
    assert all(
        f"{source}: NOT_OBSERVED" in row["issues"]
        for row in observed["correlation"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
@pytest.mark.parametrize("source", probe._SOURCES)
@pytest.mark.parametrize(
    "mutation",
    [
        "status_go",
        "approved_topology",
        "production_go",
        "extra_row",
        "missing_role",
        "unknown_snapshot",
        "foreign_digest",
        "leaked_failure",
    ],
)
def test_malformed_or_promoted_subreports_refused(source, mutation):
    reports = _reports()
    field, _ = probe._FIELDS[source]
    role = probe._ROLES[0][0]
    if mutation == "status_go":
        reports[source]["status"] = "GO"
    elif mutation == "approved_topology":
        reports[source]["topology"] = "QUALIFIED"
    elif mutation == "production_go":
        reports[source]["production_emission"] = "ALLOWED"
    elif mutation == "extra_row":
        reports[source][field][role]["private"] = "PRIVATE_UNTRUSTED_INPUT"
    elif mutation == "missing_role":
        reports[source][field].pop(role)
    elif mutation == "unknown_snapshot":
        reports[source]["snapshot"] = "PRODUCTION_PINNED"
    elif mutation == "foreign_digest":
        row = reports[source][field][role]
        if source == "fragment":
            row["fragment_sha256"] = "PRIVATE_UNTRUSTED_INPUT"
        elif source == "executable":
            row["executable_sha256"] = "PRIVATE_UNTRUSTED_INPUT"
        elif source == "identity":
            row["identity"]["namespaces"] = {"secret": "PRIVATE_UNTRUSTED_INPUT"}
        elif source == "pidfd":
            row["pidfd_poll_alive"] = False
        else:
            row["start_time_ticks"] = "PRIVATE_UNTRUSTED_INPUT"
    else:
        reports[source]["observed_failures"] = ["PRIVATE_UNTRUSTED_INPUT" * 20]
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert f"{source}: OBSERVER_FAILED_OR_INVALID" in observed["observed_failures"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_duplicate_pids_across_roles_are_not_promoted_as_consistency():
    pids = (7777, 7777, 7777)
    reports = {source: _make_report(source, pids=pids) for source in probe._SOURCES}
    reports["pidfd"]["observed_failures"] = ["shared systemd pid hint"]
    observed = probe.inspect_host(observers=_observers(reports))
    _blocked(observed)
    assert all(row["observation"] == "NOT_QUALIFIED" for row in observed["correlation"].values())


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_missing_or_incorrect_backend_map_fails_closed_without_invocation():
    reports = _reports()
    for invalid in (None, {}, {"identity": lambda: reports["identity"]}, {"evil": "x"}):
        if invalid is None:
            continue  # None intentionally selects real native observers
        observed = probe.inspect_host(observers=invalid)
        _blocked(observed)
        assert observed["observed_failures"] == ["INVALID_OBSERVER_CONFIGURATION"]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_non_linux_rejects_before_accessing_any_observer(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden():
        pytest.fail("Observer called on a non-Linux platform")

    observed = probe.inspect_host(observers={name: forbidden for name in probe._SOURCES})
    _blocked(observed)
    assert observed["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux CLI test")
def test_cli_always_returns_exit_two(monkeypatch, capsys):
    original = probe.inspect_host
    monkeypatch.setattr(probe, "inspect_host", lambda: original(observers=_observers(_reports())))
    assert probe.main() == 2
    observed = json.loads(capsys.readouterr().out)
    _blocked(observed)
    assert observed["snapshot"] == "SYNTHETIC_TEST_INPUT"


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_cross_role_shared_pid_is_detected_without_subreport_warning():
    reports = {
        source: _make_report(source, pids=(7777, 7777, 4323))
        for source in probe._SOURCES
    }
    assert all(not report["observed_failures"] for report in reports.values())
    correlated = probe.inspect_host(observers=_observers(reports))
    _blocked(correlated)
    roles = [item[0] for item in probe._ROLES]
    for role in roles[:2]:
        assert "CROSS_ROLE_SHARED_PID_HINT" in correlated["correlation"][role]["issues"]
        assert correlated["correlation"][role]["observation"] == "NOT_QUALIFIED"
    assert correlated["correlation"][roles[2]]["observation"] == (
        "NO_CROSS_SOURCE_CONTRADICTION_OBSERVED"
    )
    assert not any(
        "SHARED_PID" in issue
        for issue in correlated["correlation"][roles[2]]["issues"]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_cross_role_shared_pid_is_diagnosed_once_per_role():
    reports = {source: _make_report(source, pids=(7777, 7777, 7777))
               for source in probe._SOURCES}
    correlated = probe.inspect_host(observers=_observers(reports))
    _blocked(correlated)
    assert len([
        item for item in correlated["observed_failures"]
        if item.endswith(": CROSS_ROLE_SHARED_PID_HINT")
    ]) == 3
