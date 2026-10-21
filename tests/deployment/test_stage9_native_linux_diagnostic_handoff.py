"""S9-RPI-32 handoff never promotes untrusted observations."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_diagnostic_handoff as handoff
from deployment import stage9_native_linux_source_cca_preflight as preflight

_ROLES = handoff._ROLES
_REFERENCES = handoff._REFERENCES


def _preflight():
    return {
        "schema": "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1",
        **handoff._COMMON,
        "blockers": list(preflight._BLOCKERS),
        "principal_observations": {
            role: {
                "name": name,
                "uid": 4000 + index,
                "gid": 4000 + index,
                "supplementary_gids": [],
            }
            for index, (role, name, _subdir) in enumerate(preflight._PROPOSED)
        },
        "reference_principal_observations": {
            role: {
                "name": name,
                "uid": 5000 + index,
                "gid": 5000 + index,
                "supplementary_gids": [],
            }
            for index, (role, name) in enumerate(preflight._KNOWN_REFERENCE_AUTHORITIES)
        },
        "unobserved_reference_principals": [],
        "systemd_observations": {
            role: {
                "unit": unit,
                "observation": "PROPOSED_DECLARATIONS_MATCHED_ONLY",
                "issues": [],
            }
            for role, unit, _name in preflight._PROPOSED_SYSTEMD_UNITS
        },
        "paths": {
            f"{role}.{kind}": {
                "path": f"{prefix}/{subdir}",
                "observation": "MATCHED_PROPOSED_METADATA_ONLY",
                "reason": "no private contents inspected",
            }
            for role, _name, subdir in preflight._PROPOSED
            for kind, prefix in (
                ("configuration", "/etc/cryptohunter"),
                ("private_state", "/var/lib/cryptohunter"),
            )
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
        row
        == {
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
@pytest.mark.parametrize("field", ["principal_observations", "systemd_observations", "paths"])
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
        output["two_pass"]["roles"][other]["observation"] == "UNTRUSTED_NO_CONTRADICTION"
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
        handoff,
        "inspect_host",
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


def _assert_preflight_rejected(source):
    report = _run(source, _drift())
    _blocked(report)
    assert report["preflight"]["observation"] == "NOT_QUALIFIED"
    assert report["preflight"]["reason"] in {
        "INVALID_PREFLIGHT_REPORT",
        "PREFLIGHT_INCOMPLETE_OR_ANOMALOUS",
    }
    assert report["observed_failures"] == ["PREFLIGHT_NOT_QUALIFIED"]
    assert report["two_pass"]["observation"] == "UNTRUSTED_TWO_PASS_METADATA_ONLY"
    serialized = json.dumps(report, allow_nan=False)
    assert "PRIVATE_SOURCE_SECRET" not in serialized
    assert "/etc/cryptohunter" not in serialized
    assert "ch_root_custody_source" not in serialized
    assert "4000" not in serialized


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["principal_observations", "reference_principal_observations"])
@pytest.mark.parametrize(
    "replacement",
    [{}, {"uid": 4000, "gid": 4000}, [], None, {"private": "PRIVATE_SOURCE_SECRET"}],
)
def test_empty_or_incomplete_identity_is_never_complete(field, replacement):
    source = _preflight()
    role = next(iter(source[field]))
    source[field][role] = copy.deepcopy(replacement)
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["principal_observations", "reference_principal_observations"])
@pytest.mark.parametrize("numeric", ["uid", "gid"])
@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 2**32 - 1, 1.5, "4000", None, float("nan"), float("inf")]
)
def test_identity_numbers_reject_bool_privileged_and_noninteger_values(field, numeric, value):
    source = _preflight()
    source[field][next(iter(source[field]))][numeric] = value
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["principal_observations", "reference_principal_observations"])
@pytest.mark.parametrize(
    "value", [None, "PRIVATE_SOURCE_SECRET", (), [True], [False], [-1], [0], [2**32 - 1]]
)
def test_identity_groups_reject_malformed_or_privileged_values(field, value):
    source = _preflight()
    source[field][next(iter(source[field]))]["supplementary_gids"] = value
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["principal_observations", "reference_principal_observations"])
@pytest.mark.parametrize("value", [[6100, 6100], [6101, 6100], "primary"])
def test_supplementary_groups_are_unique_sorted_and_exclude_primary(field, value):
    source = _preflight()
    row = source[field][next(iter(source[field]))]
    row["supplementary_gids"] = [row["gid"]] if value == "primary" else value
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("gid", [5000, 6199])
def test_proposed_principals_cannot_claim_unapproved_supplementary_groups(gid):
    source = _preflight()
    source["principal_observations"][_ROLES[0]]["supplementary_gids"] = [gid]
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "field",
    ["principal_observations", "reference_principal_observations", "systemd_observations", "paths"],
)
def test_nested_unknown_fields_and_unknown_role_keys_are_rejected(field):
    source = _preflight()
    key = next(iter(source[field]))
    source[field][key]["private"] = "PRIVATE_SOURCE_SECRET"
    _assert_preflight_rejected(source)
    source = _preflight()
    source[field]["PRIVATE_SOURCE_SECRET"] = source[field].pop(key)
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["principal_observations", "reference_principal_observations"])
@pytest.mark.parametrize("name", ["PRIVATE_SOURCE_SECRET", "root", None, True])
def test_identity_account_names_must_match_fixed_roles(field, name):
    source = _preflight()
    source[field][next(iter(source[field]))]["name"] = name
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "left,right",
    [
        (("principal_observations", _ROLES[0]), ("principal_observations", _ROLES[1])),
        (
            ("principal_observations", _ROLES[0]),
            ("reference_principal_observations", _REFERENCES[0]),
        ),
        (
            ("reference_principal_observations", _REFERENCES[0]),
            ("reference_principal_observations", _REFERENCES[1]),
        ),
    ],
)
@pytest.mark.parametrize("numeric", ["uid", "gid"])
def test_pairwise_identity_collisions_cannot_be_hidden_by_empty_failures(left, right, numeric):
    source = _preflight()
    source[right[0]][right[1]][numeric] = source[left[0]][left[1]][numeric]
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "other_field,other_role",
    [("principal_observations", _ROLES[0]), ("reference_principal_observations", _REFERENCES[1])],
)
def test_reference_supplementary_groups_cannot_overlap_other_effective_groups(
    other_field, other_role
):
    source = _preflight()
    source["reference_principal_observations"][_REFERENCES[0]]["supplementary_gids"] = [
        source[other_field][other_role]["gid"]
    ]
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_reference_principals_allow_disjoint_nonprivileged_supplementary_groups():
    source = _preflight()
    source["reference_principal_observations"][_REFERENCES[0]]["supplementary_gids"] = [6100, 6101]
    report = _run(source, _drift())
    _blocked(report)
    assert report["preflight"]["observation"] == "UNTRUSTED_DECLARATIONS_ONLY"
    assert "6100" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "patch",
    [
        {"unit": "PRIVATE_SOURCE_SECRET.service"},
        {"unit": None},
        {"issues": ["PRIVATE_SOURCE_SECRET"]},
        {"issues": ["MAIN_PID_NOT_OBSERVED"]},
        {"issues": "PRIVATE_SOURCE_SECRET"},
        {"observation": "NOT_QUALIFIED", "issues": []},
        {"observation": "NOT_OBSERVED_TEST_FIXTURE", "issues": []},
        {"observation": "READY"},
    ],
)
def test_matching_systemd_declarations_require_exact_unit_and_consistent_issues(patch):
    source = _preflight()
    source["systemd_observations"][_ROLES[0]].update(patch)
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "patch",
    [
        {"path": "/PRIVATE_SOURCE_SECRET"},
        {"path": "/etc/cryptohunter/../root-custody-source"},
        {"path": "/etc/cryptohunter//root-custody-source"},
        {"path": None},
        {"reason": "PRIVATE_SOURCE_SECRET"},
        {"reason": "path missing or inaccessible"},
        {"observation": "NOT_QUALIFIED", "reason": "no private contents inspected"},
        {"observation": "READY"},
    ],
)
def test_matching_path_metadata_requires_exact_candidate_and_consistent_reason(patch):
    source = _preflight()
    source["paths"][f"{_ROLES[0]}.configuration"].update(patch)
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_matching_labels_do_not_override_top_level_failures():
    source = _preflight()
    source["observed_failures"] = ["PRIVATE_SOURCE_SECRET"]
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_matching_labels_do_not_override_native_linux_blocker():
    source = _preflight()
    source["blockers"].insert(0, "NATIVE_LINUX_REQUIRED")
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("backend_failure", [None, "systemd", "path", "principal", "reference"])
def test_actual_preflight_report_schema_with_synthetic_backends(monkeypatch, backend_failure):
    import subprocess

    expected = _preflight()
    accounts = {
        row["name"]: (row["uid"], row["gid"], (row["gid"],))
        for field in ("principal_observations", "reference_principal_observations")
        for row in expected[field].values()
    }
    if backend_failure == "principal":
        accounts.pop(preflight._PROPOSED[0][1])
    elif backend_failure == "reference":
        accounts.pop(preflight._KNOWN_REFERENCE_AUTHORITIES[0][1])
    monkeypatch.setattr(preflight, "_native_accounts", lambda: accounts)
    monkeypatch.setattr(
        preflight,
        "_inspect_directory",
        lambda *args, **kwargs: (
            ("NOT_QUALIFIED", "path missing or inaccessible")
            if backend_failure == "path"
            else ("MATCHED_PROPOSED_METADATA_ONLY", "no private contents inspected")
        ),
    )
    unit_accounts = {unit: account for _role, unit, account in preflight._PROPOSED_SYSTEMD_UNITS}
    calls = []

    def systemctl(command, **kwargs):
        calls.append(command[-1])
        unit = command[-1]
        values = dict.fromkeys(preflight._SYSTEMD_PROPERTIES, "")
        values.update(
            Id=unit,
            LoadState="loaded",
            ActiveState="active",
            User=unit_accounts[unit],
            Group=unit_accounts[unit],
            DynamicUser="no",
            MainPID="12345",
            FragmentPath=f"/etc/systemd/system/{unit}",
            NoNewPrivileges="yes",
            PrivateNetwork="yes",
            PrivateIPC="yes",
            PrivateMounts="yes",
            PrivateTmp="yes",
            ProtectSystem="strict",
            ProtectHome="yes",
            RestrictAddressFamilies="AF_UNIX",
        )
        return SimpleNamespace(
            returncode=1 if backend_failure == "systemd" else 0,
            stdout="\n".join(f"{key}={value}" for key, value in values.items()),
        )

    monkeypatch.setattr(subprocess, "run", systemctl)
    source = preflight.inspect_host()
    assert calls == [unit for _role, unit, _account in preflight._PROPOSED_SYSTEMD_UNITS]
    if backend_failure is None:
        assert source == expected
        report = _run(source, _drift())
        _blocked(report)
        assert report["preflight"]["observation"] == "UNTRUSTED_DECLARATIONS_ONLY"
    else:
        _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("malformed", [False, True])
def test_default_runner_propagates_synthetic_child_even_if_child_is_invalid(monkeypatch, malformed):
    child = _drift()
    if malformed:
        child["untrusted_private"] = "PRIVATE_SOURCE_SECRET"
    monkeypatch.setattr(handoff.preflight, "inspect_host", _preflight)
    monkeypatch.setattr(handoff.drift, "inspect_host", lambda: child)
    report = handoff.inspect_host()
    _blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert "PRIVATE_SOURCE_SECRET" not in json.dumps(report)
    assert report["two_pass"]["observation"] == (
        "NOT_QUALIFIED" if malformed else "UNTRUSTED_TWO_PASS_METADATA_ONLY"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("snapshot", [None, "UNKNOWN", "missing"])
def test_default_runner_refuses_live_label_when_child_snapshot_is_unknown(monkeypatch, snapshot):
    child = _drift()
    if snapshot == "missing":
        child.pop("snapshot")
    else:
        child["snapshot"] = snapshot
    monkeypatch.setattr(handoff.preflight, "inspect_host", _preflight)
    monkeypatch.setattr(handoff.drift, "inspect_host", lambda: child)
    report = handoff.inspect_host()
    _blocked(report)
    assert report["snapshot"] == "NOT_QUALIFIED"
    assert report["two_pass"]["reason"] == "INVALID_DRIFT_REPORT"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_default_runner_propagates_preflight_fixture_declaration(monkeypatch):
    source = _preflight()
    source["systemd_observations"] = {
        role: {"unit": unit, "observation": "NOT_OBSERVED_TEST_FIXTURE"}
        for role, unit, _account in preflight._PROPOSED_SYSTEMD_UNITS
    }
    child = _drift()
    child["snapshot"] = "UNTRUSTED_LIVE_OBSERVATION"
    monkeypatch.setattr(handoff.preflight, "inspect_host", lambda: source)
    monkeypatch.setattr(handoff.drift, "inspect_host", lambda: child)
    report = handoff.inspect_host()
    _blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["preflight"]["observation"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize("field", ["systemd_observations", "paths"])
def test_nested_record_requires_all_actual_fields_and_dictionary_type(field):
    complete = _preflight()
    key = next(iter(complete[field]))
    for required in complete[field][key]:
        source = copy.deepcopy(complete)
        source[field][key].pop(required)
        _assert_preflight_rejected(source)
    for replacement in (None, [], "PRIVATE_SOURCE_SECRET"):
        source = copy.deepcopy(complete)
        source[field][key] = replacement
        _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
@pytest.mark.parametrize(
    "field,property_name",
    [
        ("principal_observations", "name"),
        ("reference_principal_observations", "name"),
        ("systemd_observations", "unit"),
        ("paths", "path"),
    ],
)
def test_known_declarations_cannot_be_reassigned_to_another_role(field, property_name):
    source = _preflight()
    first, second = list(source[field])[:2]
    source[field][first][property_name] = source[field][second][property_name]
    _assert_preflight_rejected(source)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic handoff")
def test_reference_supplementary_group_collision_is_not_hidden_by_distinct_primary_groups():
    source = _preflight()
    for role in _REFERENCES[:2]:
        source["reference_principal_observations"][role]["supplementary_gids"] = [6100]
    _assert_preflight_rejected(source)
