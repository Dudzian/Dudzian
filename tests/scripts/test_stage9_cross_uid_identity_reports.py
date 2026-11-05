"""Format i odmowy spreparowanych raportów; żaden przypadek nie kwalifikuje VM."""

from __future__ import annotations

from copy import deepcopy
import errno
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("Raport próby odwołuje się do interfejsów Linux.", allow_module_level=True)

from scripts.diagnostics.stage9_cross_uid_identity import format_reports, validate_reports


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def report_format_examples() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Syntetyczne rekordy parsera; identyfikatory nie pochodzą z obserwacji hosta."""
    trial = {
        "scope": "TEST_ONLY_STAGE9_CROSS_UID_DENIAL",
        "trial_id": "12345678-1234-1234-1234-123456789abc",
        "revision": "a" * 40,
        "environment_id": "noble-cross-uid-001",
        "machine_id": "b" * 32,
        "boot_id": "12345678-1234-1234-1234-123456789def",
        "vm_image_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
    }
    manager = {
        "MainPID": "202",
        "ExecMainPID": "202",
        "ExecMainStartTimestampMonotonic": "100000",
        "ControlGroup": "/system.slice/cryptohunter-freshness-verifier.service",
        "ExecStart": "syntetyczny rekord zatwierdzonego fixture",
    }
    process = {
        "pid": 202,
        "uid": 1002,
        "gid": 1003,
        "groups": [1003],
        "start_time_ticks": 10,
        "exe_sha256": "e" * 64,
        "cgroup": manager["ControlGroup"],
    }
    snapshot: dict[str, Any] = {
        "scope": "TEST_ONLY_ROOT_SNAPSHOT",
        "trial": deepcopy(trial),
        "systemd": deepcopy(manager),
        "process": deepcopy(process),
    }
    observed = {**deepcopy(process), "exe_sha256": None}
    client: dict[str, Any] = {
        "scope": "TEST_ONLY_CLIENT_DENIAL",
        "trial": deepcopy(trial),
        "systemd": deepcopy(manager),
        "root_snapshot_sha256": snapshot_digest(snapshot),
        "pid": 101,
        "uid": 1001,
        "gid": 1001,
        "peer_credentials": {"pid": 202, "uid": 1002, "gid": 1003},
        "peer_process": deepcopy(observed),
        "exe_errno": errno.EACCES,
        "server_denial": "APPROVED_VERIFIER_PROCESS_REQUIRED",
        "server_denial_errno": None,
        "socket_bound_pidfd": True,
        "actual_server_guard": {
            "calls": [deepcopy(observed)],
            "exceptions": [
                {
                    "type": "PermissionError",
                    "reason": "APPROVED_VERIFIER_PROCESS_REQUIRED",
                    "errno": None,
                }
            ],
        },
        "outcome": "UNAVAILABLE",
        "reason": "TRANSPORT_OR_IDENTITY_UNAVAILABLE",
        "product_qualification": "BLOCKED",
    }
    server: dict[str, Any] = {
        "scope": "TEST_ONLY_LISTENER",
        "trial": deepcopy(trial),
        "systemd": deepcopy(manager),
        "process": deepcopy(process),
        "pid": 202,
        "connections": [
            {"pid": 101, "uid": 1001, "gid": 1001, "bytes": 0, "eof": True},
            {"pid": 101, "uid": 1001, "gid": 1001, "bytes": 0, "eof": True},
        ],
        "preparation_operation": "NIE_ISTNIEJE_W_LISTENERZE",
        "product_qualification": "BLOCKED",
    }
    return client, server, snapshot


def validate_examples(
    client: dict[str, Any], server: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    validate_reports(client, server, snapshot, snapshot_sha256=snapshot_digest(snapshot))


def write_examples(
    tmp_path: Path,
    client: dict[str, Any],
    server: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[Path, Path, Path]:
    paths = (tmp_path / "client.json", tmp_path / "listener.json", tmp_path / "before.json")
    for path, value in zip(paths, (client, server, snapshot), strict=True):
        path.write_text(json.dumps(value, sort_keys=True))
    return paths


def test_formatting_synthetic_json_never_claims_native_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dowolne spójne JSON-y zaliczają wyłącznie format, bez wyniku natywnego PASS."""
    format_reports(*write_examples(tmp_path, *report_format_examples()))
    output = json.loads(capsys.readouterr().out)
    assert output["scope"] == "TEST_ONLY_REPORT_FORMAT"
    assert output["result"] == "FORMAT_CONSISTENT"
    assert output["native_status"] == "NOT_RUN"
    assert output["product_qualification"] == "BLOCKED"
    assert "PASS" not in json.dumps(output)


def test_complete_synthetic_fixture_only_satisfies_pure_validator() -> None:
    validate_examples(*report_format_examples())


@pytest.mark.parametrize("trial", [{}, {"scope": "INNY_ZAKRES"}])
def test_consistently_missing_trial_binding_is_still_refused(trial: dict[str, Any]) -> None:
    client, server, snapshot = report_format_examples()
    for record in (client, server, snapshot):
        record["trial"] = deepcopy(trial)
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError, match="Niepełne powiązanie zatwierdzonej próby"):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize(
    "damage",
    [
        "bytes",
        "eof",
        "pidfd",
        "executable",
        "uid",
        "gid",
        "qualification",
        "guard_calls",
        "guard_exception",
        "guard_record",
        "trace_missing",
        "snapshot_hash",
        "outcome",
        "reason",
        "unexpected_connection",
        "missing_connection",
    ],
)
def test_validator_rejects_false_positive_reports(damage: str) -> None:
    """Mutacje kompletnego fixture nie mogą uzyskać nawet zgodności raportów."""
    client, server, snapshot = report_format_examples()
    if damage == "bytes":
        server["connections"][1]["bytes"] = 1
    elif damage == "eof":
        server["connections"][0]["eof"] = False
    elif damage == "pidfd":
        client["socket_bound_pidfd"] = False
    elif damage == "executable":
        client["peer_process"]["exe_sha256"] = "0" * 64
        client["actual_server_guard"]["calls"][0]["exe_sha256"] = "0" * 64
    elif damage == "uid":
        client["uid"] = 1002
    elif damage == "gid":
        client["gid"] = 1003
    elif damage == "qualification":
        server["product_qualification"] = "PASS"
    elif damage == "guard_calls":
        client["actual_server_guard"]["calls"] = []
    elif damage == "guard_exception":
        client["actual_server_guard"]["exceptions"][0]["reason"] = "INNA_ODMOWA"
    elif damage == "guard_record":
        client["actual_server_guard"]["calls"][0]["pid"] = 303
    elif damage == "trace_missing":
        del client["actual_server_guard"]
    elif damage == "snapshot_hash":
        client["root_snapshot_sha256"] = "0" * 64
    elif damage == "outcome":
        client["outcome"] = "PREPARED"
    elif damage == "reason":
        client["reason"] = "AMBIGUOUS_EXCHANGE"
    elif damage == "unexpected_connection":
        server["connections"].append(deepcopy(server["connections"][0]))
    else:
        server["connections"].pop()
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize("report", ["client", "listener", "snapshot"])
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("scope", "INNY_ZAKRES"),
        ("trial_id", "12345678-1234-1234-1234-123456789aaa"),
        ("revision", "0" * 40),
        ("environment_id", "inny-runtime"),
        ("machine_id", "0" * 32),
        ("boot_id", "12345678-1234-1234-1234-123456789bbb"),
        ("vm_image_sha256", "0" * 64),
        ("manifest_sha256", "1" * 64),
    ],
)
def test_trial_binding_rejects_reports_from_another_attempt(
    report: str, field: str, replacement: str
) -> None:
    client, server, snapshot = report_format_examples()
    records = {"client": client, "listener": server, "snapshot": snapshot}
    records[report]["trial"][field] = replacement
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize("report", ["client", "listener", "snapshot"])
@pytest.mark.parametrize(
    "field",
    ["MainPID", "ExecMainPID", "ExecMainStartTimestampMonotonic", "ControlGroup", "ExecStart"],
)
def test_systemd_drift_rejects_combined_reports(report: str, field: str) -> None:
    client, server, snapshot = report_format_examples()
    records = {"client": client, "listener": server, "snapshot": snapshot}
    records[report]["systemd"][field] = "inne"
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize("report", ["client", "listener", "snapshot"])
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pid", 303),
        ("uid", 1004),
        ("gid", 1004),
        ("groups", [1003, 1004]),
        ("start_time_ticks", 11),
        ("exe_sha256", "0" * 64),
        ("cgroup", "/system.slice/inny.service"),
    ],
)
def test_process_drift_rejects_combined_reports(report: str, field: str, replacement: Any) -> None:
    client, server, snapshot = report_format_examples()
    records = {"client": client, "listener": server, "snapshot": snapshot}
    key = "peer_process" if report == "client" else "process"
    records[report][key][field] = replacement
    if report == "client":
        client["actual_server_guard"]["calls"][0] = deepcopy(client["peer_process"])
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize("report", ["client", "listener", "snapshot"])
@pytest.mark.parametrize("section", ["trial", "systemd", "process"])
def test_missing_binding_evidence_is_not_accepted(report: str, section: str) -> None:
    client, server, snapshot = report_format_examples()
    records = {"client": client, "listener": server, "snapshot": snapshot}
    key = "peer_process" if report == "client" and section == "process" else section
    del records[report][key]
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize(
    "field", ["pid", "uid", "gid", "groups", "start_time_ticks", "exe_sha256", "cgroup"]
)
def test_snapshot_requires_every_process_field(field: str) -> None:
    client, server, snapshot = report_format_examples()
    del snapshot["process"][field]
    client["root_snapshot_sha256"] = snapshot_digest(snapshot)
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)


@pytest.mark.parametrize("field", ["pid", "uid", "gid"])
def test_kernel_credentials_must_include_every_scalar(field: str) -> None:
    client, server, snapshot = report_format_examples()
    del client["peer_credentials"][field]
    with pytest.raises(ValueError):
        validate_examples(client, server, snapshot)
