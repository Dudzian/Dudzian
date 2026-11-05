"""Wyłącznie format i odmowy spreparowanych raportów; bez kwalifikacji jądra."""

from __future__ import annotations

import errno
import json
from pathlib import Path
import sys
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("Raport próby odwołuje się do boot ID Linux.", allow_module_level=True)

from scripts.diagnostics.stage9_cross_uid_identity import boot_id, compare


def report_format_examples() -> tuple[dict[str, Any], dict[str, Any]]:
    """Syntetyczne dane parsera; numery nie są obserwacją UID/PID ani testem klienta."""
    client: dict[str, Any] = {
        "scope": "TEST_ONLY_CLIENT_DENIAL",
        "boot_id": boot_id(),
        "pid": 101,
        "uid": 1001,
        "gid": 1001,
        "peer_credentials": {"pid": 202, "uid": 1002, "gid": 1002},
        "peer_process": {"pid": 202, "uid": 1002, "gid": 1002, "exe_sha256": None},
        "exe_errno": errno.EACCES,
        "server_denial": "APPROVED_VERIFIER_PROCESS_REQUIRED",
        "server_denial_errno": None,
        "socket_bound_pidfd": True,
        "outcome": "UNAVAILABLE",
        "reason": "TRANSPORT_OR_IDENTITY_UNAVAILABLE",
        "product_qualification": "BLOCKED",
    }
    client["actual_server_guard"] = {
        "calls": [dict(client["peer_process"])],
        "exceptions": [
            {
                "type": "PermissionError",
                "reason": "APPROVED_VERIFIER_PROCESS_REQUIRED",
                "errno": None,
            }
        ],
    }
    server: dict[str, Any] = {
        "scope": "TEST_ONLY_LISTENER",
        "boot_id": boot_id(),
        "pid": 202,
        "connections": [
            {"pid": 101, "uid": 1001, "gid": 1001, "bytes": 0, "eof": True},
            {"pid": 101, "uid": 1001, "gid": 1001, "bytes": 0, "eof": True},
        ],
        "preparation_operation": "NIE_ISTNIEJE_W_LISTENERZE",
        "product_qualification": "BLOCKED",
    }
    return client, server


def write_examples(
    tmp_path: Path, client: dict[str, Any], server: dict[str, Any]
) -> tuple[Path, Path]:
    client_path = tmp_path / "client.json"
    server_path = tmp_path / "server.json"
    client_path.write_text(json.dumps(client))
    server_path.write_text(json.dumps(server))
    return client_path, server_path


def test_comparison_accepts_consistent_format_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Kontrola formatu bazowego zapobiega pozornym sukcesom testów uszkodzeń."""
    compare(*write_examples(tmp_path, *report_format_examples()))
    output = json.loads(capsys.readouterr().out)
    assert output["scope"] == "TEST_ONLY_NATIVE_DENIAL"
    assert output["product_qualification"] == "BLOCKED"
    assert output["postgresql_no_preparation_proof"] == "NOT_RUN"


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("bytes", "Połączenia przesłały dane"),
        ("eof", "Połączenia przesłały dane"),
        ("pidfd", "Brak dowodu socket-bound pidfd"),
        ("executable", "Brak dowodu odmowy procfs"),
        ("uid", "Raporty nie dotyczą odrębnych UID/GID"),
        ("qualification", "Raport rozszerza zakres kwalifikacji"),
        ("guard_calls", "Brak dokładnego wywołania kontroli serwera"),
        ("guard_exception", "Brak konkretnej odmowy kontroli serwera"),
        ("guard_record", "Brak dokładnego wywołania kontroli serwera"),
        ("trace_missing", "Brak rejestru wykonania klienta"),
    ],
)
def test_comparison_rejects_false_positive_reports(
    tmp_path: Path, damage: str, reason: str
) -> None:
    """Każda mutacja musi otrzymać własną odmowę, a nie błąd niepełnego fixture."""
    client, server = report_format_examples()
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
    elif damage == "qualification":
        server["product_qualification"] = "PASS"
    elif damage == "guard_calls":
        client["actual_server_guard"]["calls"] = []
    elif damage == "guard_exception":
        client["actual_server_guard"]["exceptions"][0]["reason"] = "INNA_ODMOWA"
    elif damage == "guard_record":
        client["actual_server_guard"]["calls"][0]["pid"] = 303
    else:
        del client["actual_server_guard"]
    with pytest.raises(ValueError, match=reason):
        compare(*write_examples(tmp_path, client, server))
