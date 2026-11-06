"""Statyczny inwentarz S9-ID-01; bez obserwacji hosta i kwalifikacji bezpieczeństwa.

Sukces tych testów oznacza spójność dokumentu. Natywne przypadki N01–N18
pozostają NOT_RUN; testy nie wydają decyzji o tożsamości ani autorytecie.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs/architecture/stage9_verifier_identity_acceptance_contract_v1.md"
SCALARS = {
    "contract_id": "S9-ID-01/K4-v2",
    "schema_version": 1,
    "revision": "1.0.0",
    "scope": "TEST_ONLY",
    "active": False,
    "production_loadable": False,
    "implementation_authorized": False,
    "stage1_decision": "PENDING_OWNER_ACCEPT",
    "identity_status": "NOT_PROVEN",
    "authority_status": "BLOCKED",
    "stage9_status": "IN_PROGRESS / PRODUCTION_NO_GO",
    "candidate": "EXISTING_PYTHON",
    "base_commit": "d86eafbfd06696bdc298703bc154b2019807a26a",
    "predicate": "ALL_REQUIRED_EVIDENCE",
}
EVIDENCE = {
    "E01": {
        "independent_approval",
        "installation_manifest",
        "runtime_application_closure",
        "protected_bytes",
    },
    "E02": {
        "system_bus",
        "name_owner_credentials",
        "system_manager_identity",
        "manager_policy",
    },
    "E03": {
        "effective_unit",
        "exec_start_pre",
        "exec_start",
        "environment",
        "drop_ins",
        "no_delegation",
    },
    "E04": {
        "uids_gids_groups",
        "main_pid",
        "exec_main_pid",
        "cgroup",
        "start_ticks",
        "boot_id",
        "invocation_id",
    },
    "E05": {
        "connected_af_unix",
        "so_peercred",
        "socket_bound_so_peerpidfd",
        "pidfd_liveness",
    },
    "E06": {
        "socket_path",
        "ancestor_handles",
        "socket_inode",
        "owners_modes_acl",
        "namespace_mount_binding",
    },
    "E07": {
        "startup_policy",
        "protection_from_birth",
        "no_prior_tracer",
        "inheritance_inventory",
        "native_startup_qualification",
    },
    "E08": {
        "reviewed_startup_order",
        "seal_before_listen",
        "all_threads_abi",
        "no_exec_fork_fd_delegation",
        "native_continuity_qualification",
    },
    "E09": {
        "pre_payload_recheck",
        "post_response_recheck",
        "epoch_revocation",
        "deadline_no_retry",
        "independent_evidence_binding",
    },
}
MANDATORY_TESTS = {
    "identity": [f"N{number:02}" for number in range(1, 16)],
    "authority": [f"N{number:02}" for number in range(16, 19)],
}
OWNER_DECISIONS = [f"D{number:02}" for number in range(1, 9)]
FORBIDDEN_SUBSTITUTES = [
    "SERVER_HASH",
    "READY_FILE",
    "SECCOMP_MODE_ONLY",
    "PIDFD_OPEN_FALLBACK",
    "EXPECTED_AS_OBSERVED",
]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Powtórzone pole inwentarza dokumentacji")
        result[name] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("Niedopuszczalna stała JSON")


def _parse_inventory(raw: str) -> dict:
    return json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def _read_inventory() -> dict:
    blocks = re.findall(r"^```json\n(.*?)^```$", DOCUMENT.read_text(encoding="utf-8"), re.M | re.S)
    assert len(blocks) == 1
    return _parse_inventory(blocks[0])


def _validate_inventory(inventory: dict) -> None:
    """Sprawdzaj wyłącznie zamknięty schemat danych dokumentu, nie dowody hosta."""
    assert type(inventory) is dict
    assert set(inventory) == set(SCALARS) | {
        "evidence",
        "mandatory_tests",
        "owner_decisions",
        "forbidden_substitutes",
    }
    for name, expected in SCALARS.items():
        assert type(inventory[name]) is type(expected)
        assert inventory[name] == expected

    evidence = inventory["evidence"]
    assert type(evidence) is dict and set(evidence) == set(EVIDENCE)
    for evidence_id, expected_fields in EVIDENCE.items():
        fields = evidence[evidence_id]
        assert type(fields) is list
        assert all(type(field) is str for field in fields)
        assert len(fields) == len(set(fields))
        assert set(fields) == expected_fields

    mandatory = inventory["mandatory_tests"]
    assert type(mandatory) is dict and set(mandatory) == set(MANDATORY_TESTS)
    for scope, expected_ids in MANDATORY_TESTS.items():
        assert type(mandatory[scope]) is list and mandatory[scope] == expected_ids
    assert type(inventory["owner_decisions"]) is list
    assert inventory["owner_decisions"] == OWNER_DECISIONS
    assert type(inventory["forbidden_substitutes"]) is list
    assert inventory["forbidden_substitutes"] == FORBIDDEN_SUBSTITUTES


def test_document_has_complete_inactive_test_only_inventory():
    _validate_inventory(_read_inventory())


@pytest.mark.parametrize(
    "name", [*SCALARS, "evidence", "mandatory_tests", "owner_decisions", "forbidden_substitutes"]
)
def test_missing_top_level_field_is_rejected(name):
    inventory = _read_inventory()
    del inventory[name]
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize("evidence_id", EVIDENCE)
def test_each_missing_evidence_group_is_rejected(evidence_id):
    inventory = _read_inventory()
    del inventory["evidence"][evidence_id]
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize(
    ("evidence_id", "field"),
    [(evidence_id, field) for evidence_id, fields in EVIDENCE.items() for field in sorted(fields)],
)
def test_each_missing_evidence_field_is_rejected(evidence_id, field):
    inventory = _read_inventory()
    inventory["evidence"][evidence_id].remove(field)
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize(
    ("scope", "test_id"),
    [(scope, test_id) for scope, test_ids in MANDATORY_TESTS.items() for test_id in test_ids],
)
def test_each_missing_mandatory_native_test_is_rejected(scope, test_id):
    inventory = _read_inventory()
    inventory["mandatory_tests"][scope].remove(test_id)
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize("decision_id", OWNER_DECISIONS)
def test_each_missing_owner_decision_is_rejected(decision_id):
    inventory = _read_inventory()
    inventory["owner_decisions"].remove(decision_id)
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize("substitute", FORBIDDEN_SUBSTITUTES)
def test_each_missing_substitute_prohibition_is_rejected(substitute):
    inventory = _read_inventory()
    inventory["forbidden_substitutes"].remove(substitute)
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", "1"),
        ("schema_version", 2),
        ("revision", "1.0.1"),
        ("active", True),
        ("active", 0),
        ("production_loadable", True),
        ("implementation_authorized", True),
        ("stage1_decision", "ACCEPT"),
        ("identity_status", "IDENTITY_ACCEPT"),
        ("authority_status", "AUTHORITY_ACCEPT"),
        ("scope", "PRODUCTION"),
        ("candidate", "RUST"),
        ("predicate", "ANY_EVIDENCE"),
    ],
)
def test_unsupported_version_or_activation_is_rejected(name, value):
    inventory = _read_inventory()
    inventory[name] = value
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize("location", ["root", "evidence", "mandatory_tests"])
def test_unknown_fields_are_rejected(location):
    inventory = _read_inventory()
    target = inventory if location == "root" else inventory[location]
    target["unknown"] = []
    with pytest.raises(AssertionError):
        _validate_inventory(inventory)


@pytest.mark.parametrize("evidence_id", EVIDENCE)
def test_duplicate_or_unknown_evidence_fields_are_rejected(evidence_id):
    original = _read_inventory()
    for extra in (original["evidence"][evidence_id][0], "unknown_evidence"):
        inventory = deepcopy(original)
        inventory["evidence"][evidence_id].append(extra)
        with pytest.raises(AssertionError):
            _validate_inventory(inventory)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"evidence":{"E01":[],"E01":[]}}',
        '{"mandatory_tests":{"identity":[],"identity":[]}}',
        '{"schema_version":NaN}',
        '{"schema_version":Infinity}',
        '{"schema_version":-Infinity}',
    ],
)
def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(raw):
    with pytest.raises(ValueError):
        _parse_inventory(raw)


def test_every_inventory_id_has_one_normative_section_or_table_row():
    document = DOCUMENT.read_text(encoding="utf-8")
    evidence_ids = re.findall(r"^### (E\d{2}) — ", document, re.M)
    assert len(evidence_ids) == len(set(evidence_ids))
    assert set(evidence_ids) == set(EVIDENCE)
    test_ids = re.findall(r"^\| (N\d{2}) \|", document, re.M)
    assert test_ids == [test_id for test_ids in MANDATORY_TESTS.values() for test_id in test_ids]
    decision_ids = re.findall(r"^\| (D\d{2}) \|", document, re.M)
    assert decision_ids == OWNER_DECISIONS


def test_relative_document_links_resolve_inside_repository():
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", DOCUMENT.read_text(encoding="utf-8"))
    assert links
    for link in links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (DOCUMENT.parent / unquote(parsed.path)).resolve()
        assert target.is_relative_to(ROOT)
        assert target.is_file(), link
