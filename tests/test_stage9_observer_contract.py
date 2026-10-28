"""Rootless synthetic contract tests; no artifacts or host processes execute."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_contract as observer
from deployment.canonical_json import canonical_json_bytes

ContractRefusal = observer.ContractRefusal
_FIXTURES = Path(__file__).parent / "fixtures" / "stage9_observer"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_bytes())


def _wire(value: Any) -> bytes:
    # json.dumps permits intentionally invalid numeric variants for negative tests.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def validate_launch_policy(value):
    observer.parse_launch_policy(_wire(value))


def _validate_scope_binding(value):
    observer.parse_scope_binding(_wire(value))


def _validate_run_context(value):
    observer.parse_run_context(_wire(value))


def _validate_offline_result(value):
    observer.parse_offline_result(_wire(value))


@pytest.fixture
def launch_policy():
    return _fixture("launch_policy")


@pytest.fixture
def scope_fixture():
    return _fixture("scope_binding")


@pytest.fixture
def run_fixture():
    return _fixture("run_context")


@pytest.fixture
def offline_fixture():
    return _fixture("offline_result")


@pytest.fixture
def acceptance():
    return _fixture("acceptance")


def _rebind_acceptance(value):
    value["payload_digest"] = hashlib.sha256(canonical_json_bytes(value["payload"])).hexdigest()


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "interpreter",
        "elf",
        "imports",
        "environment",
        "cwd",
        "namespace_scope",
        "sandbox",
        "file_policy",
        "runtime_dependencies",
        "resources",
        "reporting",
        "run_context",
    ],
)
def test_launch_nested_fields_closed(launch_policy, field):
    launch_policy[field]["unexpected"] = "secret-marker"
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "interpreter",
        "elf",
        "imports",
        "environment",
        "cwd",
        "namespace_scope",
        "sandbox",
        "file_policy",
        "runtime_dependencies",
        "resources",
        "reporting",
        "run_context",
    ],
)
def test_launch_nested_fields_required(launch_policy, field):
    del launch_policy[field][next(iter(launch_policy[field]))]
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("environment", "inherit", True),
        ("interpreter", "version", "3.13.0"),
        ("interpreter", "version", "3.12"),
        ("interpreter", "version", "3.12.03"),
        ("interpreter", "frozen_modules", "off"),
        ("interpreter", "package_version", "3.12.*"),
        ("namespace_scope", "pid", "PRIVATE"),
        ("namespace_scope", "user", "PRIVATE"),
        ("namespace_scope", "visibility", "INCOMPLETE"),
        ("sandbox", "uid", 0),
        ("sandbox", "uid", True),
        ("sandbox", "no_new_privileges", False),
        ("sandbox", "network", "ALLOW"),
        ("sandbox", "devices", "ALLOW"),
        ("file_policy", "regular_nlink", True),
        ("file_policy", "regular_nlink", 2),
        ("resources", "fd_limit", 7),
        ("resources", "fd_limit", 65),
        ("resources", "fd_limit", True),
        ("resources", "memory_max_bytes", 268435457),
        ("resources", "run_timeout_s", 301),
        ("resources", "receive_timeout_s", 31),
        ("resources", "process_limit", 2),
        ("resources", "core_dump_bytes", 1),
        ("reporting", "fd", 4),
        ("reporting", "max_payload_bytes", 65537),
        ("reporting", "eof", "OPTIONAL"),
        ("run_context", "fd", 3),
        ("run_context", "max_bytes", 16385),
        ("run_context", "source", "OPERATOR"),
    ],
)
def test_launch_forbidden_variants(launch_policy, record, field, value):
    launch_policy[record][field] = value
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_extra_environment_variable(launch_policy):
    launch_policy["environment"]["values"]["PYTHONPATH"] = "/operator"
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


@pytest.mark.parametrize("field", ["roots"])
def test_launch_duplicate_import_root(launch_policy, field):
    launch_policy["imports"][field] *= 2
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_missing_deny_operation(launch_policy):
    launch_policy["sandbox"]["deny_operations"].pop()
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_duplicate_deny_operation(launch_policy):
    launch_policy["sandbox"]["deny_operations"][-1] = "NEW_EXEC"
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_unordered_fds(launch_policy):
    launch_policy["allowed_fds"].reverse()
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_unordered_seals(launch_policy):
    launch_policy["run_context"]["seals"].reverse()
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


def test_launch_dependency_projection_substitution(launch_policy):
    launch_policy["runtime_dependencies"]["imports_sha256"] = "a" * 64
    with pytest.raises(ContractRefusal):
        validate_launch_policy(launch_policy)


@pytest.mark.parametrize("fd_limit", [8, 64])
def test_launch_fd_boundary(launch_policy, fd_limit):
    launch_policy["resources"]["fd_limit"] = fd_limit
    validate_launch_policy(launch_policy)


def test_scope_synthetic_binding_does_not_create_authority(scope_fixture):
    assert _validate_scope_binding(scope_fixture) is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema", "HostScopeBindingV2"),
        ("version", True),
        ("version", 2),
        ("host_anchor_id", ""),
        ("host_anchor_id", "x" * 129),
        ("source", "nie-ascii-ł"),
        ("boot_id", "12345678-1234-1234-ABCD-123456789ABC"),
        ("boot_id", "{12345678-1234-1234-abcd-123456789abc}"),
        ("run_nonce", "ab" * 31),
        ("run_nonce", [1] * 32),
        ("observer_start_time_ticks", 0),
        ("observer_start_time_ticks", True),
        ("observer_start_time_ticks", 2**53),
        ("authentication_evidence_sha256", "0" * 64),
        ("authentication_evidence_sha256", "SHA256:" + "a" * 64),
    ],
)
def test_scope_rejects_invalid_top_level(scope_fixture, field, replacement):
    scope_fixture[field] = replacement
    with pytest.raises(observer.ContractRefusal):
        _validate_scope_binding(scope_fixture)


@pytest.mark.parametrize("namespace", ["pid_namespace", "user_namespace", "mount_namespace"])
@pytest.mark.parametrize(
    "replacement",
    [
        {"device": True, "inode": 2},
        {"device": 1, "inode": 0},
        {"device": -1, "inode": 2},
        {"device": 1, "inode": 2**53},
        {"device": 1, "inode": 2, "host_initial": True},
        {"inode": 2},
    ],
)
def test_namespace_identity_is_closed_and_bounded(scope_fixture, namespace, replacement):
    scope_fixture[namespace] = replacement
    with pytest.raises(observer.ContractRefusal):
        _validate_scope_binding(scope_fixture)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("mount_id", 0),
        ("mount_id", False),
        ("fs_type", "overlay"),
        ("root", "/proc"),
        ("pid_view", "CONTAINER_INITIAL_COMPLETE"),
        ("mount_namespace", {"device": 1, "inode": 99}),
        ("options", ["ro", "rw"]),
        ("options", ["nodev"]),
        ("options", ["hidepid=2", "ro"]),
        ("options", ["ro", "ro"]),
        ("options", ["ro", "nodev"]),
        ("options", ["ro", "secret-marker"]),
    ],
)
def test_procfs_identity_cannot_claim_arbitrary_host_view(scope_fixture, field, replacement):
    scope_fixture["procfs_identity"][field] = replacement
    with pytest.raises(observer.ContractRefusal):
        _validate_scope_binding(scope_fixture)


def test_scope_numeric_boundaries(scope_fixture):
    scope_fixture["pid_namespace"] = {"device": 0, "inode": 2**53 - 1}
    scope_fixture["user_namespace"] = {"device": 2**53 - 1, "inode": 1}
    scope_fixture["observer_start_time_ticks"] = 2**53 - 1
    scope_fixture["procfs_identity"]["mount_id"] = 2**53 - 1
    scope_fixture["procfs_identity"]["options"] = ["rw"]
    assert _validate_scope_binding(scope_fixture) is None


def test_run_scope_is_bound_to_nonce_and_canonical_digest(run_fixture):
    assert _validate_run_context(run_fixture) is None
    run_fixture["run_nonce"] = "cd" * 32
    with pytest.raises(observer.ContractRefusal):
        _validate_run_context(run_fixture)


def test_run_scope_mutation_preserves_no_stale_digest_acceptance(run_fixture):
    run_fixture["scope_binding"]["observer_start_time_ticks"] += 1
    with pytest.raises(observer.ContractRefusal):
        _validate_run_context(run_fixture)


def test_run_scope_digest_is_not_hash_of_noncanonical_text(run_fixture):
    run_fixture["scope_binding_sha256"] = hashlib.sha256(
        b" " + canonical_json_bytes(run_fixture["scope_binding"])
    ).hexdigest()
    with pytest.raises(observer.ContractRefusal):
        _validate_run_context(run_fixture)


@pytest.mark.parametrize("field", ["scope_binding", "authentication_evidence_sha256", "run_nonce"])
def test_run_refuses_missing_required_material(run_fixture, field):
    del run_fixture[field]
    with pytest.raises(observer.ContractRefusal):
        _validate_run_context(run_fixture)


def test_offline_result_synthetic_only_keeps_all_blocks(offline_fixture):
    assert _validate_offline_result(offline_fixture) is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("synthetic", False),
        ("synthetic", 1),
        ("acceptance", "VERIFIED"),
        ("full_confirmation", True),
        ("coexistence_authorized", True),
        ("preflight_exception", True),
        ("qualification", "PASS"),
        ("physical_test_status", "PASS"),
        ("production_status", "GO"),
        ("runtime_attestation", "VERIFIED"),
        ("host_pid_namespace", "VERIFIED"),
        ("manifest_sha256", "0" * 64),
        ("artifact_consistency", "SUCCESS"),
        ("reason_codes", []),
        ("reason_codes", ["RUNTIME_LAUNCH_UNVERIFIED", "secret-marker"]),
        ("reason_codes", ["RUNTIME_LAUNCH_UNVERIFIED"] * 65),
    ],
)
def test_offline_result_cannot_upgrade_qualification(offline_fixture, field, replacement):
    offline_fixture[field] = replacement
    with pytest.raises(observer.ContractRefusal):
        _validate_offline_result(offline_fixture)


def _scope_frame(payload=b"{}"):
    return len(payload).to_bytes(4, "big") + payload


def _decode_scope_frame(raw, **updates):
    parameters = {
        "eof": True,
        "process_exit_code": 0,
        "run_elapsed_ms": 30000,
        "receive_elapsed_ms": 30000,
    }
    parameters.update(updates)
    return observer.decode_json(observer.decode_synthetic_frame(raw, **parameters))


def test_frame_single_payload_with_eof_at_receive_deadline():
    assert _decode_scope_frame(_scope_frame()) == {}


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x00",
        b"\x00\x00\x00",
        b"\x00\x00\x00\x00",
        b"\x00\x00\x00\x03{}",
        b"\x00\x00\x00\x01{}",
        b"\x00\x01\x00\x01{}",
        b"\xff\xff\xff\xff{}",
        _scope_frame() + b"x",
        _scope_frame() + _scope_frame(),
        _scope_frame(b" " * 65537),
    ],
)
def test_frame_refuses_truncation_extra_bytes_or_bad_length(raw):
    with pytest.raises(observer.ContractRefusal) as refused:
        _decode_scope_frame(raw)
    assert str(refused.value) == "CHANNEL_REJECTED"


@pytest.mark.parametrize(
    "updates",
    [
        {"eof": False},
        {"eof": 1},
        {"process_exit_code": 1},
        {"process_exit_code": -9},
        {"process_exit_code": False},
        {"run_elapsed_ms": 300000},
        {"run_elapsed_ms": -1},
        {"run_elapsed_ms": True},
        {"receive_elapsed_ms": 30001},
        {"receive_elapsed_ms": -1},
        {"receive_elapsed_ms": 1.5},
        {"run_elapsed_ms": 1, "receive_elapsed_ms": 2},
    ],
)
def test_frame_requires_successful_end_and_bounded_deadlines(updates):
    with pytest.raises(observer.ContractRefusal) as refused:
        _decode_scope_frame(_scope_frame(), **updates)
    assert str(refused.value) == "CHANNEL_REJECTED"


@pytest.mark.parametrize("payload", [b"\xff", b"\xef\xbb\xbf{}", b'{"x":1,"x":2}', b"[]"])
def test_frame_still_requires_strict_json_object(payload):
    with pytest.raises(observer.ContractRefusal):
        _decode_scope_frame(_scope_frame(payload))


def test_frame_maximum_payload_boundary():
    # A valid object with leading whitespace exercises exact wire byte count.
    assert _decode_scope_frame(_scope_frame(b" " * (65536 - 2) + b"{}")) == {}


def test_nested_scope_unknown_field_cannot_be_ignored(run_fixture):
    mutated = copy.deepcopy(run_fixture)
    mutated["scope_binding"]["procfs_identity"]["extra"] = "secret-marker"
    with pytest.raises(observer.ContractRefusal):
        _validate_run_context(mutated)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b" ",
        b"\xff",
        b"\xc0\xaf",
        b"\xef\xbb\xbf{}",
        b'{"x":1,"x":2}',
        b'{"x":{"y":1,"y":2}}',
        b'{"x":1,"\\u0078":2}',
        b'{"x":null}',
        b'{"x":1.0}',
        b'{"x":1e0}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":-1}',
        b'{"x":-0}',
        b'{"x":9007199254740992}',
        b'{"x":"\\ud800"}',
        b'{"x":"\\udfff"}',
        b'{"\\ud800":1}',
        b'{"x":"\\u0000"}',
        b'{"\\u0000":1}',
        b'{"x":"\x00"}',
        b'{"x":01}',
        b'{"x":true} trailing',
        b'{"x":}',
        b'{"x":"unterminated}',
        b'{"x":"bad\\q"}',
        b"[}",
    ],
)
def test_decoder_rejects_non_v1_json(raw):
    with pytest.raises(ContractRefusal) as refused:
        observer.decode_json(raw)
    assert refused.value.code in observer.REASON_CODES


@pytest.mark.parametrize("value", [None, {}, [], "{}", bytearray(b"{}"), memoryview(b"{}")])
def test_decoder_requires_exact_bytes(value):
    with pytest.raises(ContractRefusal):
        observer.decode_json(value)


def test_decoder_does_not_coerce_boolean_or_integer():
    decoded = observer.decode_json(b'{"integer":1,"boolean":true}')
    assert type(decoded["integer"]) is int
    assert type(decoded["boolean"]) is bool


def test_decoder_unescaped_unicode_and_json_escapes_agree():
    assert observer.decode_json('{"x":"ł"}'.encode()) == observer.decode_json(b'{"x":"\\u0142"}')


def test_decoder_depth_exact_boundary():
    assert observer.decode_json(b"[" * 16 + b"0" + b"]" * 16)


@pytest.mark.parametrize(
    "raw",
    [
        b"[" * 17 + b"0" + b"]" * 17,
        b" " * (observer.MAX_MANIFEST_BYTES + 1),
        b"[" + b"0," * observer.MAX_FILES + b"0]",
        b'{"x":' + b"9" * 4096 + b"}",
    ],
    ids=["depth", "document-bytes", "container-cardinality", "integer-token"],
)
def test_decoder_limits_precede_json_allocation(monkeypatch, raw):
    calls = []

    def decoder_must_not_run(*args, **kwargs):
        calls.append(True)
        raise AssertionError("JSON allocation happened before the lexical limit")

    monkeypatch.setattr(observer.json, "loads", decoder_must_not_run)
    with pytest.raises(ContractRefusal) as refused:
        observer.decode_json(raw)
    assert refused.value.code == "LIMIT_EXCEEDED"
    assert calls == []


def test_decoder_document_byte_boundary():
    raw = b" " * (observer.MAX_MANIFEST_BYTES - 2) + b"{}"
    assert observer.decode_json(raw) == {}


def test_decoder_container_cardinality_boundary():
    raw = b"[" + b",".join([b"0"] * observer.MAX_FILES) + b"]"
    assert len(observer.decode_json(raw)) == observer.MAX_FILES


@pytest.mark.parametrize("value", [0, 2**53 - 1])
def test_exact_json_integer_boundaries(value):
    assert observer.decode_json(str(value).encode()) == value


@pytest.mark.parametrize(
    "replacement",
    [
        "",
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "0" * 64,
        "sha256:" + "a" * 64,
        " " + "a" * 64,
        "a" * 64 + "\n",
        "a" * 63 + "g",
        [1] * 32,
        None,
        True,
        123,
    ],
)
def test_acceptance_digest_representation_is_exact(acceptance, replacement):
    acceptance["payload"]["manifest_sha256"] = replacement
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "version",
        "purpose",
        "manifest_sha256",
        "policy_sha256",
        "root_set_id",
        "sequence",
        "revocation_checkpoint",
        "issued_at",
        "expires_at",
    ],
)
def test_acceptance_payload_requires_every_field(acceptance, field):
    del acceptance["payload"][field]
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


@pytest.mark.parametrize("record", ["envelope", "payload", "signature", "checkpoint"])
def test_acceptance_all_nested_records_are_closed(acceptance, record):
    targets = {
        "envelope": acceptance,
        "payload": acceptance["payload"],
        "signature": acceptance["signatures"][0],
        "checkpoint": acceptance["payload"]["revocation_checkpoint"],
    }
    targets[record]["unexpected"] = "secret-marker"
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


@pytest.mark.parametrize("replacement", [0, 1, 3, True, False, 2.0, "2"])
def test_acceptance_threshold_is_exactly_integer_two(acceptance, replacement):
    acceptance["threshold"] = replacement
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("schema", "CryptoHunter.WindowsAcceptanceV1"),
        ("version", 2),
        ("version", True),
        ("purpose", "PRODUCTION"),
        ("sequence", True),
        ("sequence", -1),
        ("root_set_id", "x" * 129),
        ("root_set_id", "ł"),
        ("issued_at", "2026-02-30T00:00:00Z"),
        ("issued_at", "2026-10-27T00:00:00+00:00"),
        ("issued_at", "2026-10-28T00:00:00Z"),
        ("expires_at", "2026-10-26T00:00:00Z"),
    ],
)
def test_acceptance_known_fields_fail_closed(acceptance, field, replacement):
    acceptance["payload"][field] = replacement
    _rebind_acceptance(acceptance)
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


def test_acceptance_single_field_substitution_invalidates_payload_digest(acceptance):
    acceptance["payload"]["root_set_id"] = "different-root"
    with pytest.raises(ContractRefusal) as refused:
        observer.parse_acceptance(_wire(acceptance))
    assert refused.value.code == "ACCEPTANCE_REJECTED"


def test_acceptance_payload_digest_is_not_entire_envelope_digest(acceptance):
    acceptance["payload_digest"] = hashlib.sha256(_wire(acceptance)).hexdigest()
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


def test_acceptance_requires_canonical_envelope_without_changing_payload_digest(acceptance):
    raw = json.dumps(acceptance, indent=2).encode()
    digest = hashlib.sha256(canonical_json_bytes(acceptance["payload"])).hexdigest()
    assert digest == acceptance["payload_digest"]
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(raw)
    with pytest.raises(ContractRefusal):
        observer.acceptance_signed_message(raw)
    canonical = canonical_json_bytes(acceptance)
    parsed = observer.parse_acceptance(canonical)
    assert parsed.raw == parsed.canonical == canonical
    assert observer.acceptance_signed_message(
        canonical
    ) == observer.ACCEPTANCE_DOMAIN + bytes.fromhex(digest)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unsorted", "missing", "bad_hex", "short_hex", "unknown_signer_field"]
)
def test_acceptance_signature_inventory_is_strict(acceptance, mutation):
    if mutation == "duplicate":
        acceptance["signatures"][1]["signer_id"] = acceptance["signatures"][0]["signer_id"]
    elif mutation == "unsorted":
        acceptance["signatures"].reverse()
    elif mutation == "missing":
        acceptance["signatures"].pop()
    elif mutation == "bad_hex":
        acceptance["signatures"][0]["signature_hex"] = "AA" * 64
    elif mutation == "short_hex":
        acceptance["signatures"][0]["signature_hex"] = "aa" * 63
    else:
        acceptance["signatures"][0]["public_key_hex"] = "11" * 32
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance(_wire(acceptance))


def test_acceptance_domain_has_one_nul_and_raw_digest_bytes(acceptance):
    message = observer.acceptance_signed_message(_wire(acceptance))
    assert message == b"CryptoHunter.Stage9ObserverAcceptanceV1\x00" + bytes.fromhex(
        acceptance["payload_digest"]
    )
    assert len(message) == len(observer.ACCEPTANCE_DOMAIN) + 32
    assert not message.endswith(acceptance["payload_digest"].encode())


def test_syntactically_valid_signatures_never_authorize_acceptance(acceptance):
    parsed = observer.parse_acceptance(_wire(acceptance))
    assert acceptance["signatures"][0]["signature_hex"] == "0" * 128
    assert parsed.authorization == "UNVERIFIED"


@pytest.mark.parametrize(
    "mutation",
    [
        "signer",
        "reviewer",
        "key",
        "unsorted",
        "extra",
        "missing",
        "short_key",
        "uppercase_key",
        "one",
    ],
)
def test_reviewer_keys_cannot_count_duplicate_identity_or_key(mutation):
    keys = _fixture("refused_policy")["reviewer_keys"]
    if mutation in {"signer", "reviewer", "key"}:
        field = {"signer": "signer_id", "reviewer": "reviewer_id", "key": "public_key_hex"}[
            mutation
        ]
        keys[1][field] = keys[0][field]
    elif mutation == "unsorted":
        keys.reverse()
    elif mutation == "extra":
        keys[0]["unknown"] = True
    elif mutation == "missing":
        del keys[0]["reviewer_id"]
    elif mutation == "short_key":
        keys[0]["public_key_hex"] = "11" * 31
    elif mutation == "uppercase_key":
        keys[0]["public_key_hex"] = "AA" * 32
    else:
        keys.pop()
    with pytest.raises(ContractRefusal):
        observer._reviewer_keys(keys)


def test_reviewer_key_fragment_does_not_authorize_policy():
    keys = _fixture("refused_policy")["reviewer_keys"]
    assert observer._reviewer_keys(keys) == keys
    with pytest.raises(ContractRefusal):
        observer.parse_acceptance_policy((_FIXTURES / "refused_policy.json").read_bytes())


@pytest.mark.parametrize(
    "name,parser",
    [
        ("acceptance", observer.parse_acceptance),
        ("launch_policy", observer.parse_launch_policy),
        ("scope_binding", observer.parse_scope_binding),
        ("run_context", observer.parse_run_context),
        ("offline_result", observer.parse_offline_result),
    ],
)
def test_parsed_contract_is_immutable_and_never_authorizing(name, parser):
    raw = (_FIXTURES / f"{name}.json").read_bytes()
    parsed = parser(raw)
    assert parsed.authorization == "UNVERIFIED"
    assert parsed.raw == raw
    assert type(parsed.canonical) is bytes
    assert parsed.sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FrozenInstanceError):
        parsed.authorization = "VERIFIED"
    document = parsed.as_dict()
    document.clear()
    assert parsed.as_dict() == _fixture(name)


@pytest.mark.parametrize(
    "name,parser",
    [
        ("launch_policy", observer.parse_launch_policy),
        ("scope_binding", observer.parse_scope_binding),
        ("run_context", observer.parse_run_context),
    ],
)
def test_canonical_documents_are_rejected_without_normalizing(name, parser):
    document = _fixture(name)
    for raw in (
        b" " + _wire(document),
        _wire(document) + b"\n",
        json.dumps(document, indent=2).encode(),
    ):
        with pytest.raises(ContractRefusal):
            parser(raw)


@pytest.mark.parametrize(
    "name,parser,maximum",
    [
        ("acceptance", observer.parse_acceptance, 65536),
        ("launch_policy", observer.parse_launch_policy, 65536),
        ("scope_binding", observer.parse_scope_binding, 65536),
        ("run_context", observer.parse_run_context, 16384),
        ("offline_result", observer.parse_offline_result, 65536),
    ],
)
def test_contract_specific_byte_limits_precede_json_decode(monkeypatch, name, parser, maximum):
    def decoder_must_not_run(*args, **kwargs):
        raise AssertionError("unbounded JSON decoding")

    monkeypatch.setattr(observer.json, "loads", decoder_must_not_run)
    with pytest.raises(ContractRefusal) as refused:
        parser(b" " * (maximum + 1))
    assert refused.value.code == "LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("run_id", "other-run"),
        ("run_nonce", "cd" * 32),
        ("observer_manifest_sha256", hashlib.sha256(b"other-manifest").hexdigest()),
        ("policy_sha256", hashlib.sha256(b"other-policy").hexdigest()),
        ("launch_policy_sha256", hashlib.sha256(b"other-launch").hexdigest()),
        ("authentication_evidence_sha256", hashlib.sha256(b"other-evidence").hexdigest()),
    ],
)
def test_synthetic_run_comparison_rejects_crossrun_tampering(run_fixture, field, replacement):
    expected = _wire(run_fixture)
    run_fixture[field] = replacement
    if field == "run_nonce":
        run_fixture["scope_binding"]["run_nonce"] = replacement
        run_fixture["scope_binding_sha256"] = hashlib.sha256(
            canonical_json_bytes(run_fixture["scope_binding"])
        ).hexdigest()
    with pytest.raises(ContractRefusal):
        observer.compare_synthetic_run_context(_wire(run_fixture), expected)


def test_synthetic_run_comparison_never_authenticates_its_own_input(run_fixture):
    raw = _wire(run_fixture)
    assert observer.compare_synthetic_run_context(raw, raw).authorization == "UNVERIFIED"


@pytest.mark.parametrize(
    "path",
    ["relative", "/x/../y", "/x/./y", "/x//y", "/x/", "//x", "/x\\y", "/x\ty", "/" + "x" * 4096],
)
def test_launch_rejects_noncanonical_or_oversized_paths(launch_policy, path):
    launch_policy["interpreter"]["argv"][0] = path
    with pytest.raises(ContractRefusal):
        observer.parse_launch_policy(_wire(launch_policy))


@pytest.mark.parametrize("length", [127, 128])
def test_identifier_ascii_size_boundaries(scope_fixture, length):
    scope_fixture["host_anchor_id"] = "x" * length
    observer.parse_scope_binding(_wire(scope_fixture))


@pytest.mark.parametrize(
    "name,parser",
    [
        ("refused_manifest", observer.parse_manifest),
        ("refused_policy", observer.parse_acceptance_policy),
        ("refused_result", observer.parse_observer_result),
    ],
)
def test_underspecified_contracts_are_never_accepted(name, parser):
    with pytest.raises(ContractRefusal) as refused:
        parser((_FIXTURES / f"{name}.json").read_bytes())
    assert refused.value.code in observer.REASON_CODES


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("full_confirmation", True),
        ("production_status", "GO"),
        ("completeness", "COMPLETE"),
        ("runtime_attestation", "VERIFIED"),
        ("host_pid_namespace", "VERIFIED"),
        ("coexistence_authorized", True),
        ("preflight_exception", True),
    ],
)
def test_underspecified_result_cannot_upgrade_any_safety_block(field, replacement):
    result = _fixture("refused_result")
    result[field] = replacement
    with pytest.raises(ContractRefusal):
        observer.parse_observer_result(_wire(result))


def test_transport_success_does_not_accept_missing_result_contract():
    raw = _wire(_fixture("refused_result"))
    frame = _scope_frame(raw)
    parameters = {"eof": True, "process_exit_code": 0, "run_elapsed_ms": 1, "receive_elapsed_ms": 1}
    assert observer.decode_synthetic_frame(frame, **parameters) == raw
    with pytest.raises(ContractRefusal):
        observer.parse_report_frame(frame, **parameters)


@pytest.mark.parametrize(
    "parser",
    [
        observer.parse_acceptance,
        observer.parse_launch_policy,
        observer.parse_scope_binding,
        observer.parse_run_context,
        observer.parse_offline_result,
        observer.parse_manifest,
        observer.parse_acceptance_policy,
        observer.parse_observer_result,
    ],
)
def test_diagnostics_never_echo_secret_input_or_exception(parser, capsys, caplog):
    marker = "stage9-secret-marker-1e5c9"
    raw = ('{"' + marker + '":NaN}').encode()
    with pytest.raises(ContractRefusal) as refused:
        parser(raw)
    assert str(refused.value) in observer.REASON_CODES
    assert marker not in repr(refused.value)
    assert marker not in str(refused.value)
    assert capsys.readouterr() == ("", "")
    assert marker not in caplog.text


def test_unknown_diagnostic_is_replaced_with_fixed_code():
    refusal = ContractRefusal("stage9-secret-marker")
    assert str(refusal) == "FORMAT_REJECTED"


@pytest.mark.parametrize("field", ["uid", "gid"])
@pytest.mark.parametrize("value", [1, 2**32 - 2])
def test_sandbox_linux_identity_boundaries(launch_policy, field, value):
    launch_policy["sandbox"][field] = value
    observer.parse_launch_policy(_wire(launch_policy))


@pytest.mark.parametrize("field", ["uid", "gid"])
@pytest.mark.parametrize("value", [0, True, 2**32 - 1, 2**32, 2**53 - 1])
def test_sandbox_linux_identity_cannot_truncate_or_use_reserved_value(launch_policy, field, value):
    launch_policy["sandbox"][field] = value
    with pytest.raises(ContractRefusal):
        observer.parse_launch_policy(_wire(launch_policy))


@pytest.mark.parametrize("control", ["\x1f", "\x7f", "\x80", "\x9f"])
def test_launch_paths_reject_ascii_and_c1_controls(launch_policy, control):
    launch_policy["interpreter"]["argv"][0] = "/observer/" + control
    with pytest.raises(ContractRefusal):
        observer.parse_launch_policy(_wire(launch_policy))


@pytest.mark.parametrize("codepoint", list(range(32)) + [127])
def test_identifiers_refuse_every_ascii_control(scope_fixture, codepoint):
    scope_fixture["host_anchor_id"] = "host" + chr(codepoint) + "anchor"
    with pytest.raises(ContractRefusal):
        observer.parse_scope_binding(canonical_json_bytes(scope_fixture))


@pytest.mark.parametrize(
    "identifier",
    ["", " ", " padded", "padded ", ".hidden", "-prefix", "path/name", "wild*card", "unicode-é"],
)
def test_identifier_token_grammar_rejects_ambiguous_forms(scope_fixture, identifier):
    scope_fixture["host_anchor_id"] = identifier
    with pytest.raises(ContractRefusal):
        observer.parse_scope_binding(canonical_json_bytes(scope_fixture))


@pytest.mark.parametrize(
    "identifier", ["Host-1", "host.namespace", "host_anchor", "host:anchor", "0" + "a" * 127]
)
def test_identifier_token_grammar_preserves_explicit_valid_forms(scope_fixture, identifier):
    scope_fixture["host_anchor_id"] = identifier
    assert (
        observer.parse_scope_binding(canonical_json_bytes(scope_fixture)).authorization
        == "UNVERIFIED"
    )


def proposed_manifest_launch_fixture(
    launch_template: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    launch = copy.deepcopy(launch_template)

    def digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def projection(value: Any) -> str:
        return digest(canonical_json_bytes(value))

    sources = []
    for source_id, kind in (
        ("git-source", "git_object_proof"),
        ("wheel-source", "pinned_wheel"),
        ("apt-source", "ubuntu_apt_snapshot"),
    ):
        raw = ("synthetic artifact " + source_id).encode()
        sources.append(
            {
                "source_id": source_id,
                "kind": kind,
                "artifact_sha256": digest(raw),
                "artifact_size": len(raw),
                "proof_record_sha256": digest(("synthetic proof " + source_id).encode()),
            }
        )
    files = []
    payloads: dict[str, bytes] = {}

    def directory(file_id: str, path: str) -> None:
        files.append(
            {
                "file_id": file_id,
                "path": path,
                "type": "directory",
                "uid": 0,
                "gid": 0,
                "mode": 0o555,
                "size": 0,
                "provenance": {
                    "kind": "source_member",
                    "source_id": "git-source",
                    "member": path[1:],
                },
            }
        )

    def regular(
        file_id: str,
        path: str,
        raw: bytes,
        source_id: str = "git-source",
        transform_id: str | None = None,
    ) -> None:
        payloads[file_id] = raw
        files.append(
            {
                "file_id": file_id,
                "path": path,
                "type": "regular",
                "uid": 0,
                "gid": 0,
                "mode": 0o555 if file_id == "python" else 0o444,
                "size": len(raw),
                "sha256": digest(raw),
                "provenance": {"kind": "transform_output", "transform_id": transform_id}
                if transform_id is not None
                else {
                    "kind": "source_member",
                    "source_id": source_id,
                    "member": path[1:],
                },
            }
        )

    directory("image-root", "/observer")
    directory("bin-root", "/observer/bin")
    directory("lib-root", "/observer/lib")
    directory("stdlib-root", "/observer/lib/python3.12")
    directory("config-root", "/observer/config")
    regular("python", "/observer/bin/python3.12", b"synthetic CPython binary", "apt-source")
    regular(
        "loader",
        "/observer/lib/ld-linux-x86-64.so.2",
        b"synthetic ELF loader",
        "apt-source",
    )
    regular(
        "libpython",
        "/observer/lib/libpython3.12.so.1",
        b"synthetic libpython",
        "apt-source",
    )
    regular("libc", "/observer/lib/libc.so.6", b"synthetic libc", "apt-source")
    regular(
        "entrypoint",
        "/observer/lib/python3.12/observer.py",
        b"# synthetic observer source\n",
    )
    regular(
        "helper",
        "/observer/lib/python3.12/helper.py",
        b"# synthetic wheel source\n",
        "wheel-source",
    )
    regular("seccomp", "/observer/config/seccomp.json", b'{"synthetic":"seccomp"}')
    regular("lsm", "/observer/config/lsm.json", b'{"synthetic":"lsm"}')
    regular("integrity", "/observer/config/integrity.json", b'{"synthetic":"integrity"}')
    regular(
        "materialized",
        "/observer/config/materialized.json",
        b'{"synthetic":"materialized"}',
        transform_id="materialize",
    )
    regular(
        "pyc-output",
        "/observer/config/helper.pyc",
        b"synthetic pyc output",
        transform_id="compile-pyc",
    )
    regular(
        "materialized-copy",
        "/observer/config/materialized-copy.json",
        payloads["materialized"],
        "materialized-source",
    )
    sources.append(
        {
            "source_id": "materialized-source",
            "kind": "approved_transform",
            "artifact_sha256": digest(payloads["materialized"]),
            "artifact_size": len(payloads["materialized"]),
            "proof_record_sha256": digest(b"synthetic materialized source proof"),
            "transform_id": "materialize",
        }
    )
    by_id = {file["file_id"]: file for file in files}
    build_id = launch["interpreter"]["build_id"]
    transforms = [
        {
            "transform_id": "materialize",
            "kind": "materialize_member",
            "inputs": [
                {
                    "kind": "source_member",
                    "source_id": "git-source",
                    "member": "data/materialized.json",
                    "sha256": by_id["materialized"]["sha256"],
                    "size": by_id["materialized"]["size"],
                }
            ],
            "tool": {
                "source_id": "git-source",
                "member": "tools/materialize.py",
                "sha256": digest(b"synthetic materialization tool"),
                "build_id": "fixture-materializer",
                "runtime_source_ids": ["git-source"],
            },
            "parameters": {"input_index": 0},
            "output": {
                "file_id": "materialized",
                "sha256": by_id["materialized"]["sha256"],
                "size": by_id["materialized"]["size"],
            },
            "reconstruction_proof_sha256": digest(b"synthetic materialization proof"),
        },
        {
            "transform_id": "compile-pyc",
            "kind": "pyc",
            "inputs": [
                {
                    "kind": "file",
                    "file_id": "helper",
                    "sha256": by_id["helper"]["sha256"],
                    "size": by_id["helper"]["size"],
                }
            ],
            "tool": {
                "source_id": "apt-source",
                "member": "usr/bin/python3.12",
                "sha256": digest(b"synthetic compiler"),
                "build_id": build_id,
                "runtime_source_ids": ["apt-source"],
            },
            "parameters": {
                "source_input_index": 0,
                "cpython_magic_hex": "cb0d0d0a",
                "build_id": build_id,
                "optimization": 0,
                "invalidation_mode": "checked_hash",
                "co_filename": by_id["helper"]["path"],
            },
            "output": {
                "file_id": "pyc-output",
                "sha256": by_id["pyc-output"]["sha256"],
                "size": by_id["pyc-output"]["size"],
            },
            "reconstruction_proof_sha256": digest(b"synthetic pyc reconstruction proof"),
        },
    ]
    runtime = {
        "image_root_file_id": "image-root",
        "interpreter_file_id": "python",
        "entrypoint_file_id": "entrypoint",
        "libpython_file_ids": ["libpython"],
        "stdlib_file_ids": ["entrypoint", "helper"],
        "loader_file_id": "loader",
        "library_file_ids": ["libc"],
        "dependency_edges": [
            {
                "requester_file_id": "python",
                "needed": "/observer/lib/ld-linux-x86-64.so.2",
                "resolved_file_id": "loader",
            },
            {
                "requester_file_id": "python",
                "needed": "libpython3.12.so.1",
                "resolved_file_id": "libpython",
            },
            {
                "requester_file_id": "libpython",
                "needed": "libc.so.6",
                "resolved_file_id": "libc",
            },
        ],
        "loader_config_file_ids": [],
        "absent_paths": ["/etc/ld.so.cache", "/etc/ld.so.preload"],
        "hwcaps_file_ids": [],
        "dlopen_file_ids": [],
        "resolution_root_ids": ["lib-root"],
        "rpath_runpath_policy": "IN_IMAGE_ONLY",
        "import_root_ids": ["stdlib-root"],
        "data_file_ids": [
            "seccomp",
            "lsm",
            "integrity",
            "materialized",
            "pyc-output",
            "materialized-copy",
        ],
        "embedded_modules": [
            {
                "embedded_module_id": "builtin-sys",
                "name": "sys",
                "kind": "builtin",
                "build_id": build_id,
                "container_file_ids": ["python"],
            },
            {
                "embedded_module_id": "frozen-bootstrap",
                "name": "_frozen_importlib",
                "kind": "frozen",
                "build_id": build_id,
                "container_file_ids": ["libpython"],
            },
        ],
    }
    imports = [
        {"name": "sys", "kind": "builtin", "embedded_module_id": "builtin-sys"},
        {
            "name": "_frozen_importlib",
            "kind": "frozen",
            "embedded_module_id": "frozen-bootstrap",
        },
        {
            "name": "observer",
            "kind": "source",
            "file_id": "entrypoint",
            "sha256": by_id["entrypoint"]["sha256"],
        },
        {
            "name": "helper",
            "kind": "source",
            "file_id": "helper",
            "sha256": by_id["helper"]["sha256"],
        },
    ]
    manifest = {
        "schema": "CryptoHunter.Stage9ObserverManifestV1",
        "version": 1,
        "purpose": "STAGE9_OBSERVER_TEST_ONLY",
        "manifest_id": "synthetic-proposed-manifest",
        "sequence": 1,
        "source_revision": "1234567890abcdef1234567890abcdef12345678",
        "target": {
            **launch["target"],
            "interpreter": {
                key: launch["interpreter"][key]
                for key in ("version", "package_version", "abi", "build_id")
            }
            | {"profile": "synthetic-cpython-profile"},
        },
        "sources": sources,
        "files": files,
        "transforms": transforms,
        "runtime": runtime,
        "imports": imports,
        "launch_policy_sha256": digest(b"pending synthetic launch digest"),
    }
    launch["interpreter"]["file_id"] = "python"
    launch["interpreter"]["entrypoint_file_id"] = "entrypoint"
    launch["interpreter"]["libpython_file_ids"] = ["libpython"]
    launch["interpreter"]["argv"] = [
        by_id["python"]["path"],
        "-I",
        "-S",
        "-B",
        by_id["entrypoint"]["path"],
    ]
    for key in launch["elf"]:
        launch["elf"][key] = copy.deepcopy(runtime[key])
    launch["imports"]["roots"] = ["stdlib-root"]
    launch["imports"]["modules_sha256"] = projection(imports)
    launch["cwd"]["directory_file_id"] = "image-root"
    for kind in ("seccomp", "lsm"):
        launch["sandbox"][kind]["profile_file_id"] = kind
        launch["sandbox"][kind]["profile_sha256"] = by_id[kind]["sha256"]
    launch["file_policy"]["integrity_profile_file_id"] = "integrity"
    launch["file_policy"]["integrity_profile_sha256"] = by_id["integrity"]["sha256"]
    launch["file_policy"]["files_sha256"] = projection(files)
    launch["runtime_dependencies"] = {
        "files_sha256": projection(files),
        "runtime_sha256": projection(runtime),
        "imports_sha256": projection(imports),
        "unlisted_dependencies": "FORBIDDEN",
    }
    manifest["launch_policy_sha256"] = projection(launch)
    return {"manifest": manifest, "launch": launch}


# Proposed-only manifest mutations. Standard parse_manifest remains fail-closed.


@pytest.fixture
def proposed_pair():
    return proposed_manifest_launch_fixture(_fixture("launch_policy"))


def _proposed_manifest_wire(value):
    return observer.parse_proposed_manifest(canonical_json_bytes(value))


def _proposed_manifest_item(manifest, section, item_id):
    field = {"files": "file_id", "sources": "source_id", "transforms": "transform_id"}[section]
    return next(item for item in manifest[section] if item[field] == item_id)


def _proposed_path(document, path):
    for key in path:
        document = document[key]
    return document


def test_proposed_manifest_full_synthetic_inventory(proposed_pair):
    manifest = proposed_pair["manifest"]
    parsed = _proposed_manifest_wire(manifest)
    assert parsed.authorization == "UNVERIFIED"
    assert {source["kind"] for source in manifest["sources"]} == {
        "git_object_proof",
        "pinned_wheel",
        "ubuntu_apt_snapshot",
        "approved_transform",
    }
    assert {item["kind"] for item in manifest["transforms"]} == {
        "materialize_member",
        "pyc",
    }
    assert {item["kind"] for item in manifest["imports"]} == {
        "source",
        "builtin",
        "frozen",
    }
    observer._validate_manifest_launch(manifest, proposed_pair["launch"])
    with pytest.raises(ContractRefusal):
        observer.parse_manifest(canonical_json_bytes(manifest))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("target",),
        ("target", "interpreter"),
        ("sources", 0),
        ("sources", 3),
        ("files", 0),
        ("files", 5),
        ("files", 5, "provenance"),
        ("files", 14, "provenance"),
        ("transforms", 0),
        ("transforms", 0, "inputs", 0),
        ("transforms", 1, "inputs", 0),
        ("transforms", 0, "tool"),
        ("transforms", 0, "parameters"),
        ("transforms", 1, "parameters"),
        ("transforms", 0, "output"),
        ("runtime",),
        ("runtime", "embedded_modules", 0),
        ("runtime", "dependency_edges", 0),
        ("imports", 0),
        ("imports", 2),
    ],
)
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_proposed_manifest_every_record_is_closed(proposed_pair, path, mutation):
    record = _proposed_path(proposed_pair["manifest"], path)
    if mutation == "extra":
        record["unknown"] = "never-accepted"
    else:
        del record[next(iter(record))]
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(proposed_pair["manifest"])


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("version",), True),
        (("version",), 2),
        (("purpose",), "PRODUCTION"),
        (("source_revision",), "782816540373d7b31f426dab645ad115320d1234"),
        (("target", "architecture"), "aarch64"),
        (("target", "interpreter", "version"), "3.13.0"),
        (("target", "interpreter", "build_id"), "wrong-build"),
        (("sources", 0, "kind"), "git"),
        (("sources", 0, "artifact_sha256"), "A" * 64),
        (("sources", 0, "artifact_size"), True),
        (("sources", 0, "artifact_size"), 0),
        (("sources", 0, "artifact_size"), 536870913),
        (("sources", 1, "artifact_size"), 16777217),
        (("sources", 3, "transform_id"), "unknown"),
        (("sources", 3, "artifact_sha256"), "f" * 64),
        (("files", 0, "size"), 1),
        (("files", 5, "type"), "symlink"),
        (("files", 5, "type"), "hardlink"),
        (("files", 5, "type"), "socket"),
        (("files", 5, "uid"), 2**32 - 1),
        (("files", 5, "gid"), True),
        (("files", 5, "mode"), 0o777),
        (("files", 5, "mode"), 0o4555),
        (("files", 5, "size"), 2147483649),
        (("files", 5, "sha256"), "0" * 64),
        (("files", 5, "path"), "/observer/../python"),
        (("files", 5, "path"), "/observer//python"),
        (("files", 5, "provenance", "kind"), "implicit"),
        (("files", 5, "provenance", "source_id"), "unknown"),
        (("files", 5, "provenance", "member"), "../escape"),
        (("transforms", 0, "kind"), "shell-command"),
        (("transforms", 0, "tool", "source_id"), "unknown"),
        (("transforms", 0, "tool", "member"), "/absolute"),
        (("transforms", 0, "tool", "runtime_source_ids"), ["unknown"]),
        (("transforms", 0, "parameters", "input_index"), True),
        (("transforms", 0, "parameters", "input_index"), 1),
        (("transforms", 0, "output", "file_id"), "unknown"),
        (("transforms", 0, "output", "sha256"), "f" * 64),
        (("transforms", 1, "inputs", 0, "sha256"), "f" * 64),
        (("transforms", 1, "parameters", "build_id"), "different-build"),
        (("transforms", 1, "parameters", "optimization"), True),
        (("transforms", 1, "parameters", "optimization"), 3),
        (("transforms", 1, "parameters", "invalidation_mode"), "unknown"),
        (("transforms", 1, "parameters", "co_filename"), "../escape.py"),
        (("runtime", "image_root_file_id"), "python"),
        (("runtime", "interpreter_file_id"), "missing"),
        (("runtime", "libpython_file_ids"), ["python"]),
        (("runtime", "entrypoint_file_id"), "pyc-output"),
        (("runtime", "import_root_ids"), ["python"]),
        (("runtime", "library_file_ids"), ["libc", "libc"]),
        (("runtime", "rpath_runpath_policy"), "ANYWHERE"),
        (("runtime", "embedded_modules", 0, "build_id"), "different-build"),
        (("runtime", "embedded_modules", 0, "container_file_ids"), []),
        (("runtime", "embedded_modules", 0, "container_file_ids"), ["helper"]),
        (("runtime", "dependency_edges", 0, "resolved_file_id"), "missing"),
        (("runtime", "dependency_edges", 0, "needed"), "bad\nsoname"),
        (("imports", 0, "kind"), "namespace"),
        (("imports", 0, "embedded_module_id"), "missing"),
        (("imports", 0, "name"), "other_builtin"),
        (("imports", 2, "file_id"), "image-root"),
        (("imports", 2, "sha256"), "f" * 64),
        (("imports", 2, "name"), "unknown_alias"),
    ],
)
def test_proposed_manifest_refuses_invalid_values_and_references(proposed_pair, path, replacement):
    record = _proposed_path(proposed_pair["manifest"], path[:-1])
    record[path[-1]] = replacement
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(proposed_pair["manifest"])


@pytest.mark.parametrize("section", ["sources", "files", "transforms", "imports"])
def test_proposed_manifest_duplicate_inventory_records(proposed_pair, section):
    manifest = proposed_pair["manifest"]
    manifest[section].append(copy.deepcopy(manifest[section][0]))
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize("section", ["sources", "files", "imports"])
def test_proposed_manifest_required_inventory_is_nonempty(proposed_pair, section):
    proposed_pair["manifest"][section] = []
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(proposed_pair["manifest"])


def test_proposed_manifest_extra_unreferenced_source_is_not_implicitly_allowed(
    proposed_pair,
):
    manifest = proposed_pair["manifest"]
    unused = copy.deepcopy(manifest["sources"][0])
    unused["source_id"] = "unused-source"
    manifest["sources"].append(unused)
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize("path", ["/observer", "/observer/lib/python3.12/helper.py"])
def test_proposed_manifest_duplicate_paths_even_with_unique_ids(proposed_pair, path):
    manifest = proposed_pair["manifest"]
    manifest["files"][5]["path"] = path
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def test_proposed_manifest_requires_complete_in_image_ancestry(proposed_pair):
    manifest = proposed_pair["manifest"]
    manifest["files"] = [file for file in manifest["files"] if file["file_id"] != "bin-root"]
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def test_proposed_manifest_rejects_unused_directory(proposed_pair):
    manifest = proposed_pair["manifest"]
    directory = copy.deepcopy(manifest["files"][0])
    directory["file_id"], directory["path"] = "unused-dir", "/observer/unused"
    manifest["files"].append(directory)
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def test_proposed_manifest_root_is_explicit_not_host_root(proposed_pair):
    manifest = proposed_pair["manifest"]
    manifest["files"][0]["path"] = "/"
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def test_proposed_manifest_total_regular_size_limit(proposed_pair):
    manifest = proposed_pair["manifest"]
    manifest["files"][5]["size"] = 2147483648
    with pytest.raises(ContractRefusal) as refused:
        _proposed_manifest_wire(manifest)
    assert refused.value.code == "LIMIT_EXCEEDED"


@pytest.mark.parametrize("graph_kind", ["transform-file", "transform-source", "elf"])
def test_proposed_manifest_rejects_real_declared_dependency_cycles(proposed_pair, graph_kind):
    manifest = proposed_pair["manifest"]
    if graph_kind == "transform-file":
        transform = manifest["transforms"][1]
        output = transform["output"]
        transform["inputs"] = [{"kind": "file", **output}]
    elif graph_kind == "transform-source":
        manifest["transforms"][0]["inputs"][0]["source_id"] = "materialized-source"
    else:
        manifest["runtime"]["dependency_edges"].append(
            {
                "requester_file_id": "loader",
                "needed": "python-back-reference",
                "resolved_file_id": "python",
            }
        )
    with pytest.raises(ContractRefusal) as refused:
        _proposed_manifest_wire(manifest)
    assert refused.value.code == "CLOSURE_INCOMPLETE"


def test_proposed_manifest_acyclic_graph_is_iterative_not_recursive():
    graph = {("node", str(index)): {("node", str(index + 1))} for index in range(20000)}
    graph[("node", "20000")] = set()
    observer._proposed_acyclic(graph)
    graph[("node", "20000")] = {("node", "0")}
    with pytest.raises(ContractRefusal):
        observer._proposed_acyclic(graph)


@pytest.mark.parametrize("mode", ["checked_hash", "unchecked_hash", "timestamp"])
@pytest.mark.parametrize("optimization", [0, 1, 2])
def test_proposed_manifest_exact_pyc_parameter_variants(proposed_pair, mode, optimization):
    manifest = proposed_pair["manifest"]
    parameters = manifest["transforms"][1]["parameters"]
    parameters["invalidation_mode"] = mode
    parameters["optimization"] = optimization
    if mode == "timestamp":
        parameters["source_timestamp"] = 2**32 - 1
        parameters["source_size"] = manifest["transforms"][1]["inputs"][0]["size"]
    assert _proposed_manifest_wire(manifest).authorization == "UNVERIFIED"


@pytest.mark.parametrize(
    "field,value",
    [("source_timestamp", True), ("source_timestamp", 2**32), ("source_size", 0)],
)
def test_proposed_manifest_timestamp_pyc_fields_are_exact(proposed_pair, field, value):
    manifest = proposed_pair["manifest"]
    parameters = manifest["transforms"][1]["parameters"]
    parameters.update(
        invalidation_mode="timestamp",
        source_timestamp=0,
        source_size=manifest["transforms"][1]["inputs"][0]["size"],
    )
    parameters[field] = value
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def test_proposed_manifest_hash_pyc_forbids_timestamp_fields(proposed_pair):
    proposed_pair["manifest"]["transforms"][1]["parameters"]["source_timestamp"] = 0
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(proposed_pair["manifest"])


@pytest.mark.parametrize("field", ["inputs", "outputs"])
def test_proposed_manifest_does_not_allow_undefined_transform_arity(proposed_pair, field):
    transform = proposed_pair["manifest"]["transforms"][0]
    if field == "inputs":
        transform["inputs"] *= 2
    else:
        transform["outputs"] = [transform.pop("output")]
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(proposed_pair["manifest"])


def test_proposed_manifest_directory_cannot_be_transform_output(proposed_pair):
    manifest = proposed_pair["manifest"]
    manifest["files"][0]["provenance"] = {
        "kind": "transform_output",
        "transform_id": "materialize",
    }
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize("mutation", ["missing", "extra", "kind", "name", "container", "fake-file"])
def test_proposed_manifest_embedded_inventory_exact_and_no_fictional_sources(
    proposed_pair, mutation
):
    manifest = proposed_pair["manifest"]
    modules = manifest["runtime"]["embedded_modules"]
    if mutation == "missing":
        modules.pop()
    elif mutation == "extra":
        module = copy.deepcopy(modules[0])
        module["embedded_module_id"] = "extra"
        module["name"] = "extra"
        modules.append(module)
    elif mutation == "kind":
        modules[0]["kind"] = "frozen"
    elif mutation == "name":
        modules[0]["name"] = "other"
    elif mutation == "container":
        modules[0]["container_file_ids"] = ["python", "python"]
    else:
        manifest["imports"][0]["file_id"] = "helper"
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("interpreter", "build_id"), "different"),
        (("interpreter", "version"), "3.12.99"),
        (("interpreter", "package_version"), "3.12.3-foreign"),
        (("interpreter", "abi"), "cpython-312-foreign"),
        (("interpreter", "argv", 0), "/foreign/python"),
        (("interpreter", "argv", 4), "/foreign/observer.py"),
        (("elf", "loader_file_id"), "libc"),
        (("elf", "library_file_ids"), []),
        (("imports", "roots"), ["lib-root"]),
        (("imports", "modules_sha256"), "f" * 64),
        (("sandbox", "seccomp", "profile_sha256"), "f" * 64),
        (("file_policy", "integrity_profile_sha256"), "f" * 64),
        (("cwd", "directory_file_id"), "python"),
        (("runtime_dependencies", "runtime_sha256"), "f" * 64),
    ],
)
def test_proposed_manifest_launch_exact_cross_projection(proposed_pair, path, replacement):
    launch = proposed_pair["launch"]
    record = _proposed_path(launch, path[:-1])
    record[path[-1]] = replacement
    with pytest.raises(ContractRefusal):
        observer._validate_manifest_launch(proposed_pair["manifest"], launch)


def test_proposed_manifest_launch_policy_digest_binds_actual_canonical_policy(
    proposed_pair,
):
    proposed_pair["manifest"]["launch_policy_sha256"] = "f" * 64
    with pytest.raises(ContractRefusal):
        observer._validate_manifest_launch(proposed_pair["manifest"], proposed_pair["launch"])


def test_proposed_manifest_does_not_claim_opaque_artifact_reference_scan(proposed_pair):
    parsed = _proposed_manifest_wire(proposed_pair["manifest"])
    assert parsed.authorization == "UNVERIFIED"
    assert parsed.review_status == "PROPOSED_UNAPPROVED"


def test_proposed_manifest_noncanonical_raw_is_rejected(proposed_pair):
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_manifest(json.dumps(proposed_pair["manifest"], indent=2).encode())


@pytest.mark.parametrize(
    "role",
    [
        "libpython_file_ids",
        "stdlib_file_ids",
        "library_file_ids",
        "loader_config_file_ids",
        "hwcaps_file_ids",
        "dlopen_file_ids",
    ],
)
def test_proposed_pyc_is_exclusive_inert_data_not_any_runtime_code_role(proposed_pair, role):
    manifest = proposed_pair["manifest"]
    manifest["runtime"][role].append("pyc-output")
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize(
    "path",
    [
        "/observer/lib/python3.12/helper.pyc",
        "/observer/lib/python3.12/__pycache__/helper.cpython-312.pyc",
    ],
)
def test_proposed_pyc_cannot_be_discovered_under_import_roots(proposed_pair, path):
    manifest = proposed_pair["manifest"]
    _proposed_manifest_item(manifest, "files", "pyc-output")["path"] = path
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


def _proposed_add_regular(manifest, file_id, path, raw=b"synthetic extra source"):
    template = copy.deepcopy(_proposed_manifest_item(manifest, "files", "helper"))
    template.update(
        file_id=file_id,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
    )
    template["provenance"]["member"] = path[1:]
    manifest["files"].append(template)
    return template


def _proposed_add_package(manifest):
    directory = copy.deepcopy(manifest["files"][0])
    directory.update(file_id="package-dir", path="/observer/lib/python3.12/pkg")
    manifest["files"].append(directory)
    package = _proposed_add_regular(manifest, "package", "/observer/lib/python3.12/pkg/__init__.py")
    child = _proposed_add_regular(
        manifest, "package-child", "/observer/lib/python3.12/pkg/child.py"
    )
    manifest["imports"].extend(
        [
            {
                "name": "pkg",
                "kind": "source",
                "file_id": "package",
                "sha256": package["sha256"],
            },
            {
                "name": "pkg.child",
                "kind": "source",
                "file_id": "package-child",
                "sha256": child["sha256"],
            },
        ]
    )


def test_proposed_manifest_standard_source_package_and_child_positive(proposed_pair):
    manifest = proposed_pair["manifest"]
    _proposed_add_package(manifest)
    assert _proposed_manifest_wire(manifest).authorization == "UNVERIFIED"


@pytest.mark.parametrize(
    "mutation", ["missing", "module-not-package", "builtin", "frozen", "different-root"]
)
def test_proposed_manifest_dotted_parent_resolution_is_explicit_and_same_root(
    proposed_pair, mutation
):
    manifest = proposed_pair["manifest"]
    _proposed_add_package(manifest)
    parent = manifest["imports"][-2]
    if mutation == "missing":
        manifest["imports"].remove(parent)
        manifest["runtime"]["data_file_ids"].append("package")
    elif mutation == "module-not-package":
        _proposed_manifest_item(manifest, "files", "package")["path"] = (
            "/observer/lib/python3.12/pkg.py"
        )
    elif mutation in {"builtin", "frozen"}:
        existing = manifest["imports"][0 if mutation == "builtin" else 1]
        manifest["runtime"]["embedded_modules"][0 if mutation == "builtin" else 1]["name"] = "pkg"
        existing["name"] = "pkg"
        manifest["imports"].remove(parent)
        manifest["runtime"]["data_file_ids"].append("package")
    else:
        manifest["runtime"]["import_root_ids"].append("lib-root")
        directory = copy.deepcopy(manifest["files"][0])
        directory.update(file_id="outer-package-dir", path="/observer/lib/pkg")
        manifest["files"].append(directory)
        _proposed_manifest_item(manifest, "files", "package")["path"] = (
            "/observer/lib/pkg/__init__.py"
        )
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize("shadow", ["source-other-root", "native", "native-abi", "package", "pyc"])
def test_proposed_manifest_import_shadowing_is_rejected_even_if_labeled_data(proposed_pair, shadow):
    manifest = proposed_pair["manifest"]
    paths = {
        "source-other-root": "/observer/lib/helper.py",
        "native": "/observer/lib/python3.12/helper.so",
        "native-abi": "/observer/lib/python3.12/helper."
        + manifest["target"]["interpreter"]["abi"]
        + ".so",
        "package": "/observer/lib/python3.12/helper/__init__.py",
        "pyc": "/observer/lib/python3.12/helper.pyc",
    }
    if shadow == "source-other-root":
        manifest["runtime"]["import_root_ids"].append("lib-root")
    file = _proposed_add_regular(manifest, "shadow", paths[shadow])
    manifest["runtime"]["data_file_ids"].append(file["file_id"])
    with pytest.raises(ContractRefusal):
        _proposed_manifest_wire(manifest)


@pytest.mark.parametrize("suffix", [".so", ".cpython-312-x86_64-linux-gnu.so"])
def test_proposed_manifest_native_module_exact_filename_and_dependency_positive(
    proposed_pair, suffix
):
    manifest = proposed_pair["manifest"]
    file = _proposed_add_regular(
        manifest,
        "native",
        "/observer/lib/python3.12/native" + suffix,
        b"synthetic native ELF",
    )
    manifest["imports"].append(
        {
            "name": "native",
            "kind": "native",
            "file_id": "native",
            "sha256": file["sha256"],
        }
    )
    manifest["runtime"]["dependency_edges"].append(
        {
            "requester_file_id": "native",
            "needed": "libc.so.6",
            "resolved_file_id": "libc",
        }
    )
    assert _proposed_manifest_wire(manifest).authorization == "UNVERIFIED"


def test_proposed_manifest_resolution_work_budget_accepts_exact_boundary():
    observer._proposed_resolution_budget(100, 156, 256)
    with pytest.raises(ContractRefusal) as refused:
        observer._proposed_resolution_budget(100, 157, 256)
    assert refused.value.code == "LIMIT_EXCEEDED"


def test_proposed_manifest_resolution_work_budget_precedes_import_resolution(
    proposed_pair, monkeypatch
):
    manifest = proposed_pair["manifest"]
    for index in range(256):
        directory = copy.deepcopy(manifest["files"][0])
        file_id = "broad-root-" + str(index)
        directory.update(file_id=file_id, path="/observer/importroot-" + str(index))
        manifest["files"].append(directory)
        manifest["runtime"]["import_root_ids"].append(file_id)

    def unexpected_resolution(*args, **kwargs):
        raise AssertionError(
            "expensive import/module resolution started before work-budget refusal"
        )

    monkeypatch.setattr(observer, "_proposed_module_name", unexpected_resolution)
    with pytest.raises(ContractRefusal) as refused:
        _proposed_manifest_wire(manifest)
    assert refused.value.code == "LIMIT_EXCEEDED"


@pytest.fixture
def proposed_policy_document():
    return {
        "schema": "CryptoHunter.Stage9ObserverAcceptancePolicyV1",
        "version": 1,
        "purpose": "STAGE9_OBSERVER_TEST_ONLY",
        "root_set_id": "synthetic-roots",
        "reviewer_keys": [
            {"signer_id": "a", "reviewer_id": "reviewer-a", "public_key_hex": "11" * 32},
            {"signer_id": "b", "reviewer_id": "reviewer-b", "public_key_hex": "22" * 32},
        ],
        "threshold": 2,
        "allowed_sources": [
            {
                "source_id": "synthetic-source",
                "kind": "git_object_proof",
                "artifact_sha256": hashlib.sha256(b"synthetic-artifact").hexdigest(),
                "artifact_size": 64,
                "proof_record_sha256": hashlib.sha256(b"synthetic-proof").hexdigest(),
            }
        ],
        "minimum_sequence": 0,
        "revocation_checkpoint": {
            "sequence": 0,
            "state_sha256": hashlib.sha256(b"synthetic-revocation").hexdigest(),
        },
        "valid_until": "2026-10-28T00:00:00Z",
        "target": {
            "os": "Linux",
            "distribution": "Ubuntu",
            "distribution_version": "24.04",
            "architecture": "x86_64",
            "interpreter": {
                "version": "3.12.7",
                "package_version": "3.12.7-0ubuntu1",
                "abi": "cpython-312-x86_64-linux-gnu",
                "build_id": "synthetic-build",
                "profile": "synthetic-profile",
            },
        },
        "limits": {
            "manifest_bytes": 4194304,
            "acceptance_policy_bytes": 65536,
            "launch_policy_bytes": 65536,
            "acceptance_bytes": 65536,
            "result_bytes": 65536,
            "run_context_bytes": 16384,
            "depth": 16,
            "files": 32768,
            "sources": 128,
            "reason_codes": 64,
            "identifier_ascii_chars": 128,
            "path_bytes": 4096,
            "archive_bytes": 536870912,
            "unpacked_bytes": 2147483648,
            "archive_members": 65536,
            "compression_ratio": 100,
            "artifact_timeout_s": 60,
            "bundle_timeout_s": 300,
            "working_memory_bytes": 268435456,
            "wheel_archive_bytes": 16777216,
            "wheel_unpacked_bytes": 33554432,
            "buffered_descriptor_bytes": 16777216,
        },
        "launch_policy_sha256": hashlib.sha256(b"synthetic-launch").hexdigest(),
    }


@pytest.fixture
def proposed_result_document():
    return {
        "schema": "ObserverResultV1",
        "version": 1,
        "synthetic": True,
        "run_id": "synthetic-run",
        "run_nonce": "ab" * 32,
        "observer_manifest_sha256": hashlib.sha256(b"synthetic-manifest").hexdigest(),
        "policy_sha256": hashlib.sha256(b"synthetic-policy").hexdigest(),
        "launch_policy_sha256": hashlib.sha256(b"synthetic-launch").hexdigest(),
        "installation_commits": [
            "782816540373d7b31f426dab645ad115320d1234",
            "26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f",
        ],
        "scope_binding_sha256": hashlib.sha256(b"synthetic-binding").hexdigest(),
        "boot_id": "12345678-1234-1234-abcd-123456789abc",
        "started_at": "2026-10-27T23:00:00Z",
        "finished_at": "2026-10-27T23:00:01Z",
        "counters": {
            "pids_seen": 0,
            "tids_seen": 0,
            "identities_checked": 0,
            "references_checked": 0,
            "matching_observations": 0,
            "permission_denied": 0,
            "identity_changes": 0,
            "binding_changes": 0,
            "incomplete_reads": 0,
            "limits_hit": 0,
        },
        "observations": {"match_seen": False, "identity_stable": True, "bindings_stable": True},
        "reason_codes": ["RUNTIME_LAUNCH_UNVERIFIED", "HOST_SCOPE_UNVERIFIED"],
        "completeness": "INCOMPLETE",
        "qualification": "BLOCKED",
        "full_confirmation": False,
        "coexistence_authorized": False,
        "preflight_exception": False,
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "status": "IN_PROGRESS",
        "production_status": "PRODUCTION_NO_GO",
        "runtime_attestation": "MISSING",
        "host_pid_namespace": "UNVERIFIED",
    }


def test_proposed_policy_structural_record_does_not_authorize_roots(proposed_policy_document):
    assert observer._validate_proposed_acceptance_policy(proposed_policy_document) is None


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "version",
        "purpose",
        "root_set_id",
        "reviewer_keys",
        "threshold",
        "allowed_sources",
        "minimum_sequence",
        "revocation_checkpoint",
        "valid_until",
        "target",
        "limits",
        "launch_policy_sha256",
    ],
)
def test_proposed_policy_requires_each_top_level_field(proposed_policy_document, field):
    del proposed_policy_document[field]
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize("threshold", [True, 0, 1, 3, "2", 2.0])
def test_proposed_policy_rejects_changed_quorum(proposed_policy_document, threshold):
    proposed_policy_document["threshold"] = threshold
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize("field", ["signer_id", "reviewer_id", "public_key_hex"])
def test_proposed_policy_rejects_duplicate_reviewer_binding(proposed_policy_document, field):
    keys = proposed_policy_document["reviewer_keys"]
    keys[1][field] = keys[0][field]
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


def test_proposed_policy_rejects_unsorted_reviewers(proposed_policy_document):
    proposed_policy_document["reviewer_keys"].reverse()
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize("field", ["target", "limits", "revocation_checkpoint"])
def test_proposed_policy_nested_records_are_closed(proposed_policy_document, field):
    proposed_policy_document[field]["unknown"] = "secret-marker"
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize(
    "field",
    [
        "manifest_bytes",
        "acceptance_policy_bytes",
        "launch_policy_bytes",
        "acceptance_bytes",
        "result_bytes",
        "run_context_bytes",
        "depth",
        "files",
        "sources",
        "reason_codes",
        "identifier_ascii_chars",
        "path_bytes",
        "archive_bytes",
        "unpacked_bytes",
        "archive_members",
        "compression_ratio",
        "artifact_timeout_s",
        "bundle_timeout_s",
        "working_memory_bytes",
        "wheel_archive_bytes",
        "wheel_unpacked_bytes",
        "buffered_descriptor_bytes",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "increased", "boolean"])
def test_proposed_policy_limits_are_exact(proposed_policy_document, field, mutation):
    limits = proposed_policy_document["limits"]
    if mutation == "missing":
        del limits[field]
    elif mutation == "increased":
        limits[field] += 1
    else:
        limits[field] = True
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("os", "Windows"),
        ("distribution", "Debian"),
        ("distribution_version", "*"),
        ("architecture", "aarch64"),
    ],
)
def test_proposed_policy_rejects_foreign_target(proposed_policy_document, field, replacement):
    proposed_policy_document["target"][field] = replacement
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize(
    "mutation", ["duplicate_source", "unknown_kind", "uppercase_digest", "unknown_field"]
)
def test_proposed_policy_source_inventory_is_closed(proposed_policy_document, mutation):
    sources = proposed_policy_document["allowed_sources"]
    if mutation == "duplicate_source":
        sources.append(copy.deepcopy(sources[0]))
    elif mutation == "unknown_kind":
        sources[0]["kind"] = "local_receipt"
    elif mutation == "uppercase_digest":
        sources[0]["artifact_sha256"] = sources[0]["artifact_sha256"].upper()
    else:
        sources[0]["accepted"] = True
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("minimum_sequence", True),
        ("minimum_sequence", -1),
        ("minimum_sequence", 2**53),
        ("launch_policy_sha256", "0" * 64),
        ("valid_until", "2026-10-28T00:00:00+00:00"),
        ("purpose", "PRODUCT_RELEASE"),
    ],
)
def test_proposed_policy_rejects_invalid_constraint(proposed_policy_document, field, replacement):
    proposed_policy_document[field] = replacement
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


def test_proposed_result_stays_incomplete_with_no_hits(proposed_result_document):
    assert observer._validate_proposed_observer_result(proposed_result_document) is None


@pytest.mark.parametrize(
    "field",
    [
        "pids_seen",
        "tids_seen",
        "identities_checked",
        "references_checked",
        "matching_observations",
        "permission_denied",
        "identity_changes",
        "binding_changes",
        "incomplete_reads",
        "limits_hit",
    ],
)
@pytest.mark.parametrize("replacement", ["missing", True, -1, 2**53])
def test_proposed_result_counter_schema_and_bounds(proposed_result_document, field, replacement):
    if replacement == "missing":
        del proposed_result_document["counters"][field]
    else:
        proposed_result_document["counters"][field] = replacement
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize("field", ["match_seen", "identity_stable", "bindings_stable"])
def test_proposed_result_observation_flags_require_boolean(proposed_result_document, field):
    proposed_result_document["observations"][field] = 1
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_categories_overlap_without_summing(proposed_result_document):
    proposed_result_document["counters"].update(
        pids_seen=1,
        tids_seen=1,
        identities_checked=1,
        references_checked=7,
        matching_observations=7,
        permission_denied=7,
    )
    proposed_result_document["observations"]["match_seen"] = True
    assert observer._validate_proposed_observer_result(proposed_result_document) is None


def test_proposed_result_successful_identity_checks_need_enumeration(proposed_result_document):
    proposed_result_document["counters"]["identities_checked"] = 1
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_maximum_counter_boundary(proposed_result_document):
    proposed_result_document["counters"] = {
        name: 2**53 - 1 for name in proposed_result_document["counters"]
    }
    proposed_result_document["observations"] = {
        "match_seen": True,
        "identity_stable": False,
        "bindings_stable": False,
    }
    proposed_result_document["reason_codes"].append("LIMIT_EXCEEDED")
    assert observer._validate_proposed_observer_result(proposed_result_document) is None


@pytest.mark.parametrize("field", ["counters", "observations"])
def test_proposed_result_nested_records_are_closed(proposed_result_document, field):
    proposed_result_document[field]["unknown"] = 0
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize(
    ("counter", "flag"),
    [
        ("matching_observations", "match_seen"),
        ("identity_changes", "identity_stable"),
        ("binding_changes", "bindings_stable"),
    ],
)
def test_proposed_result_rejects_counter_observation_conflict(
    proposed_result_document, counter, flag
):
    proposed_result_document["counters"][counter] = 1
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_true_match_needs_nonzero_count(proposed_result_document):
    proposed_result_document["observations"]["match_seen"] = True
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_unknown_stability_is_not_complete(proposed_result_document):
    proposed_result_document["observations"]["identity_stable"] = False
    assert observer._validate_proposed_observer_result(proposed_result_document) is None
    proposed_result_document["completeness"] = "COMPLETE"
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize("missing_reason", ["RUNTIME_LAUNCH_UNVERIFIED", "HOST_SCOPE_UNVERIFIED"])
def test_proposed_result_preserves_missing_evidence_reasons(
    proposed_result_document, missing_reason
):
    proposed_result_document["reason_codes"].remove(missing_reason)
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_limits_hit_is_not_silent(proposed_result_document):
    proposed_result_document["counters"]["limits_hit"] = 1
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("synthetic", False),
        ("full_confirmation", True),
        ("coexistence_authorized", True),
        ("preflight_exception", True),
        ("qualification", "PASS"),
        ("production_status", "GO"),
        ("runtime_attestation", "VERIFIED"),
        ("host_pid_namespace", "VERIFIED"),
        ("completeness", "COMPLETE"),
        ("finished_at", "2026-10-27T22:59:59Z"),
        ("finished_at", "2026-10-27T23:05:01Z"),
        ("run_nonce", "a" * 63),
        ("scope_binding_sha256", "0" * 64),
    ],
)
def test_proposed_result_rejects_upgrades_and_invalid_bindings(
    proposed_result_document, field, replacement
):
    proposed_result_document[field] = replacement
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize("finish", ["2026-10-27T23:00:00Z", "2026-10-27T23:04:59Z"])
def test_proposed_result_window_boundaries(proposed_result_document, finish):
    proposed_result_document["finished_at"] = finish
    assert observer._validate_proposed_observer_result(proposed_result_document) is None


def test_proposed_sequence_preserves_match_after_later_failure(proposed_result_document):
    previous = copy.deepcopy(proposed_result_document)
    previous["counters"].update(pids_seen=1, references_checked=1, matching_observations=1)
    previous["observations"]["match_seen"] = True
    current = copy.deepcopy(previous)
    current["finished_at"] = "2026-10-27T23:00:02Z"
    current["counters"]["permission_denied"] = 1
    result = observer.parse_proposed_observer_sequence(
        canonical_json_bytes(previous), canonical_json_bytes(current)
    )
    assert result.as_dict()["observations"]["match_seen"] is True
    assert result.authorization == "UNVERIFIED"


def test_proposed_sequence_refuses_reset_match_after_error(proposed_result_document):
    previous = copy.deepcopy(proposed_result_document)
    previous["counters"].update(pids_seen=1, references_checked=1, matching_observations=1)
    previous["observations"]["match_seen"] = True
    current = copy.deepcopy(proposed_result_document)
    current["counters"]["permission_denied"] = 1
    current["finished_at"] = "2026-10-27T23:00:02Z"
    with pytest.raises(observer.ContractRefusal):
        observer.parse_proposed_observer_sequence(
            canonical_json_bytes(previous), canonical_json_bytes(current)
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "other-run"),
        ("run_nonce", "cd" * 32),
        ("boot_id", "12345678-1234-1234-abcd-123456789abd"),
        ("policy_sha256", "bc" * 32),
        ("started_at", "2026-10-27T22:59:59Z"),
        ("finished_at", "2026-10-27T23:00:00Z"),
    ],
)
def test_proposed_sequence_refuses_cross_run_or_replayed_window(
    proposed_result_document, field, replacement
):
    current = copy.deepcopy(proposed_result_document)
    current[field] = replacement
    with pytest.raises(observer.ContractRefusal):
        observer.parse_proposed_observer_sequence(
            canonical_json_bytes(proposed_result_document), canonical_json_bytes(current)
        )


@pytest.mark.parametrize("counter", ["tids_seen", "references_checked"])
def test_proposed_result_process_counters_require_enumeration(proposed_result_document, counter):
    proposed_result_document["counters"][counter] = 1
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_match_events_cannot_exceed_checked_references(proposed_result_document):
    proposed_result_document["counters"].update(
        pids_seen=1, references_checked=1, matching_observations=2
    )
    proposed_result_document["observations"]["match_seen"] = True
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_deadline_is_exclusive(proposed_result_document):
    proposed_result_document["finished_at"] = "2026-10-27T23:05:00Z"
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize("flag", ["identity_stable", "bindings_stable"])
def test_proposed_sequence_cannot_erase_observed_instability(proposed_result_document, flag):
    previous = copy.deepcopy(proposed_result_document)
    previous["observations"][flag] = False
    current = copy.deepcopy(previous)
    current["observations"][flag] = True
    current["finished_at"] = "2026-10-27T23:00:02Z"
    with pytest.raises(observer.ContractRefusal):
        observer.parse_proposed_observer_sequence(
            canonical_json_bytes(previous), canonical_json_bytes(current)
        )


def test_proposed_sequence_cannot_erase_later_failure_reasons(proposed_result_document):
    previous = copy.deepcopy(proposed_result_document)
    previous["reason_codes"].append("FILE_CHANGED")
    current = copy.deepcopy(previous)
    current["reason_codes"].remove("FILE_CHANGED")
    current["finished_at"] = "2026-10-27T23:00:02Z"
    with pytest.raises(observer.ContractRefusal):
        observer.parse_proposed_observer_sequence(
            canonical_json_bytes(previous), canonical_json_bytes(current)
        )


@pytest.mark.parametrize(
    "counter",
    [
        "pids_seen",
        "tids_seen",
        "identities_checked",
        "references_checked",
        "matching_observations",
        "permission_denied",
        "identity_changes",
        "binding_changes",
        "incomplete_reads",
        "limits_hit",
    ],
)
def test_proposed_sequence_cumulative_counters_never_decrease(proposed_result_document, counter):
    previous = copy.deepcopy(proposed_result_document)
    previous["counters"].update({name: 2 for name in previous["counters"]})
    previous["observations"] = {
        "match_seen": True,
        "identity_stable": False,
        "bindings_stable": False,
    }
    previous["reason_codes"].append("LIMIT_EXCEEDED")
    current = copy.deepcopy(previous)
    current["counters"][counter] = 1
    current["finished_at"] = "2026-10-27T23:00:02Z"
    with pytest.raises(observer.ContractRefusal):
        observer.parse_proposed_observer_sequence(
            canonical_json_bytes(previous), canonical_json_bytes(current)
        )


def test_proposed_sequence_retains_instability_and_new_diagnostic(proposed_result_document):
    previous = copy.deepcopy(proposed_result_document)
    previous["observations"]["identity_stable"] = False
    current = copy.deepcopy(previous)
    current["observations"]["bindings_stable"] = False
    current["counters"]["permission_denied"] = 1
    current["reason_codes"].append("FILE_CHANGED")
    current["finished_at"] = "2026-10-27T23:00:02Z"
    result = observer.parse_proposed_observer_sequence(
        canonical_json_bytes(previous), canonical_json_bytes(current)
    )
    assert result.authorization == "UNVERIFIED"
    assert result.as_dict()["observations"]["identity_stable"] is False


@pytest.mark.parametrize(
    "bad_key", [True, "", "a" * 63, "a" * 65, "AB" * 32, "gg" * 32, "11" * 31 + "\\n"]
)
def test_proposed_policy_public_key_encoding_is_strict(proposed_policy_document, bad_key):
    proposed_policy_document["reviewer_keys"][0]["public_key_hex"] = bad_key
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_acceptance_policy(proposed_policy_document)


@pytest.mark.parametrize(
    "bad_status", ["INCOMPLETE", "UNVERIFIED", "BLOCKED", "PERMISSION_DENIED", "PASS"]
)
def test_proposed_result_status_words_are_not_reason_codes(proposed_result_document, bad_status):
    proposed_result_document["reason_codes"].append(bad_status)
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.mark.parametrize("kind", ["policy", "result"])
def test_proposed_top_level_schema_rejects_extra_fields(
    proposed_policy_document, proposed_result_document, kind
):
    value = proposed_policy_document if kind == "policy" else proposed_result_document
    validator = (
        observer._validate_proposed_acceptance_policy
        if kind == "policy"
        else observer._validate_proposed_observer_result
    )
    value["independent_approval"] = True
    with pytest.raises(observer.ContractRefusal):
        validator(value)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "version",
        "synthetic",
        "run_id",
        "run_nonce",
        "observer_manifest_sha256",
        "policy_sha256",
        "launch_policy_sha256",
        "installation_commits",
        "scope_binding_sha256",
        "boot_id",
        "started_at",
        "finished_at",
        "counters",
        "observations",
        "reason_codes",
        "completeness",
        "qualification",
        "full_confirmation",
        "coexistence_authorized",
        "preflight_exception",
        "physical_test_status",
        "status",
        "production_status",
        "runtime_attestation",
        "host_pid_namespace",
    ],
)
def test_proposed_result_requires_each_field(proposed_result_document, field):
    del proposed_result_document[field]
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


def test_proposed_result_duplicate_reasons_do_not_satisfy_schema(proposed_result_document):
    proposed_result_document["reason_codes"].append("RUNTIME_LAUNCH_UNVERIFIED")
    with pytest.raises(observer.ContractRefusal):
        observer._validate_proposed_observer_result(proposed_result_document)


@pytest.fixture
def proposed_bundle_documents(proposed_policy_document, proposed_result_document):
    pair = proposed_manifest_launch_fixture(_fixture("launch_policy"))
    manifest, launch = pair["manifest"], pair["launch"]
    policy = copy.deepcopy(proposed_policy_document)
    policy["target"] = copy.deepcopy(manifest["target"])
    policy["allowed_sources"] = copy.deepcopy(manifest["sources"])
    policy["launch_policy_sha256"] = manifest["launch_policy_sha256"]
    policy["minimum_sequence"] = manifest["sequence"]
    acceptance = _fixture("acceptance")
    acceptance["payload"].update(
        manifest_sha256=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        root_set_id=policy["root_set_id"],
        sequence=manifest["sequence"],
        revocation_checkpoint=copy.deepcopy(policy["revocation_checkpoint"]),
        issued_at="2026-10-27T22:00:00Z",
        expires_at="2026-10-27T23:59:00Z",
    )
    for signature, key in zip(acceptance["signatures"], policy["reviewer_keys"]):
        signature["signer_id"] = key["signer_id"]
    _rebind_acceptance(acceptance)
    scope = _fixture("scope_binding")
    run = _fixture("run_context")
    run.update(
        run_id="synthetic-run",
        run_nonce=scope["run_nonce"],
        observer_manifest_sha256=acceptance["payload"]["manifest_sha256"],
        policy_sha256=acceptance["payload"]["policy_sha256"],
        launch_policy_sha256=manifest["launch_policy_sha256"],
        scope_binding=copy.deepcopy(scope),
        scope_binding_sha256=hashlib.sha256(canonical_json_bytes(scope)).hexdigest(),
    )
    result = copy.deepcopy(proposed_result_document)
    for key in (
        "run_id",
        "run_nonce",
        "observer_manifest_sha256",
        "policy_sha256",
        "launch_policy_sha256",
        "scope_binding_sha256",
    ):
        result[key] = run[key]
    result["boot_id"] = scope["boot_id"]
    return dict(
        manifest=manifest,
        policy=policy,
        launch_policy=launch,
        acceptance=acceptance,
        scope_binding=scope,
        run_context=run,
        result=result,
    )


def _bundle_wire(documents):
    return {key + "_raw": canonical_json_bytes(value) for key, value in documents.items()} | {
        "synthetic_now": "2026-10-27T23:00:02Z"
    }


def _rebind_bundle_acceptance(documents):
    documents["acceptance"]["payload"]["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(documents["manifest"])
    ).hexdigest()
    documents["acceptance"]["payload"]["policy_sha256"] = hashlib.sha256(
        canonical_json_bytes(documents["policy"])
    ).hexdigest()
    _rebind_acceptance(documents["acceptance"])


def test_proposed_bundle_is_consistent_but_never_authorizes(proposed_bundle_documents):
    parsed = observer.parse_proposed_bundle(**_bundle_wire(proposed_bundle_documents))
    assert parsed.authorization == "UNVERIFIED"
    assert parsed.review_status == "PROPOSED_UNAPPROVED"
    assert parsed.synthetic is True
    for name in ("authorization", "review_status", "synthetic"):
        with pytest.raises(FrozenInstanceError):
            setattr(parsed, name, "FORGED")
    assert parsed.result.as_dict()["production_status"] == "PRODUCTION_NO_GO"
    for record in (parsed.manifest, parsed.policy, parsed.result):
        assert record.authorization == "UNVERIFIED"
        assert record.synthetic is True
        assert record.review_status == "PROPOSED_UNAPPROVED"
        with pytest.raises(FrozenInstanceError):
            record.synthetic = False


@pytest.mark.parametrize(
    "document,field,replacement",
    [
        ("acceptance", "root_set_id", "foreign-roots"),
        ("acceptance", "sequence", 0),
        ("acceptance", "policy_sha256", "ab" * 32),
        ("acceptance", "manifest_sha256", "ab" * 32),
    ],
)
def test_proposed_bundle_checks_envelope_bindings_after_payload_rehash(
    proposed_bundle_documents, document, field, replacement
):
    proposed_bundle_documents[document]["payload"][field] = replacement
    _rebind_acceptance(proposed_bundle_documents["acceptance"])
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_bundle(**_bundle_wire(proposed_bundle_documents))


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint",
        "minimum_sequence",
        "source_bytes",
        "target_build",
        "launch_pin",
        "unknown_signer",
        "expired",
        "future",
        "report_after_now",
    ],
)
def test_proposed_bundle_crossrecord_mutations_fail_closed(proposed_bundle_documents, mutation):
    docs = proposed_bundle_documents
    if mutation == "checkpoint":
        docs["policy"]["revocation_checkpoint"]["sequence"] += 1
    elif mutation == "minimum_sequence":
        docs["policy"]["minimum_sequence"] += 1
    elif mutation == "source_bytes":
        docs["policy"]["allowed_sources"][0]["artifact_sha256"] = "ab" * 32
    elif mutation == "target_build":
        docs["policy"]["target"]["interpreter"]["build_id"] = "foreign-build"
    elif mutation == "launch_pin":
        docs["policy"]["launch_policy_sha256"] = "ab" * 32
    elif mutation == "unknown_signer":
        docs["acceptance"]["signatures"][1]["signer_id"] = "unknown-reviewer"
    elif mutation == "expired":
        docs["acceptance"]["payload"]["expires_at"] = "2026-10-27T22:59:00Z"
    elif mutation == "future":
        docs["acceptance"]["payload"]["issued_at"] = "2026-10-27T23:30:00Z"
    else:
        docs["result"]["finished_at"] = "2026-10-27T23:00:03Z"
    _rebind_bundle_acceptance(docs)
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_bundle(**_bundle_wire(docs))


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "run_nonce",
        "boot_id",
        "observer_manifest_sha256",
        "policy_sha256",
        "launch_policy_sha256",
        "scope_binding_sha256",
    ],
)
def test_proposed_result_must_bind_expected_run(proposed_bundle_documents, field):
    result = proposed_bundle_documents["result"]
    result[field] = (
        "foreign-run"
        if field == "run_id"
        else "87654321-1234-1234-abcd-123456789abc"
        if field == "boot_id"
        else "cd" * 32
    )
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_bundle(**_bundle_wire(proposed_bundle_documents))


@pytest.mark.parametrize("document", ["manifest", "policy", "result"])
def test_proposed_records_require_canonical_bytes_and_explicit_review_api(
    proposed_bundle_documents, document
):
    raw = canonical_json_bytes(proposed_bundle_documents[document])
    parser = {
        "manifest": observer.parse_proposed_manifest,
        "policy": observer.parse_proposed_acceptance_policy,
        "result": observer.parse_proposed_observer_result,
    }[document]
    standard = {
        "manifest": observer.parse_manifest,
        "policy": observer.parse_acceptance_policy,
        "result": observer.parse_observer_result,
    }[document]
    assert parser(raw).review_status == "PROPOSED_UNAPPROVED"
    for wire in (b" " + raw, raw + b"\n"):
        with pytest.raises(ContractRefusal):
            parser(wire)
    with pytest.raises(ContractRefusal):
        standard(raw)


def test_proposed_full_report_frame_stays_untrusted_and_standard_gate_closed(
    proposed_bundle_documents,
):
    raw = canonical_json_bytes(proposed_bundle_documents["result"])
    frame = len(raw).to_bytes(4, "big") + raw
    args = dict(eof=True, process_exit_code=0, run_elapsed_ms=2000, receive_elapsed_ms=1000)
    assert observer.parse_proposed_report_frame(frame, **args).authorization == "UNVERIFIED"
    with pytest.raises(ContractRefusal):
        observer.parse_report_frame(frame, **args)


@pytest.mark.parametrize(
    "key",
    [
        "manifest_raw",
        "policy_raw",
        "launch_policy_raw",
        "acceptance_raw",
        "scope_binding_raw",
        "run_context_raw",
        "result_raw",
    ],
)
def test_proposed_bundle_does_not_construct_missing_document_digests(
    proposed_bundle_documents, key
):
    wire = _bundle_wire(proposed_bundle_documents)
    wire[key] = b""
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_bundle(**wire)


def test_contract_architecture_keeps_only_stdlib_and_shared_canonicalizer():
    tree = ast.parse(Path(observer.__file__).read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "hashlib",
        "json",
        "re",
        "dataclasses",
        "datetime",
        "typing",
        "deployment.canonical_json",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "eval", "exec", "__import__", "compile"}


def test_proposed_bundle_rejects_explicit_scope_metadata_as_source_proof(proposed_bundle_documents):
    docs = proposed_bundle_documents
    docs["manifest"]["sources"][0]["proof_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(docs["scope_binding"])
    ).hexdigest()
    docs["policy"]["allowed_sources"] = copy.deepcopy(docs["manifest"]["sources"])
    _rebind_bundle_acceptance(docs)
    for record in (docs["run_context"], docs["result"]):
        record["observer_manifest_sha256"] = docs["acceptance"]["payload"]["manifest_sha256"]
        record["policy_sha256"] = docs["acceptance"]["payload"]["policy_sha256"]
    with pytest.raises(ContractRefusal) as refused:
        observer.parse_proposed_bundle(**_bundle_wire(docs))
    assert refused.value.code == "CLOSURE_INCOMPLETE"


def test_proposed_bundle_does_not_accept_expiry_as_current_time(proposed_bundle_documents):
    wire = _bundle_wire(proposed_bundle_documents)
    wire["synthetic_now"] = proposed_bundle_documents["acceptance"]["payload"]["expires_at"]
    with pytest.raises(ContractRefusal):
        observer.parse_proposed_bundle(**wire)
