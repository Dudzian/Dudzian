"""Rootless synthetic contract tests; no artifacts or host processes execute."""

from __future__ import annotations

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
