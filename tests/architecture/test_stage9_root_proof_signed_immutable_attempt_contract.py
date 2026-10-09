"""Child freeze, historical byte identity, current status and forbidden scope."""

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from bot_core import (
    cha_attempt_signatures,
    cha_attempt_store,
    cha_issuance_request,
    cha_issuance_execution,
    cha_root_proof_signing_custody,
)
from bot_core.licensing import cha_root_proof_signed_attempt

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_root_proof_signed_immutable_attempt_contract.json"
FREEZE_PATH = DOCS / "stage9_root_proof_signed_immutable_attempt_freeze.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())
FREEZE = json.loads(FREEZE_PATH.read_bytes())

_EXECUTION_OWNER = "bot_core.cha_issuance_execution"
_EXECUTION_CONSUMERS = {
    _EXECUTION_OWNER,
    "bot_core.licensing.cha_root_proof_attempt_reservation",
    "bot_core.licensing.cha_root_proof_signed_attempt",
}
_ISSUANCE_EFFECTS = {
    "reserve_or_resolve_attempt_id",
    "prepare_signature",
    "sign_issuance_request",
    "authorize_entitlement_claim",
    "persist_signature",
    "finalize_attempt",
}
_EXECUTION_PRIVATE_REFERENCES = {
    "_INSTALLED_OPERATIONS",
    "_LOCAL_OPERATIONS",
    "_LOCAL_GRANT_BY_OPERATION",
    "_qualified_operations",
}
_LOCAL_ISSUANCE_MODULES = _EXECUTION_CONSUMERS | {
    "bot_core.cha_attempt_store",
    "bot_core.cha_attempt_signatures",
    "bot_core.cha_issuance_request",
    "bot_core.cha_root_proof_signing_custody",
}
_TRANSPORT_IMPORTS = {"requests", "httpx", "http", "grpc", "socket", "aiohttp", "urllib"}
_INFRASTRUCTURE_SOCKET_MODULES = {
    "bot_core.security.fingerprint",  # Hostname collection, not issuer transport.
    "deployment.production_pdsa_signing",  # Existing protected local AF_UNIX signing IPC.
    "deployment.windows_stage8_postgresql_probe",  # Existing PostgreSQL connectivity probe.
}
_RETAINED_REQUEST_REFERENCES = {
    "RetainedIssuanceRequest",
    "VerifiedSignedImmutableRootProofIssuanceAttempt",
    "canonical_request_bytes",
    "signed_request",
}


def _source_imports(tree, path):
    imports = set()
    package = Path(path).parent.parts
    literal_importers = {"__import__"} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "importlib" and not node.level
        for alias in node.names
        if alias.name == "import_module"
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else ()
            module = ".".join((*prefix, node.module)) if node.module else ".".join(prefix)
            imports.add(module)
            imports.update(f"{module}.{alias.name}" for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id in literal_importers
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            )
        ):
            imports.add(node.args[0].value)
    return imports


def _assert_local_issuance_execution_ownership(sources):
    """Review gate for static production paths, including future importers.

    References are checked, not just calls: taking a bound-method alias must not
    escape dispatch. Import closures cover request/handle re-exports and delegated sends.
    This protects accidental architectural changes, not hostile dynamic Python.
    """
    dependencies = {}
    consumers = set(_LOCAL_ISSUANCE_MODULES)
    for path, source in sources.items():
        module = ".".join(Path(path).with_suffix("").parts).removesuffix(".__init__")
        tree = ast.parse(source, filename=path)
        imports = _source_imports(tree, path)
        dependencies[module] = imports
        references = (
            {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
        )
        references.update(
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr"}
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        )
        if references & _RETAINED_REQUEST_REFERENCES:
            consumers.add(module)
        if module != _EXECUTION_OWNER:
            assert not references & (_ISSUANCE_EFFECTS | _EXECUTION_PRIVATE_REFERENCES), (
                f"issuance effect bypass in {path}"
            )
        if module not in _EXECUTION_CONSUMERS:
            assert "execute_local_issuance" not in references, (
                f"unreviewed issuance dispatcher consumer in {path}"
            )

    while True:
        expanded = consumers | {
            module for module, imports in dependencies.items() if imports & consumers
        }
        if expanded == consumers:
            break
        consumers = expanded
    # A new boundary consumer must not delegate transport to a generic helper.
    reachable = set(consumers)
    while True:
        expanded = reachable | {
            imported
            for module in reachable
            for imported in dependencies.get(module, ())
            if imported in dependencies
        }
        if expanded == reachable:
            break
        reachable = expanded
    for module in reachable & dependencies.keys():
        forbidden = {imported.split(".")[0] for imported in dependencies[module]}
        if module in _INFRASTRUCTURE_SOCKET_MODULES and module not in consumers:
            forbidden.discard("socket")
        assert not forbidden & _TRANSPORT_IMPORTS, f"issuer transport bypass in {module}"


def test_production_issuance_effects_and_consumers_preserve_local_dispatch_ownership():
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8-sig")
        for root in ("bot_core", "core", "scripts", "deployment", "deploy", "ui")
        for path in (ROOT / root).rglob("*.py")
    }
    assert sources
    _assert_local_issuance_execution_ownership(sources)


@pytest.mark.parametrize(
    "source",
    [
        "def issue(store, identity):\n    return store.finalize_attempt(identity)\n",
        "def issue(store, auth):\n    return store.reserve_or_resolve_attempt_id(auth)\n",
        "def issue(store, signature):\n    return store.persist_signature(signature)\n",
        "def issue(signer):\n    aliased = signer.sign_issuance_request\n    return aliased\n",
        "from bot_core.cha_attempt_store import SQLiteCHAAttemptStore as Store\n"
        "prepare = Store.prepare_signature\n",
        "def issue(claimant):\n    return getattr(claimant, 'authorize_entitlement_claim')\n",
        "from bot_core.cha_issuance_execution import execute_local_issuance as issue\n",
        "from bot_core.cha_issuance_execution import *\n"
        "def issue(store):\n    return execute_local_issuance('RESERVE', store)\n",
        "import bot_core.cha_issuance_execution as executor\n"
        "issue = executor.execute_local_issuance\n",
        "from bot_core.cha_issuance_execution import _INSTALLED_OPERATIONS as routes\n",
        "from bot_core.licensing.cha_root_proof_signed_attempt import "
        "VerifiedSignedImmutableRootProofIssuanceAttempt as Attempt\n"
        "import httpx\n"
        "def send(attempt: Attempt):\n"
        "    return httpx.post('https://issuer.invalid', content=attempt.canonical_request_bytes)\n",
        "from bot_core.licensing import cha_root_proof_signed_attempt as attempts\n"
        "from urllib.request import urlopen as send\n",
        "import importlib\n"
        "attempts = importlib.import_module('bot_core.licensing.cha_root_proof_signed_attempt')\n"
        "import socket\n",
        "from importlib import import_module as load\n"
        "attempts = load('bot_core.licensing.cha_root_proof_signed_attempt')\n"
        "import httpx\n",
        "from bot_core.licensing.cha_root_proof_signed_attempt import "
        "VerifiedSignedImmutableRootProofIssuanceAttempt as Attempt\n"
        "from http.client import HTTPSConnection\n",
        "import httpx\n"
        "def issue(attempt):\n"
        "    return httpx.post('https://issuer.invalid', content=attempt.canonical_request_bytes)\n",
    ],
)
def test_added_execution_paths_cannot_bypass_local_only_review_gate(source):
    with pytest.raises(AssertionError, match="issuance|issuer transport"):
        _assert_local_issuance_execution_ownership({"bot_core/future_issuer_transport.py": source})


def test_retained_request_reexports_cannot_hide_a_future_transport():
    with pytest.raises(AssertionError, match="issuer transport"):
        _assert_local_issuance_execution_ownership(
            {
                "bot_core/retained_attempt_api.py": (
                    "from bot_core.cha_attempt_signatures import RetainedIssuanceRequest as Request\n"
                ),
                "bot_core/issuance/future_transport.py": (
                    "from ..retained_attempt_api import Request\nimport aiohttp\n"
                ),
            }
        )


@pytest.mark.parametrize("transport", ["httpx", "http.client", "grpc", "socket"])
def test_delegating_a_signed_attempt_to_a_transport_helper_fails(transport):
    with pytest.raises(AssertionError, match="issuer transport"):
        _assert_local_issuance_execution_ownership(
            {
                "bot_core/future_issuer.py": (
                    "from bot_core.licensing.cha_root_proof_signed_attempt import "
                    "VerifiedSignedImmutableRootProofIssuanceAttempt as Attempt\n"
                    "from core.future_network_helper import publish\n"
                    "def issue(attempt: Attempt):\n"
                    "    return publish(attempt.canonical_request_bytes)\n"
                ),
                "core/future_network_helper.py": f"import {transport}\n",
            }
        )


def test_child_freeze_and_historical_parent_bytes():
    assert FREEZE["contract_bytes_sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert FREEZE["parent_sha256"] == CONTRACT["frozen_parent_sha256"]
    for parent in FREEZE["parent_sha256"]:
        digest = hashlib.sha256((DOCS / parent["path"]).read_bytes()).hexdigest()
        assert digest == parent["bytes_sha256"]
    assert FREEZE["boundary"] == CONTRACT["boundary"] == "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"


def test_exact_request_fields_domains_profiles_and_authority_namespace():
    parent = json.loads(
        (DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json").read_bytes()
    )
    assert (
        list(cha_issuance_request.REQUEST_FIELDS)
        == CONTRACT["request"]["signed_payload_fields"]
        == parent["issuance_request"]["signed_payload_fields"]
    )
    assert (
        CONTRACT["request"]["issuer_target_namespace"]
        == cha_issuance_request.ISSUER_TARGET_NAMESPACE
    )
    for role in cha_issuance_request.IssuanceSigningRole:
        contract = CONTRACT["signatures"][
            "requester"
            if role is cha_issuance_request.IssuanceSigningRole.REQUESTER
            else "claimant"
        ]
        assert contract["domain"] == role.domain
        assert contract["profile"] == role.profile
        assert (
            contract["preimage"] == "ASCII(domain) || 0x00 || exact same UTF-8 JCS signed payload"
        )
        assert contract["algorithm"] == "Ed25519"
    assert (
        cha_attempt_store._ATTEMPT_DOMAIN
        == b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_ATTEMPT_IDENTITY_V1\0"
    )
    assert CONTRACT["request"]["attempt_digest_in_request"] is False
    assert CONTRACT["request"]["namespace_caller_selected"] is False


def test_schema_and_immutable_signing_tables():
    assert cha_attempt_store._SCHEMA_VERSION == CONTRACT["durability"]["schema_version"] == 6
    assert list(cha_attempt_signatures.TABLES) == CONTRACT["durability"]["tables"][:3]
    assert CONTRACT["durability"]["migration"]["from_version"] == 5
    assert "resigning_allowed" not in CONTRACT["durability"]
    assert (
        CONTRACT["durability"]["reference"]
        == "immutable:req:sha256:<lowercase SHA256 of exact canonical bytes>"
    )
    assert CONTRACT["durability"]["states"] == [
        cha_attempt_store.AttemptState.RESERVED_AWAITING_SIGNATURES.value,
        cha_attempt_store.AttemptState.REQUEST_SIGNED_BY_REQUESTER.value,
        cha_attempt_store.AttemptState.CLAIMANT_AUTHORIZED.value,
        cha_attempt_store.AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
    ]


def assert_status(current):
    assert CONTRACT["current_status"] == FREEZE["current_status"]
    for key, value in CONTRACT["current_status"].items():
        assert type(current[key]) is type(value) and current[key] == value


def test_current_status_keeps_implementation_and_live_provisioning_separate():
    current = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    assert_status(current)
    assert current["signed_immutable_attempt"] == "IMPLEMENTED"
    assert (
        current["signed_immutable_attempt_live_availability"]
        == "BLOCKED_UNTIL_PRODUCTION_CREDENTIAL_PROVISIONING"
    )
    assert current["production_preaccount_credentials"] == "NOT_PROVISIONED"
    assert current["production_provisioning_ready"] is False
    assert current["windows_0_14"] == "10/15 DONE"
    assert current["root_proof"] == "NOT_ISSUED"
    assert current["prepared"] == "NOT_STARTED"


@pytest.mark.parametrize("field", CONTRACT["current_status"])
def test_status_conflicts_fail_closed(field):
    current = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    current[field] = "CONTRADICTORY"
    with pytest.raises(AssertionError):
        assert_status(current)


def test_public_api_has_only_existing_upstream_capability_inputs():
    assert list(
        inspect.signature(cha_root_proof_signed_attempt.sign_root_proof_issuance_attempt).parameters
    ) == ["reservation"]
    assert list(
        inspect.signature(
            cha_root_proof_signed_attempt.resume_root_proof_issuance_attempt
        ).parameters
    ) == ["binding", "authorization"]
    assert (
        CONTRACT["public_api"][
            "caller_selected_private_key_path_dsn_signer_key_version_attempt_id_reference_signature_namespace"
        ]
        is False
    )


@pytest.mark.parametrize(
    "module",
    [
        cha_issuance_request,
        cha_attempt_signatures,
        cha_issuance_execution,
        cha_root_proof_signing_custody,
        cha_root_proof_signed_attempt,
    ],
)
def test_child_scope_contains_no_transport_bind_or_proof_authority(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports & {"requests", "httpx", "grpc", "socket", "aiohttp", "urllib"}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (attrs | names) & {
        "compare_and_swap_bind",
        "sign_root_proof",
        "sign_history_head",
        "send",
        "send_issuance_request",
        "sign_freshness_proposal",
        "sign_finalization",
        "LocalRootProofSigningProvider",
        "RootProofIssuer",
        "admit_root_proof",
        "MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN",
        "EXACT_BOUND_RECOVERED",
        "PREPARED",
    }
    assert not any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("rpf_")
        for node in ast.walk(tree)
    )


def test_custody_runtime_ports_do_not_expose_provisioning_or_private_export():
    for cls, expected in (
        (cha_root_proof_signing_custody.LocalCHARequesterSigningCustody, "sign_issuance_request"),
        (
            cha_root_proof_signing_custody.LocalPreaccountClaimantAuthorizationCustody,
            "authorize_entitlement_claim",
        ),
    ):
        public = {name for name in dir(cls) if not name.startswith("_")}
        assert public == {"identity", expected}
    assert (
        CONTRACT["custody"]["runtime_can_generate_rotate_delete_export_or_change_lifecycle"]
        is False
    )
    assert CONTRACT["provisioning"]["distributed_atomicity"] is False


def assert_parent_recovery_parity(contract):
    parent = json.loads(
        (DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json").read_bytes()
    )
    recovery = contract["durability"]["recovery"]
    assert recovery["parent_reservation_crash_recovery"] == parent["reservation_crash_recovery"]
    assert recovery["pre_finalization_missing_signature_reobtain_allowed"] is True
    assert recovery["same_rpa_request_key_role_required"] is True
    assert recovery["durable_checkpoint_always_reused"] is True
    assert recovery["permanent_intent_exhaustion"] is False
    assert recovery["post_finalization_resigning_allowed"] is False
    assert recovery["post_send_resigning_allowed"] is False
    assert recovery["credential_substitution_requires_safe_supersession"] is True
    assert "resigning_allowed" not in contract["durability"]
    supersession = contract["durability"]["pre_send_reservation_supersession"]
    for field, value in parent["pre_send_reservation_supersession"].items():
        assert type(supersession[field]) is type(value) and supersession[field] == value
    assert (
        supersession["state"]
        == cha_attempt_store.AttemptState.SUPERSEDED_PRE_SEND_PROVEN_UNSENDABLE.value
    )
    assert supersession["state"] in parent["attempt_subordinate_states"]


def test_phase_specific_recovery_conforms_to_frozen_parent():
    assert_parent_recovery_parity(CONTRACT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pre_finalization_missing_signature_reobtain_allowed", False),
        ("durable_checkpoint_always_reused", False),
        ("permanent_intent_exhaustion", True),
        ("post_finalization_resigning_allowed", True),
        ("post_send_resigning_allowed", True),
        ("credential_substitution_requires_safe_supersession", False),
    ],
)
def test_child_recovery_mutations_cannot_redefine_parent(field, value):
    import copy

    changed = copy.deepcopy(CONTRACT)
    changed["durability"]["recovery"][field] = value
    with pytest.raises(AssertionError):
        assert_parent_recovery_parity(changed)


@pytest.mark.parametrize(
    "field",
    [
        "old_history_retained",
        "new_rpa_id_required",
        "unresolved_reservation_allows_hidden_new_id",
        "distinct_from_post_send_replacement",
    ],
)
def test_child_supersession_mutations_cannot_redefine_parent(field):
    import copy

    changed = copy.deepcopy(CONTRACT)
    changed["durability"]["pre_send_reservation_supersession"][field] = not changed["durability"][
        "pre_send_reservation_supersession"
    ][field]
    with pytest.raises(AssertionError):
        assert_parent_recovery_parity(changed)


def test_current_platform_support_matches_native_custody_and_child(tmp_path):
    from tests.security._local_signing_platform import requires_native_custody_locking

    assert (
        CONTRACT["current_status"]["signed_immutable_attempt_platform_support"]
        == "POSIX_AND_WINDOWS_NATIVE"
    )
    from bot_core.local_signing_custody import _LOCK_FILENAME

    with cha_root_proof_signing_custody._custody_lock(tmp_path, exclusive=False):
        descriptor = os.open(tmp_path / _LOCK_FILENAME, os.O_RDWR)
        try:
            with pytest.raises(OSError):
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    assert requires_native_custody_locking.args == (False,)


def test_child_never_claims_permanent_pre_sign_intent_exhaustion():
    assert_parent_recovery_parity(CONTRACT)
    text = json.dumps(CONTRACT["durability"]).lower()
    for forbidden in (
        "blocks forever",
        "permanently blocks",
        "invocation cannot be repeated",
        "latch stays spent",
    ):
        assert forbidden not in text


def test_runtime_reuses_an_exact_intent_and_rejects_unequal_identity(tmp_path):
    from dataclasses import replace
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from tests.licensing.test_cha_attempt_store_stage9 import authorization
    from tests.licensing.test_cha_root_proof_signed_attempt import signer_identity

    auth = authorization()
    signer = signer_identity(
        cha_issuance_request.IssuanceSigningRole.REQUESTER, auth, Ed25519PrivateKey.generate()
    )
    with cha_attempt_store.SQLiteCHAAttemptStore(
        tmp_path / "intent.sqlite3", auth.trust_domain
    ) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        original = store.prepare_signature(current, signer)
        assert store.prepare_signature(current, signer) == original
        with pytest.raises(cha_attempt_store.AttemptConflictError, match="EXACT_SIGNING_OPERATION"):
            store.prepare_signature(current, replace(signer, service_namespace="changed"))
        assert store.attempt(auth.logical_operation_id) == current
