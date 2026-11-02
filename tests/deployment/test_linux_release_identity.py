"""Certyfikat wydania nie zastępuje fizycznej kwalifikacji usług produktu."""

from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from deployment import linux_release_identity as release
from deployment.linux_physical_evidence import canonical_bytes, digest
from tests.deployment.test_linux_physical_collector import installation_fixture


SOURCE_REVISION = "a" * 40


def _public(key: Ed25519PrivateKey) -> tuple[str, str]:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii")


def _sign(context: dict[str, Any]) -> None:
    envelope = context["installation"]["release_identity"]
    envelope["signature"] = base64.b64encode(
        context["release_key"].sign(canonical_bytes(envelope["payload"]))
    ).decode("ascii")
    context["materials"]["installation_contract_sha256"] = digest(context["installation"])


def release_context() -> dict[str, Any]:
    """Dwa niezależne, wyłącznie syntetyczne klucze i kompletny certyfikat."""
    producer_key, release_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    _producer_id, producer_public = _public(producer_key)
    key_id, release_public = _public(release_key)
    installation = installation_fixture(Path("/tmp/cryptohunter-release-unit-fixture"))
    payload = {
        "schema_version": 1,
        "product_id": release.PRODUCT_ID,
        "contract": release.RELEASE_CONTRACT,
        "security_profile": release.SECURITY_PROFILE,
        "repository": "Dudzian/Dudzian",
        "source_revision": SOURCE_REVISION,
        "release_id": "1.2.3",
        "installation_profile_sha256": digest(installation),
        "artifacts": {
            phase: {"sha256": spec["sha256"], "version": spec["version"]}
            for phase, spec in installation["packages"].items()
        },
        "components": [
            {
                "role": spec["role"],
                "unit": spec["name"],
                "executable_sha256": spec["executable_sha256"],
                "update_executable_sha256": spec["update_executable_sha256"],
                "contracts": list(release.REQUIRED_COMPONENT_CONTRACTS[spec["role"]]),
            }
            for spec in installation["units"]
        ],
    }
    installation["release_identity"] = {"key_id": key_id, "payload": payload, "signature": ""}
    materials = {
        "source_revision": SOURCE_REVISION,
        "collector_sha256": "b" * 64,
        "artifact_sha256": installation["packages"]["baseline"]["sha256"],
        "update_artifact_sha256": installation["packages"]["update"]["sha256"],
        "failed_update_artifact_sha256": installation["packages"]["failed_update"]["sha256"],
        "configuration_sha256": digest(
            {item["path"]: item["sha256"] for item in installation["configuration"]}
        ),
        "installation_contract_sha256": "c" * 64,
    }
    context = {
        "installation": installation,
        "materials": materials,
        "policy": {
            "hosts": [{"public_key": producer_public}],
            "release_authorities": [{"key_id": key_id, "public_key": release_public}],
        },
        "producer_key": producer_key,
        "release_key": release_key,
    }
    _sign(context)
    return context


def _verify(context: dict[str, Any], *, source_revision: str = SOURCE_REVISION) -> dict[str, Any]:
    return release.verify_release_identity(
        context["installation"],
        context["policy"],
        source_revision=source_revision,
        materials=context["materials"],
    )


def test_independent_release_authority_verifies_certificate_without_qualifying_product() -> None:
    context = release_context()
    payload = _verify(context)
    assert payload == context["installation"]["release_identity"]["payload"]
    assert payload["product_id"] == release.PRODUCT_ID
    assert "status" not in payload
    assert "LINUX_PRODUCTION_READY" not in payload


def test_release_certificate_allows_reviewed_packaging_names_without_changing_logical_roles() -> (
    None
):
    context = release_context()
    installation = context["installation"]
    installation["package_name"] = "cryptohunter-reviewed-successor"
    payload = installation["release_identity"]["payload"]
    for spec, component in zip(installation["units"], payload["components"], strict=True):
        name = "cryptohunter-successor-" + spec["role"] + ".service"
        spec["name"] = name
        spec["unit_file"]["path"] = "/etc/systemd/system/" + name
        component["unit"] = name
    installation["clean_paths"] = [installation["state_directory"]] + [
        unit["unit_file"]["path"] for unit in installation["units"]
    ]
    payload["installation_profile_sha256"] = digest(
        {name: value for name, value in installation.items() if name != "release_identity"}
    )
    _sign(context)
    assert _verify(context)["release_id"] == "1.2.3"


@pytest.mark.parametrize("authorities", [None, {}, "keys", True, [None], [{}]])
def test_release_authority_inventory_refuses_malformed_types(authorities: object) -> None:
    with pytest.raises(ValueError):
        release.validate_release_authorities(authorities, producer_keys=set())


def test_empty_authority_inventory_is_only_an_unqualified_registry() -> None:
    release.validate_release_authorities([], producer_keys=set())
    context = release_context()
    context["policy"]["release_authorities"] = []
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update(key_id="not-a-digest"),
        lambda item: item.update(key_id="f" * 64),
        lambda item: item.update(public_key="invalid-base64"),
        lambda item: item.update(public_key=base64.b64encode(b"p" * 31).decode("ascii")),
        lambda item: item.update(public_key=[]),
        lambda item: item.update(purpose="self-approved"),
    ],
)
def test_release_authority_inventory_refuses_ambiguous_or_unpinned_key(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    context = release_context()
    authorities = copy.deepcopy(context["policy"]["release_authorities"])
    mutate(authorities[0])
    with pytest.raises(ValueError):
        release.validate_release_authorities(authorities, producer_keys=set())


def test_release_authority_inventory_refuses_duplicate_authorities() -> None:
    context = release_context()
    authorities = context["policy"]["release_authorities"]
    with pytest.raises(ValueError):
        release.validate_release_authorities(authorities + authorities, producer_keys=set())


def test_producer_key_cannot_be_an_independent_release_authority() -> None:
    context = release_context()
    key_id, public = _public(context["producer_key"])
    context["policy"]["release_authorities"] = [{"key_id": key_id, "public_key": public}]
    context["installation"]["release_identity"]["key_id"] = key_id
    context["release_key"] = context["producer_key"]
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize("field", ["release_identity"])
def test_certificate_is_mandatory_at_release_qualification_boundary(field: str) -> None:
    context = release_context()
    del context["installation"][field]
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize("value", [None, [], True, {}, {"status": "PASS"}])
def test_certificate_cannot_be_replaced_with_a_pass_claim(value: object) -> None:
    context = release_context()
    context["installation"]["release_identity"] = value
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda envelope: envelope.update(key_id="f" * 64),
        lambda envelope: envelope.update(signature="unverified"),
        lambda envelope: envelope.update(signature=base64.b64encode(b"s" * 64).decode("ascii")),
        lambda envelope: envelope.update(public_key="self-authorized"),
        lambda envelope: envelope.pop("signature"),
    ],
)
def test_unsigned_self_authorized_or_substituted_certificate_refuses(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    context = release_context()
    mutate(context["installation"]["release_identity"])
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("product_id", "urn:unrelated:test-package"),
        ("contract", "arbitrary-services-v1"),
        ("security_profile", "TEST_ONLY"),
        ("repository", "unrelated/package"),
        ("source_revision", "b" * 40),
        ("source_revision", "a" * 39),
        ("release_id", ""),
        ("release_id", "x" * 1024),
        ("installation_profile_sha256", "e" * 64),
    ],
)
def test_valid_signature_cannot_authorize_the_wrong_product_or_revision(
    field: str,
    value: object,
) -> None:
    context = release_context()
    context["installation"]["release_identity"]["payload"][field] = value
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    "field",
    [
        "product_id",
        "contract",
        "security_profile",
        "repository",
        "source_revision",
        "release_id",
        "installation_profile_sha256",
        "artifacts",
        "components",
    ],
)
def test_certificate_rejects_incomplete_release_identity_even_if_signed(field: str) -> None:
    context = release_context()
    del context["installation"]["release_identity"]["payload"][field]
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize("phase", ["baseline", "update", "failed_update"])
@pytest.mark.parametrize("field", ["sha256", "version"])
def test_signed_certificate_must_match_every_approved_artifact(phase: str, field: str) -> None:
    context = release_context()
    artifact = context["installation"]["release_identity"]["payload"]["artifacts"][phase]
    artifact[field] = "e" * 64 if field == "sha256" else "99.0.0"
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize("inventory", [[], {}, None, ["placeholder"]])
def test_product_requires_complete_component_inventory(inventory: object) -> None:
    context = release_context()
    context["installation"]["release_identity"]["payload"]["components"] = inventory
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda components: components.pop(),
        lambda components: components.append(copy.deepcopy(components[0])),
        lambda components: components[0].update(role="administrator"),
        lambda components: components[0].update(unit="unrelated-placeholder.service"),
        lambda components: components[0].update(executable_sha256="e" * 64),
        lambda components: components[0].update(update_executable_sha256="e" * 64),
        lambda components: components[0].update(contracts=[]),
        lambda components: components[0]["contracts"].pop(),
        lambda components: components[0]["contracts"].append("arbitrary.functional.claim"),
        lambda components: components[0]["contracts"].append(components[0]["contracts"][0]),
        lambda components: components[0].update(status="PASS"),
    ],
)
def test_signed_placeholder_components_do_not_satisfy_product_contracts(
    mutate: Callable[[list[dict[str, Any]]], None],
) -> None:
    context = release_context()
    mutate(context["installation"]["release_identity"]["payload"]["components"])
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize(
    "field",
    [
        "source_revision",
        "artifact_sha256",
        "update_artifact_sha256",
        "failed_update_artifact_sha256",
    ],
)
def test_release_certificate_and_evidence_materials_must_agree(field: str) -> None:
    context = release_context()
    context["materials"][field] = "d" * (40 if field == "source_revision" else 64)
    with pytest.raises(ValueError):
        _verify(context)


def test_certificate_cannot_be_reused_to_qualify_a_new_source_revision() -> None:
    context = release_context()
    with pytest.raises(ValueError):
        _verify(context, source_revision="b" * 40)


def test_installation_replacement_is_detected_even_when_its_envelope_is_unchanged() -> None:
    context = release_context()
    context["installation"]["package_name"] = "arbitrary-test-only-package"
    context["materials"]["installation_contract_sha256"] = digest(context["installation"])
    with pytest.raises(ValueError):
        _verify(context)


@pytest.mark.parametrize("roles", [("runtime", "administrator"), ("runtime", "runtime")])
def test_independently_signed_generic_inventory_cannot_replace_required_product_roles(
    roles: tuple[str, str],
) -> None:
    context = release_context()
    installation = context["installation"]
    payload = installation["release_identity"]["payload"]
    for unit, component, role in zip(
        installation["units"], payload["components"], roles, strict=True
    ):
        unit["role"] = component["role"] = role
        component["contracts"] = list(release.REQUIRED_COMPONENT_CONTRACTS.get(role, ()))
    payload["installation_profile_sha256"] = digest(
        {name: value for name, value in installation.items() if name != "release_identity"}
    )
    _sign(context)
    with pytest.raises(ValueError):
        _verify(context)


def runtime_release_context() -> dict[str, Any]:
    context = release_context()
    binding = {
        "environment_id": "approved-env-1",
        "manifest_sha256": "1" * 64,
        "source_revision": SOURCE_REVISION,
        "executable": "/opt/cryptohunter/freshness-runtime/releases/approved-env-1/bin/python3.12",
        "base_executable": "/usr/bin/python3.12",
        "base_executable_sha256": "2" * 64,
    }
    installation = context["installation"]
    installation["freshness_runtime"] = binding
    payload = installation["release_identity"]["payload"]
    payload.update(
        schema_version=2, contract=release.RELEASE_CONTRACT_V2, freshness_runtime=binding.copy()
    )
    payload["installation_profile_sha256"] = digest(
        {k: v for k, v in installation.items() if k != "release_identity"}
    )
    _sign(context)
    return context


def test_release_v2_independently_signs_runtime_binding_without_product_pass():
    context = runtime_release_context()
    statement = _verify(context)
    assert statement["freshness_runtime"] == context["installation"]["freshness_runtime"]
    assert statement["schema_version"] == 2
    assert "status" not in statement


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_id", "../foreign"),
        ("manifest_sha256", "0" * 63),
        ("source_revision", "0" * 40),
        ("executable", "/usr/bin/python3.12"),
        ("base_executable", "/tmp/python3.12"),
        ("base_executable_sha256", "0" * 63),
    ],
)
def test_release_v2_refuses_unbound_runtime_even_after_independent_resigning(field, value):
    context = runtime_release_context()
    installation = context["installation"]
    installation["freshness_runtime"][field] = value
    installation["release_identity"]["payload"]["freshness_runtime"][field] = value
    installation["release_identity"]["payload"]["installation_profile_sha256"] = digest(
        {k: v for k, v in installation.items() if k != "release_identity"}
    )
    _sign(context)
    with pytest.raises(release.ReleaseIdentityError, match="runtime"):
        _verify(context)


def test_release_v1_refuses_adding_runtime_as_arbitrary_optional_field():
    context = runtime_release_context()
    payload = context["installation"]["release_identity"]["payload"]
    payload.update(schema_version=1, contract=release.RELEASE_CONTRACT)
    del payload["freshness_runtime"]
    _sign(context)
    with pytest.raises(release.ReleaseIdentityError, match="v2"):
        _verify(context)
