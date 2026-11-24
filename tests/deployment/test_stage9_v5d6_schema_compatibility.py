"""Legacy proposal digests and receipts cannot approve a V5d6 header omission."""

from __future__ import annotations

import copy

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d4_environment as env
from scripts import stage9_v5d4_launch as launch
from tests.deployment.test_stage9_v5d4_dependencies import material, native_inventory
from tests.deployment.test_stage9_v5d4_environment import staging
from tests.deployment.test_stage9_v5d5_environment import shared_staging
from tests.deployment.test_stage9_v5d5_shared import shared_material
from tests.deployment.test_stage9_v5d6_environment import header_staging
from tests.deployment.test_stage9_v5d6_headers import header_material

pytestmark = [pytest.mark.security, pytest.mark.unit]


@pytest.mark.parametrize(
    "profile,material_fixture", (("v5d4", "material"), ("v5d5", "shared_material"))
)
def test_legacy_proposal_requires_regeneration_and_new_approved_sha256(
    profile, material_fixture, request
):
    fixture = request.getfixturevalue(material_fixture)
    if profile == "v5d5":
        project, directory, origins, proposal, approved_shared = fixture
    else:
        project, directory, origins = fixture
        proposal = deps.propose(project, directory, origins)
        approved_shared = None
    assert proposal["profile"] == f"stage9-{profile}-dependency-proposal-v1"
    legacy = copy.deepcopy(proposal)
    assert legacy.pop("header_omissions")["status"] == "NO_HEADER_OMISSIONS"
    for package in legacy["packages"]:
        assert package.pop("header_omissions") == []
    legacy_sha = deps.digest(deps.canonical(legacy))
    regenerated_sha = deps.digest(deps.canonical(proposal))
    assert regenerated_sha != legacy_sha

    def verify(candidate, expected_sha):
        return deps.verify(
            project,
            directory,
            origins,
            candidate,
            expected_sha,
            native_inventory(proposal),
            approved_shared=approved_shared,
        )

    with pytest.raises(deps.Refusal, match="^PROPOSAL_CONTENT_MISMATCH$"):
        verify(legacy, legacy_sha)
    with pytest.raises(deps.Refusal, match="^PROPOSAL_SHA256_MISMATCH$"):
        verify(proposal, legacy_sha)
    # The synthetic reviewer binds the regenerated artifact for this test only.
    verified = verify(proposal, regenerated_sha)
    assert verified["profile"] == f"stage9-{profile}-dependency-verification-v1"
    assert verified["proposal_sha256"] == regenerated_sha
    assert verified["header_omissions"] == proposal["header_omissions"]
    assert verified["header_expectations"] is None
    assert verified["header_policy"] == proposal["header_omissions"]["policy"]
    assert verified["authorization"] == "NOT_AUTHORIZED"


def remove_new_verification_fields(receipt):
    verification = receipt["dependency_verification"]
    for field in ("header_omissions", "header_expectations", "header_policy"):
        verification.pop(field)


def test_legacy_shared_receipt_without_header_candidate_passes_static_checks(shared_staging):
    inputs, approval, output, _ = shared_staging
    receipt = env.build(inputs, approval, output)
    remove_new_verification_fields(receipt)
    for package in receipt["wheel_materialization"]:
        assert package.pop("header_omissions") == []
    assert "header_omissions" not in receipt
    assert "header-audit.py" not in receipt["files"]
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    launch.verify_shared_receipt(output, receipt)
    launch.verify_header_receipt(output, receipt)
    assert receipt["authorization"] == "NOT_AUTHORIZED"


@pytest.mark.parametrize(
    "legacy_part,reason",
    (
        ("verification", "HEADER_OMISSION_VERIFICATION_MISMATCH"),
        ("receipt", "HEADER_OMISSION_RECEIPT_INCOMPLETE"),
    ),
)
def test_header_candidate_refuses_legacy_receipt_or_embedded_verification(
    header_staging, legacy_part, reason
):
    inputs, approval, output, _ = header_staging
    receipt = env.build(inputs, approval, output)
    remove_new_verification_fields(receipt)
    if legacy_part == "receipt":
        for field in (
            "header_omissions",
            "header_omission_expectations",
            "header_omission_policy",
            "header_runtime_inclusion",
        ):
            receipt.pop(field)
        receipt["wheel_materialization"][0].pop("header_omissions")
        receipt["files"].pop("header-audit.py")
        (output / "header-audit.py").unlink()
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    launch.verify_shared_receipt(output, receipt)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        launch.verify_header_receipt(output, receipt)
