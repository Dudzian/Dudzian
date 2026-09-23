"""Executable checks for the M0.3/account root-of-trust reconciliation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path

import pytest

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    ConsumedBootstrapAuthority,
    FirstRunBootstrapClaim,
    ProvisioningBoundary,
    ProvisioningMembershipBinding,
)

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_root_of_trust_reconciliation.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_root_of_trust_reconciliation.md"
ACCOUNT_DESIGN = DOCS / "m05_cryptohunter_account_authority_contract_design.json"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    return (
        "# M0.5 CryptoHunterAccount root-of-trust reconciliation\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_root_of_trust_reconciliation.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate_cross_artifact(account_design: dict, reconciliation: dict) -> None:
    reservation = reconciliation["account_id_reservation_problem"]
    assert reservation["prior_account_design_owner_before_reconciliation"] == (
        "future genuine CryptoHunterAccountAuthority only"
    )
    assert reservation["prior_owner_status"] == "REOPENED / SUPERSEDED"
    assert reservation["end_to_end_account_id_ownership_protocol"] == "UNRESOLVED"
    assert reservation["end_to_end_account_id_mint_protocol"] == "NOT_FROZEN"
    assert reservation["models"]["E_DESIGN_BLOCKED"] == "SELECTED"

    minting = account_design["id_minting"]
    assert minting["account_id_owner"] in {
        "UNRESOLVED_PENDING_ROOT_OF_TRUST_RECONCILIATION",
        "NOT_FROZEN",
    }
    assert (
        reconciliation["impact_on_current_account_design"]["account_id_owner"]
        == (minting["account_id_owner"])
    )
    assert reservation["current_account_id_owner"] == minting["account_id_owner"]
    assert minting["status"] == "DESIGN_BLOCKED"
    assert minting["mint_reservation_protocol"] == "NOT_FROZEN"
    assert minting["entropy_clock_owner"] == "NOT_FROZEN"
    assert minting["caller_selected_genuine_account_id"] == "FORBIDDEN"
    assert "account_id" not in account_design["admission"]["request_must_not_contain"]
    assert account_design["admission"]["authority_bound_account_id_input_presence"] == (
        "NOT_FROZEN"
    )
    assert account_design["account_id_ownership_reopened_by"] == MACHINE.name
    assert reconciliation["supersedes_or_reopens"]["artifact"] == ACCOUNT_DESIGN.name
    assert reconciliation["supersedes_or_reopens"]["whole_prior_design_invalid"] is False


def validate(data: dict) -> None:
    assert data["iteration"] == "ROOT_OF_TRUST_DISCOVERY_RECONCILIATION_ONLY"
    assert data["provenance"]["classification"] == "UNKNOWN"
    assert data["production_implementation_search"]["classification"] == (
        "SEMANTIC_KERNEL_FOUND_BUT_UPSTREAM_PORT_EXTERNAL"
    )
    assert data["upstream_dependency_status"] == "BLOCKED_UPSTREAM"
    assert data["upstream_dependency"] == (
        "external_product_provisioning_boundary production implementation"
    )
    assert data["production_provisioning_authority_availability"] == ("NOT_FOUND / NOT_AVAILABLE")
    provenance = data["account_id_provenance"]
    assert provenance["handoff_supplies_account_id"] is True
    assert provenance["handoff_mints_account_id"] == "NOT_PROVEN"
    assert provenance["classification"] == "ACCOUNT_ID_PROVENANCE_STOPS_UPSTREAM"
    assert data["account_id_mint_owner"]["existing_owner"] == "NOT_FOUND"
    assert data["m03_existing_authorities"]["FirstRunBootstrapAuthority"]["purpose"] == (
        "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert data["root_candidate_evaluation"]["result"] == ("VIABLE_ONLY_AFTER_ADDITIONAL_AUTHORITY")
    assert data["circularity_analysis"]["classification"] == "UNRESOLVED"
    assert data["account_vs_device_ordering"]["classification"] == "UNRESOLVED"
    assert data["impact_on_current_account_design"]["classification"] == "UNRESOLVED"
    assert data["impact_on_current_account_design"]["account_id_owner"] == (
        "UNRESOLVED_PENDING_ROOT_OF_TRUST_RECONCILIATION"
    )
    assert data["impact_on_current_account_design"]["competing_mint_owner_conflict"] == "NO"
    assert data["impact_on_current_account_design"]["narrow_finding"] == (
        "NO_COMPETING_MINT_OWNER_FOUND"
    )
    reservation = data["account_id_reservation_problem"]
    assert reservation["prior_owner_status"] == "REOPENED / SUPERSEDED"
    assert reservation["current_account_id_owner"] == (
        "UNRESOLVED_PENDING_ROOT_OF_TRUST_RECONCILIATION"
    )
    assert reservation["end_to_end_account_id_mint_protocol"] == "NOT_FROZEN"
    assert reservation["end_to_end_account_id_ownership_protocol"] == "UNRESOLVED"
    assert reservation["models"]["E_DESIGN_BLOCKED"] == "SELECTED"
    assert reservation["syntax_or_reservation_establishes_authority"] is False
    assert data["impact_on_current_account_design"]["implementation_allowed"] is False
    assert data["intrinsic_root_proof_semantics"] == "INSUFFICIENT"
    assert len(data["intrinsic_semantic_gaps"]) == 7
    assert data["upstream_only_resolution_sufficient"] is False
    assert data["root_candidate_evaluation"]["additional_authority_alone_sufficient"] is False
    assert data["root_candidate_evaluation"]["condition"] == (
        "AND_AFTER_INTRINSIC_GENESIS_SEMANTICS_ARE_FROZEN"
    )
    assert data["result"] == "ROOT_PROOF_INSUFFICIENT_SEMANTICS"
    assert data["supersedes_or_reopens"]["scope_only"] == [
        "id_minting.account_id_owner",
        "id_minting.algorithm ownership and mint/reservation protocol",
        "admission authority-bound account_id input presence",
    ]
    assert data["supersedes_or_reopens"]["whole_prior_design_invalid"] is False
    statuses = data["preserved_status"]
    assert statuses["CryptoHunterAccountAuthority"] == "NOT_AVAILABLE"
    assert statuses["WorkspaceAuthority"] == "NOT_AVAILABLE"
    assert statuses["FullFillAuthority"] == "NOT_AVAILABLE"
    assert statuses["production_M0.5"] == "NOT_AVAILABLE"
    assert statuses["M0.8"] == "NOT_AVAILABLE"

    failures = data["mandatory_redteam_mutations"]
    assert len(failures) == 10
    assert set(failures.values()) == {"FAIL"}
    assert data["authentication"]["caller_generated_correct_sha"] == "DENY"
    roles = data["m03_existing_authorities"]
    assert roles["FirstRunBootstrapAuthority"]["may_mint_account_id"] is False
    assert roles["Bootstrapper"]["authority_owner"] is False
    assert roles["CoreHost"]["may_accept_account_identity"] is False
    assert data["authentication"]["claim_fingerprint_role"].endswith("only")
    assert len(data["multi_device_semantics"]["forbidden"]) == 3
    assert "existing genuine acct_A" in data["multi_device_semantics"]["required_proof"]
    assert "must never mint another account" in data["replay"]["account_admission_gap"]
    assert data["provisioning_claim_contract"]["public_fingerprint_is_authenticity"] is False
    assert roles["M0.11_durable_registry"]["authority_owner"] is False

    # An external adapter becoming available cannot manufacture missing genesis semantics.
    intrinsic_unresolved = (
        data["account_vs_device_ordering"]["classification"] == "UNRESOLVED"
        or data["circularity_analysis"]["classification"] == "UNRESOLVED"
        or data["account_id_mint_owner"]["existing_owner"] == "NOT_FOUND"
    )
    if intrinsic_unresolved:
        assert data["result"] != "ROOT_PROOF_BLOCKED_UPSTREAM"
        assert data["upstream_only_resolution_sufficient"] is False

    assert (
        data["circularity_analysis"]["production_ProvisioningBoundary_AVAILABLE_solves_circularity"]
        is False
    )
    assert (
        data["account_vs_device_ordering"][
            "production_provisioning_authority_available_selects_model"
        ]
        is False
    )
    assert data["account_vs_device_ordering"]["models"] == {
        "ACCOUNT_FIRST": "NOT_SELECTED / NOT_PROVEN",
        "ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE": "NOT_SELECTED / NOT_PROVEN",
        "DESIGN_BLOCKED": "SELECTED",
    }
    distinct_race = data["account_vs_device_ordering"]["first_device_race"]
    assert distinct_race["claims"] == ["acct_A / dev_1", "acct_B / dev_2"]
    assert distinct_race["result"] == "DESIGN_BLOCKED"
    assert distinct_race["winner_selection_forbidden"] == [
        "arrival order",
        "local timestamp",
        "device startup order",
        "last writer wins",
    ]
    same_account_race = data["account_vs_device_ordering"]["same_account_first_device_race"]
    assert same_account_race["claims"] == ["acct_A / dev_1", "acct_A / dev_2"]
    assert same_account_race["result"] == "DESIGN_BLOCKED"
    assert "must not create two genesis facts" in same_account_race["rule"]
    assert set(data["account_genesis_semantics"].values()) == {
        "NOT_DEFINED",
        "DESIGN_BLOCKED",
    }


def test_markdown_is_deterministic_complete_projection() -> None:
    data = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(data)
    validate(data)


def test_frozen_schemas_match_production_code_exactly() -> None:
    data = load()
    assert data["provisioning_claim_contract"]["fields"] == [
        field.name for field in fields(FirstRunBootstrapClaim)
    ]
    assert data["membership_contract"]["fields"] == [
        field.name for field in fields(ProvisioningMembershipBinding)
    ]
    assert data["membership_contract"]["consumed_fields"] == [
        field.name for field in fields(ConsumedBootstrapAuthority)
    ]
    assert data["membership_contract"]["authority_source_exact"] == AUTHORITY_SOURCE
    assert data["m03_existing_authorities"]["FirstRunBootstrapAuthority"]["purpose"] == (
        INITIAL_SECURITY_ESTABLISHMENT_ONLY
    )
    assert ProvisioningBoundary.__doc__ and "pre-existing external" in ProvisioningBoundary.__doc__


def test_account_design_and_reconciliation_have_account_id_parity() -> None:
    account_design = json.loads(ACCOUNT_DESIGN.read_text(encoding="utf-8"))
    reconciliation = load()
    validate_cross_artifact(account_design, reconciliation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda design, _: design["id_minting"].update(
            account_id_owner="future genuine CryptoHunterAccountAuthority only"
        ),
        lambda design, _: design["id_minting"].update(status="PARTIALLY_FROZEN_WITH_OWNER"),
        lambda design, _: design["admission"]["request_must_not_contain"].append("account_id"),
        lambda design, reconciliation: (
            reconciliation["account_id_reservation_problem"]["models"].update(
                B_EXTERNAL_AUTHORITY_MINTS_ACCOUNT_ID="POSSIBLE",
                C_ATOMIC_ISSUER_ACCOUNT_AUTHORITY_PROTOCOL="POSSIBLE",
            ),
            design["id_minting"].update(
                account_id_owner="future genuine CryptoHunterAccountAuthority only"
            ),
        ),
        lambda _, reconciliation: reconciliation["account_id_reservation_problem"].update(
            current_account_id_owner="future genuine CryptoHunterAccountAuthority only"
        ),
        lambda _, reconciliation: reconciliation["account_id_reservation_problem"].update(
            prior_owner_status="CURRENT"
        ),
    ],
    ids=[
        "concrete-owner-while-unresolved",
        "frozen-owner-status-while-protocol-open",
        "permanent-request-account-id-ban",
        "external-or-atomic-model-versus-account-authority-only",
        "stale-owner-masquerades-as-current",
        "historical-owner-loses-superseded-marker",
    ],
)
def test_cross_artifact_account_id_regressions_fail(mutation) -> None:
    account_design = json.loads(ACCOUNT_DESIGN.read_text(encoding="utf-8"))
    reconciliation = load()
    mutation(account_design, reconciliation)
    with pytest.raises(AssertionError):
        validate_cross_artifact(account_design, reconciliation)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "caller-hash-is-authority",
            lambda x: x["authentication"].update(caller_generated_correct_sha="ALLOW"),
        ),
        (
            "bootstrap-is-account-authority",
            lambda x: x["m03_existing_authorities"]["FirstRunBootstrapAuthority"].update(
                may_mint_account_id=True
            ),
        ),
        (
            "bootstrapper-mints-membership",
            lambda x: x["m03_existing_authorities"]["Bootstrapper"].update(authority_owner=True),
        ),
        (
            "core-self-enrolls",
            lambda x: x["m03_existing_authorities"]["CoreHost"].update(
                may_accept_account_identity=True
            ),
        ),
        (
            "syntax-proves-membership",
            lambda x: x["authentication"].update(claim_fingerprint_role="authority"),
        ),
        ("second-genesis", lambda x: x["multi_device_semantics"].update(forbidden=[])),
        (
            "device-two-new-account",
            lambda x: x["multi_device_semantics"].update(required_proof="mint acct_B"),
        ),
        (
            "replay-mints-account",
            lambda x: x["replay"].update(account_admission_gap="replay mints another account"),
        ),
        (
            "public-sha-is-membership",
            lambda x: x["provisioning_claim_contract"].update(
                public_fingerprint_is_authenticity=True
            ),
        ),
        (
            "carrier-is-root",
            lambda x: x["m03_existing_authorities"]["M0.11_durable_registry"].update(
                authority_owner=True
            ),
        ),
        ("fake-upstream-only-result", lambda x: x.update(result="ROOT_PROOF_BLOCKED_UPSTREAM")),
        (
            "upstream-magically-selects-ordering",
            lambda x: x["account_vs_device_ordering"].update(
                production_provisioning_authority_available_selects_model=True
            ),
        ),
        (
            "ownership-falsely-complete",
            lambda x: x["account_id_reservation_problem"].update(
                end_to_end_account_id_ownership_protocol="FROZEN"
            ),
        ),
    ],
)
def test_mandatory_redteam_mutations_fail(name, mutate) -> None:
    altered = deepcopy(load())
    mutate(altered)
    with pytest.raises(AssertionError):
        validate(altered)


def test_required_sections_are_present() -> None:
    required = {
        "repository_head_examined",
        "provenance",
        "m03_existing_authorities",
        "provisioning_claim_contract",
        "membership_contract",
        "production_implementation_search",
        "account_id_provenance",
        "account_id_mint_owner",
        "account_vs_device_ordering",
        "multi_device_semantics",
        "root_candidate_evaluation",
        "circularity_analysis",
        "replay",
        "expiry",
        "restart_provenance",
        "authentication",
        "rollback_freshness",
        "impact_on_current_account_design",
        "impact_on_workspace",
        "account_id_reservation_problem",
        "account_genesis_semantics",
        "intrinsic_root_proof_semantics",
        "intrinsic_semantic_gaps",
        "upstream_dependency_status",
        "upstream_dependency",
        "production_provisioning_authority_availability",
        "upstream_only_resolution_sufficient",
        "supersedes_or_reopens",
        "result",
        "next_stage",
        "preserved_status",
    }
    assert required <= load().keys()
