"""Executable checks for the CryptoHunterAccount genesis authority model."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_genesis_authority_model.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_genesis_authority_model.md"
ROOT_RECONCILIATION = DOCS / "m05_cryptohunter_account_root_of_trust_reconciliation.json"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    return (
        "# M0.5 CryptoHunterAccount genesis authority model design\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_genesis_authority_model.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(d: dict) -> None:
    assert d["repository_head_examined"] == "643680bf56b5a16d7ff67acd63de61456eb60a43"
    assert d["account_subject_identity"]["classification"] == (
        "ACCOUNT_GENESIS_SUBJECT_IDENTITY = NOT_FOUND"
    )
    assert d["account_subject_identity"]["subject_to_account_cardinality"] == "NOT_FROZEN"
    assert d["account_subject_identity"]["one_subject_one_account"] == "NOT_PROVEN"
    assert (
        d["account_subject_identity"]["missing_subject_identity_alone_makes_genesis_impossible"]
        is False
    )
    assert d["selected_or_blocked_mint_model"]["selection"] == "E_DESIGN_BLOCKED"
    assert d["selected_or_blocked_mint_model"]["account_id_mint_owner"] == "NOT_FROZEN"
    states = d["account_id_state_machine"]["states"]
    assert len(states) == 4
    assert [state["authority"] for state in states] == [False, False, False, True]
    assert (
        d["account_device_ordering"]["ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE"]["authority_protocol"]
        == "NOT_FOUND"
    )
    assert d["selected_or_blocked_ordering_model"]["selection"] == "DESIGN_BLOCKED"
    assert d["circularity"]["non_circular_root_establishment"] == "NOT_PROVEN"
    assert d["genesis_decision_authority"]["owner"] == "NOT_FOUND"
    assert d["serialization"]["selected_scope"] == "DESIGN_BLOCKED"
    assert d["serialization"]["arrival_order_selects_winner"] is False
    assert d["serialization"]["local_timestamp_selects_winner"] is False
    assert d["serialization"]["last_writer_wins"] is False
    assert d["idempotency"]["selected_identity"] == "DESIGN_BLOCKED"
    assert d["concurrency"]["same_account"]["second_genesis"] == "FORBIDDEN"
    assert d["concurrency"]["distinct_account_same_subject"]["closure_status"].startswith(
        "DESIGN_BLOCKED"
    )
    assert d["concurrency"]["distinct_account_same_subject"]["classification"] == (
        "DOMAIN_SEMANTICS_NOT_FROZEN"
    )
    assert d["subject_cardinality_parity"]["subject_to_account_cardinality"] == "NOT_FROZEN"
    assert d["subject_cardinality_parity"]["one_subject_one_account"] == "NOT_PROVEN"
    assert d["subject_cardinality_parity"]["canonical_vocabulary_defines_cardinality"] is False
    assert d["reservation_crash_recovery"]["reuse"] == "FORBIDDEN by default"
    assert d["genesis_crash_recovery"]["new_account"] == "FORBIDDEN"
    assert d["multi_device_after_genesis"]["account_genesis"] == "FORBIDDEN"
    root_evidence = d["account_root_provisioning_evidence"]
    assert root_evidence["role"] == "conditional root-proof candidate only"
    assert root_evidence["timing"] == "NOT_FROZEN"
    assert root_evidence["before_account_genesis"].startswith("NOT_FROZEN")
    assert root_evidence["inside_atomic_genesis_protocol"].startswith("NOT_FROZEN")
    assert root_evidence["selected_timing"] == "NONE"
    assert root_evidence["account_authority"] is False
    device_membership = d["first_device_provisioning_membership"]
    assert device_membership["role"] == (
        "DeviceInstallation / M0.3 initial-security admission evidence"
    )
    assert device_membership["references_account_id_is_account_authority"] is False
    assert device_membership["before_account_genesis"] == "NOT_SELECTED / NOT_PROVEN"
    relationship = d["root_evidence_device_membership_relationship"]
    assert relationship["same_artifact_or_authority"] == "NOT_FROZEN"
    assert relationship["equivalence_asserted"] is False
    assert relationship["difference_asserted"] is False
    assert d["provisioning_timing"]["all_timing_paths_simultaneously_selected"] is False
    assert "NEVER trigger account genesis" in d["multi_device_after_genesis"]["invariant"]
    assert d["impact_on_M03"]["FirstRunBootstrapAuthority_scope_expanded"] is False
    assert d["authentication_requirements"]["public_SHA_is_authenticity"] is False
    assert d["authentication_requirements"]["M0.11_carrier_is_root_proof"] is False
    assert d["result"]["primary_result"] == "ACCOUNT_GENESIS_MODEL_DESIGN_BLOCKED"
    assert all(value is False for value in d["implementation_allowed"].values())
    assert set(d["mandatory_redteam_mutations"].values()) == {"FAIL"}
    assert len(d["mandatory_redteam_mutations"]) == 18


def validate_root_candidate_timing_parity(genesis: dict, root: dict) -> None:
    candidate = root["root_candidate_evaluation"]
    reconciliation = genesis["root_candidate_reconciliation"]
    assert reconciliation["root_candidate_source"] == ROOT_RECONCILIATION.name
    assert reconciliation["root_candidate"] == candidate["candidate"]
    assert candidate["result"] == "VIABLE_ONLY_AFTER_ADDITIONAL_AUTHORITY"
    assert candidate["condition"] == "AND_AFTER_INTRINSIC_GENESIS_SEMANTICS_ARE_FROZEN"
    assert reconciliation["root_candidate_status"] == (
        f"{candidate['result']} {candidate['condition']}"
    )
    assert reconciliation["superseded"] is False
    assert root["circularity_analysis"]["pre_admission_external_establishment"] == "NOT_PROVEN"
    assert genesis["account_root_provisioning_evidence"]["before_account_genesis"].startswith(
        "NOT_FROZEN / CONDITIONAL_CANDIDATE"
    )


def test_markdown_is_deterministic_complete_projection() -> None:
    data = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(data)
    validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["account_id_state_machine"]["states"][0].update(authority=True),
        lambda d: d["account_id_state_machine"]["states"][1].update(authority=True),
        lambda d: d["impact_on_M03"].update(FirstRunBootstrapAuthority_scope_expanded=True),
        lambda d: d["concurrency"]["same_account"].update(second_genesis="ALLOW"),
        lambda d: d["concurrency"]["distinct_account_same_subject"].update(closure_status="ALLOW"),
        lambda d: d["concurrency"]["distinct_account_same_subject"].update(
            classification="SECURITY_CONFLICT"
        ),
        lambda d: d["account_subject_identity"].update(one_subject_one_account=True),
        lambda d: d["account_subject_identity"].update(
            missing_subject_identity_alone_makes_genesis_impossible=True
        ),
        lambda d: d["serialization"].update(arrival_order_selects_winner=True),
        lambda d: d["serialization"].update(local_timestamp_selects_winner=True),
        lambda d: d["serialization"].update(last_writer_wins=True),
        lambda d: d["genesis_crash_recovery"].update(new_account="ALLOW"),
        lambda d: d["multi_device_after_genesis"].update(account_genesis="ALLOW"),
        lambda d: d["authentication_requirements"].update(public_SHA_is_authenticity=True),
        lambda d: d["authentication_requirements"].update({"M0.11_carrier_is_root_proof": True}),
        lambda d: d["root_evidence_device_membership_relationship"].update(
            equivalence_asserted=True
        ),
        lambda d: d["first_device_provisioning_membership"].update(
            references_account_id_is_account_authority=True
        ),
    ],
)
def test_required_redteam_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises((AssertionError, KeyError)):
        validate(altered)


def test_cross_artifact_status_parity() -> None:
    data = load()
    for filename in (
        "m05_cryptohunter_account_root_of_trust_reconciliation.json",
        "m05_cryptohunter_account_authority_contract_design.json",
        "m05_cryptohunter_account_authority_discovery.json",
    ):
        other = json.loads((DOCS / filename).read_text(encoding="utf-8"))
        statuses = other["preserved_status"]
        for key in (
            "CryptoHunterAccountAuthority",
            "WorkspaceAuthority",
            "FullFillAuthority",
            "M0.8",
        ):
            if key in statuses:
                assert statuses[key].startswith(data["preserved_status"][key])


def test_root_candidate_and_genesis_timing_semantic_parity() -> None:
    genesis = load()
    root = json.loads(ROOT_RECONCILIATION.read_text(encoding="utf-8"))
    validate_root_candidate_timing_parity(genesis, root)


def test_pre_account_root_evidence_rejection_breaks_semantic_parity() -> None:
    genesis = deepcopy(load())
    root = json.loads(ROOT_RECONCILIATION.read_text(encoding="utf-8"))
    genesis["account_root_provisioning_evidence"]["before_account_genesis"] = "NOT_AUTHORIZED"
    with pytest.raises(AssertionError):
        validate_root_candidate_timing_parity(genesis, root)
