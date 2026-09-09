"""Executable foundation checks for the canonical M0.12 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE_PATH = DOCS / "audit_observability_alerts_and_updater.json"
MARKDOWN_PATH = MACHINE_PATH.with_suffix(".md")
MACHINE: dict[str, Any] = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))

TOP_LEVEL_KEYS = {
    "schema_version",
    "m0_element",
    "status",
    "contract_identity",
    "upstream_dependencies",
    "canonical_vocabulary",
    "ownership_boundaries",
    "trust_boundaries",
    "audit_model",
    "observability_model",
    "health_readiness_model",
    "alert_model",
    "updater_model",
    "release_artifact_authority",
    "failure_policy",
    "rollback_persistence_rules",
    "cross_contract_invariants",
    "current_state_classification",
    "non_goals",
    "open_items",
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _render_markdown(contract: dict[str, Any]) -> str:
    lines = [
        "# M0.12 — Audit, observability, alerts and updater",
        "",
        "> Deterministyczna projekcja kanonicznego JSON. Nie jest niezależnym źródłem semantyki.",
        "",
    ]
    # Preserve canonical JSON insertion order; changing the machine contract changes the projection.
    for key, value in contract.items():
        lines.extend(
            [
                f"## `{key}`",
                "",
                "```json",
                json.dumps(value, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def test_identity_status_and_exact_top_level_shape() -> None:
    assert set(MACHINE) == TOP_LEVEL_KEYS
    assert MACHINE["schema_version"] == "cryptohunter.audit_observability_alerts_and_updater.v1"
    assert MACHINE["m0_element"] == "M0.12"
    assert MACHINE["status"] == "IN_PROGRESS_FOUNDATION"
    assert MACHINE["contract_identity"] == {
        "contract_id": "M0.12-audit-observability-alerts-updater",
        "version": "0.1.0",
        "phase": "S9A_CONTRACT_FOUNDATION",
        "machine_source_of_truth": True,
        "markdown_is_projection_only": True,
    }


def test_upstream_references_are_complete_and_resolve_to_frozen_contracts() -> None:
    dependencies = MACHINE["upstream_dependencies"]
    assert [item["milestone"] for item in dependencies] == [
        "M0.2",
        "M0.3",
        "M0.4",
        "M0.5",
        "M0.6",
        "M0.7",
        "M0.8",
        "M0.9",
        "M0.10",
        "M0.11",
    ]
    freeze = _load("architecture_baseline_freeze.json")
    frozen = {
        item["milestone_id"]: Path(item["canonical_artifact"]).name for item in freeze["milestones"]
    }
    assert {item["milestone"]: item["artifact"] for item in dependencies} == frozen
    assert all((DOCS / item["artifact"]).is_file() for item in dependencies)


def test_canonical_sets_are_closed_unique_and_not_parallel_upstream_vocabulary() -> None:
    vocabulary = MACHINE["canonical_vocabulary"]
    for values in vocabulary.values():
        assert len(values) == len(set(values))
    assert vocabulary["health_dimensions"] == ["LIVENESS", "HEALTH", "READINESS"]
    assert vocabulary["condition_states"] == ["UNKNOWN", "OK", "DEGRADED", "BLOCKED"]
    assert vocabulary["alert_lifecycle_states"] == ["RAISED", "ACKNOWLEDGED", "RESOLVED"]

    capabilities = _load("environment_and_product_capabilities.json")
    environments = [item["environment_id"] for item in capabilities["execution_environments"]]
    identity = _load("identity_device_authentication_and_secrets.json")
    risk = _load("risk_hierarchy_kill_switch_and_execution_lease.json")
    assert environments == identity["registries"]["environments"]
    assert environments == risk["environment_isolation"]["environments"]
    assert "environments" not in vocabulary  # environments stay upstream-owned


def test_no_duplicate_record_surface_or_classification_identities() -> None:
    vocabulary = MACHINE["canonical_vocabulary"]
    assert len(vocabulary["record_types"]) == len(set(vocabulary["record_types"]))
    surfaces = [item["surface"] for item in MACHINE["ownership_boundaries"]]
    boundaries = [item["boundary"] for item in MACHINE["trust_boundaries"]]
    subsystems = [item["subsystem"] for item in MACHINE["current_state_classification"]]
    assert len(surfaces) == len(set(surfaces))
    assert len(boundaries) == len(set(boundaries))
    assert len(subsystems) == len(set(subsystems))


def test_every_ownership_surface_closes_required_architectural_dimensions() -> None:
    exact = {
        "surface",
        "owner",
        "authority_source",
        "durability",
        "ordering",
        "identity",
        "retention",
        "rebuildability",
        "trust_boundary",
        "failure_policy",
    }
    assert all(set(item) == exact for item in MACHINE["ownership_boundaries"])
    by_surface = {item["surface"]: item for item in MACHINE["ownership_boundaries"]}
    audit_owner = by_surface["DURABLE_AUDIT_EVIDENCE"]["owner"]
    assert "M0.2 owns canonical AuditEvent identity/schema" in audit_owner
    assert "lifecycle-phase dependent" in audit_owner
    assert by_surface["UI_PROJECTIONS"]["durability"] == "cache only"
    topology = _load("process_topology_and_lifecycle.json")
    role_by_name = {item["name"]: item for item in topology["process_roles"]}
    assert role_by_name["core_host"]["owns_trading_state"] is True
    assert role_by_name["bootstrapper"]["may_consume_maintenance_authorization"] is True
    assert role_by_name["desktop_shell"]["owns_trading_state"] is False


def test_audit_is_distinct_from_domain_events_and_all_telemetry_forms() -> None:
    distinctions = MACHINE["audit_model"]["distinctions"]
    assert set(distinctions) == {
        "AuditEvent",
        "DomainEventReference",
        "StructuredLogRecord",
        "MetricSample",
        "TraceSpan",
        "HealthObservation",
        "UI_notification",
    }
    assert "canonical M0.2 durable evidence" in distinctions["AuditEvent"]
    assert "M0.7 domain event" in distinctions["DomainEventReference"]
    assert "not copied as audit-store authority" in distinctions["DomainEventReference"]
    properties = set(MACHINE["audit_model"]["required_properties"])
    assert {
        "canonical M0.2 AuditEvent identity/schema",
        "privileged trusted writer appropriate to lifecycle phase",
        "durable",
        "identity-bound",
        "device-bound",
        "environment-bound where applicable",
    } <= properties
    assert "tamper-evident" in properties


def test_audit_event_identity_is_bound_exactly_to_actual_m02_entity() -> None:
    vocabulary = _load("canonical_domain_vocabulary.json")
    matches = [
        entity for entity in vocabulary["entity_kinds"] if entity["canonical_name"] == "AuditEvent"
    ]
    assert len(matches) == 1
    upstream = matches[0]
    binding = MACHINE["canonical_vocabulary"]["audit_event_upstream_binding"]
    assert binding["entity_pointer"] == (
        "/entity_kinds entry selected where canonical_name == AuditEvent"
    )
    assert binding["identity_redefinition_by_M0.12"] is False
    assert {key: upstream[key] for key in ("id_field", "id_prefix", "parent", "persistence")} == {
        "id_field": "audit_event_id",
        "id_prefix": "evt",
        "parent": "DeviceInstallation",
        "persistence": True,
    }
    assert "AuditEvent" in MACHINE["canonical_vocabulary"]["record_types"]


def test_phase_aware_writers_follow_actual_m03_bootstrap_boundary() -> None:
    topology = _load("process_topology_and_lifecycle.json")
    upstream = topology["first_run_bootstrap_authority_contract"]
    assert upstream["audit_boundary"]["uses"].startswith("canonical M0.2 AuditEvent")
    assert "no artificial pre-RuntimeSession" in upstream["audit_boundary"]["ordering"]
    assert upstream["bootstrapper_role"]["mode"] == "transport_and_discovery_only"
    assert upstream["bootstrapper_role"]["cannot"] == ["mint", "accept", "elevate"]

    phases = {item["phase"]: item for item in MACHINE["audit_model"]["writer_authority"]["phases"]}
    assert phases["PRE_CORE"]["runtime_session_required"] is False
    assert "external product provisioning boundary" in phases["PRE_CORE"]["trusted_append_path"]
    assert "cannot mint, accept or elevate" in " ".join(phases["PRE_CORE"]["constraints"])
    assert phases["CORE_RUNTIME"]["trusted_append_path"] == "CoreHost"
    maintenance = phases["MAINTENANCE_UPDATE_RECOVERY"]
    assert (
        "Bootstrapper only as the M0.3 maintenance/update participant"
        in maintenance["trusted_append_path"]
    )
    assert "no trading-state authority" in maintenance["constraints"]
    assert MACHINE["audit_model"]["writer_authority"]["direct_writer_non_authorities"][:2] == [
        "DesktopShell",
        "TrayAgent",
    ]


def test_m07_m08_and_m09_consume_the_same_upstream_audit_event_identity() -> None:
    commands = _load("commands_events_order_lifecycle_and_idempotency.json")
    m07 = commands["identity_policy"]["durable_ids"]["AuditEvent"]
    vocabulary = _load("canonical_domain_vocabulary.json")
    upstream = next(
        entity for entity in vocabulary["entity_kinds"] if entity["canonical_name"] == "AuditEvent"
    )
    assert m07 == {"field": upstream["id_field"], "prefix": upstream["id_prefix"]}

    ledger = _load("ledger_portfolio_capital_and_pnl.json")
    assert ledger["source_registry"]["deposit"]["identity"] == upstream["id_field"]
    assert "exact-economic AuditEvent" in ledger["source_registry"]["deposit"]["authority"]
    risk = _load("risk_hierarchy_kill_switch_and_execution_lease.json")
    reservation = risk["reservation_relation_to_m08"]
    assert upstream["id_field"] in reservation["exact_binding"]
    assert "AuditEvent-derived fact" in reservation["identity"]


def test_no_parallel_audit_identity_or_new_entity_prefix_and_other_ids_are_classified() -> None:
    serialized = json.dumps(MACHINE, ensure_ascii=False)
    assert "audit_" + "record_id" not in serialized
    assert "AuditRecord" not in MACHINE["canonical_vocabulary"]["record_types"]
    classification = MACHINE["canonical_vocabulary"]["identity_classification"]
    assert classification["new_M02_style_prefixes_declared"] is False
    classified = {
        entry["field"]
        for key, entries in classification.items()
        if key != "new_M02_style_prefixes_declared"
        for entry in entries
    }
    assert classified == {
        "audit_event_id",
        "runtime_session_id",
        "contract_id",
        "correlation_id",
        "observation_id",
        "log_record_id",
        "alert_id",
        "release_id",
        "build_id",
        "artifact_id",
        "update_attempt_id",
        "signing_key_id",
    }


def test_m011_registry_remains_frozen_and_audit_journal_is_explicitly_deferred() -> None:
    registry_source = (ROOT / "bot_core/persistence/record_registry.py").read_text(encoding="utf-8")
    assert '"AuditEvent"' not in registry_source
    assert '"AuditRecord"' not in registry_source
    persistence = _load("persistence_versioning_migrations_backup_and_recovery.json")
    assert (
        persistence["state_store_physical_schema_registry"]["current_state_store_schema_version"]
        == 2
    )
    open_items = " ".join(MACHINE["open_items"])
    assert "physical durable AuditEvent journal/carrier" in open_items
    assert "without adding a registry entry or StateStore schema v3" in open_items


def test_observations_and_alerts_never_create_upstream_authority() -> None:
    rules = " ".join(MACHINE["observability_model"]["observation_rules"])
    alert_principles = " ".join(MACHINE["alert_model"]["principles"])
    assert "never mutate upstream authority" in rules
    assert "never mint upstream authority" in " ".join(MACHINE["cross_contract_invariants"])
    for authority in ("ProductCapabilities", "readiness", "authentication", "risk", "execution"):
        assert authority in alert_principles
    lifecycle = MACHINE["alert_model"]["lifecycle"]
    assert "source fault remains active" in lifecycle["ACKNOWLEDGED"]
    assert "trusted resolution evidence" in lifecycle["RESOLVED"]


def test_health_readiness_composes_upstream_lifecycle_and_capabilities_fail_closed() -> None:
    model = MACHINE["health_readiness_model"]
    assert set(model["dimensions"]) == set(MACHINE["canonical_vocabulary"]["health_dimensions"])
    assert set(model["condition_states"]) == set(
        MACHINE["canonical_vocabulary"]["condition_states"]
    )
    composition = " ".join(model["composition"])
    assert "never collapse dimensions into one healthy boolean" in composition
    assert "M0.3 startup readiness" in composition
    assert "M0.4 ProductCapabilities" in composition
    assert "UNKNOWN safety evidence is BLOCKED" in composition


def test_download_is_not_install_authority_and_exact_verification_precedes_install() -> None:
    phases = {item["phase"]: item for item in MACHINE["updater_model"]["phase_rules"]}
    assert phases["DOWNLOAD"]["authority_created"] is False
    assert phases["INSTALLATION"]["authority_created"] is False
    assert "exact bytes" in phases["VERIFICATION"]["output"]
    assert list(phases).index("VERIFICATION") < list(phases).index("AUTHORIZATION")
    assert list(phases).index("AUTHORIZATION") < list(phases).index("INSTALLATION")
    eligibility = " ".join(MACHINE["release_artifact_authority"]["install_eligibility_requires"])
    assert "recomputed exact artifact digest and size" in eligibility
    assert "trusted non-revoked signing identity" in eligibility
    assert "M0.11 StateStore schema compatibility" in eligibility
    assert "never installable authority" in MACHINE["release_artifact_authority"]["candidate_rule"]


def test_rollback_respects_actual_m011_forward_only_schema_authority() -> None:
    persistence = _load("persistence_versioning_migrations_backup_and_recovery.json")
    versions = {
        item["state_store_schema_version"]
        for item in persistence["state_store_physical_schema_registry"]["entries"]
    }
    current = persistence["state_store_physical_schema_registry"][
        "current_state_store_schema_version"
    ]
    assert current in versions
    assert persistence["migration_protocol"]["rollback_policy"] == "FORWARD_ONLY"
    rules = " ".join(MACHINE["rollback_persistence_rules"])
    assert "FORWARD_ONLY" in rules
    assert "never downgrade StateStore schema" in rules
    assert "rollback is BLOCKED" in rules
    assert "M0.3 protected restore-freshness authority" in rules


def test_identity_artifact_authenticity_and_operator_authorization_are_separate() -> None:
    identity = _load("identity_device_authentication_and_secrets.json")
    assert identity["authority"]["owner"] == "CoreHost"
    trust_model = " ".join(MACHINE["updater_model"]["trust_model"])
    assert "release authenticity is distinct from M0.10 user authentication" in trust_model
    assert "signature never substitutes for operator authorization" in trust_model
    assert "never substitutes for artifact authenticity" in trust_model


def test_live_remains_blocked_without_an_alternate_gate() -> None:
    capabilities = _load("environment_and_product_capabilities.json")
    live = capabilities["current_product_edition"]["environment_capabilities"]["LIVE"]
    assert live["executable"] is False
    assert live["denial_code"] == "LIVE_BLOCKED_BY_EDITION"
    invariants = " ".join(MACHINE["cross_contract_invariants"])
    assert "LIVE remains target-capable but blocked" in invariants
    assert "no alternate LIVE gate" in invariants


def test_inventory_classifications_are_closed_and_evidence_backed() -> None:
    allowed = {"KEEP", "ADAPT", "REWRITE", "DELETE", "SOURCE_ONLY"}
    classifications = MACHINE["current_state_classification"]
    assert {item["classification"] for item in classifications} == allowed
    for item in classifications:
        assert set(item) == {"subsystem", "classification", "evidence", "reason"}
        assert item["evidence"]
        assert item["reason"]
        for evidence in item["evidence"]:
            assert (ROOT / evidence.rstrip("/")).exists(), evidence


def test_markdown_is_an_exact_deterministic_projection() -> None:
    assert MARKDOWN_PATH.read_text(encoding="utf-8") == _render_markdown(MACHINE)
