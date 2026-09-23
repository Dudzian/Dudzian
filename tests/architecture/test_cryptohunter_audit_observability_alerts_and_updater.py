"""Executable foundation checks for the canonical M0.12 contract."""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from dataclasses import fields

import pytest
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bot_core.alerts.store import (
    DeliveryAttempt,
    HistoricalSourceDecision,
    MutationHistoryEntry,
    OperatorReplayEntry,
    PRODUCTION_SOURCE_RESOLUTION_POLICIES,
)
from bot_core.alerts.s9c_source import S9C_PRODUCTION_PROJECTION, S9C_RESOLUTION_POLICY_ID
from bot_core.observability.authority import (
    FreshnessPolicy,
    FrozenEnvironmentRegistryBinding,
    InMemoryObservationAuthorityCarrier,
    ObservationAuthority,
)

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
    "audit_journal_contract",
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
    "downstream_operation_declarations",
    "current_downstream_operation_definition_revisions",
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
    assert MACHINE["status"] == "IN_PROGRESS_S9D_C25_BLOCKED_CATALOG_RUNTIME_ACCEPTANCE"
    assert MACHINE["contract_identity"] == {
        "contract_id": "M0.12-audit-observability-alerts-updater",
        "version": "1.46.0",
        "phase": "S9D_C25_BLOCKED_CATALOG_RUNTIME_ACCEPTANCE",
        "machine_source_of_truth": True,
        "markdown_is_projection_only": True,
    }


def test_replay_schema_exactly_matches_executable_record() -> None:
    schema = MACHINE["alert_model"]["durability"]["replay_entry_schema"]
    assert schema["required_fields"] == [item.name for item in fields(OperatorReplayEntry)]
    assert schema["optional_fields"] == []
    assert schema["additional_fields"] == "REJECT"


def test_delivery_and_history_schemas_exactly_match_executable_records() -> None:
    delivery = MACHINE["alert_model"]["delivery_contract"]["attempt_schema"]
    history = MACHINE["alert_model"]["durability"]["history_entry_schema"]
    assert delivery["required_fields"] == [item.name for item in fields(DeliveryAttempt)]
    assert history["required_fields"] == [item.name for item in fields(MutationHistoryEntry)]
    for schema in (delivery, history):
        assert schema["optional_fields"] == []
        assert schema["additional_fields"] == "REJECT"


def test_historical_source_decision_schema_matches_executable_record() -> None:
    schema = MACHINE["alert_model"]["durability"]["historical_source_decision_schema"]
    assert schema["required_fields"] == [item.name for item in fields(HistoricalSourceDecision)]
    assert schema["optional_fields"] == []
    assert schema["additional_fields"] == "REJECT"


def test_production_source_policy_registry_matches_machine_source_of_truth() -> None:
    machine = MACHINE["alert_model"]["executable_authority"][
        "production_source_resolution_policies"
    ]
    executable = {
        alert_type: {
            "canonical_corrective_authority": policy,
            "executable_status": status,
        }
        for alert_type, (policy, status) in PRODUCTION_SOURCE_RESOLUTION_POLICIES.items()
    }
    assert executable == machine
    assert set(machine) == set(MACHINE["alert_model"]["resolution_contract"]["typed_paths"])


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
    assert MACHINE["audit_journal_contract"]["physical_carrier"]["decision"] == (
        "OPEN_FOR_M1_IMPLEMENTATION"
    )
    assert (
        "StateStore schema v3" in MACHINE["audit_journal_contract"]["physical_carrier"]["forbidden"]
    )


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


JOURNAL = MACHINE["audit_journal_contract"]
ZERO_HASH = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
PROOF_FIELDS = set(JOURNAL["schema"]["proof_fields_excluded_from_content"])


class ContractInconsistent(ValueError):
    pass


def _contracts() -> dict[str, dict[str, Any]]:
    return {
        "m02": _load("canonical_domain_vocabulary.json"),
        "m03": _load("process_topology_and_lifecycle.json"),
        "m04": _load("environment_and_product_capabilities.json"),
        "m07": _load("commands_events_order_lifecycle_and_idempotency.json"),
        "m08": _load("ledger_portfolio_capital_and_pnl.json"),
        "m09": _load("risk_hierarchy_kill_switch_and_execution_lease.json"),
        "m10": _load("identity_device_authentication_and_secrets.json"),
        "m11": _load("persistence_versioning_migrations_backup_and_recovery.json"),
    }


def _canonical(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, bool) or isinstance(item, int):
            return item
        if isinstance(item, float):
            raise ValueError("floats are not canonical")
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("keys must be strings")
            result = {
                unicodedata.normalize("NFC", key): normalize(child) for key, child in item.items()
            }
            if len(result) != len(item):
                raise ValueError("normalization created duplicate keys")
            return result
        raise ValueError("unsupported canonical type")

    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_entity(m02: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in m02["entity_kinds"] if item["canonical_name"] == name]
    if len(matches) != 1:
        raise ContractInconsistent(f"M0.2 entity mismatch: {name}")
    return matches[0]


def canonical_id_contract(m02: dict[str, Any], name: str) -> tuple[str, str]:
    entity = canonical_entity(m02, name)
    return entity["id_field"], entity["id_prefix"]


def validate_prefixed_id(value: Any, prefix: str, regex: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(regex, value):
        raise ValueError("invalid canonical id")
    if not value.startswith(prefix + "_"):
        raise ValueError("wrong canonical prefix")


def validate_canonical_id(value: Any, entity_name: str, m02: dict[str, Any]) -> None:
    _, prefix = canonical_id_contract(m02, entity_name)
    validate_prefixed_id(value, prefix, m02["identifier_policy"]["regex"])


def _resolve_pointer(contracts: dict[str, dict[str, Any]], source: str) -> Any:
    filename, pointer = source.split("#")
    key = {
        "identity_device_authentication_and_secrets.json": "m10",
        "risk_hierarchy_kill_switch_and_execution_lease.json": "m09",
        "audit_observability_alerts_and_updater.json": "m012",
    }[filename]
    value: Any = MACHINE if key == "m012" else contracts[key]
    for part in pointer.strip("/").split("/"):
        value = value[part]
    return value


def _timestamp(value: Any) -> None:
    from datetime import datetime

    if not isinstance(value, str) or not value.endswith("Z") or "+" in value:
        raise ValueError("timestamp must use canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("naive timestamp")


def _schema_value(
    value: Any,
    schema: dict[str, Any],
    *,
    registries: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
) -> None:
    kind = schema.get("type")
    if kind == "id":
        validate_prefixed_id(
            value, schema["prefix"], contracts["m02"]["identifier_policy"]["regex"]
        )
    elif kind == "enum":
        allowed = schema.get("values")
        if allowed is None:
            allowed = registries[schema["registry"]]
        if value not in allowed:
            raise ValueError("unknown enum value")
    elif kind == "constant":
        if value != schema["value"]:
            raise ValueError("wrong constant")
    elif kind == "positive_integer":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("positive non-bool integer required")
    elif kind == "non_negative_integer":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("non-negative non-bool integer required")
    elif kind == "timestamp":
        _timestamp(value)
    elif kind == "sha256_hex":
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError("lowercase SHA-256 required")
    elif kind in {"string", "non_empty_string"}:
        if not isinstance(value, str) or (kind == "non_empty_string" and not value):
            raise ValueError("string required")
    elif kind == "decimal":
        if not isinstance(value, str) or not DECIMAL_RE.fullmatch(value):
            raise ValueError("canonical decimal string required")
    elif kind == "safe_code":
        if not isinstance(value, str) or not SAFE_CODE_RE.fullmatch(value):
            raise ValueError("SAFE_CODE required")
    elif kind == "canonical_id":
        if not isinstance(value, str) or not re.fullmatch(
            contracts["m02"]["identifier_policy"]["regex"], value
        ):
            raise ValueError("canonical ID required")
    elif kind == "event_safe_payload":
        if not isinstance(value, dict):
            raise ValueError("event safe payload object required")
    else:
        raise ContractInconsistent(f"unknown schema type: {kind}")


def _m07_fingerprint(envelope: dict[str, Any], m07: dict[str, Any]) -> str:
    excluded = set(m07["event_contract"]["envelope_schema"]["fingerprint_excluded_fields"])
    return _sha256(
        _canonical({key: value for key, value in envelope.items() if key not in excluded})
    )


def validate_m07_envelope(envelope: Any, contracts: dict[str, dict[str, Any]]) -> None:
    m07 = contracts["m07"]["event_contract"]
    schema = m07["envelope_schema"]
    if type(envelope) is not dict or set(envelope) != set(schema["fields"]):
        raise ValueError("exact M0.7 envelope key set required")
    nullable = set(schema["nullable_fields"])
    for field in schema["fields"]:
        value = envelope[field]
        if value is None:
            if field not in nullable:
                raise ValueError(f"non-null M0.7 field: {field}")
            continue
        _schema_value(value, schema["field_schemas"][field], registries=m07, contracts=contracts)
    event_schema = m07["event_schema_registry"][envelope["event_type"]]
    payload = envelope["safe_payload"]
    if set(payload) != set(event_schema["safe_payload_fields"]):
        raise ValueError("exact M0.7 payload required")
    for field, value in payload.items():
        if value is None:
            if field not in event_schema["nullable_fields"]:
                raise ValueError("non-null M0.7 payload field")
        else:
            _schema_value(
                value, event_schema["field_schemas"][field], registries=m07, contracts=contracts
            )
    if envelope["event_fingerprint_sha256"] != _m07_fingerprint(envelope, contracts["m07"]):
        raise ValueError("invalid M0.7 event fingerprint")


def validate_m012_upstream_bindings(contracts: dict[str, dict[str, Any]]) -> None:
    expected = JOURNAL["upstream_executable_bindings"]
    m02 = contracts["m02"]
    actual_entities = {
        name: {key: canonical_entity(m02, name)[key] for key in ("id_field", "id_prefix")}
        for name in expected["M02"]["entities"]
    }
    audit = canonical_entity(m02, "AuditEvent")
    actual_audit = {
        key: audit[key]
        for key in (
            "id_prefix",
            "parent",
            "persistence",
            "audit_event_categories",
            "optional_references",
            "relationships",
        )
    }
    checks = [
        actual_entities == expected["M02"]["entities"],
        actual_audit == expected["M02"]["audit_event"],
        m02["identifier_policy"] == expected["M02"]["identifier_policy"],
    ]
    m03 = contracts["m03"]
    bootstrap = m03["first_run_bootstrap_authority_contract"]
    role = next(item for item in m03["process_roles"] if item["name"] == "bootstrapper")
    checks += [
        bootstrap["bootstrapper_role"]["mode"] == expected["M03"]["bootstrapper_mode"],
        bootstrap["audit_boundary"] == expected["M03"]["audit_boundary"],
        {key: role[key] for key in expected["M03"]["maintenance_authorization"]}
        == expected["M03"]["maintenance_authorization"],
        [item["environment_id"] for item in contracts["m04"]["execution_environments"]]
        == expected["M04"]["execution_environments"],
    ]
    m07 = contracts["m07"]
    checks += [
        m07["event_contract"][key] == expected["M07"][key]
        for key in ("envelope_schema", "event_types", "event_schema_registry", "fingerprint")
    ]
    checks.append(m07["identity_policy"] == expected["M07"]["identity_policy"])
    m11 = contracts["m11"]
    checks += [
        m11["state_store_physical_schema_registry"]["current_state_store_schema_version"]
        == expected["M11"]["current_state_store_schema_version"],
        m11["state_store_physical_schema_registry"]["entries"]
        == expected["M11"]["physical_schema_registry_entries"],
        m11["migration_protocol"]["rollback_policy"] == expected["M11"]["rollback_policy"],
    ]
    if not all(checks):
        raise ContractInconsistent("M0.12 upstream binding drift")


def _content_fingerprint(event: dict[str, Any]) -> str:
    content = {key: value for key, value in event.items() if key not in PROOF_FIELDS}
    return _sha256(b"CryptoHunter/M0.12/AuditEventContent/v1\0" + _canonical(content))


def _chain_fingerprint(sequence: int, previous: str, content: str) -> str:
    return _sha256(
        b"CryptoHunter/M0.12/AuditEventChain/v1\0"
        + str(sequence).encode()
        + b"\0"
        + previous.encode()
        + b"\0"
        + content.encode()
    )


def _segment_fingerprint(events: list[dict[str, Any]]) -> str:
    return _sha256(_canonical(events))


class AuditJournalReference:
    """Pure source-injected oracle; production authority is never imported."""

    def __init__(self, installation_id: str, contracts: dict[str, dict[str, Any]]) -> None:
        validate_m012_upstream_bindings(contracts)
        validate_canonical_id(installation_id, "DeviceInstallation", contracts["m02"])
        self.installation_id = installation_id
        self.contracts = contracts
        self.events: list[dict[str, Any]] = []
        self.active_by_id: dict[str, dict[str, Any]] = {}
        self.compact_by_id: dict[str, dict[str, Any]] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.pending_pruning: dict[str, Any] | None = None

    @property
    def recovery_required(self) -> bool:
        return self.pending_pruning is not None

    def validate(self, event: dict[str, Any]) -> None:
        if set(event) != set(JOURNAL["schema"]["fields"]):
            raise ValueError("exact top-level fields required")
        m02 = self.contracts["m02"]
        validate_canonical_id(event["audit_event_id"], "AuditEvent", m02)
        validate_canonical_id(event["device_installation_id"], "DeviceInstallation", m02)
        if event["device_installation_id"] != self.installation_id:
            raise ValueError("wrong journal parent")
        entity_for_optional = {
            "runtime_session_id": "RuntimeSession",
            "operator_id": "OperatorIdentity",
            "workspace_id": "Workspace",
            "exchange_account_id": "ExchangeAccount",
            "order_id": "Order",
            "ledger_entry_id": "LedgerEntry",
        }
        for field, entity in entity_for_optional.items():
            if event[field] is not None:
                validate_canonical_id(event[field], entity, m02)
        m07_fields = self.contracts["m07"]["event_contract"]["envelope_schema"]["field_schemas"]
        for field in ("correlation_id", "causation_id"):
            if event[field] is not None:
                _schema_value(
                    event[field],
                    m07_fields[field],
                    registries={},
                    contracts=self.contracts,
                )
        _timestamp(event["occurred_at_utc"])
        environments = [
            x["environment_id"] for x in self.contracts["m04"]["execution_environments"]
        ]
        if event["environment"] is not None and event["environment"] not in environments:
            raise ValueError("unknown environment")
        if event["writer_phase"] not in {
            x["phase"] for x in MACHINE["audit_model"]["writer_authority"]["phases"]
        }:
            raise ValueError("unknown writer phase")
        if event["payload_family"] not in JOURNAL["payload_families"]:
            raise ValueError("unknown payload family")
        if event["outcome"] not in JOURNAL["outcomes"]:
            raise ValueError("unknown outcome")
        if event["reason_code"] is not None and not SAFE_CODE_RE.fullmatch(event["reason_code"]):
            raise ValueError("invalid reason code")
        if (
            event["outcome"] in {"DENIED", "FAILED", "RECOVERY_REQUIRED"}
            and event["reason_code"] is None
        ):
            raise ValueError("reason code required")
        for field in (
            "content_fingerprint_sha256",
            "previous_chain_fingerprint_sha256",
            "chain_fingerprint_sha256",
        ):
            if not isinstance(event[field], str) or not SHA256_RE.fullmatch(event[field]):
                raise ValueError(f"invalid proof: {field}")
        if (
            isinstance(event["sequence"], bool)
            or not isinstance(event["sequence"], int)
            or event["sequence"] < 1
        ):
            raise ValueError("positive non-bool sequence required")
        family = event["payload_family"]
        if family == "CORE_DOMAIN_EVENT":
            if event["writer_phase"] != "CORE_RUNTIME" or event["runtime_session_id"] is None:
                raise ValueError("Core runtime/session required")
            validate_m07_envelope(event["m07_event_envelope"], self.contracts)
            envelope = event["m07_event_envelope"]
            if event["audit_event_id"] != envelope["audit_event_id"]:
                raise ValueError("outer/nested identity mismatch")
            for field in (
                "occurred_at_utc",
                "environment",
                "workspace_id",
                "exchange_account_id",
                "order_id",
                "correlation_id",
                "causation_id",
            ):
                if field in envelope and event[field] != envelope[field]:
                    raise ValueError(f"outer/nested mismatch: {field}")
            if event["event_type"] != envelope["event_type"] or event["category"] != "trading":
                raise ValueError("M0.7 event/category mismatch")
            if event["safe_payload"] != envelope["safe_payload"]:
                raise ValueError("outer/nested payload mismatch")
        else:
            if event["m07_event_envelope"] is not None:
                raise ValueError("M0.7 envelope forbidden for non-M0.7 family")
            family_contract = JOURNAL["payload_families"][family]
            if event["event_type"] not in family_contract["allowed_event_types"]:
                raise ValueError("event/family mismatch")
            if family == "PRE_CORE_SECURITY_EVENT" and (
                event["writer_phase"] != "PRE_CORE" or event["runtime_session_id"] is not None
            ):
                raise ValueError("PRE_CORE phase/session mismatch")
            if family == "CORE_CONTROL_EVENT" and (
                event["writer_phase"] != "CORE_RUNTIME" or event["runtime_session_id"] is None
            ):
                raise ValueError("Core control phase/session required")
            if (
                family in {"MAINTENANCE_UPDATE_EVENT", "RECOVERY_EVENT"}
                and event["writer_phase"] != "MAINTENANCE_UPDATE_RECOVERY"
            ):
                raise ValueError("maintenance phase required")
            schema = JOURNAL["non_m07_event_schema_registry"][event["event_type"]]
            if event["category"] != schema["exact_category"]:
                raise ValueError("event/category retention mismatch")
            if set(event["safe_payload"]) != set(schema["required_fields"]):
                raise ValueError("exact safe payload required")
            for name, value in event["safe_payload"].items():
                field_schema = schema["field_schemas"][name]
                if field_schema["type"] == "enum" and "source" in field_schema:
                    if value not in _resolve_pointer(
                        {**self.contracts, "m012": MACHINE}, field_schema["source"]
                    ):
                        raise ValueError("unknown upstream enum")
                else:
                    _schema_value(value, field_schema, registries={}, contracts=self.contracts)
        if event["content_fingerprint_sha256"] != _content_fingerprint(event):
            raise ValueError("invalid M0.12 content fingerprint")

    def append(
        self,
        event: dict[str, Any],
        *,
        writer: str,
        simulate_crash_before_pending_clear: bool = False,
    ) -> dict[str, Any]:
        self.validate(event)
        expected_writer = {
            "PRE_CORE": "PROVISIONING_AUTHORITY",
            "CORE_RUNTIME": "CoreHost",
            "MAINTENANCE_UPDATE_RECOVERY": "AUTHORIZED_BOOTSTRAPPER",
        }[event["writer_phase"]]
        if writer != expected_writer:
            raise ValueError("unauthorized writer")
        if self.pending_pruning is None:
            self.verify()
        elif self.pending_pruning["phase"] == "BODIES_REMOVED_EVIDENCE_PENDING":
            self.verify(allow_latest_missing_evidence=True)
        else:
            self.verify()
        prior = self.active_by_id.get(event["audit_event_id"])
        compact = self.compact_by_id.get(event["audit_event_id"])
        is_evidence = self._is_pruning_evidence(event)
        if self.pending_pruning is not None:
            pending = self.pending_pruning
            if pending["phase"] != "BODIES_REMOVED_EVIDENCE_PENDING":
                raise ValueError("RECOVERY_REQUIRED_BODY_REMOVAL_RETRY")
            if (
                not is_evidence
                or event["safe_payload"] != pending["expected_payload"]
                or event["sequence"] != pending["expected_sequence"]
                or event["previous_chain_fingerprint_sha256"] != pending["expected_predecessor"]
            ):
                raise ValueError("RECOVERY_REQUIRED_EXACT_CHECKPOINT_EVIDENCE")
        elif is_evidence and prior is None and compact is None:
            raise ValueError("UNBOUND_OR_DUPLICATE_PRUNING_EVIDENCE")
        if prior is not None:
            if _content_fingerprint(prior) != _content_fingerprint(event):
                raise ValueError("IDENTITY_CONFLICT")
            if any(
                prior[field] != event[field]
                for field in PROOF_FIELDS - {"content_fingerprint_sha256"}
            ):
                raise ValueError("REPLAY_PROOF_CONFLICT")
            if self.pending_pruning is not None and not simulate_crash_before_pending_clear:
                self.pending_pruning = None
            return {"outcome": "REPLAY", "proof": {field: prior[field] for field in PROOF_FIELDS}}
        if compact is not None:
            if compact["content_fingerprint_sha256"] != event["content_fingerprint_sha256"]:
                raise ValueError("IDENTITY_CONFLICT")
            if any(compact[field] != event[field] for field in PROOF_FIELDS):
                raise ValueError("REPLAY_PROOF_CONFLICT")
            if self.pending_pruning is not None and not simulate_crash_before_pending_clear:
                self.pending_pruning = None
            return {"outcome": "REPLAY", "proof": deepcopy(compact)}
        base_sequence, previous = (
            (0, ZERO_HASH)
            if not self.checkpoints
            else (
                self.checkpoints[-1]["last_sequence"],
                self.checkpoints[-1]["terminal_chain_fingerprint_sha256"],
            )
        )
        if self.events:
            base_sequence, previous = (
                self.events[-1]["sequence"],
                self.events[-1]["chain_fingerprint_sha256"],
            )
        if event["sequence"] != base_sequence + 1:
            raise ValueError("SEQUENCE_CONFLICT")
        if event["previous_chain_fingerprint_sha256"] != previous:
            raise ValueError("WRONG_PREDECESSOR")
        if event["chain_fingerprint_sha256"] != _chain_fingerprint(
            event["sequence"], previous, event["content_fingerprint_sha256"]
        ):
            raise ValueError("BROKEN_CHAIN")
        accepted = deepcopy(event)
        self.events.append(accepted)
        self.active_by_id[accepted["audit_event_id"]] = accepted
        if is_evidence and not simulate_crash_before_pending_clear:
            self.pending_pruning = None
        return {"outcome": "APPENDED", "proof": {field: accepted[field] for field in PROOF_FIELDS}}

    def verify(self, *, allow_latest_missing_evidence: bool = False) -> None:
        self._verify_identity_indexes()
        self.verify_retained_evidence(allow_latest_missing_evidence=allow_latest_missing_evidence)
        sequence, previous = (
            (0, ZERO_HASH)
            if not self.checkpoints
            else (
                self.checkpoints[-1]["last_sequence"],
                self.checkpoints[-1]["terminal_chain_fingerprint_sha256"],
            )
        )
        for event in self.events:
            self.validate(event)
            sequence += 1
            if (
                event["sequence"] != sequence
                or event["previous_chain_fingerprint_sha256"] != previous
            ):
                raise ValueError("broken sequence")
            if event["chain_fingerprint_sha256"] != _chain_fingerprint(
                sequence, previous, event["content_fingerprint_sha256"]
            ):
                raise ValueError("broken chain")
            previous = event["chain_fingerprint_sha256"]

    def _verify_identity_indexes(self) -> None:
        active_ids = [event["audit_event_id"] for event in self.events]
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("duplicate active AuditEvent identity")
        if set(self.active_by_id) != set(active_ids):
            raise ValueError("active/index divergence")
        for key, value in self.active_by_id.items():
            validate_canonical_id(key, "AuditEvent", self.contracts["m02"])
            if value.get("audit_event_id") != key:
                raise ValueError("active identity-index key mismatch")
            matches = [event for event in self.events if event["audit_event_id"] == key]
            if len(matches) != 1 or matches[0] != value:
                raise ValueError("active index value mismatch")
        compact_fields = set(JOURNAL["pruning"]["compact_replay_metadata"]["fields"])
        binding_schema = JOURNAL["non_m07_event_schema_registry"]["PERSISTENCE_TRANSITION"]
        compact_sequences: list[int] = []
        for key, value in self.compact_by_id.items():
            validate_canonical_id(key, "AuditEvent", self.contracts["m02"])
            if type(value) is not dict or set(value) != compact_fields:
                raise ValueError("compact/index structure divergence")
            if value["audit_event_id"] != key:
                raise ValueError("compact identity-index key mismatch")
            sequence = value["sequence"]
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                raise ValueError("invalid compact sequence")
            compact_sequences.append(sequence)
            for field in (
                "content_fingerprint_sha256",
                "previous_chain_fingerprint_sha256",
                "chain_fingerprint_sha256",
            ):
                if not isinstance(value[field], str) or not SHA256_RE.fullmatch(value[field]):
                    raise ValueError("invalid compact proof")
            matches = [
                checkpoint
                for checkpoint in self.checkpoints
                if checkpoint["first_sequence"] <= sequence <= checkpoint["last_sequence"]
            ]
            if len(matches) != 1:
                raise ValueError("orphan or multiply-covered compact record")
            binding = value["pruning_evidence_binding"]
            if binding is not None:
                if type(binding) is not dict or set(binding) != set(
                    binding_schema["required_fields"]
                ):
                    raise ValueError("malformed compact pruning binding")
                for field, field_value in binding.items():
                    _schema_value(
                        field_value,
                        binding_schema["field_schemas"][field],
                        registries={},
                        contracts=self.contracts,
                    )
                if (
                    binding["operation_code"] != "AUDIT_SEGMENT_PRUNED"
                    or binding["result_code"] != "COMPLETED"
                ):
                    raise ValueError("non-success compact pruning binding")
        if set(self.active_by_id) & set(self.compact_by_id):
            raise ValueError("active/compact identity overlap")
        if len(compact_sequences) != len(set(compact_sequences)):
            raise ValueError("duplicate compact append sequence")
        expected_sequences = {
            sequence
            for checkpoint in self.checkpoints
            for sequence in range(checkpoint["first_sequence"], checkpoint["last_sequence"] + 1)
        }
        if set(compact_sequences) != expected_sequences:
            raise ValueError("compact/checkpoint coverage divergence")

    @staticmethod
    def _compact(event: dict[str, Any]) -> dict[str, Any]:
        compact = {
            field: event[field]
            for field in JOURNAL["pruning"]["compact_replay_metadata"]["fields"]
            if field != "pruning_evidence_binding"
        }
        compact["pruning_evidence_binding"] = (
            deepcopy(event["safe_payload"])
            if AuditJournalReference._is_pruning_evidence(event)
            else None
        )
        return compact

    @staticmethod
    def _is_pruning_evidence(event: dict[str, Any]) -> bool:
        return (
            event["event_type"] == "PERSISTENCE_TRANSITION"
            and event["safe_payload"].get("operation_code") == "AUDIT_SEGMENT_PRUNED"
            and event["safe_payload"].get("result_code") == "COMPLETED"
        )

    def seal_prefix(self, count: int) -> dict[str, Any]:
        if self.recovery_required:
            raise ValueError("pending pruning must be resolved")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or count > len(self.events)
        ):
            raise ValueError("verified unsealed prefix required")
        self.verify()
        segment = deepcopy(self.events[:count])
        previous_checkpoint = self.checkpoints[-1] if self.checkpoints else None
        first = 1 if previous_checkpoint is None else previous_checkpoint["last_sequence"] + 1
        predecessor = (
            ZERO_HASH
            if previous_checkpoint is None
            else previous_checkpoint["terminal_chain_fingerprint_sha256"]
        )
        if (
            segment[0]["sequence"] != first
            or segment[0]["previous_chain_fingerprint_sha256"] != predecessor
        ):
            raise ValueError("segment gap, overlap or predecessor mismatch")
        compact = [self._compact(event) for event in segment]
        checkpoint = {
            "device_installation_id": self.installation_id,
            "first_sequence": first,
            "last_sequence": segment[-1]["sequence"],
            "predecessor_chain_fingerprint_sha256": predecessor,
            "terminal_chain_fingerprint_sha256": segment[-1]["chain_fingerprint_sha256"],
            "segment_content_fingerprint_sha256": _segment_fingerprint(segment),
            "journal_version": JOURNAL["version"],
            "retention_policy_version": JOURNAL["retention"]["policy_version"],
            "compact_replay_fingerprint_sha256": _segment_fingerprint(compact),
        }
        self.verify_checkpoint(
            checkpoint,
            bodies=segment,
            compact=compact,
            previous_checkpoint=previous_checkpoint,
        )
        return checkpoint

    def verify_checkpoint(
        self,
        checkpoint: dict[str, Any],
        *,
        bodies: list[dict[str, Any]] | None = None,
        compact: list[dict[str, Any]] | None = None,
        previous_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        required = set(JOURNAL["pruning"]["sealed_checkpoint"]["fields"])
        if type(checkpoint) is not dict or set(checkpoint) != required:
            raise ValueError("exact checkpoint fields required")
        validate_canonical_id(
            checkpoint["device_installation_id"], "DeviceInstallation", self.contracts["m02"]
        )
        if checkpoint["device_installation_id"] != self.installation_id:
            raise ValueError("wrong checkpoint installation")
        if checkpoint["journal_version"] != JOURNAL["version"]:
            raise ValueError("unsupported journal version")
        if checkpoint["retention_policy_version"] != JOURNAL["retention"]["policy_version"]:
            raise ValueError("unaccepted retention policy version")
        for field in (
            "predecessor_chain_fingerprint_sha256",
            "terminal_chain_fingerprint_sha256",
            "segment_content_fingerprint_sha256",
            "compact_replay_fingerprint_sha256",
        ):
            if not isinstance(checkpoint[field], str) or not SHA256_RE.fullmatch(checkpoint[field]):
                raise ValueError("malformed checkpoint hash")
        first, last = checkpoint["first_sequence"], checkpoint["last_sequence"]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in (first, last)
            )
            or last < first
        ):
            raise ValueError("invalid checkpoint range")
        expected_first = (
            1 if previous_checkpoint is None else previous_checkpoint["last_sequence"] + 1
        )
        expected_predecessor = (
            ZERO_HASH
            if previous_checkpoint is None
            else previous_checkpoint["terminal_chain_fingerprint_sha256"]
        )
        if first != expected_first:
            raise ValueError("checkpoint gap or overlap")
        if checkpoint["predecessor_chain_fingerprint_sha256"] != expected_predecessor:
            raise ValueError("cross-segment predecessor mismatch")
        evidence = (
            compact
            if compact is not None
            else ([] if bodies is None else [self._compact(body) for body in bodies])
        )
        if len(evidence) != last - first + 1:
            raise ValueError("checkpoint evidence count mismatch")
        previous = expected_predecessor
        for sequence, item in enumerate(evidence, first):
            if set(item) != set(JOURNAL["pruning"]["compact_replay_metadata"]["fields"]):
                raise ValueError("non-compact replay evidence")
            validate_canonical_id(item["audit_event_id"], "AuditEvent", self.contracts["m02"])
            for field in (
                "content_fingerprint_sha256",
                "previous_chain_fingerprint_sha256",
                "chain_fingerprint_sha256",
            ):
                if not isinstance(item[field], str) or not SHA256_RE.fullmatch(item[field]):
                    raise ValueError("malformed compact proof")
            if isinstance(item["sequence"], bool) or item["sequence"] != sequence:
                raise ValueError("non-contiguous compact sequence")
            if item["previous_chain_fingerprint_sha256"] != previous:
                raise ValueError("compact predecessor mismatch")
            if item["chain_fingerprint_sha256"] != _chain_fingerprint(
                sequence, previous, item["content_fingerprint_sha256"]
            ):
                raise ValueError("compact chain mismatch")
            previous = item["chain_fingerprint_sha256"]
        if previous != checkpoint["terminal_chain_fingerprint_sha256"]:
            raise ValueError("terminal anchor mismatch")
        if _segment_fingerprint(evidence) != checkpoint["compact_replay_fingerprint_sha256"]:
            raise ValueError("compact replay fingerprint mismatch")
        if bodies is not None:
            if len(bodies) != len(evidence):
                raise ValueError("body count mismatch")
            for body in bodies:
                self.validate(body)
                if body["device_installation_id"] != checkpoint["device_installation_id"]:
                    raise ValueError("body installation mismatch")
            if _segment_fingerprint(bodies) != checkpoint["segment_content_fingerprint_sha256"]:
                raise ValueError("segment body fingerprint mismatch")

    def verify_retained_evidence(self, *, allow_latest_missing_evidence: bool = False) -> None:
        if allow_latest_missing_evidence:
            if (
                self.pending_pruning is None
                or self.pending_pruning["phase"] != "BODIES_REMOVED_EVIDENCE_PENDING"
                or not self.checkpoints
                or self.pending_pruning["checkpoint"] != self.checkpoints[-1]
                or self.pending_pruning["expected_payload"]
                != self._expected_evidence_payload(self.checkpoints[-1])
            ):
                raise ValueError("latest evidence exception requires exact pending checkpoint")
        previous_checkpoint = None
        for checkpoint in self.checkpoints:
            compact = [
                value
                for value in self.compact_by_id.values()
                if checkpoint["first_sequence"] <= value["sequence"] <= checkpoint["last_sequence"]
            ]
            compact.sort(key=lambda value: value["sequence"])
            self.verify_checkpoint(
                checkpoint, compact=compact, previous_checkpoint=previous_checkpoint
            )
            previous_checkpoint = checkpoint
        evidence_payloads: list[dict[str, Any]] = []
        for event in self.events:
            if self._is_pruning_evidence(event):
                self.validate(event)
                evidence_payloads.append(event["safe_payload"])
        evidence_payloads.extend(
            item["pruning_evidence_binding"]
            for item in self.compact_by_id.values()
            if item["pruning_evidence_binding"] is not None
        )
        expected_payloads = [
            self._expected_evidence_payload(checkpoint) for checkpoint in self.checkpoints
        ]
        for payload in evidence_payloads:
            if sum(payload == expected for expected in expected_payloads) != 1:
                raise ValueError("orphan or ambiguous pruning evidence")
        for index, expected in enumerate(expected_payloads):
            matches = sum(payload == expected for payload in evidence_payloads)
            if allow_latest_missing_evidence and index == len(expected_payloads) - 1:
                if matches not in {0, 1}:
                    raise ValueError("ambiguous latest checkpoint pruning evidence")
            elif matches != 1:
                raise ValueError("missing or ambiguous checkpoint pruning evidence")

    @staticmethod
    def _expected_evidence_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation_code": "AUDIT_SEGMENT_PRUNED",
            "first_sequence": checkpoint["first_sequence"],
            "last_sequence": checkpoint["last_sequence"],
            "checkpoint_fingerprint_sha256": _sha256(_canonical(checkpoint)),
            "retention_policy_version": checkpoint["retention_policy_version"],
            "result_code": "COMPLETED",
        }

    def prune_with_checkpoint(
        self,
        checkpoint: dict[str, Any],
        evidence: dict[str, Any] | None,
        *,
        writer: str = "AUTHORIZED_BOOTSTRAPPER",
        removal_succeeds: bool = True,
    ) -> dict[str, Any]:
        self.verify()
        previous_checkpoint = self.checkpoints[-1] if self.checkpoints else None
        if self.pending_pruning is not None:
            if (
                self.pending_pruning["phase"] != "CHECKPOINT_DURABLE_BODIES_RETAINED"
                or checkpoint != self.pending_pruning["checkpoint"]
            ):
                raise ValueError("DIFFERENT_PENDING_CHECKPOINT")
        count = checkpoint["last_sequence"] - checkpoint["first_sequence"] + 1
        bodies = self.events[:count]
        compact = [self._compact(body) for body in bodies]
        self.verify_checkpoint(
            checkpoint,
            bodies=bodies,
            compact=compact,
            previous_checkpoint=previous_checkpoint,
        )
        expected_payload = self._expected_evidence_payload(checkpoint)
        tail = self.events[-1]
        descriptor = {
            "phase": "CHECKPOINT_DURABLE_BODIES_RETAINED",
            "checkpoint": deepcopy(checkpoint),
            "checkpoint_fingerprint_sha256": expected_payload["checkpoint_fingerprint_sha256"],
            "first_sequence": checkpoint["first_sequence"],
            "last_sequence": checkpoint["last_sequence"],
            "retention_policy_version": checkpoint["retention_policy_version"],
            "expected_payload": expected_payload,
            "expected_sequence": tail["sequence"] + 1,
            "expected_predecessor": tail["chain_fingerprint_sha256"],
        }
        if self.pending_pruning is None:
            self.pending_pruning = descriptor
        if not removal_succeeds:
            raise ValueError("RECOVERY_REQUIRED_BODY_REMOVAL_FAILED")
        self.checkpoints.append(deepcopy(checkpoint))
        for body, metadata in zip(bodies, compact, strict=True):
            del self.active_by_id[body["audit_event_id"]]
            self.compact_by_id[body["audit_event_id"]] = metadata
        self.events = self.events[count:]
        self.pending_pruning = {
            **descriptor,
            "phase": "BODIES_REMOVED_EVIDENCE_PENDING",
        }
        if evidence is None:
            raise ValueError("RECOVERY_REQUIRED_PRUNING_EVIDENCE")
        return self.append(evidence, writer=writer)

    def restore_decision(self, sequence: int, fingerprint: str) -> str:
        if self.recovery_required:
            raise ValueError("RECOVERY_REQUIRED_PENDING_CHECKPOINT_NOT_RESTORE_AUTHORITY")
        self.verify()
        checkpoint = next(
            (item for item in self.checkpoints if item["last_sequence"] == sequence), None
        )
        if checkpoint is not None:
            expected = checkpoint["terminal_chain_fingerprint_sha256"]
        else:
            found = next((item for item in self.events if item["sequence"] == sequence), None)
            expected = (
                ZERO_HASH
                if sequence == 0
                else None
                if found is None
                else found["chain_fingerprint_sha256"]
            )
        if fingerprint != expected:
            raise ValueError("DIVERGENT_RESTORE")
        return "PRESERVE_CURRENT_NEWER_EXTENSION"


def _blank_event(suffix: str, **updates: Any) -> dict[str, Any]:
    event = {key: None for key in JOURNAL["schema"]["fields"]}
    event.update(
        audit_event_id=f"evt_0000000{suffix}-0000-7000-8000-00000000000{suffix}",
        category="authentication",
        event_type="AUTHENTICATION_DECIDED",
        payload_family="PRE_CORE_SECURITY_EVENT",
        occurred_at_utc="2026-09-09T12:00:00Z",
        device_installation_id="dev_00000001-0000-7000-8000-000000000001",
        writer_phase="PRE_CORE",
        outcome="ACCEPTED",
        safe_payload={"method": "PIN", "result_code": "SUCCESS"},
        sequence=int(suffix),
        previous_chain_fingerprint_sha256=ZERO_HASH,
    )
    event.update(updates)
    event["content_fingerprint_sha256"] = _content_fingerprint(event)
    event["chain_fingerprint_sha256"] = _chain_fingerprint(
        event["sequence"],
        event["previous_chain_fingerprint_sha256"],
        event["content_fingerprint_sha256"],
    )
    return event


def _m07_event(suffix: str = "2", sequence: int = 2, previous: str = ZERO_HASH) -> dict[str, Any]:
    c = _contracts()
    schema = c["m07"]["event_contract"]["envelope_schema"]
    values = {
        "audit_event_id": f"evt_0000000{suffix}-0000-7000-8000-00000000000{suffix}",
        "event_type": "ORDER_REJECTED",
        "order_id": "ord_00000001-0000-7000-8000-000000000001",
        "aggregate_version": 7,
        "correlation_id": "corr_00000001-0000-7000-8000-000000000001",
        "causation_id": None,
        "command_id": "cmd_00000001-0000-7000-8000-000000000001",
        "environment": "PAPER",
        "workspace_id": "ws_00000001-0000-7000-8000-000000000001",
        "portfolio_id": "port_00000001-0000-7000-8000-000000000001",
        "exchange_account_id": "xacc_00000001-0000-7000-8000-000000000001",
        "exchange_id": "kraken",
        "instrument_id": "instr_00000001-0000-7000-8000-000000000001",
        "execution_route_id": "xroute_00000001-0000-7000-8000-000000000001",
        "occurred_at_utc": "2020-01-01T00:00:00Z",
        "safe_payload": {"reason_code": "VENUE_DENIED"},
        "event_fingerprint_sha256": ZERO_HASH,
    }
    envelope = {field: values[field] for field in schema["fields"]}
    envelope["event_fingerprint_sha256"] = _m07_fingerprint(envelope, c["m07"])
    event = _blank_event(
        suffix,
        category="trading",
        event_type=envelope["event_type"],
        payload_family="CORE_DOMAIN_EVENT",
        occurred_at_utc=envelope["occurred_at_utc"],
        runtime_session_id="run_00000001-0000-7000-8000-000000000001",
        environment=envelope["environment"],
        workspace_id=envelope["workspace_id"],
        exchange_account_id=envelope["exchange_account_id"],
        order_id=envelope["order_id"],
        correlation_id=envelope["correlation_id"],
        causation_id=envelope["causation_id"],
        writer_phase="CORE_RUNTIME",
        safe_payload=envelope["safe_payload"],
        m07_event_envelope=envelope,
        sequence=sequence,
        previous_chain_fingerprint_sha256=previous,
    )
    event["content_fingerprint_sha256"] = _content_fingerprint(event)
    event["chain_fingerprint_sha256"] = _chain_fingerprint(
        sequence, previous, event["content_fingerprint_sha256"]
    )
    return event


def _pruning_evidence(
    suffix: str, sequence: int, previous: str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    return _blank_event(
        suffix,
        category="recovery",
        event_type="PERSISTENCE_TRANSITION",
        payload_family="MAINTENANCE_UPDATE_EVENT",
        writer_phase="MAINTENANCE_UPDATE_RECOVERY",
        safe_payload={
            "operation_code": "AUDIT_SEGMENT_PRUNED",
            "first_sequence": checkpoint["first_sequence"],
            "last_sequence": checkpoint["last_sequence"],
            "checkpoint_fingerprint_sha256": _sha256(_canonical(checkpoint)),
            "retention_policy_version": JOURNAL["retention"]["policy_version"],
            "result_code": "COMPLETED",
        },
        sequence=sequence,
        previous_chain_fingerprint_sha256=previous,
    )


def _assert_rejected(journal: AuditJournalReference, event: dict[str, Any]) -> None:
    with pytest.raises((ValueError, ContractInconsistent, KeyError)):
        journal.validate(event)


def test_source_first_ids_exact_m07_coverage_and_actual_bindings() -> None:
    c = _contracts()
    validate_m012_upstream_bindings(c)
    assert {
        name: canonical_id_contract(c["m02"], name)[1]
        for name in JOURNAL["upstream_executable_bindings"]["M02"]["entities"]
    } == {
        "AuditEvent": "evt",
        "DeviceInstallation": "dev",
        "RuntimeSession": "run",
        "OperatorIdentity": "op",
        "Workspace": "ws",
        "ExchangeAccount": "xacc",
        "Order": "ord",
        "LedgerEntry": "led",
    }
    bound = JOURNAL["upstream_executable_bindings"]["M07"]
    actual = c["m07"]["event_contract"]
    assert bound["envelope_schema"]["fields"] == actual["envelope_schema"]["fields"]
    assert (
        bound["envelope_schema"]["nullable_fields"] == actual["envelope_schema"]["nullable_fields"]
    )
    assert bound["envelope_schema"]["field_schemas"] == actual["envelope_schema"]["field_schemas"]
    assert bound["event_schema_registry"] == actual["event_schema_registry"]


def test_exact_m07_envelope_fingerprints_ordering_and_replay_proof() -> None:
    c = _contracts()
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", c)
    first = _blank_event("1")
    assert journal.append(first, writer="PROVISIONING_AUTHORITY")["outcome"] == "APPENDED"
    core = _m07_event(previous=first["chain_fingerprint_sha256"])
    validate_m07_envelope(core["m07_event_envelope"], c)
    assert core["m07_event_envelope"]["aggregate_version"] == 7 and core["sequence"] == 2
    assert (
        core["m07_event_envelope"]["event_fingerprint_sha256"] != core["content_fingerprint_sha256"]
    )
    assert journal.append(core, writer="CoreHost")["outcome"] == "APPENDED"
    journal.verify()
    replay = deepcopy(core)
    result = journal.append(replay, writer="CoreHost")
    assert result["outcome"] == "REPLAY" and result["proof"]["sequence"] == 2
    replay["sequence"] = 3
    with pytest.raises(ValueError, match="REPLAY_PROOF_CONFLICT"):
        journal.append(replay, writer="CoreHost")


def test_m07_each_authority_field_mutation_fails_without_recomputation() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    core = _m07_event(sequence=1)
    mutations = {
        "aggregate_version": 8,
        "command_id": None,
        "portfolio_id": "port_00000002-0000-7000-8000-000000000002",
        "exchange_id": "coinbase",
        "instrument_id": "instr_00000002-0000-7000-8000-000000000002",
        "execution_route_id": "xroute_00000002-0000-7000-8000-000000000002",
        "environment": "TESTNET",
        "event_fingerprint_sha256": "f" * 64,
        "safe_payload": {"reason_code": "OTHER"},
    }
    for field, value in mutations.items():
        changed = deepcopy(core)
        changed["m07_event_envelope"][field] = value
        _assert_rejected(journal, changed)


def test_context_ids_timestamp_environment_category_and_secret_value_fail_closed() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    journal.validate(_blank_event("1"))
    legacy_runtime = "se" + "ss_00000001-0000-7000-8000-000000000001"
    for runtime in (None, legacy_runtime, "run_bad"):
        core = _m07_event(sequence=1)
        core["runtime_session_id"] = runtime
        core["content_fingerprint_sha256"] = _content_fingerprint(core)
        _assert_rejected(journal, core)
    for timestamp in (
        "anything",
        "2026-01-01T00:00:00",
        "2026-01-01T01:00:00+01:00",
        "2026-99-01T00:00:00Z",
    ):
        event = _blank_event("1", occurred_at_utc=timestamp)
        event["content_fingerprint_sha256"] = _content_fingerprint(event)
        _assert_rejected(journal, event)
    for change in (
        {"environment": "SANDBOX"},
        {"category": "runtime"},
        {"reason_code": "free text"},
        {"sequence": True},
    ):
        event = _blank_event("1", **change)
        event["content_fingerprint_sha256"] = _content_fingerprint(event)
        _assert_rejected(journal, event)
    secret = _blank_event("1", safe_payload={"method": "PIN", "result_code": "my raw password"})
    secret["content_fingerprint_sha256"] = _content_fingerprint(secret)
    _assert_rejected(journal, secret)


def test_all_non_m07_categories_are_closed_and_retention_downgrade_fails() -> None:
    schemas = JOURNAL["non_m07_event_schema_registry"]
    covered = [
        event
        for family in JOURNAL["payload_families"].values()
        for event in family.get("allowed_event_types", [])
    ]
    assert set(covered) == set(schemas) and all(covered.count(event) >= 1 for event in schemas)
    upstream = set(canonical_entity(_contracts()["m02"], "AuditEvent")["audit_event_categories"])
    assert {x["exact_category"] for x in schemas.values()} <= upstream
    cases = [
        (
            "AUTHENTICATION_DECIDED",
            "PRE_CORE_SECURITY_EVENT",
            {"method": "PIN", "result_code": "SUCCESS"},
        ),
        (
            "SECURITY_MUTATION_RECORDED",
            "PRE_CORE_SECURITY_EVENT",
            {
                "mutation_code": "DEVICE_REVOKED",
                "target_reference": "dev_00000001-0000-7000-8000-000000000001",
                "revision": 1,
            },
        ),
        (
            "RISK_CONTROL_TRANSITION",
            "CORE_CONTROL_EVENT",
            {"control_code": "KILL_SWITCH", "state_code": "ACTIVE"},
        ),
        (
            "UPDATE_TRANSITION",
            "MAINTENANCE_UPDATE_EVENT",
            {
                "update_attempt_reference": "UPDATE_ATTEMPT",
                "phase_code": "INSTALLATION",
                "artifact_fingerprint_sha256": "a" * 64,
            },
        ),
    ]
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    for typ, family, payload in cases:
        phase = (
            "PRE_CORE"
            if family == "PRE_CORE_SECURITY_EVENT"
            else "CORE_RUNTIME"
            if family == "CORE_CONTROL_EVENT"
            else "MAINTENANCE_UPDATE_RECOVERY"
        )
        event = _blank_event(
            "1",
            event_type=typ,
            payload_family=family,
            category="runtime",
            safe_payload=payload,
            writer_phase=phase,
            runtime_session_id="run_00000001-0000-7000-8000-000000000001"
            if phase == "CORE_RUNTIME"
            else None,
        )
        event["content_fingerprint_sha256"] = _content_fingerprint(event)
        _assert_rejected(journal, event)


def test_field_contract_execution_mutation_matrix_and_unknown_type() -> None:
    assert set(JOURNAL["schema"]["fields"]) == set(JOURNAL["schema"]["field_contracts"])
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    base = _blank_event("1")
    bad = {
        "audit_event_id": "evt_bad",
        "category": "other",
        "event_type": "UNKNOWN",
        "payload_family": "UNKNOWN",
        "occurred_at_utc": "bad",
        "device_installation_id": "dev_bad",
        "runtime_session_id": "run_bad",
        "operator_id": "op_bad",
        "environment": "BAD",
        "workspace_id": "ws_bad",
        "exchange_account_id": "xacc_bad",
        "order_id": "ord_bad",
        "ledger_entry_id": "led_bad",
        "correlation_id": 7,
        "causation_id": 7,
        "writer_phase": "UI",
        "outcome": "OTHER",
        "reason_code": "free text",
        "safe_payload": [],
        "m07_event_envelope": {},
        "content_fingerprint_sha256": "A" * 64,
        "sequence": True,
        "previous_chain_fingerprint_sha256": "x",
        "chain_fingerprint_sha256": "x",
    }
    for field, value in bad.items():
        candidate = deepcopy(base)
        candidate[field] = value
        if field not in PROOF_FIELDS:
            candidate["content_fingerprint_sha256"] = _content_fingerprint(candidate)
        _assert_rejected(journal, candidate)
    with pytest.raises(ContractInconsistent):
        _schema_value("x", {"type": "future_type"}, registries={}, contracts=_contracts())


def test_m07_object_order_is_irrelevant_but_exact_field_set_remains_closed() -> None:
    contracts = _contracts()
    core = _m07_event(sequence=1)
    original = core["m07_event_envelope"]
    reordered = dict(reversed(list(original.items())))
    assert list(reordered) != list(original)
    assert _m07_fingerprint(reordered, contracts["m07"]) == original["event_fingerprint_sha256"]
    core["m07_event_envelope"] = reordered
    core["content_fingerprint_sha256"] = _content_fingerprint(core)
    AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", contracts).validate(core)
    candidates = []
    missing = deepcopy(reordered)
    missing.pop("exchange_id")
    candidates.append(missing)
    extra = deepcopy(reordered)
    extra["extra"] = "x"
    candidates.append(extra)
    wrong_name = deepcopy(reordered)
    wrong_name["exchange"] = wrong_name.pop("exchange_id")
    candidates.append(wrong_name)
    wrong_null = deepcopy(reordered)
    wrong_null["order_id"] = None
    candidates.append(wrong_null)
    wrong_type = deepcopy(reordered)
    wrong_type["aggregate_version"] = True
    candidates.append(wrong_type)
    for envelope in candidates:
        with pytest.raises(ValueError):
            validate_m07_envelope(envelope, contracts)


def test_checkpoint_fields_real_pruning_post_prune_replay_and_attacks() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    checkpoint = journal.seal_prefix(1)
    evidence = _pruning_evidence("2", 2, first["chain_fingerprint_sha256"], checkpoint)
    assert journal.prune_with_checkpoint(checkpoint, evidence)["outcome"] == "APPENDED"
    journal.verify()
    assert any(journal._is_pruning_evidence(event) for event in journal.events)
    assert first not in journal.events
    assert first["audit_event_id"] not in journal.active_by_id
    compact = deepcopy(journal.compact_by_id[first["audit_event_id"]])
    assert set(compact) == set(JOURNAL["pruning"]["compact_replay_metadata"]["fields"])
    assert "safe_payload" not in compact and "m07_event_envelope" not in compact
    assert journal.append(deepcopy(first), writer="PROVISIONING_AUTHORITY")["outcome"] == "REPLAY"
    changed = deepcopy(first)
    changed["safe_payload"]["result_code"] = "DENIED"
    changed["content_fingerprint_sha256"] = _content_fingerprint(changed)
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT"):
        journal.append(changed, writer="PROVISIONING_AUTHORITY")
    changed_proof = deepcopy(first)
    changed_proof["chain_fingerprint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="REPLAY_PROOF_CONFLICT"):
        journal.append(changed_proof, writer="PROVISIONING_AUTHORITY")
    attacks = (
        ("device_installation_id", "dev_00000002-0000-7000-8000-000000000002"),
        ("device_installation_id", "dev_bad"),
        ("journal_version", "FUTURE_UNKNOWN"),
        ("retention_policy_version", "UNKNOWN"),
        ("terminal_chain_fingerprint_sha256", "A" * 64),
        ("compact_replay_fingerprint_sha256", "not-hex"),
    )
    for field, value in attacks:
        bad = deepcopy(checkpoint)
        bad[field] = value
        with pytest.raises(ValueError):
            journal.verify_checkpoint(bad, compact=[compact])
    journal.compact_by_id[first["audit_event_id"]]["content_fingerprint_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        journal.verify_retained_evidence()
    journal.compact_by_id[first["audit_event_id"]] = compact
    malformed = deepcopy(first)
    malformed["category"] = "runtime"
    malformed["content_fingerprint_sha256"] = _content_fingerprint(malformed)
    malformed["chain_fingerprint_sha256"] = _chain_fingerprint(
        1, ZERO_HASH, malformed["content_fingerprint_sha256"]
    )
    forged_compact = [journal._compact(malformed)]
    forged = deepcopy(checkpoint)
    forged["terminal_chain_fingerprint_sha256"] = malformed["chain_fingerprint_sha256"]
    forged["segment_content_fingerprint_sha256"] = _segment_fingerprint([malformed])
    forged["compact_replay_fingerprint_sha256"] = _segment_fingerprint(forged_compact)
    with pytest.raises(ValueError):
        journal.verify_checkpoint(forged, bodies=[malformed], compact=forged_compact)


def test_successive_checkpoint_chain_pruning_evidence_and_restore() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    second = _blank_event("2", previous_chain_fingerprint_sha256=first["chain_fingerprint_sha256"])
    second["content_fingerprint_sha256"] = _content_fingerprint(second)
    second["chain_fingerprint_sha256"] = _chain_fingerprint(
        2, first["chain_fingerprint_sha256"], second["content_fingerprint_sha256"]
    )
    journal.append(second, writer="PROVISIONING_AUTHORITY")
    cp1 = journal.seal_prefix(1)
    prune1 = _pruning_evidence("3", 3, second["chain_fingerprint_sha256"], cp1)
    journal.prune_with_checkpoint(cp1, prune1)
    fourth = _blank_event(
        "4", sequence=4, previous_chain_fingerprint_sha256=prune1["chain_fingerprint_sha256"]
    )
    journal.append(fourth, writer="PROVISIONING_AUTHORITY")
    cp2 = journal.seal_prefix(2)
    assert cp2["first_sequence"] == 2 and cp2["last_sequence"] == 3
    prune2 = _pruning_evidence("5", 5, fourth["chain_fingerprint_sha256"], cp2)
    journal.prune_with_checkpoint(cp2, prune2)
    sixth = _blank_event(
        "6", sequence=6, previous_chain_fingerprint_sha256=prune2["chain_fingerprint_sha256"]
    )
    journal.append(sixth, writer="PROVISIONING_AUTHORITY")
    journal.verify()
    assert len(journal.checkpoints) == 2
    assert (
        journal.restore_decision(1, cp1["terminal_chain_fingerprint_sha256"])
        == "PRESERVE_CURRENT_NEWER_EXTENSION"
    )
    assert (
        journal.restore_decision(3, cp2["terminal_chain_fingerprint_sha256"])
        == "PRESERVE_CURRENT_NEWER_EXTENSION"
    )
    compact2 = [value for value in journal.compact_by_id.values() if 2 <= value["sequence"] <= 3]
    compact2.sort(key=lambda value: value["sequence"])
    for field, value in (
        ("first_sequence", 1),
        ("first_sequence", 4),
        ("predecessor_chain_fingerprint_sha256", ZERO_HASH),
    ):
        bad = deepcopy(cp2)
        bad[field] = value
        with pytest.raises(ValueError):
            journal.verify_checkpoint(bad, compact=compact2, previous_checkpoint=cp1)


def test_body_removal_failure_retains_pending_checkpoint_and_exact_retry_completes() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    checkpoint = journal.seal_prefix(1)
    with pytest.raises(ValueError, match="BODY_REMOVAL_FAILED"):
        journal.prune_with_checkpoint(checkpoint, None, removal_succeeds=False)
    assert journal.recovery_required
    assert journal.pending_pruning is not None
    assert journal.pending_pruning["phase"] == "CHECKPOINT_DURABLE_BODIES_RETAINED"
    assert journal.pending_pruning["checkpoint"] == checkpoint
    assert journal.events == [first] and first["audit_event_id"] in journal.active_by_id
    assert journal.checkpoints == [] and journal.compact_by_id == {}
    with pytest.raises(ValueError, match="BODY_REMOVAL_RETRY"):
        journal.append(
            _blank_event(
                "2", sequence=2, previous_chain_fingerprint_sha256=first["chain_fingerprint_sha256"]
            ),
            writer="PROVISIONING_AUTHORITY",
        )
    with pytest.raises(ValueError, match="PENDING_CHECKPOINT_NOT_RESTORE_AUTHORITY"):
        journal.restore_decision(1, checkpoint["terminal_chain_fingerprint_sha256"])
    different = deepcopy(checkpoint)
    different["segment_content_fingerprint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="DIFFERENT_PENDING_CHECKPOINT"):
        journal.prune_with_checkpoint(different, None)
    evidence = _pruning_evidence("2", 2, first["chain_fingerprint_sha256"], checkpoint)
    assert journal.prune_with_checkpoint(checkpoint, evidence)["outcome"] == "APPENDED"
    assert not journal.recovery_required and journal.pending_pruning is None
    third = _blank_event(
        "3", sequence=3, previous_chain_fingerprint_sha256=evidence["chain_fingerprint_sha256"]
    )
    assert journal.append(third, writer="PROVISIONING_AUTHORITY")["outcome"] == "APPENDED"


def test_missing_evidence_fake_bypass_exact_completion_and_crash_replay() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    checkpoint = journal.seal_prefix(1)
    with pytest.raises(ValueError, match="PRUNING_EVIDENCE"):
        journal.prune_with_checkpoint(checkpoint, None)
    pending = deepcopy(journal.pending_pruning)
    assert pending is not None
    assert pending["phase"] == "BODIES_REMOVED_EVIDENCE_PENDING" and journal.recovery_required
    assert journal.checkpoints == [checkpoint] and not journal.events
    exact = _pruning_evidence("2", 2, checkpoint["terminal_chain_fingerprint_sha256"], checkpoint)
    for field, value in (
        ("first_sequence", 2),
        ("last_sequence", 2),
        ("checkpoint_fingerprint_sha256", "f" * 64),
        ("retention_policy_version", "UNKNOWN"),
    ):
        fake = deepcopy(exact)
        fake["safe_payload"][field] = value
        fake["content_fingerprint_sha256"] = _content_fingerprint(fake)
        fake["chain_fingerprint_sha256"] = _chain_fingerprint(
            2, checkpoint["terminal_chain_fingerprint_sha256"], fake["content_fingerprint_sha256"]
        )
        with pytest.raises(ValueError):
            journal.append(fake, writer="AUTHORIZED_BOOTSTRAPPER")
        assert journal.pending_pruning == pending and not journal.events
    result = journal.append(
        exact, writer="AUTHORIZED_BOOTSTRAPPER", simulate_crash_before_pending_clear=True
    )
    assert result["outcome"] == "APPENDED" and journal.recovery_required
    replay = journal.append(deepcopy(exact), writer="AUTHORIZED_BOOTSTRAPPER")
    assert replay["outcome"] == "REPLAY" and not journal.recovery_required
    duplicate = deepcopy(exact)
    duplicate["audit_event_id"] = "evt_00000009-0000-7000-8000-000000000009"
    duplicate["content_fingerprint_sha256"] = _content_fingerprint(duplicate)
    duplicate["sequence"] = 3
    duplicate["previous_chain_fingerprint_sha256"] = exact["chain_fingerprint_sha256"]
    duplicate["chain_fingerprint_sha256"] = _chain_fingerprint(
        3, exact["chain_fingerprint_sha256"], duplicate["content_fingerprint_sha256"]
    )
    with pytest.raises(ValueError, match="UNBOUND_OR_DUPLICATE"):
        journal.append(duplicate, writer="AUTHORIZED_BOOTSTRAPPER")


def test_checkpoint_evidence_graph_detects_mutation_missing_and_pruned_binding_attack() -> None:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    cp1 = journal.seal_prefix(1)
    evidence1 = _pruning_evidence("2", 2, first["chain_fingerprint_sha256"], cp1)
    journal.prune_with_checkpoint(cp1, evidence1)
    journal.verify()
    original_cp1 = deepcopy(journal.checkpoints[0])
    journal.checkpoints[0]["segment_content_fingerprint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="pruning evidence"):
        journal.verify()
    journal.checkpoints[0] = original_cp1
    removed = journal.active_by_id.pop(evidence1["audit_event_id"])
    journal.events.remove(removed)
    with pytest.raises(ValueError):
        journal.verify_retained_evidence()
    journal.events.append(removed)
    journal.active_by_id[evidence1["audit_event_id"]] = removed
    third = _blank_event(
        "3", sequence=3, previous_chain_fingerprint_sha256=evidence1["chain_fingerprint_sha256"]
    )
    journal.append(third, writer="PROVISIONING_AUTHORITY")
    cp2 = journal.seal_prefix(2)
    evidence2 = _pruning_evidence("4", 4, third["chain_fingerprint_sha256"], cp2)
    journal.prune_with_checkpoint(cp2, evidence2)
    journal.verify()
    assert evidence1["audit_event_id"] not in journal.active_by_id
    binding = journal.compact_by_id[evidence1["audit_event_id"]]["pruning_evidence_binding"]
    assert binding == AuditJournalReference._expected_evidence_payload(cp1)
    original_cp1 = deepcopy(journal.checkpoints[0])
    journal.checkpoints[0]["segment_content_fingerprint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="pruning evidence"):
        journal.verify_retained_evidence()
    journal.checkpoints[0] = original_cp1
    original_binding = deepcopy(binding)
    binding["checkpoint_fingerprint_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        journal.verify_retained_evidence()
    journal.compact_by_id[evidence1["audit_event_id"]]["pruning_evidence_binding"] = (
        original_binding
    )
    journal.compact_by_id[evidence1["audit_event_id"]]["pruning_evidence_binding"] = None
    with pytest.raises(ValueError):
        journal.verify_retained_evidence()
    journal.compact_by_id[evidence1["audit_event_id"]]["pruning_evidence_binding"] = (
        original_binding
    )
    journal.verify()
    assert (
        journal.restore_decision(1, cp1["terminal_chain_fingerprint_sha256"])
        == "PRESERVE_CURRENT_NEWER_EXTENSION"
    )


def test_cross_contract_mutations_are_detected_without_mutating_frozen_sources() -> None:
    actual = _contracts()
    validate_m012_upstream_bindings(actual)
    mutations = []
    for entity, key, value in (
        ("AuditEvent", "id_prefix", "bad"),
        ("DeviceInstallation", "id_prefix", "bad"),
        ("RuntimeSession", "id_prefix", "bad"),
        ("AuditEvent", "parent", "RuntimeSession"),
    ):

        def mutate(c: dict[str, dict[str, Any]], entity=entity, key=key, value=value) -> None:
            canonical_entity(c["m02"], entity)[key] = value

        mutations.append(mutate)
    mutations += [
        lambda c: canonical_entity(c["m02"], "AuditEvent")["audit_event_categories"].pop(),
        lambda c: canonical_entity(c["m02"], "AuditEvent")["optional_references"].pop(),
        lambda c: c["m03"]["first_run_bootstrap_authority_contract"]["bootstrapper_role"].update(
            mode="authority"
        ),
        lambda c: c["m03"]["first_run_bootstrap_authority_contract"]["audit_boundary"][
            "allowed"
        ].pop(),
        lambda c: next(x for x in c["m03"]["process_roles"] if x["name"] == "bootstrapper").update(
            may_consume_maintenance_authorization=False
        ),
        lambda c: c["m07"]["event_contract"]["envelope_schema"]["fields"].pop(),
        lambda c: c["m07"]["event_contract"]["envelope_schema"]["nullable_fields"].pop(),
        lambda c: c["m07"]["event_contract"]["envelope_schema"]["field_schemas"]["order_id"].update(
            prefix="bad"
        ),
        lambda c: c["m07"]["event_contract"]["envelope_schema"]["field_schemas"][
            "aggregate_version"
        ].update(type="string"),
        lambda c: c["m07"]["event_contract"]["envelope_schema"]["field_schemas"][
            "event_fingerprint_sha256"
        ].update(type="string"),
        lambda c: c["m07"]["event_contract"]["event_schema_registry"]["ORDER_REJECTED"][
            "safe_payload_fields"
        ].append("extra"),
        lambda c: c["m11"]["state_store_physical_schema_registry"].update(
            current_state_store_schema_version=3
        ),
        lambda c: c["m11"]["state_store_physical_schema_registry"]["entries"].pop(),
        lambda c: c["m11"]["migration_protocol"].update(rollback_policy="BACKWARD"),
    ]
    for mutate in mutations:
        changed = deepcopy(actual)
        mutate(changed)
        with pytest.raises(ContractInconsistent):
            validate_m012_upstream_bindings(changed)


def _completed_single_prune() -> tuple[AuditJournalReference, dict[str, Any], dict[str, Any]]:
    journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    journal.append(first, writer="PROVISIONING_AUTHORITY")
    checkpoint = journal.seal_prefix(1)
    evidence = _pruning_evidence("2", 2, first["chain_fingerprint_sha256"], checkpoint)
    journal.prune_with_checkpoint(checkpoint, evidence)
    return journal, checkpoint, evidence


def test_restore_and_append_self_verify_missing_evidence_and_mutated_checkpoint() -> None:
    journal, checkpoint, evidence = _completed_single_prune()
    assert (
        journal.restore_decision(1, checkpoint["terminal_chain_fingerprint_sha256"])
        == "PRESERVE_CURRENT_NEWER_EXTENSION"
    )
    removed = journal.active_by_id.pop(evidence["audit_event_id"])
    journal.events.remove(removed)
    with pytest.raises(ValueError):
        journal.restore_decision(1, checkpoint["terminal_chain_fingerprint_sha256"])
    ordinary = _blank_event(
        "3",
        sequence=3,
        previous_chain_fingerprint_sha256=evidence["chain_fingerprint_sha256"],
    )
    before = len(journal.events)
    with pytest.raises(ValueError):
        journal.append(ordinary, writer="PROVISIONING_AUTHORITY")
    assert len(journal.events) == before

    mutated, checkpoint, _ = _completed_single_prune()
    mutated.checkpoints[0]["segment_content_fingerprint_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        mutated.restore_decision(1, checkpoint["terminal_chain_fingerprint_sha256"])


def test_active_identity_index_corruption_missing_entry_and_duplicate_fail_authority() -> None:
    for mode in ("wrong_key", "missing"):
        journal = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
        first = _blank_event("1")
        journal.append(first, writer="PROVISIONING_AUTHORITY")
        indexed = journal.active_by_id.pop(first["audit_event_id"])
        if mode == "wrong_key":
            journal.active_by_id["evt_00000009-0000-7000-8000-000000000009"] = indexed
        with pytest.raises(ValueError):
            journal.verify()
        with pytest.raises(ValueError):
            journal.append(deepcopy(first), writer="PROVISIONING_AUTHORITY")

    duplicate = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    first = _blank_event("1")
    duplicate.append(first, writer="PROVISIONING_AUTHORITY")
    second = deepcopy(first)
    second["sequence"] = 2
    second["previous_chain_fingerprint_sha256"] = first["chain_fingerprint_sha256"]
    second["chain_fingerprint_sha256"] = _chain_fingerprint(
        2, first["chain_fingerprint_sha256"], second["content_fingerprint_sha256"]
    )
    duplicate.events.append(second)
    with pytest.raises(ValueError, match="duplicate active"):
        duplicate.verify()


def test_compact_key_and_binding_corruption_block_verify_replay_and_append() -> None:
    journal, checkpoint, evidence = _completed_single_prune()
    metadata = journal.compact_by_id.pop("evt_00000001-0000-7000-8000-000000000001")
    journal.compact_by_id["evt_00000009-0000-7000-8000-000000000009"] = metadata
    with pytest.raises(ValueError, match="key mismatch"):
        journal.verify()
    with pytest.raises(ValueError):
        journal.append(_blank_event("1"), writer="PROVISIONING_AUTHORITY")

    journal, checkpoint, evidence = _completed_single_prune()
    third = _blank_event(
        "3",
        sequence=3,
        previous_chain_fingerprint_sha256=evidence["chain_fingerprint_sha256"],
    )
    journal.append(third, writer="PROVISIONING_AUTHORITY")
    checkpoint2 = journal.seal_prefix(2)
    evidence2 = _pruning_evidence("4", 4, third["chain_fingerprint_sha256"], checkpoint2)
    journal.prune_with_checkpoint(checkpoint2, evidence2)
    compact_evidence = journal.compact_by_id[evidence["audit_event_id"]]
    compact_evidence["pruning_evidence_binding"]["checkpoint_fingerprint_sha256"] = "f" * 64
    fifth = _blank_event(
        "5",
        sequence=5,
        previous_chain_fingerprint_sha256=evidence2["chain_fingerprint_sha256"],
    )
    before = len(journal.events)
    with pytest.raises(ValueError):
        journal.append(fifth, writer="PROVISIONING_AUTHORITY")
    assert len(journal.events) == before


def test_compact_binding_is_none_or_exact_closed_typed_object() -> None:
    journal, _, _ = _completed_single_prune()
    metadata = next(iter(journal.compact_by_id.values()))
    for malformed in (
        [],
        "binding",
        {"operation_code": "AUDIT_SEGMENT_PRUNED"},
        {
            "operation_code": "AUDIT_SEGMENT_PRUNED",
            "first_sequence": True,
            "last_sequence": 1,
            "checkpoint_fingerprint_sha256": "a" * 64,
            "retention_policy_version": JOURNAL["retention"]["policy_version"],
            "result_code": "COMPLETED",
        },
    ):
        metadata["pruning_evidence_binding"] = malformed
        with pytest.raises(ValueError):
            journal.verify()
    metadata["pruning_evidence_binding"] = None
    journal._verify_identity_indexes()


def _orphan_compact(audit_event_id: str, sequence: int) -> dict[str, Any]:
    return {
        "audit_event_id": audit_event_id,
        "content_fingerprint_sha256": "a" * 64,
        "sequence": sequence,
        "previous_chain_fingerprint_sha256": "b" * 64,
        "chain_fingerprint_sha256": "c" * 64,
        "pruning_evidence_binding": None,
    }


def test_orphan_compact_fails_verify_and_all_authority_paths_before_lookup() -> None:
    journal, checkpoint, evidence = _completed_single_prune()
    orphan_id = "evt_00000009-0000-7000-8000-000000000009"
    journal.compact_by_id[orphan_id] = _orphan_compact(orphan_id, checkpoint["last_sequence"] + 100)
    with pytest.raises(ValueError, match="orphan"):
        journal.verify()
    candidate = _blank_event(
        "9",
        sequence=3,
        previous_chain_fingerprint_sha256=evidence["chain_fingerprint_sha256"],
    )
    before = len(journal.events)
    with pytest.raises(ValueError, match="orphan"):
        journal.append(candidate, writer="PROVISIONING_AUTHORITY")
    assert len(journal.events) == before
    with pytest.raises(ValueError, match="orphan"):
        journal.restore_decision(1, checkpoint["terminal_chain_fingerprint_sha256"])
    with pytest.raises(ValueError, match="orphan"):
        journal.seal_prefix(1)


def test_no_checkpoint_compact_and_duplicate_compact_sequences_fail_closed() -> None:
    no_checkpoint = AuditJournalReference("dev_00000001-0000-7000-8000-000000000001", _contracts())
    orphan_id = "evt_00000009-0000-7000-8000-000000000009"
    no_checkpoint.compact_by_id[orphan_id] = _orphan_compact(orphan_id, 1)
    with pytest.raises(ValueError, match="orphan"):
        no_checkpoint.verify()

    duplicate, _, _ = _completed_single_prune()
    existing = next(iter(duplicate.compact_by_id.values()))
    duplicate_id = "evt_00000008-0000-7000-8000-000000000008"
    duplicate.compact_by_id[duplicate_id] = {
        **deepcopy(existing),
        "audit_event_id": duplicate_id,
    }
    with pytest.raises(ValueError, match="duplicate compact"):
        duplicate.verify()


def test_pending_latest_exception_rejects_orphan_active_evidence_and_requires_real_state() -> None:
    journal, _, evidence1 = _completed_single_prune()
    third = _blank_event(
        "3",
        sequence=3,
        previous_chain_fingerprint_sha256=evidence1["chain_fingerprint_sha256"],
    )
    journal.append(third, writer="PROVISIONING_AUTHORITY")
    checkpoint2 = journal.seal_prefix(2)
    with pytest.raises(ValueError, match="PRUNING_EVIDENCE"):
        journal.prune_with_checkpoint(checkpoint2, None)
    orphan = _pruning_evidence("4", 4, third["chain_fingerprint_sha256"], checkpoint2)
    orphan["safe_payload"]["checkpoint_fingerprint_sha256"] = "f" * 64
    orphan["content_fingerprint_sha256"] = _content_fingerprint(orphan)
    orphan["chain_fingerprint_sha256"] = _chain_fingerprint(
        4, checkpoint2["terminal_chain_fingerprint_sha256"], orphan["content_fingerprint_sha256"]
    )
    journal.events.append(orphan)
    journal.active_by_id[orphan["audit_event_id"]] = orphan
    with pytest.raises(ValueError, match="orphan"):
        journal.verify(allow_latest_missing_evidence=True)

    completed, _, _ = _completed_single_prune()
    with pytest.raises(ValueError, match="exact pending"):
        completed.verify(allow_latest_missing_evidence=True)


def test_compact_orphan_pruning_binding_fails_coverage_and_graph() -> None:
    journal, checkpoint, _ = _completed_single_prune()
    orphan_id = "evt_00000009-0000-7000-8000-000000000009"
    orphan = _orphan_compact(orphan_id, checkpoint["last_sequence"] + 1)
    orphan["pruning_evidence_binding"] = {
        "operation_code": "AUDIT_SEGMENT_PRUNED",
        "first_sequence": 99,
        "last_sequence": 99,
        "checkpoint_fingerprint_sha256": "a" * 64,
        "retention_policy_version": JOURNAL["retention"]["policy_version"],
        "result_code": "COMPLETED",
    }
    journal.compact_by_id[orphan_id] = orphan
    with pytest.raises(ValueError, match="orphan"):
        journal.verify()


# S9C-C1 pure executable oracle. It intentionally has no production imports or side effects.
class ContractInconsistent(ValueError):
    """An injected upstream contract no longer satisfies the M0.12 binding."""


UUID7 = "00000000-0000-7000-8000-000000000001"
IDS = {
    "workspace_id": f"ws_{UUID7}",
    "exchange_account_id": f"xacc_{UUID7}",
    "instrument_id": f"instr_{UUID7}",
    "market_data_route_id": f"mdr_{UUID7}",
    "execution_route_id": f"xroute_{UUID7}",
    "portfolio_id": f"port_{UUID7}",
    "device_installation_id": f"dev_{UUID7}",
}
SAFE_HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
HEX_ID = re.compile(r"^[0-9a-f]{16,64}$")
UUID7_BODY = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _json_pointer(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer.strip("/").split("/") if pointer != "/" else []:
        if token == "entity_kinds[canonical_name=RuntimeSession]":
            value = next(
                x for x in value["entity_kinds"] if x["canonical_name"] == "RuntimeSession"
            )
        elif token.startswith("entity_kinds["):
            value = value["entity_kinds"]
        else:
            value = value[token]
    return value


def _all_contracts() -> dict[str, dict[str, Any]]:
    return {item["milestone"]: _load(item["artifact"]) for item in MACHINE["upstream_dependencies"]}


def _utc(value: Any) -> datetime:
    if type(value) is not str or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
        value,
    ):
        raise ValueError("MALFORMED_TIMESTAMP")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("MALFORMED_TIMESTAMP") from error
    if result.microsecond:
        raise ValueError("MALFORMED_TIMESTAMP")
    return result


def _route_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("INVALID_UPSTREAM_EVIDENCE") from error


def _safe_scalar(value: Any) -> None:
    if (
        value is None
        or isinstance(value, bool)
        or (isinstance(value, int) and not isinstance(value, bool))
    ):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("INVALID_VALUE")
        return
    if not isinstance(value, str) or len(value) > 256 or any(ord(char) < 32 for char in value):
        raise ValueError("UNSAFE_VALUE")
    # Defense in depth after closed schema/type/domain validation.
    if re.search(
        r"(?i)(password|api[_ -]?secret|api[_ -]?key|private[_ -]?key|bearer|token|pin|biometric)",
        value,
    ):
        raise ValueError("SECRET_CONTENT")


def _canonical_id(value: Any, prefix: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix + "_")
        or not UUID7_BODY.fullmatch(value[len(prefix) + 1 :])
    ):
        raise ValueError("WRONG_SCOPE")


def _correlation(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or value.get("kind") not in {"LOCAL", "CANONICAL"}:
        raise ValueError("INVALID_CORRELATION")
    expected = {"kind", "value"} if value["kind"] == "LOCAL" else {"kind", "entity", "value"}
    if set(value) != expected:
        raise ValueError("INVALID_CORRELATION")
    if value["kind"] == "LOCAL":
        if not isinstance(value["value"], str) or not SAFE_HANDLE.fullmatch(value["value"]):
            raise ValueError("INVALID_CORRELATION")
        _safe_scalar(value["value"])
        return
    entities = {
        "RuntimeSession": "run",
        "ExchangeAccount": "xacc",
        "Instrument": "instr",
        "MarketDataRoute": "mdr",
        "ExecutionRoute": "xroute",
    }
    if type(value["entity"]) is not str or value["entity"] not in entities:
        raise ValueError("INVALID_CORRELATION")
    try:
        _canonical_id(value["value"], entities[value["entity"]])
    except ValueError as exc:
        raise ValueError("INVALID_CORRELATION") from exc


def _validate_closed(value: dict[str, Any], required: set[str], optional: set[str]) -> None:
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - optional
    ):
        raise ValueError("CLOSED_SCHEMA")


class ObservationReference:
    """Field-complete, source-first current observation ingress."""

    def __init__(
        self, contracts: dict[str, dict[str, Any]], policies: list[dict[str, Any]]
    ) -> None:
        self.contracts = contracts
        self.policies = {policy["policy_id"]: deepcopy(policy) for policy in policies}
        self.current: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.accepted_by_sequence: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.last_sequence: dict[tuple[str, str], int] = {}
        self.validate_bindings()
        for policy in policies:
            self._validate_policy(policy)

    def validate_bindings(self) -> None:
        artifacts = {
            item["milestone"]: item
            for item in MACHINE["health_readiness_model"]["upstream_binding_manifest"]
        }
        for milestone, binding in artifacts.items():
            actual = self.contracts.get(milestone)
            baseline = _load(binding["artifact"])
            if actual is None:
                raise ContractInconsistent(f"missing {milestone}")
            for pointer in binding["pointers"]:
                try:
                    if _json_pointer(actual, pointer) != _json_pointer(baseline, pointer):
                        raise ContractInconsistent(f"changed {milestone}{pointer}")
                except (KeyError, StopIteration, TypeError) as error:
                    raise ContractInconsistent(f"unresolved {milestone}{pointer}") from error

    @staticmethod
    def _validate_policy(policy: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["freshness_policy_contract"]
        if set(policy) != set(schema["required_fields"]):
            raise ValueError("INVALID_FRESHNESS_POLICY")
        if not SAFE_CODE.fullmatch(policy["policy_id"]):
            raise ValueError("INVALID_FRESHNESS_POLICY")
        for field in schema["required_fields"][1:2] + schema["required_fields"][4:]:
            if isinstance(policy[field], bool) or not isinstance(policy[field], int):
                raise ValueError("INVALID_FRESHNESS_POLICY")
        if (
            policy["version"] <= 0
            or policy["validity_horizon_seconds"] <= 0
            or any(policy[x] < 0 for x in schema["required_fields"][5:])
        ):
            raise ValueError("INVALID_FRESHNESS_POLICY")
        try:
            for field in schema["required_fields"][4:]:
                timedelta(seconds=policy[field])
            json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (OverflowError, ValueError) as error:
            raise ValueError("INVALID_FRESHNESS_POLICY") from error

    @staticmethod
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item["category"],
            item["source_component"],
            item["source_instance_id"],
            item["environment"],
            json.dumps(item["scope"], sort_keys=True, separators=(",", ":")),
        )

    def _validate_scope(self, item: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["category_scope_registry"][item["category"]]
        scope = item["scope"]
        if not isinstance(scope, dict) or set(scope) - set(schema["required_keys"]) - set(
            schema["optional_keys"]
        ):
            raise ValueError("WRONG_SCOPE")
        if not set(schema["required_keys"]) <= set(scope):
            raise ValueError("WRONG_SCOPE")
        if schema.get("at_least_one_of") and not set(schema["at_least_one_of"]) & set(scope):
            raise ValueError("WRONG_SCOPE")
        requirement = schema["environment_requirement"]
        if (requirement == "REQUIRED" and item["environment"] is None) or (
            requirement == "FORBIDDEN" and item["environment"] is not None
        ):
            raise ValueError("ILLEGAL_ENVIRONMENT")
        if item["source_component"] not in schema["allowed_source_components"]:
            raise ValueError("WRONG_SOURCE")
        for field, value in scope.items():
            contract = schema["field_contracts"][field]
            identity_type = contract["identity_or_type"]
            if contract["milestone"] in {"M0.2", "M0.5", "M0.6"}:
                if contract["milestone"] == "M0.2":
                    entity_name = contract["pointer"].split("=")[-1].rstrip("]")
                    upstream = next(
                        entity
                        for entity in self.contracts["M0.2"]["entity_kinds"]
                        if entity["canonical_name"] == entity_name
                    )
                else:
                    upstream = _json_pointer(
                        self.contracts[contract["milestone"]], contract["pointer"]
                    )
                if upstream.get("id_field") != field or upstream.get("id_prefix") != identity_type:
                    raise ContractInconsistent(f"unresolved identity contract for {field}")
                _canonical_id(value, upstream["id_prefix"])
            elif identity_type == "lowercase_hex_64":
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("WRONG_SCOPE")
            elif identity_type == "enum" and field == "component":
                if (
                    value
                    not in MACHINE["observability_model"]["source_component_registry"]["values"]
                ):
                    raise ValueError("WRONG_SCOPE")
            elif identity_type == "enum" and field == "resource_class":
                if value not in MACHINE["observability_model"]["resource_class_registry"]:
                    raise ValueError("WRONG_SCOPE")
            elif identity_type == "safe_handle":
                if not isinstance(value, str) or not SAFE_HANDLE.fullmatch(value):
                    raise ValueError("WRONG_SCOPE")
            _safe_scalar(value)

    def _validate_value(self, item: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["category_value_schemas"][item["category"]]
        value = item["value"]
        if (
            not isinstance(value, dict)
            or not set(schema["required_keys"]) <= set(value)
            or set(value) - set(schema["allowed_keys"])
        ):
            raise ValueError("INVALID_VALUE")
        for scalar in value.values():
            _safe_scalar(scalar)
        try:
            serialized = json.dumps(value, allow_nan=False)
        except (OverflowError, ValueError) as error:
            raise ValueError("INVALID_VALUE") from error
        if len(serialized) > schema["max_serialized_bytes"]:
            raise ValueError("INVALID_VALUE")

    def ingest(self, item: dict[str, Any], now: str) -> dict[str, Any]:
        fields = {
            field["name"]
            for field in MACHINE["observability_model"]["observation_envelope"]["fields"]
        }
        _validate_closed(item, fields, set())
        if not isinstance(item["observation_id"], str) or not SAFE_HANDLE.fullmatch(
            item["observation_id"]
        ):
            raise ValueError("INVALID_OBSERVATION_ID")
        _safe_scalar(item["observation_id"])
        if (
            type(item["category"]) is not str
            or item["category"] not in MACHINE["observability_model"]["categories"]
        ):
            raise ValueError("UNKNOWN_CATEGORY")
        if (
            item["source_component"]
            not in MACHINE["observability_model"]["source_component_registry"]["values"]
        ):
            raise ValueError("WRONG_SOURCE")
        session = item["source_instance_id"]
        policy_name = MACHINE["observability_model"]["category_scope_registry"][item["category"]][
            "source_instance_policy"
        ]
        policy = MACHINE["observability_model"]["source_instance_policy_interpreter"][policy_name]
        interpretation = policy[
            "core_host" if item["source_component"] == "core_host" else "non_core"
        ]
        if interpretation == "CANONICAL_RUNTIME_SESSION_REQUIRED":
            runtime = next(
                entity
                for entity in self.contracts["M0.2"]["entity_kinds"]
                if entity["canonical_name"] == "RuntimeSession"
            )
            try:
                _canonical_id(session, runtime["id_prefix"])
            except ValueError as exc:
                raise ValueError("WRONG_SOURCE_INSTANCE") from exc
        elif interpretation == "LOCAL_SAFE_SESSION_REQUIRED" and session is None:
            raise ValueError("WRONG_SOURCE_INSTANCE")
        elif interpretation.startswith("LOCAL_SAFE_SESSION") and session is not None:
            if not isinstance(session, str) or not SAFE_HANDLE.fullmatch(session):
                raise ValueError("WRONG_SOURCE_INSTANCE")
            _safe_scalar(session)
        elif interpretation == "FORBIDDEN_BY_CATEGORY_SOURCE_POLICY":
            raise ValueError("WRONG_SOURCE")
        environments = {
            x["environment_id"] for x in self.contracts["M0.4"]["execution_environments"]
        }
        if item["environment"] is not None and (
            type(item["environment"]) is not str or item["environment"] not in environments
        ):
            raise ValueError("ILLEGAL_ENVIRONMENT")
        self._validate_scope(item)
        if item["condition"] not in MACHINE["canonical_vocabulary"]["condition_states"]:
            raise ValueError("UNKNOWN_CONDITION")
        if not isinstance(item["reason_code"], str) or not SAFE_CODE.fullmatch(item["reason_code"]):
            raise ValueError("INVALID_REASON_CODE")
        _safe_scalar(item["reason_code"])
        if item["source_quality"] not in {"DIRECT", "DERIVED", "CACHED"}:
            raise ValueError("UNKNOWN_SOURCE_QUALITY")
        _correlation(item["correlation_reference"])
        self._validate_value(item)
        sequence = item["source_sequence"]
        if sequence is not None and (type(sequence) is not int or sequence < 0):
            raise ValueError("INVALID_SOURCE_SEQUENCE")
        if sequence is not None:
            try:
                json.dumps(sequence, allow_nan=False)
            except (OverflowError, ValueError) as error:
                raise ValueError("INVALID_SOURCE_SEQUENCE") from error
        if sequence is not None:
            replay_key = (item["source_component"], session, sequence)
            historical = self.accepted_by_sequence.get(replay_key)
            if historical is not None:
                # Independent exact JSON-content identity: unlike Python mapping
                # equality, canonical JSON distinguishes integer, decimal and bool.
                historical_fingerprint = hashlib.sha256(
                    json.dumps(
                        historical["observation"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                candidate_fingerprint = hashlib.sha256(
                    json.dumps(
                        item, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest()
                if historical_fingerprint != candidate_fingerprint:
                    raise ValueError("DUPLICATE_SEQUENCE_CONFLICT")
                replay = deepcopy(historical)
                replay["replayed"] = True
                return {"acceptance": replay, "current": deepcopy(self.current.get(self.key(item)))}
        if type(item["freshness_policy_id"]) is not str:
            raise ValueError("UNKNOWN_FRESHNESS_POLICY")
        observed, ingested, expires, logical_now = map(
            _utc, (item["observed_at_utc"], item["ingested_at_utc"], item["expires_at_utc"], now)
        )
        source_event = (
            _utc(item["source_event_at_utc"]) if item["source_event_at_utc"] is not None else None
        )
        policy = self.policies.get(item["freshness_policy_id"])
        if (
            policy is None
            or policy["category"] != item["category"]
            or policy["source_class"] != item["source_component"]
        ):
            raise ValueError("UNKNOWN_FRESHNESS_POLICY")
        if expires <= observed or expires - observed != timedelta(
            seconds=policy["validity_horizon_seconds"]
        ):
            raise ValueError("INVALID_EXPIRY")
        if observed - logical_now > timedelta(
            seconds=policy["allowed_observed_future_skew_seconds"]
        ):
            raise ValueError("FUTURE_TIMESTAMP")
        if source_event and source_event - logical_now > timedelta(
            seconds=policy["allowed_source_event_future_skew_seconds"]
        ):
            raise ValueError("FUTURE_SOURCE_EVENT")
        if observed - ingested > timedelta(
            seconds=policy["allowed_ingest_before_observed_skew_seconds"]
        ):
            raise ValueError("INGEST_SKEW")
        key = self.key(item)
        prior_current = self.current.get(key)
        if prior_current and observed < _utc(prior_current["observation"]["observed_at_utc"]):
            raise ValueError("CLOCK_REGRESSION")
        if prior_current and sequence is None:
            if observed == _utc(prior_current["observation"]["observed_at_utc"]):
                raise ValueError("SEQUENCE_REGRESSION")
            prior_source = prior_current["observation"]["source_event_at_utc"]
            if (
                source_event is not None
                and prior_source is not None
                and source_event < _utc(prior_source)
            ):
                raise ValueError("SOURCE_EVENT_REGRESSION")
        result = {
            "observation": deepcopy(item),
            "replayed": False,
            "sequence_gap": False,
            "effective_condition": item["condition"],
            "effective_reason_codes": [item["reason_code"]],
        }
        if sequence is not None:
            seq_key = (item["source_component"], session, sequence)
            source_key = (item["source_component"], session)
            last = self.last_sequence.get(source_key)
            if last is not None and sequence < last:
                raise ValueError("SEQUENCE_REGRESSION")
            if last is not None and sequence > last + 1:
                result.update(
                    sequence_gap=True,
                    effective_condition=MACHINE["observability_model"][
                        "sequence_gap_condition_composition"
                    ][item["condition"]],
                )
                result["effective_reason_codes"].append("SEQUENCE_GAP")
            self.last_sequence[source_key] = sequence
            self.accepted_by_sequence[seq_key] = deepcopy(result)
        self.current[key] = deepcopy(result)
        return {"acceptance": deepcopy(result), "current": deepcopy(result)}

    def projected(self, result: dict[str, Any] | None, now: str) -> dict[str, Any]:
        if result is None:
            return {
                "freshness": "STALE",
                "condition": "UNKNOWN",
                "reason_codes": ["MISSING_OBSERVATION"],
            }
        if _utc(now) >= _utc(result["observation"]["expires_at_utc"]):
            return {
                "freshness": "STALE",
                "condition": "UNKNOWN",
                "reason_codes": ["OBSERVATION_EXPIRED"],
            }
        return {
            "freshness": "FRESH",
            "condition": result["effective_condition"],
            "reason_codes": result["effective_reason_codes"],
        }


class HealthAggregatorReference:
    def __init__(self, observations: ObservationReference) -> None:
        self.observations = observations

    def compose(self, keys: list[tuple[Any, ...]], now: str) -> str:
        states = [
            self.observations.projected(self.observations.current.get(key), now)["condition"]
            for key in keys
        ]
        if any(x == "BLOCKED" for x in states):
            return "BLOCKED"
        if any(x == "UNKNOWN" for x in states):
            return "UNKNOWN"
        if any(x == "DEGRADED" for x in states):
            return "DEGRADED"
        return "OK"


class UpstreamReadinessEvidence:
    """Structured, scope-bound upstream-shaped evidence; never a caller boolean map."""

    REQUIRED = set(
        MACHINE["health_readiness_model"]["upstream_evidence_contract"]["required_sections"]
    )

    def __init__(self, records: dict[str, Any]) -> None:
        if not isinstance(records, dict) or set(records) != self.REQUIRED:
            raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        self.records = deepcopy(records)
        self.contracts = _all_contracts()
        self._validate_nested()

    @staticmethod
    def _closed(record: Any, fields: list[str]) -> None:
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("INVALID_UPSTREAM_EVIDENCE")

    def _canonical(self, value: Any, entity: str) -> None:
        upstream = next(
            x for x in self.contracts["M0.2"]["entity_kinds"] if x["canonical_name"] == entity
        )
        _canonical_id(value, upstream["id_prefix"])

    def _validate_nested(self) -> None:
        schemas = MACHINE["health_readiness_model"]["upstream_evidence_contract"][
            "nested_section_schemas"
        ]
        runtime = self.records["runtime_context"]
        self._closed(runtime, schemas["runtime_context"]["required_fields"])
        self._canonical(runtime["runtime_session_id"], "RuntimeSession")
        self._canonical(runtime["device_installation_id"], "DeviceInstallation")
        if not isinstance(
            runtime["state_store_identity_fingerprint_sha256"], str
        ) or not re.fullmatch(r"[0-9a-f]{64}", runtime["state_store_identity_fingerprint_sha256"]):
            raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        readiness = {item["name"] for item in self.contracts["M0.3"]["startup_readiness_states"]}
        if (
            runtime["startup_readiness_state"] not in readiness
            or type(runtime["process_lock_owned"]) is not bool
        ):
            raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        capability = self.records["product_capabilities_evidence"]
        if capability is not None:
            self._closed(capability, schemas["product_capabilities_evidence"]["required_fields"])
            chain = self.contracts["M0.4"]["stage_evidence_chain_contract"]
            context_fields = self.contracts["M0.4"]["validation_context_contract"]["context_fields"]
            if (
                capability["carrier_kind"] != "CORE_ACCEPTED_M04_VALIDATED_SNAPSHOT_REFERENCE"
                or capability["evidence_schema_version"] != chain["evidence_schema_version"]
                or capability["stage_id"] != "VALIDATED_SNAPSHOT_CREATION"
                or capability["stage_result"] != "SNAPSHOT_CREATED"
            ):
                raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
            for field in (
                "stage_evidence_id",
                "predecessor_stage_evidence_id",
                "validation_context_id",
                "document_fingerprint",
                "signed_payload_hash",
                "capability_set_hash",
            ):
                if not isinstance(capability[field], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", capability[field]
                ):
                    raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
            if (
                capability["edition_id"]
                != self.contracts["M0.4"]["ProductCapabilities"][
                    "current_edition_capability_policy"
                ]["edition_id"]
            ):
                raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
            if (
                capability["payload_schema_version"]
                != "cryptohunter.product_capabilities.payload.v1"
                or capability["signature_schema_version"]
                != "cryptohunter.product_capabilities.signature.v1"
            ):
                raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
            if any(
                type(capability[field]) is not bool or not capability[field]
                for field in (
                    "complete_stage_evidence",
                    "same_validation_context_id",
                    "same_document_fingerprint",
                    "issuer_attestation_verified",
                )
            ):
                raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
            if not all(field in capability for field in context_fields):
                raise ValueError("INVALID_PRODUCT_CAPABILITIES_EVIDENCE")
        environments = {
            x["environment_id"] for x in self.contracts["M0.4"]["execution_environments"]
        }
        map_specs = {
            "accounts_by_id": ("ExchangeAccount", "exchange_account_id"),
            "instruments_by_id": ("Instrument", "instrument_id"),
            "market_data_routes_by_id": ("MarketDataRoute", "market_data_route_id"),
            "execution_routes_by_id": ("ExecutionRoute", "execution_route_id"),
        }
        readiness_schema = self.contracts["M0.6"]["record_schemas"]["RouteReadiness"]

        def validate_route_readiness(readiness: Any, route_id: str, route_kind: str) -> None:
            self._closed(readiness, readiness_schema["exact_fields"])
            enums = readiness_schema["enum_registry"]
            if (
                readiness["route_id"] != route_id
                or readiness["route_kind"] != route_kind
                or readiness["route_kind"] not in enums["route_kind"]
                or readiness["readiness_state"] not in enums["readiness_state"]
                or readiness["sequence_state"] not in enums["sequence_state"]
                or type(readiness["metadata_version"]) is not int
                or readiness["metadata_version"] < 1
            ):
                raise ValueError("INVALID_UPSTREAM_EVIDENCE")
            _route_utc(readiness["observed_at"])

        for section, (entity, id_field) in map_specs.items():
            schema = schemas[section]
            if not isinstance(self.records[section], dict):
                raise ValueError("INVALID_UPSTREAM_EVIDENCE")
            for key, record in self.records[section].items():
                self._closed(record, schema["required_fields"])
                self._canonical(key, entity)
                self._canonical(record[id_field], entity)
                if key != record[id_field] or record["environment"] not in environments:
                    raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                if section == "market_data_routes_by_id":
                    self._canonical(record["instrument_id"], "Instrument")
                    self._canonical(record["workspace_id"], "Workspace")
                    if record["route_kind"] != "MARKET_DATA":
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    validate_route_readiness(
                        record["route_readiness"], record["market_data_route_id"], "MARKET_DATA"
                    )
                    if not all(
                        isinstance(record[field], str) and record[field]
                        for field in (
                            "exchange_id",
                            "market_type",
                            "adapter_family_id",
                            "endpoint_class",
                        )
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    market_schema = self.contracts["M0.6"]["record_schemas"]["MarketDataRoute"]
                    for field in ("instrument_ids", "channel_types"):
                        if (
                            type(record[field]) is not list
                            or not record[field]
                            or len(record[field]) != len(set(record[field]))
                        ):
                            raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    for instrument_id in record["instrument_ids"]:
                        self._canonical(instrument_id, "Instrument")
                    if record["instrument_id"] not in record["instrument_ids"]:
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    if not set(record["channel_types"]) <= set(
                        self.contracts["M0.6"]["array_enum_registries"]["market_data_channel_types"]
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    if any(
                        record[field] not in allowed
                        for field, allowed in market_schema["enum_registry"].items()
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    freshness = record["freshness_policy"]
                    if (
                        set(freshness) != {"max_age_seconds"}
                        or type(freshness["max_age_seconds"]) is not int
                        or freshness["max_age_seconds"] < 0
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                if section == "instruments_by_id":
                    self._canonical(record["workspace_id"], "Workspace")
                    if not all(
                        isinstance(record[field], str) and record[field]
                        for field in ("exchange_id", "market_type")
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                if section == "execution_routes_by_id":
                    self._canonical(record["instrument_id"], "Instrument")
                    self._canonical(record["exchange_account_id"], "ExchangeAccount")
                    self._canonical(record["workspace_id"], "Workspace")
                    if record["route_kind"] != "EXECUTION":
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    execution_schema = self.contracts["M0.6"]["record_schemas"]["ExecutionRoute"]
                    if (
                        any(
                            not isinstance(record[field], str) or not record[field]
                            for field in (
                                "exchange_id",
                                "market_type",
                                "adapter_family_id",
                                "endpoint_class",
                                "route_status",
                            )
                        )
                        or record["endpoint_class"]
                        not in execution_schema["enum_registry"]["endpoint_class"]
                        or record["route_status"]
                        not in execution_schema["enum_registry"]["route_status"]
                        or record["market_type"]
                        not in self.contracts["M0.5"]["market_type_registry"]
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
                    validate_route_readiness(
                        record["route_readiness"], record["execution_route_id"], "EXECUTION"
                    )
                    array_registries = self.contracts["M0.6"]["array_enum_registries"]
                    arrays = {
                        "supported_instrument_types": set(
                            self.contracts["M0.5"]["instrument_type_registry"]
                        ),
                        "route_capability_ceiling": set(array_registries["route_capabilities"]),
                        "authorization_dependencies": set(
                            array_registries["authorization_dependency_names"]
                        ),
                    }
                    if any(
                        type(record[field]) is not list
                        or not record[field]
                        or len(record[field]) != len(set(record[field]))
                        or not set(record[field]) <= allowed
                        for field, allowed in arrays.items()
                    ):
                        raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        risk_schema = schemas["risk_by_scope"]
        for key, record in self.records["risk_by_scope"].items():
            self._closed(record, risk_schema["required_fields"])
            for entity, field in (
                ("ExchangeAccount", "exchange_account_id"),
                ("Instrument", "instrument_id"),
                ("ExecutionRoute", "execution_route_id"),
            ):
                self._canonical(record[field], entity)
            if (
                key != ReadinessReference.execution_scope_key(record["environment"], record)
                or record["environment"] not in environments
                or any(
                    type(record[x]) is not bool
                    for x in ("risk_allowed", "kill_switch_inactive", "execution_lease_valid")
                )
            ):
                raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        persistence = self.records["persistence_context"]
        self._closed(persistence, schemas["persistence_context"]["required_fields"])
        self._canonical(persistence["device_installation_id"], "DeviceInstallation")
        if not re.fullmatch(
            r"[0-9a-f]{64}", persistence["state_store_identity_fingerprint_sha256"]
        ) or any(
            type(persistence[x]) is not bool for x in ("accepted", "recovery_attempt_allowed")
        ):
            raise ValueError("INVALID_UPSTREAM_EVIDENCE")
        for key, record in self.records["reconciliation_by_scope"].items():
            self._closed(record, schemas["reconciliation_by_scope"]["required_fields"])
            self._canonical(record["exchange_account_id"], "ExchangeAccount")
            self._canonical(record["portfolio_id"], "Portfolio")
            expected = (
                f"{record['environment']}:{record['exchange_account_id']}:{record['portfolio_id']}"
            )
            if (
                key != expected
                or record["environment"] not in environments
                or type(record["attempt_allowed"]) is not bool
            ):
                raise ValueError("INVALID_UPSTREAM_EVIDENCE")

    @property
    def runtime(self) -> dict[str, Any]:
        return self.records["runtime_context"]

    def derive(
        self,
        check: dict[str, Any],
        row: dict[str, Any],
        environment: str | None,
        scope: dict[str, Any],
        contracts: dict[str, Any],
        validation_time_utc: str,
    ) -> bool:
        registry = MACHINE["health_readiness_model"]["upstream_evidence_contract"]["check_registry"]
        if check["check_id"] not in registry or {
            k: check[k] for k in ("milestone", "pointer", "scope_binding")
        } != {k: registry[check["check_id"]][k] for k in ("milestone", "pointer", "scope_binding")}:
            raise ContractInconsistent("unknown or changed upstream check")
        check_id = check["check_id"]
        runtime = self.runtime
        if check_id == "m03_ready":
            return runtime["startup_readiness_state"] == "READY"
        if check_id == "m03_single_instance":
            return runtime.get("process_lock_owned") is True
        if check_id == "m04_capability_allowed":
            if row["readiness_id"] == "MARKET_DATA_CONSUMPTION":
                environment_record = next(
                    (
                        item
                        for item in contracts["M0.4"]["execution_environments"]
                        if item["environment_id"] == environment
                    ),
                    None,
                )
                if not environment_record:
                    return False
                endpoint_policy = contracts["M0.4"]["endpoint_policy"]
                if environment == "PAPER":
                    return (
                        endpoint_policy["paper_execution_endpoint_class"]
                        in environment_record["endpoint_classes"]
                    )
                if environment == "TESTNET":
                    return bool(
                        set(environment_record["endpoint_classes"])
                        & set(endpoint_policy["testnet_allowed_endpoint_classes"])
                    )
                if environment == "LIVE":
                    if (
                        endpoint_policy["live_endpoint_resolution_current_edition_allowed"]
                        is not False
                    ):
                        raise ContractInconsistent(
                            "changed LIVE endpoint policy requires an explicit M0.12 update"
                        )
                    return False
                return False
            return ReadinessReference.resolve_capability_authority(
                row, environment, scope, contracts, self
            )
        if check_id == "m05_account_operable":
            account = self.records["accounts_by_id"].get(scope.get("exchange_account_id"))
            policy = contracts["M0.5"]["current_edition_account_operability_policy"]
            return bool(
                account
                and account.get("exchange_account_id") == scope.get("exchange_account_id")
                and account.get("environment") == environment
                and account.get("lifecycle_state") in policy["operational_lifecycle_states"]
                and account.get("connection_state") in policy["operational_connection_states"]
                and account.get("execution_authorization")
                in policy["operational_authorization_states"]
            )
        if check_id == "m05_instrument_operable":
            instrument = self.records["instruments_by_id"].get(scope.get("instrument_id"))
            return bool(
                instrument
                and instrument.get("instrument_id") == scope.get("instrument_id")
                and instrument.get("environment") == environment
                and instrument.get("trading_status")
                == contracts["M0.5"]["instrument_contract"]["metadata_vs_tradability"][
                    "execution_activation_requires"
                ]
            )
        if check_id == "m06_route_ready":
            validation_time = _route_utc(validation_time_utc)
            routes = []
            if scope.get("market_data_route_id"):
                routes.append(
                    self.records["market_data_routes_by_id"].get(scope["market_data_route_id"])
                )
            if scope.get("execution_route_id"):
                routes.append(
                    self.records["execution_routes_by_id"].get(scope["execution_route_id"])
                )
            registry = contracts["M0.5"]["exchange_registry_contract"]["entries"]

            def enabled_venue(route: dict[str, Any]) -> bool:
                venue = next(
                    (entry for entry in registry if entry["exchange_id"] == route["exchange_id"]),
                    None,
                )
                return bool(
                    venue
                    and venue["status"] == "ENABLED"
                    and route["environment"] in venue["supported_environments"]
                    and route["market_type"] in venue["supported_market_types"]
                    and route["adapter_family_id"] == venue["adapter_family_id"]
                )

            def route_is_source_ready(route: dict[str, Any] | None) -> bool:
                if not route:
                    return False
                if route.get("route_kind") == "MARKET_DATA":
                    readiness = route["route_readiness"]
                    age = (validation_time - _route_utc(readiness["observed_at"])).total_seconds()
                    endpoint = contracts["M0.6"]["endpoint_class_registry"].get(
                        route["endpoint_class"]
                    )
                    if not (
                        endpoint
                        and endpoint["environment"] == route["environment"]
                        and "MARKET_DATA" in endpoint["allowed_route_kinds"]
                        and endpoint["access_scope"] == route["data_scope"]
                        and not (
                            route["data_scope"] == "PUBLIC"
                            and any(
                                channel.startswith("PRIVATE_") for channel in route["channel_types"]
                            )
                        )
                        and route["data_scope"] == "PUBLIC"
                    ):
                        return False
                    if not enabled_venue(route):
                        return False
                    instruments = [
                        self.records["instruments_by_id"].get(instrument_id)
                        for instrument_id in route["instrument_ids"]
                    ]
                    instrument = self.records["instruments_by_id"].get(route["instrument_id"])
                    if not (
                        instrument
                        and all(instruments)
                        and all(
                            route["workspace_id"] == item["workspace_id"]
                            and route["exchange_id"] == item["exchange_id"]
                            and route["environment"] == item["environment"]
                            and route["market_type"] == item["market_type"]
                            and route["adapter_family_id"] == item["source_adapter_family_id"]
                            for item in instruments
                        )
                        and route["workspace_id"] == instrument["workspace_id"]
                        and route["exchange_id"] == instrument["exchange_id"]
                        and route["market_type"] == instrument["market_type"]
                        and route["route_status"] == "ACTIVE"
                        and readiness["readiness_state"] == "READY"
                        and readiness["sequence_state"] == "CONTIGUOUS"
                        and 0 <= age <= route["freshness_policy"]["max_age_seconds"]
                    ):
                        return False
                    return True
                if route.get("route_kind") == "EXECUTION":
                    endpoint = contracts["M0.6"]["endpoint_class_registry"].get(
                        route["endpoint_class"]
                    )
                    account = self.records["accounts_by_id"].get(route["exchange_account_id"])
                    instrument = self.records["instruments_by_id"].get(route["instrument_id"])
                    expected_dependencies = contracts["M0.6"][
                        "authorization_dependencies_by_environment"
                    ].get(route["environment"])
                    dependency_policy = contracts["M0.6"]["authorization_dependency_policy"]
                    environment_policy = contracts["M0.6"]["environment_policy"].get(
                        route["environment"]
                    )
                    allowed_pairs = contracts["M0.6"]["current_edition_execution_pair_policy"][
                        "allowed_pairs"
                    ]
                    readiness = route["route_readiness"]
                    age = (validation_time - _route_utc(readiness["observed_at"])).total_seconds()
                    return bool(
                        endpoint
                        and endpoint["environment"] == route["environment"]
                        and "EXECUTION" in endpoint["allowed_route_kinds"]
                        and environment_policy
                        and (
                            (
                                route["environment"] == "PAPER"
                                and endpoint["access_scope"] == "LOCAL_SIMULATION"
                                and environment_policy["execution"] == "local simulation only"
                            )
                            or (
                                route["environment"] == "TESTNET"
                                and endpoint["access_scope"] == "PRIVATE"
                                and environment_policy["live_fallback"] is False
                            )
                        )
                        and enabled_venue(route)
                        and account
                        and instrument
                        and route["workspace_id"]
                        == account["workspace_id"]
                        == instrument["workspace_id"]
                        and route["exchange_id"]
                        == account["exchange_id"]
                        == instrument["exchange_id"]
                        and route["environment"]
                        == account["environment"]
                        == instrument["environment"]
                        and route["market_type"]
                        == account["market_type"]
                        == instrument["market_type"]
                        and instrument["instrument_type"] in route["supported_instrument_types"]
                        and instrument["instrument_type"]
                        in allowed_pairs.get(route["market_type"], [])
                        and route["route_status"] == "ACTIVE"
                        and "PLACE_ORDERS" in route["route_capability_ceiling"]
                        and expected_dependencies is not None
                        and dependency_policy["exact_set_equality"] is True
                        and len(route["authorization_dependencies"])
                        == len(set(route["authorization_dependencies"]))
                        and set(route["authorization_dependencies"]) == set(expected_dependencies)
                        and readiness["route_id"] == route["execution_route_id"]
                        and readiness["route_kind"] == "EXECUTION"
                        and readiness["readiness_state"] == "READY"
                        and readiness["sequence_state"] == "NOT_APPLICABLE"
                        and 0
                        <= age
                        <= contracts["M0.6"]["execution_route_contract"][
                            "execution_readiness_max_age_seconds"
                        ]
                    )
                return False

            return bool(
                routes
                and all(
                    route
                    and route_is_source_ready(route)
                    and route.get("environment") == environment
                    and route.get("instrument_id") in {None, scope.get("instrument_id")}
                    and route.get("exchange_account_id") in {None, scope.get("exchange_account_id")}
                    for route in routes
                )
            )
        if check_id in {
            "m09_risk_allowed",
            "m09_kill_switch_inactive",
            "m09_execution_lease_valid",
        }:
            risk = self.records["risk_by_scope"].get(
                ReadinessReference.execution_scope_key(environment, scope)
            )
            field = {
                "m09_risk_allowed": "risk_allowed",
                "m09_kill_switch_inactive": "kill_switch_inactive",
                "m09_execution_lease_valid": "execution_lease_valid",
            }[check_id]
            return bool(risk and risk.get(field) is True)
        if check_id in {"m11_persistence_accepted", "m11_recovery_attempt_allowed"}:
            persistence = self.records["persistence_context"]
            exact = persistence.get("device_installation_id") == runtime.get(
                "device_installation_id"
            ) and persistence.get("state_store_identity_fingerprint_sha256") == runtime.get(
                "state_store_identity_fingerprint_sha256"
            )
            return exact and (
                persistence.get("accepted") is True
                if check_id == "m11_persistence_accepted"
                else persistence.get("recovery_attempt_allowed") is True
            )
        if check_id == "m08_reconciliation_authorized":
            value = self.records["reconciliation_by_scope"].get(
                f"{environment}:{scope.get('exchange_account_id')}:{scope.get('portfolio_id')}"
            )
            return bool(value and value.get("attempt_allowed") is True)
        if check_id == "m11_recovery_attempt_allowed":
            return self.records["persistence_context"].get("recovery_attempt_allowed") is True
        raise ContractInconsistent(f"unsupported check {check_id}")


class ReadinessReference:
    """Matrix-driven readiness derived from structured upstream evidence and current telemetry."""

    def __init__(self, observations: ObservationReference) -> None:
        self.observations = observations
        self.rows = {
            row["readiness_id"]: row
            for row in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        }
        self._validate_matrix()

    @staticmethod
    def execution_scope_key(environment: str | None, scope: dict[str, Any]) -> str:
        return ":".join(
            str(x)
            for x in (
                environment,
                scope.get("exchange_account_id"),
                scope.get("instrument_id"),
                scope.get("execution_route_id"),
            )
        )

    def _validate_matrix(self) -> None:
        modes = set(MACHINE["health_readiness_model"]["dependency_scope_binding_modes"])
        selector_registry = MACHINE["health_readiness_model"]["capability_selector_registry"]
        for row in self.rows.values():
            authority = row["capability_authority"]
            if authority["milestone"] not in self.observations.contracts:
                raise ContractInconsistent("capability milestone missing")
            try:
                _json_pointer(
                    self.observations.contracts[authority["milestone"]], authority["pointer"]
                )
            except (KeyError, TypeError) as error:
                raise ContractInconsistent("capability pointer unresolved") from error
            family = authority.get("selector_family")
            if family not in selector_registry:
                raise ContractInconsistent("unknown capability selector family")
            definition = selector_registry[family]
            selector = authority["selector"]
            if set(selector) != set(definition["exact_keys"]):
                raise ContractInconsistent("invalid capability selector shape")
            if authority["milestone"] != definition["compatible_milestone"]:
                raise ContractInconsistent("selector/pointer family mismatch")
            if family == "CAPABILITY_ENVIRONMENT":
                legal = (
                    selector["capability"] in definition["legal_capabilities"]
                    and selector["environment"] in row["allowed_environments"]
                )
            elif family == "FIELD_EQUALS_ENVIRONMENT":
                legal = selector == {
                    "field": definition["legal_field"],
                    "equals": definition["legal_equals"],
                    "environment": definition["legal_environment"],
                }
            elif family == "ROUTE_KIND":
                legal = selector["route_kind"] == definition["legal_route_kind"]
            elif family == "STARTUP_STATE":
                legal = selector["state"] == definition["legal_state"]
            elif family == "ACCEPTED_STATE_REQUIRED":
                legal = selector["accepted_state_required"] is definition["legal_value"]
            elif family == "ATTEMPT_ONLY":
                legal = selector["attempt_only"] is definition["legal_value"]
            else:
                raise ContractInconsistent("unsupported selector family")
            if not legal:
                raise ContractInconsistent("illegal capability selector")
            for dependency in row["required_observations"]:
                binding = dependency.get("scope_binding", {})
                if binding.get("mode") not in modes:
                    raise ContractInconsistent("unsupported dependency binding")
                if binding["mode"] == "MATCH_REQUEST_SCOPE_FIELDS" and not binding.get("fields"):
                    raise ContractInconsistent("empty intersection forbidden")

    @staticmethod
    def resolve_capability_authority(
        row: dict[str, Any],
        environment: str | None,
        scope: dict[str, Any],
        contracts: dict[str, Any],
        evidence: UpstreamReadinessEvidence,
    ) -> bool:
        authority = row["capability_authority"]
        try:
            source = _json_pointer(contracts[authority["milestone"]], authority["pointer"])
        except (KeyError, TypeError) as error:
            raise ContractInconsistent("capability authority unresolved") from error
        selector = authority["selector"]
        family = authority["selector_family"]
        if family == "CAPABILITY_ENVIRONMENT":
            capability = selector["capability"]
            if capability not in source["capability_set"]:
                return False
            pipeline = contracts["M0.4"]["signature_validation_pipeline"]
            if environment == "PAPER":
                fallback = contracts["M0.4"]["fail_closed_fallbacks"]["SAFE_LOCAL_ONLY"]
                return (
                    source["environment_capabilities"]["PAPER"]["local_execution_allowed"] is True
                    and fallback["paper_local_simulation_may_remain_available"] is True
                    and fallback["testnet_private_execution_allowed"] is False
                )
            if environment == "TESTNET":
                conditional = (
                    source["environment_capabilities"]["TESTNET"][
                        "private_execution_allowed_when_trust_state_valid"
                    ]
                    is True
                )
                crypto_state = pipeline["m0_4_cryptographic_signature_verification"]
                future_availability = pipeline["stage_result_registry"][
                    "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED"
                ]["availability"]
                if (
                    crypto_state != "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"
                    or future_availability != "FUTURE_ONLY"
                ):
                    raise ContractInconsistent(
                        "unrecognized M0.4 reachability; explicit binding update required"
                    )
                assert (
                    conditional
                )  # conditional policy exists, but its VALID prerequisite is unreachable
                return False
            return False
        if family == "FIELD_EQUALS_ENVIRONMENT":
            return source[selector["field"]] is selector["equals"]
        if family == "ROUTE_KIND":
            route = evidence.records["market_data_routes_by_id"].get(
                scope.get("market_data_route_id")
            )
            return bool(
                route
                and route["route_kind"] == selector["route_kind"]
                and route["environment"] == environment
                and route["instrument_id"] == scope.get("instrument_id")
                and route["route_readiness"]["route_id"] == route["market_data_route_id"]
            )
        if family == "STARTUP_STATE":
            return evidence.runtime["startup_readiness_state"] == selector["state"]
        if family == "ACCEPTED_STATE_REQUIRED":
            persistence = evidence.records["persistence_context"]
            return selector["accepted_state_required"] is True and persistence["accepted"] is True
        if family == "ATTEMPT_ONLY":
            key = f"{environment}:{scope.get('exchange_account_id')}:{scope.get('portfolio_id')}"
            reconciliation = evidence.records["reconciliation_by_scope"].get(key)
            return selector["attempt_only"] is True and bool(
                reconciliation and reconciliation["attempt_allowed"] is True
            )
        raise ContractInconsistent("unimplemented capability selector")

    def _expected_scope(
        self,
        binding: dict[str, Any],
        request_scope: dict[str, Any],
        evidence: UpstreamReadinessEvidence,
    ) -> dict[str, Any]:
        mode = binding["mode"]
        if mode == "MATCH_REQUEST_SCOPE_FIELDS":
            fields = binding["fields"]
            if not fields or not set(fields) <= set(request_scope):
                raise ContractInconsistent("empty/missing request scope binding")
            return {field: request_scope[field] for field in fields}
        if mode == "FIXED_SCOPE":
            return binding["fixed_scope"]
        if mode == "CURRENT_INSTALLATION_STATESTORE":
            runtime = evidence.runtime
            return {field: runtime[field] for field in binding["context_fields"]}
        if mode == "CURRENT_RUNTIME_CONTEXT":
            return {field: evidence.runtime[field] for field in binding["context_fields"]}
        raise ContractInconsistent("unknown binding")

    def evaluate(
        self,
        readiness_id: str,
        environment: str | None,
        scope: dict[str, Any],
        now: str,
        evidence: UpstreamReadinessEvidence,
    ) -> str:
        if not isinstance(evidence, UpstreamReadinessEvidence):
            raise TypeError("structured UpstreamReadinessEvidence required")
        row = self.rows[readiness_id]
        if (environment if environment is not None else "NONE") not in row[
            "allowed_environments"
        ] or set(scope) != set(row["scope_schema"]["required_keys"]):
            return "BLOCKED"
        if not self.resolve_capability_authority(
            row, environment, scope, self.observations.contracts, evidence
        ):
            return "BLOCKED"
        if any(
            not evidence.derive(check, row, environment, scope, self.observations.contracts, now)
            for check in row["required_upstream_checks"]
        ):
            return "BLOCKED"
        for dependency in row["required_observations"]:
            expected = self._expected_scope(dependency["scope_binding"], scope, evidence)
            category_schema = MACHINE["observability_model"]["category_scope_registry"][
                dependency["category"]
            ]
            dependency_environment = (
                None if category_schema["environment_requirement"] == "FORBIDDEN" else environment
            )
            matches = [
                value
                for key, value in self.observations.current.items()
                if key[0] == dependency["category"]
                and key[3] == dependency_environment
                and all(
                    value["observation"]["scope"].get(field) == wanted
                    for field, wanted in expected.items()
                )
            ]
            if len(matches) != 1:
                return "BLOCKED"
            condition = self.observations.projected(matches[0], now)["condition"]
            if condition in {"UNKNOWN", "BLOCKED"} or (
                condition == "DEGRADED" and dependency["degraded"] == "BLOCKED"
            ):
                return "BLOCKED"
        return "OK"


def _policy(category: str, *, horizon: int = 30, skew: int = 4) -> dict[str, Any]:
    return {
        "policy_id": f"{category}_TEST_V1",
        "version": 1,
        "category": category,
        "source_class": "core_host",
        "validity_horizon_seconds": horizon,
        "allowed_observed_future_skew_seconds": skew,
        "allowed_source_event_future_skew_seconds": skew + 1,
        "allowed_ingest_before_observed_skew_seconds": skew,
    }


def _scope(category: str) -> dict[str, Any]:
    return {
        "STRUCTURED_LOGS": {},
        "METRICS": {},
        "TRACES": {},
        "COMPONENT_STATUS": {"component": "core_host"},
        "ADAPTER_STATUS": {
            "exchange_account_id": IDS["exchange_account_id"],
            "execution_route_id": IDS["execution_route_id"],
        },
        "MARKET_DATA_FRESHNESS": {
            "market_data_route_id": IDS["market_data_route_id"],
            "instrument_id": IDS["instrument_id"],
        },
        "EXECUTION_PATH_HEALTH": {
            k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")
        },
        "PERSISTENCE_HEALTH": {
            "device_installation_id": IDS["device_installation_id"],
            "state_store_identity_fingerprint_sha256": "a" * 64,
        },
        "RECONCILIATION_HEALTH": {
            "exchange_account_id": IDS["exchange_account_id"],
            "portfolio_id": IDS["portfolio_id"],
        },
        "SECURITY_RISK_GATE_HEALTH": {
            k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")
        },
        "RESOURCE_RUNTIME_HEALTH": {"component": "core_host", "resource_class": "CPU"},
    }[category]


def _value(category: str) -> dict[str, Any]:
    return {
        "STRUCTURED_LOGS": {},
        "METRICS": {},
        "TRACES": {},
        "COMPONENT_STATUS": {"progress_counter": 1},
        "ADAPTER_STATUS": {"transport_state": "CONNECTED"},
        "MARKET_DATA_FRESHNESS": {
            "last_data_at_utc": "2030-01-01T00:00:00Z",
            "sequence_state": "CURRENT",
        },
        "EXECUTION_PATH_HEALTH": {"path_state": "AVAILABLE"},
        "PERSISTENCE_HEALTH": {"integrity_state": "ACCEPTED", "recovery_required": False},
        "RECONCILIATION_HEALTH": {"reconciliation_state": "CURRENT"},
        "SECURITY_RISK_GATE_HEALTH": {
            "risk_allowed": True,
            "kill_switch_inactive": True,
            "lease_state": "VALID",
        },
        "RESOURCE_RUNTIME_HEALTH": {"usage_ratio": 0.5},
    }[category]


def _observation(category: str = "COMPONENT_STATUS", **changes: Any) -> dict[str, Any]:
    item = {
        "observation_id": "local.observation-1",
        "category": category,
        "source_component": "core_host",
        "source_instance_id": f"run_{UUID7}",
        "environment": None
        if category in {"COMPONENT_STATUS", "PERSISTENCE_HEALTH", "RESOURCE_RUNTIME_HEALTH"}
        else "TESTNET",
        "scope": _scope(category),
        "source_event_at_utc": "2030-01-01T00:00:00Z",
        "observed_at_utc": "2030-01-01T00:00:00Z",
        "ingested_at_utc": "2030-01-01T00:00:00Z",
        "expires_at_utc": "2030-01-01T00:00:30Z",
        "freshness_policy_id": f"{category}_TEST_V1",
        "source_sequence": 1,
        "condition": "OK",
        "reason_code": "PROBE_OK",
        "value": _value(category),
        "source_quality": "DIRECT",
        "correlation_reference": {"kind": "LOCAL", "value": "request.1"},
    }
    item.update(changes)
    return item


def _production_observations(category: str):
    raw_policy = _policy(category)
    policy = FreshnessPolicy(**raw_policy)
    return ObservationAuthority.compose(
        InMemoryObservationAuthorityCarrier(),
        policies=(policy,),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({"TESTNET"}),
    )


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_s9c_c3_production_is_differentially_checked_against_c1(category: str) -> None:
    cases = [
        ({}, None),
        ({"source_component": "tray_agent"}, "WRONG_SOURCE"),
        ({"source_instance_id": "bad"}, "WRONG_SOURCE_INSTANCE"),
        ({"environment": "OTHER"}, "ILLEGAL_ENVIRONMENT"),
        ({"scope": {}}, "WRONG_SCOPE"),
        ({"scope": {**_scope(category), next(iter(_scope(category))): "bad"}}, "WRONG_SCOPE"),
        ({"value": {}}, "INVALID_VALUE"),
        (
            {"value": {**_value(category), next(iter(_value(category))): "api_secret=bad"}},
            "SECRET_CONTENT",
        ),
        ({"correlation_reference": {"kind": "LOCAL", "value": "token.value"}}, "SECRET_CONTENT"),
        (
            {
                "correlation_reference": {
                    "kind": "CANONICAL",
                    "entity": "Unknown",
                    "value": f"run_{UUID7}",
                }
            },
            "INVALID_CORRELATION",
        ),
        (
            {
                "correlation_reference": {
                    "kind": "CANONICAL",
                    "entity": "RuntimeSession",
                    "value": f"xacc_{UUID7}",
                }
            },
            "INVALID_CORRELATION",
        ),
        ({"freshness_policy_id": "UNKNOWN"}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"expires_at_utc": "2030-01-01T00:00:29Z"}, "INVALID_EXPIRY"),
        (
            {"observed_at_utc": "2030-01-01T00:00:05Z", "expires_at_utc": "2030-01-01T00:00:35Z"},
            "FUTURE_TIMESTAMP",
        ),
        ({"source_event_at_utc": "2030-01-01T00:00:06Z"}, "FUTURE_SOURCE_EVENT"),
        ({"ingested_at_utc": "2029-12-31T23:59:55Z"}, "INGEST_SKEW"),
    ]
    for changes, expected_error in cases:
        item = _observation(category, **changes)
        oracle = ObservationReference(_all_contracts(), [_policy(category)])
        _, publisher = _production_observations(category)
        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")
            with pytest.raises(ValueError, match=expected_error):
                publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
        else:
            reference = oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")["acceptance"]
            production = publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
            assert production.effective_condition == reference["effective_condition"]
            assert production.effective_reason_codes == tuple(reference["effective_reason_codes"])


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_s9c_c3_sequence_and_projection_differential(category: str) -> None:
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    authority, publisher = _production_observations(category)
    first = _observation(category)
    reference = oracle.ingest(deepcopy(first), "2030-01-01T00:00:00Z")["acceptance"]
    accepted = publisher.publish(deepcopy(first), now_utc="2030-01-01T00:00:00Z")
    assert publisher.publish(deepcopy(first), now_utc="2030-01-01T00:00:01Z") == accepted
    assert oracle.ingest(deepcopy(first), "2030-01-01T00:00:01Z")["acceptance"]["replayed"]
    for mutation, error in [
        ({"reason_code": "CHANGED"}, "DUPLICATE_SEQUENCE_CONFLICT"),
        (
            {
                "source_sequence": 0,
                "observed_at_utc": "2030-01-01T00:00:01Z",
                "expires_at_utc": "2030-01-01T00:00:31Z",
            },
            "SEQUENCE_REGRESSION",
        ),
    ]:
        item = {**first, **mutation}
        with pytest.raises(ValueError, match=error):
            oracle.ingest(deepcopy(item), "2030-01-01T00:00:01Z")
        with pytest.raises(ValueError, match=error):
            publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:01Z")
    gap = {
        **first,
        "observation_id": "gap",
        "source_sequence": 3,
        "observed_at_utc": "2030-01-01T00:00:01Z",
        "expires_at_utc": "2030-01-01T00:00:31Z",
    }
    expected = oracle.ingest(deepcopy(gap), "2030-01-01T00:00:01Z")["acceptance"]
    actual = publisher.publish(deepcopy(gap), now_utc="2030-01-01T00:00:01Z")
    assert (actual.sequence_gap, actual.effective_condition, actual.effective_reason_codes) == (
        expected["sequence_gap"],
        expected["effective_condition"],
        tuple(expected["effective_reason_codes"]),
    )
    assert (
        oracle.projected(expected, "2030-01-01T00:00:31Z")["condition"]
        == authority.resolve_current(actual.key, now_utc="2030-01-01T00:00:31Z").projected_condition
        == "UNKNOWN"
    )

    # Optional-sequence successors use clock/source-event ordering and remain history.
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    authority, publisher = _production_observations(category)
    unsequenced = {**first, "source_sequence": None}
    oracle.ingest(deepcopy(unsequenced), "2030-01-01T00:00:00Z")
    original = publisher.publish(deepcopy(unsequenced), now_utc="2030-01-01T00:00:00Z")
    for mutation, error in [
        (
            {
                "observation_id": "clock",
                "observed_at_utc": "2029-12-31T23:59:59Z",
                "ingested_at_utc": "2029-12-31T23:59:59Z",
                "expires_at_utc": "2030-01-01T00:00:29Z",
            },
            "CLOCK_REGRESSION",
        ),
        (
            {
                "observation_id": "source",
                "observed_at_utc": "2030-01-01T00:00:01Z",
                "expires_at_utc": "2030-01-01T00:00:31Z",
                "source_event_at_utc": "2029-12-31T23:59:59Z",
            },
            "SOURCE_EVENT_REGRESSION",
        ),
    ]:
        item = {**unsequenced, **mutation}
        with pytest.raises(ValueError, match=error):
            oracle.ingest(deepcopy(item), "2030-01-01T00:00:01Z")
        with pytest.raises(ValueError, match=error):
            publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:01Z")
    successor = {
        **unsequenced,
        "observation_id": "successor",
        "observed_at_utc": "2030-01-01T00:00:01Z",
        "source_event_at_utc": "2030-01-01T00:00:01Z",
        "expires_at_utc": "2030-01-01T00:00:31Z",
    }
    oracle.ingest(deepcopy(successor), "2030-01-01T00:00:01Z")
    publisher.publish(deepcopy(successor), now_utc="2030-01-01T00:00:01Z")
    assert authority.resolve_historical_acceptance(original.acceptance_id) == original


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("observation_id", "A" * 128, True),
        ("observation_id", "A" * 129, False),
        ("observation_id", "A/B", False),
        ("local_correlation", "A" * 128, True),
        ("local_correlation", "A" * 129, False),
        ("local_correlation", "A/B", False),
        ("reason_code", "A" * 64, True),
        ("reason_code", "A" * 65, False),
        ("policy_id", "A" * 64, True),
        ("policy_id", "A" * 65, False),
    ],
)
def test_s9c_c4_lexical_boundaries_have_actual_c1_parity(
    category: str, field: str, value: str, accepted: bool
) -> None:
    item = _observation(category)
    raw_policy = _policy(category)
    if field == "local_correlation":
        item["correlation_reference"] = {"kind": "LOCAL", "value": value}
    elif field == "policy_id":
        item["freshness_policy_id"] = value
        raw_policy["policy_id"] = value
    else:
        item[field] = value
    outcomes = []
    try:
        oracle = ObservationReference(_all_contracts(), [raw_policy])
        oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")
        outcomes.append(True)
    except ValueError:
        outcomes.append(False)
    try:
        policy = FreshnessPolicy(**raw_policy)
        _, publisher = ObservationAuthority.compose(
            InMemoryObservationAuthorityCarrier(),
            policies=(policy,),
            environment_binding=FrozenEnvironmentRegistryBinding(),
            enabled_environments=frozenset({"TESTNET"}),
        )
        publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
        outcomes.append(True)
    except ValueError:
        outcomes.append(False)
    assert outcomes == [accepted, accepted]


@pytest.mark.parametrize(
    "entity,prefix",
    [
        ("RuntimeSession", "run"),
        ("ExchangeAccount", "xacc"),
        ("Instrument", "instr"),
        ("MarketDataRoute", "mdr"),
        ("ExecutionRoute", "xroute"),
    ],
)
def test_s9c_c4_every_canonical_correlation_entity_has_c1_parity(entity: str, prefix: str) -> None:
    for actual_prefix, accepted in ((prefix, True), ("dev", False)):
        item = _observation(
            "MARKET_DATA_FRESHNESS",
            correlation_reference={
                "kind": "CANONICAL",
                "entity": entity,
                "value": f"{actual_prefix}_{UUID7}",
            },
        )
        oracle = ObservationReference(_all_contracts(), [_policy("MARKET_DATA_FRESHNESS")])
        _, publisher = _production_observations("MARKET_DATA_FRESHNESS")
        outcomes = []
        for operation in (
            lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
            lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
        ):
            try:
                operation()
                outcomes.append(True)
            except ValueError:
                outcomes.append(False)
        assert outcomes == [accepted, accepted]


@pytest.mark.parametrize("environment", ["PAPER", "TESTNET", "LIVE"])
def test_s9c_c4_frozen_m04_environments_have_c1_parity(environment: str) -> None:
    item = _observation("MARKET_DATA_FRESHNESS", environment=environment)
    oracle = ObservationReference(_all_contracts(), [_policy("MARKET_DATA_FRESHNESS")])
    oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")
    _, publisher = ObservationAuthority.compose(
        InMemoryObservationAuthorityCarrier(),
        policies=(FreshnessPolicy(**_policy("MARKET_DATA_FRESHNESS")),),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({environment}),
    )
    assert (
        publisher.publish(item, now_utc="2030-01-01T00:00:00Z").observation.environment
        == environment
    )


def test_s9c_c4_unknown_environment_cannot_be_configured_as_authority() -> None:
    with pytest.raises(ValueError, match="ILLEGAL_ENVIRONMENT"):
        ObservationAuthority.compose(
            InMemoryObservationAuthorityCarrier(),
            policies=(FreshnessPolicy(**_policy("MARKET_DATA_FRESHNESS")),),
            environment_binding=FrozenEnvironmentRegistryBinding(),
            enabled_environments=frozenset({"OTHER"}),
        )


def test_s9c_c5_compile_time_environment_projection_matches_frozen_m04() -> None:
    frozen_m04 = _load("environment_and_product_capabilities.json")
    assert FrozenEnvironmentRegistryBinding().canonical_environments == frozenset(
        item["environment_id"] for item in frozen_m04["execution_environments"]
    )


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("original,changed", [(1, 1.0), (0, False), (1, True)])
def test_s9c_c8_exact_typed_replay_has_c1_production_parity(
    category: str, original: Any, changed: Any
) -> None:
    item = _observation(category)
    field = next(iter(item["value"]))
    item["value"][field] = original
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    _, publisher = _production_observations(category)
    oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")
    publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
    conflict = deepcopy(item)
    conflict["value"][field] = changed
    for operation in (
        lambda: oracle.ingest(deepcopy(conflict), "2030-01-01T00:00:01Z"),
        lambda: publisher.publish(deepcopy(conflict), now_utc="2030-01-01T00:00:01Z"),
    ):
        with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
            operation()


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_s9c_c8_identical_and_reordered_replay_has_c1_production_parity(category: str) -> None:
    item = _observation(category)
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    _, publisher = _production_observations(category)
    oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")
    accepted = publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
    reordered = dict(reversed(list(item.items())))
    reordered["scope"] = dict(reversed(list(item["scope"].items())))
    reordered["value"] = dict(reversed(list(item["value"].items())))
    assert oracle.ingest(reordered, "2031-01-01T00:00:00Z")["acceptance"]["replayed"]
    assert publisher.publish(reordered, now_utc="2031-01-01T00:00:00Z") == accepted


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at_utc": "malformed"},
        {"expires_at_utc": "2030-01-01T00:00:29Z"},
        {"source_event_at_utc": "malformed"},
        {"ingested_at_utc": "2029-12-31T23:59:00Z"},
        {"freshness_policy_id": "UNKNOWN"},
    ],
)
def test_s9c_c9_known_sequence_conflict_order_has_c1_production_parity(
    category: str, changes: dict[str, Any]
) -> None:
    item = _observation(category)
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    _, publisher = _production_observations(category)
    original = oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z")["acceptance"]
    accepted = publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z")
    conflict = {**item, **changes}
    for operation in (
        lambda: oracle.ingest(deepcopy(conflict), "malformed"),
        lambda: publisher.publish(deepcopy(conflict), now_utc="malformed"),
    ):
        with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
            operation()
    assert (
        oracle.ingest(deepcopy(item), "malformed")["acceptance"]["observation"]
        == original["observation"]
    )
    assert publisher.publish(deepcopy(item), now_utc="malformed") == accepted


@pytest.mark.parametrize(
    "field,error",
    [("source_component", "WRONG_SOURCE"), ("source_instance_id", "WRONG_SOURCE_INSTANCE")],
)
def test_s9c_c9_unhashable_source_field_error_has_c1_production_parity(
    field: str, error: str
) -> None:
    item = _observation("MARKET_DATA_FRESHNESS", **{field: []})
    oracle = ObservationReference(_all_contracts(), [_policy("MARKET_DATA_FRESHNESS")])
    _, publisher = _production_observations("MARKET_DATA_FRESHNESS")
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match=error):
            operation()


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "changes,error",
    [
        ({"category": []}, "UNKNOWN_CATEGORY"),
        ({"category": {}}, "UNKNOWN_CATEGORY"),
        ({"environment": []}, "ILLEGAL_ENVIRONMENT"),
        ({"environment": {}}, "ILLEGAL_ENVIRONMENT"),
        ({"freshness_policy_id": []}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": {}}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": 1}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": True}, "UNKNOWN_FRESHNESS_POLICY"),
        (
            {"correlation_reference": {"kind": "CANONICAL", "entity": [], "value": f"run_{UUID7}"}},
            "INVALID_CORRELATION",
        ),
        (
            {"correlation_reference": {"kind": "CANONICAL", "entity": {}, "value": f"run_{UUID7}"}},
            "INVALID_CORRELATION",
        ),
    ],
)
def test_s9c_c10_hash_lookup_primitive_errors_have_c1_production_parity(
    category: str, changes: dict[str, Any], error: str
) -> None:
    item = _observation(category)
    item.update(changes)
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    _, publisher = _production_observations(category)
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match=error):
            operation()


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_s9c_c11_nonfinite_scalar_rejection_has_c1_production_parity(
    category: str, nonfinite: float
) -> None:
    item = _observation(category)
    item["value"][next(iter(item["value"]))] = nonfinite
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    _, publisher = _production_observations(category)
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match="INVALID_VALUE"):
            operation()


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "timestamp",
    [
        "2030-01-01T00:00Z",
        "2030-01-01T00Z",
        "2030-01-01Z",
        "2030-W01-1T00:00:00Z",
        "20300101T000000Z",
        "2030-01-01T000000Z",
        "20300101T00:00:00Z",
    ],
)
def test_s9c_c12_strict_rfc3339_timestamp_has_c1_production_parity(
    category: str, timestamp: str
) -> None:
    item = _observation(category, observed_at_utc=timestamp)
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = ObservationAuthority.compose(
        carrier,
        policies=(FreshnessPolicy(**_policy(category)),),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({"TESTNET"}),
    )
    before = carrier.read()
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match="MALFORMED_TIMESTAMP"):
            operation()
    assert carrier.read() == before


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("digits", [4301, 5001])
def test_s9c_c13_oversized_integer_value_has_c1_production_parity(
    category: str, digits: int
) -> None:
    item = _observation(category)
    item["value"][next(iter(item["value"]))] = 10**digits
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = ObservationAuthority.compose(
        carrier,
        policies=(FreshnessPolicy(**_policy(category)),),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({"TESTNET"}),
    )
    before = carrier.read()
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match="INVALID_VALUE"):
            operation()
    assert carrier.read() == before


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "sequence",
    [10**4301, 10**5001, True, 1.0, -1],
    ids=["digits4302", "digits5002", "bool", "float", "negative"],
)
def test_s9c_c14_source_sequence_representability_has_c1_production_parity(
    category: str, sequence: Any
) -> None:
    item = _observation(category, source_sequence=sequence)
    oracle = ObservationReference(_all_contracts(), [_policy(category)])
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = ObservationAuthority.compose(
        carrier,
        policies=(FreshnessPolicy(**_policy(category)),),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({"TESTNET"}),
    )
    before = carrier.read()
    for operation in (
        lambda: oracle.ingest(deepcopy(item), "2030-01-01T00:00:00Z"),
        lambda: publisher.publish(deepcopy(item), now_utc="2030-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError, match="INVALID_SOURCE_SEQUENCE"):
            operation()
    assert carrier.read() == before


def _store(categories: list[str]) -> ObservationReference:
    ref = ObservationReference(_all_contracts(), [_policy(x) for x in categories])
    for index, category in enumerate(categories, 1):
        ref.ingest(
            _observation(
                category, observation_id=f"local.{category.lower()}", source_sequence=index
            ),
            "2030-01-01T00:00:01Z",
        )
    return ref


def _validate_domain(value: Any, domain: Any) -> None:
    if isinstance(domain, list):
        if value not in domain:
            raise ValueError("INVALID_DOMAIN")
    elif domain == "SAFE_CODE":
        if not isinstance(value, str) or not SAFE_CODE.fullmatch(value):
            raise ValueError("INVALID_DOMAIN")
    else:
        raise ContractInconsistent(f"unsupported field domain {domain}")


def _validate_optional_source(component: str, session: Any, contracts: dict[str, Any]) -> None:
    if session is None:
        return
    if component == "core_host":
        runtime = next(
            x for x in contracts["M0.2"]["entity_kinds"] if x["canonical_name"] == "RuntimeSession"
        )
        _canonical_id(session, runtime["id_prefix"])
    else:
        if not isinstance(session, str) or not SAFE_HANDLE.fullmatch(session):
            raise ValueError("INVALID_SOURCE_INSTANCE")
        _safe_scalar(session)


class StructuredLogReference:
    @staticmethod
    def validate(record: dict[str, Any], contracts: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["structured_log_schema"]
        _validate_closed(record, set(schema["required"]), set(schema["optional"]))
        _utc(record["timestamp_utc"])
        if record["level"] not in schema["levels"] or record["component"] not in {
            x["name"] for x in contracts["M0.3"]["process_roles"]
        }:
            raise ValueError("INVALID_LOG")
        event = schema["event_registry"].get(record["event_code"])
        if event is None:
            raise ValueError("INVALID_LOG")
        _validate_closed(
            record["safe_fields"],
            set(event["required_safe_fields"]),
            set(event["optional_safe_fields"]),
        )
        for field, domain in event["field_domains"].items():
            if field in record["safe_fields"]:
                _validate_domain(record["safe_fields"][field], domain)
        for value in record["safe_fields"].values():
            _safe_scalar(value)
        _validate_optional_source(record["component"], record.get("source_instance_id"), contracts)
        _correlation(record.get("correlation_reference"))
        if record.get("environment") is not None and record["environment"] not in {
            x["environment_id"] for x in contracts["M0.4"]["execution_environments"]
        }:
            raise ValueError("INVALID_LOG")
        if "narrative_message" in record:
            _safe_scalar(record["narrative_message"])


class MetricReference:
    @staticmethod
    def validate(sample: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["metric_schema"]
        _validate_closed(sample, set(schema["required"]), set())
        definition = schema["registry"].get(sample["metric_name"])
        if definition is None or sample["unit"] != definition["unit"]:
            raise ValueError("INVALID_METRIC")
        if isinstance(sample["value"], bool) or not isinstance(sample["value"], (int, float)):
            raise ValueError("INVALID_METRIC")
        if not isinstance(sample["labels"], dict) or set(sample["labels"]) != set(
            definition["label_schema"]
        ):
            raise ValueError("INVALID_METRIC")
        for label, value in sample["labels"].items():
            if value not in definition["label_schema"][label]:
                raise ValueError("INVALID_METRIC")
            _safe_scalar(value)
        _utc(sample["timestamp_utc"])
        scope_schema = MACHINE["observability_model"]["category_scope_registry"][
            definition["scope_schema_ref"]
        ]
        if (
            not isinstance(sample["scope"], dict)
            or not set(scope_schema["required_keys"]) <= set(sample["scope"])
            or set(sample["scope"])
            - set(scope_schema["required_keys"])
            - set(scope_schema["optional_keys"])
            or sample["scope"].get("component")
            not in MACHINE["observability_model"]["source_component_registry"]["values"]
            or sample["labels"].get("component") != sample["scope"].get("component")
        ):
            raise ValueError("INVALID_METRIC")


class TraceReference:
    @staticmethod
    def validate(span: dict[str, Any], contracts: dict[str, Any]) -> None:
        schema = MACHINE["observability_model"]["trace_schema"]
        _validate_closed(span, set(schema["required"]), set(schema["optional"]))
        if not HEX_ID.fullmatch(span["trace_id"]) or not HEX_ID.fullmatch(span["span_id"]):
            raise ValueError("INVALID_TRACE")
        if span.get("parent_span_id") is not None and (
            not HEX_ID.fullmatch(span["parent_span_id"])
            or span["parent_span_id"] == span["span_id"]
        ):
            raise ValueError("INVALID_TRACE")
        if span["component"] not in {x["name"] for x in contracts["M0.3"]["process_roles"]}:
            raise ValueError("INVALID_TRACE")
        operation = schema["operation_registry"].get(span["operation_code"])
        if operation is None:
            raise ValueError("INVALID_TRACE")
        if _utc(span["ended_at_utc"]) < _utc(span["started_at_utc"]):
            raise ValueError("INVALID_TRACE")
        _validate_closed(
            span["safe_attributes"],
            set(operation["required_attributes"]),
            set(operation["optional_attributes"]),
        )
        for field, domain in operation["attribute_domains"].items():
            if field in span["safe_attributes"]:
                _validate_domain(span["safe_attributes"][field], domain)
        for value in span["safe_attributes"].values():
            _safe_scalar(value)
        environments = {x["environment_id"] for x in contracts["M0.4"]["execution_environments"]}
        if span.get("environment") is not None and span["environment"] not in environments:
            raise ValueError("INVALID_TRACE")
        _validate_optional_source(span["component"], span.get("source_instance_id"), contracts)
        _correlation(span.get("correlation_reference"))
        _correlation(span.get("causation_reference"))


def test_source_first_identities_and_category_scope_registry() -> None:
    entities = {x["canonical_name"]: x for x in _all_contracts()["M0.2"]["entity_kinds"]}
    assert (entities["ExchangeAccount"]["id_field"], entities["ExchangeAccount"]["id_prefix"]) == (
        "exchange_account_id",
        "xacc",
    )
    assert (entities["Instrument"]["id_field"], entities["Instrument"]["id_prefix"]) == (
        "instrument_id",
        "instr",
    )
    assert (entities["MarketDataRoute"]["id_field"], entities["MarketDataRoute"]["id_prefix"]) == (
        "market_data_route_id",
        "mdr",
    )
    assert (entities["ExecutionRoute"]["id_field"], entities["ExecutionRoute"]["id_prefix"]) == (
        "execution_route_id",
        "xroute",
    )
    assert (entities["RuntimeSession"]["id_field"], entities["RuntimeSession"]["id_prefix"]) == (
        "runtime_session_id",
        "run",
    )
    registry = MACHINE["observability_model"]["category_scope_registry"]
    assert set(registry) == set(MACHINE["observability_model"]["categories"])
    assert registry["MARKET_DATA_FRESHNESS"]["required_keys"] == [
        "market_data_route_id",
        "instrument_id",
    ]
    assert registry["PERSISTENCE_HEALTH"]["required_keys"] == [
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
    ]


def test_field_complete_observation_validation_and_category_source_policy() -> None:
    valid = _store(["COMPONENT_STATUS"])
    assert valid.current
    changes = [
        ({"observation_id": ""}, "INVALID_OBSERVATION_ID"),
        ({"observation_id": "bad\nhandle"}, "INVALID_OBSERVATION_ID"),
        ({"source_sequence": True}, "INVALID_SOURCE_SEQUENCE"),
        ({"source_sequence": -1}, "INVALID_SOURCE_SEQUENCE"),
        ({"source_quality": "SUPER_TRUSTED"}, "UNKNOWN_SOURCE_QUALITY"),
        ({"reason_code": "narrative reason"}, "INVALID_REASON_CODE"),
        ({"correlation_reference": "token raw"}, "INVALID_CORRELATION"),
        ({"value": {"unexpected": 1}}, "INVALID_VALUE"),
        ({"scope": {"component": "core_host", "extra": "x"}}, "WRONG_SCOPE"),
        ({"source_instance_id": "local-session"}, "WRONG_SOURCE_INSTANCE"),
    ]
    for change, error in changes:
        with pytest.raises(ValueError, match=error):
            ObservationReference(_all_contracts(), [_policy("COMPONENT_STATUS")]).ingest(
                _observation(**change), "2030-01-01T00:00:01Z"
            )
    with pytest.raises(ValueError, match="WRONG_SOURCE"):
        ObservationReference(_all_contracts(), [_policy("SECURITY_RISK_GATE_HEALTH")]).ingest(
            _observation(
                "SECURITY_RISK_GATE_HEALTH",
                source_component="bootstrapper",
                source_instance_id="boot.local",
            ),
            "2030-01-01T00:00:01Z",
        )
    with pytest.raises(ValueError, match="WRONG_SCOPE"):
        ObservationReference(_all_contracts(), [_policy("MARKET_DATA_FRESHNESS")]).ingest(
            _observation("MARKET_DATA_FRESHNESS", scope={"component": "core_host"}),
            "2030-01-01T00:00:01Z",
        )


def test_source_instance_policy_required_optional_and_secret_safe() -> None:
    contracts = _all_contracts()
    required = ObservationReference(contracts, [_policy("COMPONENT_STATUS")])
    with pytest.raises(ValueError, match="WRONG_SOURCE_INSTANCE"):
        required.ingest(
            _observation(
                source_component="tray_agent",
                source_instance_id=None,
                scope={"component": "tray_agent"},
            ),
            "2030-01-01T00:00:01Z",
        )
    with pytest.raises(ValueError, match="SECRET_CONTENT"):
        required.ingest(
            _observation(
                source_component="tray_agent",
                source_instance_id="token.raw",
                scope={"component": "tray_agent"},
            ),
            "2030-01-01T00:00:01Z",
        )
    optional = ObservationReference(
        contracts, [{**_policy("STRUCTURED_LOGS"), "source_class": "tray_agent"}]
    )
    optional.ingest(
        _observation(
            "STRUCTURED_LOGS",
            source_component="tray_agent",
            source_instance_id=None,
            scope={},
            value={},
            freshness_policy_id="STRUCTURED_LOGS_TEST_V1",
        ),
        "2030-01-01T00:00:01Z",
    )
    with pytest.raises(ValueError):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")]).ingest(
            _observation(source_instance_id="run_bad"), "2030-01-01T00:00:01Z"
        )


def test_injected_freshness_skew_expiry_and_derived_state() -> None:
    policy = _policy("COMPONENT_STATUS", skew=4)
    for observed, source, ingested in [
        ("2030-01-01T00:00:03Z", "2030-01-01T00:00:04Z", "2029-12-31T23:59:59Z")
    ]:
        ref = ObservationReference(_all_contracts(), [policy])
        item = _observation(
            observed_at_utc=observed,
            source_event_at_utc=source,
            ingested_at_utc=ingested,
            expires_at_utc="2030-01-01T00:00:33Z",
        )
        ref.ingest(item, "2030-01-01T00:00:01Z")
    for change, error in [
        (
            {"observed_at_utc": "2030-01-01T00:00:06Z", "expires_at_utc": "2030-01-01T00:00:36Z"},
            "FUTURE_TIMESTAMP",
        ),
        ({"source_event_at_utc": "2030-01-01T00:00:07Z"}, "FUTURE_SOURCE_EVENT"),
        ({"ingested_at_utc": "2029-12-31T23:59:55Z"}, "INGEST_SKEW"),
    ]:
        with pytest.raises(ValueError, match=error):
            ObservationReference(_all_contracts(), [policy]).ingest(
                _observation(**change), "2030-01-01T00:00:01Z"
            )
    ref = _store(["COMPONENT_STATUS"])
    current = next(iter(ref.current.values()))
    assert ref.projected(current, "2030-01-01T00:00:30Z") == {
        "freshness": "STALE",
        "condition": "UNKNOWN",
        "reason_codes": ["OBSERVATION_EXPIRED"],
    }


def test_sequence_replay_conflict_regression_gap_and_current_semantics() -> None:
    ref = ObservationReference(_all_contracts(), [_policy("COMPONENT_STATUS")])
    first = _observation()
    accepted = ref.ingest(first, "2030-01-01T00:00:01Z")
    replay = ref.ingest(deepcopy(first), "2030-01-01T00:00:01Z")
    assert replay["acceptance"]["replayed"] is True and replay["acceptance"]["observation"] == first
    with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
        ref.ingest(_observation(condition="DEGRADED"), "2030-01-01T00:00:01Z")
    gap = ref.ingest(
        _observation(observation_id="local.gap", source_sequence=3), "2030-01-01T00:00:01Z"
    )
    assert (
        gap["acceptance"]["sequence_gap"]
        and gap["acceptance"]["effective_condition"] == "DEGRADED"
        and "SEQUENCE_GAP" in gap["acceptance"]["effective_reason_codes"]
    )
    with pytest.raises(ValueError, match="SEQUENCE_REGRESSION"):
        ref.ingest(_observation(source_sequence=2), "2030-01-01T00:00:01Z")
    old = ref.ingest(deepcopy(first), "2030-01-01T00:00:01Z")
    assert (
        old["acceptance"]["observation"] == first
        and old["current"]["observation"]["source_sequence"] == 3
    )


def _fabricated_m04_wrapper(hex_digit: str = "1") -> dict[str, Any]:
    values = iter("123456")
    return {
        "carrier_kind": "CORE_ACCEPTED_M04_VALIDATED_SNAPSHOT_REFERENCE",
        "evidence_schema_version": "cryptohunter.stage_evidence.v1",
        "stage_id": "VALIDATED_SNAPSHOT_CREATION",
        "stage_result": "SNAPSHOT_CREATED",
        "stage_evidence_id": next(values) * 64,
        "predecessor_stage_evidence_id": next(values) * 64,
        "validation_context_id": next(values) * 64,
        "document_fingerprint": next(values) * 64,
        "capabilities_id": "capabilities-current",
        "edition_id": "CRYPTOHUNTER_TESTNET_EDITION",
        "signed_payload_hash": next(values) * 64,
        "capability_set_hash": next(values) * 64,
        "payload_schema_version": "cryptohunter.product_capabilities.payload.v1",
        "signature_schema_version": "cryptohunter.product_capabilities.signature.v1",
        "complete_stage_evidence": True,
        "same_validation_context_id": True,
        "same_document_fingerprint": True,
        "issuer_attestation_verified": True,
    }


def _evidence(
    environment: str, scope: dict[str, Any], **overrides: Any
) -> UpstreamReadinessEvidence:
    runtime = {
        "runtime_session_id": f"run_{UUID7}",
        "device_installation_id": IDS["device_installation_id"],
        "state_store_identity_fingerprint_sha256": "a" * 64,
        "startup_readiness_state": "READY",
        "process_lock_owned": True,
    }
    execution_key = ReadinessReference.execution_scope_key(environment, scope)
    records = {
        "runtime_context": runtime,
        "product_capabilities_evidence": None,
        "accounts_by_id": {
            scope.get("exchange_account_id"): {
                "exchange_account_id": scope.get("exchange_account_id"),
                "workspace_id": IDS["workspace_id"],
                "exchange_id": {
                    "PAPER": "paper_simulated_venue",
                    "TESTNET": "generic_testnet_venue",
                    "LIVE": "fabricated_live_venue",
                }[environment],
                "environment": environment,
                "market_type": "SPOT",
                "lifecycle_state": "ACTIVE",
                "connection_state": "ONLINE",
                "execution_authorization": "ORDER_ENTRY_ALLOWED",
            }
        }
        if scope.get("exchange_account_id")
        else {},
        "instruments_by_id": {
            scope.get("instrument_id"): {
                "instrument_id": scope.get("instrument_id"),
                "workspace_id": IDS["workspace_id"],
                "exchange_id": {
                    "PAPER": "paper_simulated_venue",
                    "TESTNET": "generic_testnet_venue",
                    "LIVE": "fabricated_live_venue",
                }[environment],
                "environment": environment,
                "market_type": "SPOT",
                "instrument_type": "SPOT_PAIR",
                "source_adapter_family_id": {
                    "PAPER": "paper_simulation_adapter_family",
                    "TESTNET": "generic_testnet_adapter_family",
                    "LIVE": "fabricated_live_adapter_family",
                }[environment],
                "trading_status": "TRADING",
            }
        }
        if scope.get("instrument_id")
        else {},
        "market_data_routes_by_id": {
            scope.get("market_data_route_id"): {
                "market_data_route_id": scope.get("market_data_route_id"),
                "route_kind": "MARKET_DATA",
                "workspace_id": IDS["workspace_id"],
                "exchange_id": {
                    "PAPER": "paper_simulated_venue",
                    "TESTNET": "generic_testnet_venue",
                    "LIVE": "fabricated_live_venue",
                }[environment],
                "environment": environment,
                "market_type": "SPOT",
                "adapter_family_id": {
                    "PAPER": "paper_simulation_adapter_family",
                    "TESTNET": "generic_testnet_adapter_family",
                    "LIVE": "fabricated_live_adapter_family",
                }[environment],
                "endpoint_class": {
                    "PAPER": "PAPER_PUBLIC_DATA",
                    "TESTNET": "TESTNET_PUBLIC_DATA",
                    "LIVE": "LIVE_PUBLIC_DATA",
                }[environment],
                "data_scope": "PUBLIC",
                "instrument_ids": [scope.get("instrument_id")],
                "channel_types": ["TRADES"],
                "snapshot_stream_semantics": "SNAPSHOT_THEN_STREAM",
                "sequence_policy": "MONOTONIC_NO_GAPS",
                "freshness_policy": {"max_age_seconds": 30},
                "reconnect_policy": "RESNAPSHOT",
                "route_status": "ACTIVE",
                "instrument_id": scope.get("instrument_id"),
                "route_readiness": {
                    "route_id": scope.get("market_data_route_id"),
                    "route_kind": "MARKET_DATA",
                    "readiness_state": "READY",
                    "observed_at": "2030-01-01T00:00:00Z",
                    "metadata_version": 1,
                    "sequence_state": "CONTIGUOUS",
                },
            }
        }
        if scope.get("market_data_route_id")
        else {},
        "execution_routes_by_id": {
            scope.get("execution_route_id"): {
                "execution_route_id": scope.get("execution_route_id"),
                "route_kind": "EXECUTION",
                "workspace_id": IDS["workspace_id"],
                "environment": environment,
                "exchange_id": {
                    "PAPER": "paper_simulated_venue",
                    "TESTNET": "generic_testnet_venue",
                    "LIVE": "fabricated_live_venue",
                }[environment],
                "market_type": "SPOT",
                "adapter_family_id": {
                    "PAPER": "paper_simulation_adapter_family",
                    "TESTNET": "generic_testnet_adapter_family",
                    "LIVE": "fabricated_live_adapter_family",
                }[environment],
                "endpoint_class": {
                    "PAPER": "PAPER_SIMULATION",
                    "TESTNET": "TESTNET_PRIVATE_DATA",
                    "LIVE": "LIVE_PRIVATE_DATA",
                }[environment],
                "supported_instrument_types": ["SPOT_PAIR"],
                "route_status": "ACTIVE",
                "route_capability_ceiling": ["PLACE_ORDERS"],
                "authorization_dependencies": list(
                    _load("strategy_market_data_and_execution_routing.json")[
                        "authorization_dependencies_by_environment"
                    ].get(environment, ["PRODUCT_CAPABILITIES"])
                ),
                "instrument_id": scope.get("instrument_id"),
                "exchange_account_id": scope.get("exchange_account_id"),
                "route_readiness": {
                    "route_id": scope.get("execution_route_id"),
                    "route_kind": "EXECUTION",
                    "readiness_state": "READY",
                    "observed_at": "2030-01-01T00:00:00Z",
                    "metadata_version": 1,
                    "sequence_state": "NOT_APPLICABLE",
                },
            }
        }
        if scope.get("execution_route_id")
        else {},
        "risk_by_scope": (
            {
                execution_key: {
                    "environment": environment,
                    "exchange_account_id": scope["exchange_account_id"],
                    "instrument_id": scope["instrument_id"],
                    "execution_route_id": scope["execution_route_id"],
                    "risk_allowed": True,
                    "kill_switch_inactive": True,
                    "execution_lease_valid": True,
                }
            }
            if all(
                scope.get(field)
                for field in ("exchange_account_id", "instrument_id", "execution_route_id")
            )
            else {}
        ),
        "persistence_context": {
            "device_installation_id": IDS["device_installation_id"],
            "state_store_identity_fingerprint_sha256": "a" * 64,
            "accepted": True,
            "recovery_attempt_allowed": True,
        },
        "reconciliation_by_scope": (
            {
                f"{environment}:{scope['exchange_account_id']}:{scope['portfolio_id']}": {
                    "environment": environment,
                    "exchange_account_id": scope["exchange_account_id"],
                    "portfolio_id": scope["portfolio_id"],
                    "attempt_allowed": True,
                }
            }
            if scope.get("portfolio_id")
            else {}
        ),
    }
    for path, value in overrides.items():
        section, field = path.split("__", 1)
        records[section][field] = value
    return UpstreamReadinessEvidence(records)


def _readiness(
    readiness_id: str, environment: str | None, scope: dict[str, Any], categories: list[str]
) -> tuple[ObservationReference, ReadinessReference, UpstreamReadinessEvidence]:
    evidence = _evidence(environment or "PAPER", scope)
    rebuilt = ObservationReference(_all_contracts(), [_policy(x) for x in categories])
    row = next(
        row
        for row in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        if row["readiness_id"] == readiness_id
    )
    for index, dependency in enumerate(row["required_observations"], 1):
        category = dependency["category"]
        binding = dependency["scope_binding"]
        schema = MACHINE["observability_model"]["category_scope_registry"][category]
        if binding["mode"] == "FIXED_SCOPE":
            category_scope = binding["fixed_scope"]
        elif binding["mode"] == "CURRENT_INSTALLATION_STATESTORE":
            category_scope = {field: evidence.runtime[field] for field in binding["context_fields"]}
        elif binding["mode"] == "MATCH_REQUEST_SCOPE_FIELDS":
            category_scope = {field: scope[field] for field in binding["fields"]}
        else:
            category_scope = {field: evidence.runtime[field] for field in binding["context_fields"]}
        rebuilt.ingest(
            _observation(
                category,
                environment=None
                if schema["environment_requirement"] == "FORBIDDEN"
                else environment,
                scope=category_scope,
                observation_id=f"local.ready{index}",
                source_sequence=index,
            ),
            "2030-01-01T00:00:01Z",
        )
    return rebuilt, ReadinessReference(rebuilt), evidence


def test_monotonic_sequence_gap_conditions_and_paper_fail_open_regression() -> None:
    for raw, expected in {
        "OK": "DEGRADED",
        "DEGRADED": "DEGRADED",
        "UNKNOWN": "UNKNOWN",
        "BLOCKED": "BLOCKED",
    }.items():
        ref = ObservationReference(_all_contracts(), [_policy("COMPONENT_STATUS")])
        ref.ingest(_observation(), "2030-01-01T00:00:01Z")
        result = ref.ingest(
            _observation(observation_id=f"local.{raw.lower()}", source_sequence=3, condition=raw),
            "2030-01-01T00:00:01Z",
        )["acceptance"]
        assert (
            result["sequence_gap"]
            and result["effective_condition"] == expected
            and "SEQUENCE_GAP" in result["effective_reason_codes"]
        )
    scope = {k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")}
    ref, oracle, evidence = _readiness(
        "PAPER_OPERATION", "PAPER", scope, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    component = next(value for key, value in ref.current.items() if key[0] == "COMPONENT_STATUS")
    first = component["observation"]
    first["source_sequence"] = 1
    ref.last_sequence[(first["source_component"], first["source_instance_id"])] = 1
    ref.ingest(
        _observation(
            "COMPONENT_STATUS",
            observation_id="local.blocked",
            environment="PAPER",
            scope={"component": "core_host"},
            source_sequence=3,
            condition="BLOCKED",
        ),
        "2030-01-01T00:00:01Z",
    )
    assert (
        oracle.evaluate("PAPER_OPERATION", "PAPER", scope, "2030-01-01T00:00:01Z", evidence)
        == "BLOCKED"
    )


def test_matrix_authority_is_structured_scope_bound_and_not_caller_mintable() -> None:
    paper = {k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")}
    execution = {
        k: IDS[k]
        for k in (
            "exchange_account_id",
            "instrument_id",
            "execution_route_id",
            "market_data_route_id",
            "portfolio_id",
        )
    }
    _, paper_oracle, paper_evidence = _readiness(
        "PAPER_OPERATION", "PAPER", paper, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    assert (
        paper_oracle.evaluate(
            "PAPER_OPERATION", "PAPER", paper, "2030-01-01T00:00:01Z", paper_evidence
        )
        == "OK"
    )
    with pytest.raises(TypeError):
        paper_oracle.evaluate(
            "PAPER_OPERATION", "PAPER", paper, "2030-01-01T00:00:01Z", {"all": True}
        )
    categories = [
        "ADAPTER_STATUS",
        "MARKET_DATA_FRESHNESS",
        "EXECUTION_PATH_HEALTH",
        "SECURITY_RISK_GATE_HEALTH",
        "PERSISTENCE_HEALTH",
        "RECONCILIATION_HEALTH",
    ]
    _, oracle, evidence = _readiness("TESTNET_PRIVATE_EXECUTION", "TESTNET", execution, categories)
    assert (
        oracle.evaluate(
            "TESTNET_PRIVATE_EXECUTION", "TESTNET", execution, "2030-01-01T00:00:01Z", evidence
        )
        == "BLOCKED"
    )
    wrong_account = {
        f"xacc_00000000-0000-7000-8000-000000000002": next(
            iter(evidence.records["accounts_by_id"].values())
        )
    }
    wrong_route = deepcopy(evidence.records["execution_routes_by_id"])
    wrong_route[IDS["execution_route_id"]]["instrument_id"] = (
        f"instr_00000000-0000-7000-8000-000000000002"
    )
    wrong_lease = {
        "TESTNET:other-account:other-instrument:other-route": next(
            iter(evidence.records["risk_by_scope"].values())
        )
    }
    mutations = [
        ("accounts_by_id", wrong_account),
        ("execution_routes_by_id", wrong_route),
        ("risk_by_scope", wrong_lease),
        (
            "persistence_context",
            {
                **evidence.records["persistence_context"],
                "device_installation_id": f"dev_00000000-0000-7000-8000-000000000002",
            },
        ),
    ]
    for section, value in mutations:
        changed = deepcopy(evidence.records)
        changed[section] = value
        try:
            candidate = UpstreamReadinessEvidence(changed)
        except ValueError:
            continue
        assert (
            oracle.evaluate(
                "TESTNET_PRIVATE_EXECUTION", "TESTNET", execution, "2030-01-01T00:00:01Z", candidate
            )
            == "BLOCKED"
        )
    paper_denied = deepcopy(paper_evidence.records)
    paper_denied["accounts_by_id"][IDS["exchange_account_id"]]["lifecycle_state"] = "DISABLED"
    assert (
        paper_oracle.evaluate(
            "PAPER_OPERATION",
            "PAPER",
            paper,
            "2030-01-01T00:00:01Z",
            UpstreamReadinessEvidence(paper_denied),
        )
        == "BLOCKED"
    )


def test_exact_dependency_scope_fixed_core_and_current_persistence() -> None:
    scope = {k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")}
    ref, oracle, evidence = _readiness(
        "PAPER_OPERATION", "PAPER", scope, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    component_key = next(key for key in ref.current if key[0] == "COMPONENT_STATUS")
    component = ref.current.pop(component_key)
    component["observation"]["scope"] = {"component": "tray_agent"}
    ref.current[ObservationReference.key(component["observation"])] = component
    assert (
        oracle.evaluate("PAPER_OPERATION", "PAPER", scope, "2030-01-01T00:00:01Z", evidence)
        == "BLOCKED"
    )
    ref, oracle, evidence = _readiness(
        "PAPER_OPERATION", "PAPER", scope, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    pkey = next(key for key in ref.current if key[0] == "PERSISTENCE_HEALTH")
    persistence = ref.current.pop(pkey)
    persistence["observation"]["scope"]["device_installation_id"] = (
        f"dev_00000000-0000-7000-8000-000000000002"
    )
    ref.current[ObservationReference.key(persistence["observation"])] = persistence
    assert (
        oracle.evaluate("PAPER_OPERATION", "PAPER", scope, "2030-01-01T00:00:01Z", evidence)
        == "BLOCKED"
    )


def test_clock_regression_source_event_ordering_and_new_session_reset() -> None:
    policy = _policy("COMPONENT_STATUS")
    ref = ObservationReference(_all_contracts(), [policy])
    ref.ingest(
        _observation(
            observed_at_utc="2030-01-01T00:00:10Z",
            ingested_at_utc="2030-01-01T00:00:10Z",
            expires_at_utc="2030-01-01T00:00:40Z",
        ),
        "2030-01-01T00:00:10Z",
    )
    with pytest.raises(ValueError, match="CLOCK_REGRESSION"):
        ref.ingest(
            _observation(
                observation_id="local.clock",
                source_sequence=2,
                observed_at_utc="2030-01-01T00:00:09Z",
                ingested_at_utc="2030-01-01T00:00:09Z",
                expires_at_utc="2030-01-01T00:00:39Z",
            ),
            "2030-01-01T00:00:10Z",
        )
    ref.ingest(
        _observation(
            observation_id="local.restart",
            source_instance_id="run_00000000-0000-7000-8000-000000000002",
            source_sequence=0,
            observed_at_utc="2030-01-01T00:00:09Z",
            ingested_at_utc="2030-01-01T00:00:09Z",
            expires_at_utc="2030-01-01T00:00:39Z",
        ),
        "2030-01-01T00:00:10Z",
    )
    noseq = ObservationReference(_all_contracts(), [policy])
    noseq.ingest(_observation(source_sequence=None), "2030-01-01T00:00:01Z")
    with pytest.raises(ValueError, match="SOURCE_EVENT_REGRESSION"):
        noseq.ingest(
            _observation(
                observation_id="local.source",
                source_sequence=None,
                observed_at_utc="2030-01-01T00:00:01Z",
                ingested_at_utc="2030-01-01T00:00:01Z",
                source_event_at_utc="2029-12-31T23:59:59Z",
                expires_at_utc="2030-01-01T00:00:31Z",
            ),
            "2030-01-01T00:00:01Z",
        )


def test_health_aggregator_reads_fresh_current_projection() -> None:
    ref = _store(["COMPONENT_STATUS"])
    key = next(iter(ref.current))
    health = HealthAggregatorReference(ref)
    assert health.compose([key], "2030-01-01T00:00:01Z") == "OK"
    assert health.compose([key], "2030-01-01T00:00:30Z") == "UNKNOWN"


def _log() -> dict[str, Any]:
    return {
        "timestamp_utc": "2030-01-01T00:00:00Z",
        "level": "INFO",
        "component": "core_host",
        "event_code": "COMPONENT_PROBE",
        "safe_fields": {"condition": "OK", "reason_code": "PROBE_OK"},
        "correlation_reference": {"kind": "LOCAL", "value": "request.1"},
    }


def test_structured_log_validator_and_adversarial_inputs() -> None:
    StructuredLogReference.validate(_log(), _all_contracts())
    for mutate in [
        lambda x: x.update(extra=1),
        lambda x: x.update(level="NOTICE"),
        lambda x: x.update(event_code="UNKNOWN"),
        lambda x: x["safe_fields"].update(extra="x"),
        lambda x: x["safe_fields"].update(reason_code="some narrative reason"),
        lambda x: x.update(source_instance_id="run_bad"),
        lambda x: x.update(environment="PROD"),
        lambda x: x.update(correlation_reference={"kind": "LOCAL", "value": "token.raw"}),
    ]:
        record = _log()
        mutate(record)
        with pytest.raises(ValueError):
            StructuredLogReference.validate(record, _all_contracts())


def _metric() -> dict[str, Any]:
    return {
        "metric_name": "component_probe_total",
        "value": 1,
        "unit": "count",
        "labels": {"component": "core_host", "condition": "OK"},
        "scope": {"component": "core_host"},
        "timestamp_utc": "2030-01-01T00:00:00Z",
    }


def test_metric_validator_and_cardinality_adversarial_inputs() -> None:
    MetricReference.validate(_metric())
    for mutate in [
        lambda x: x.update(metric_name="unknown"),
        lambda x: x.update(unit="bytes"),
        lambda x: x.update(value=True),
        lambda x: x["labels"].update(user="alice"),
        lambda x: x["labels"].update(component=IDS["exchange_account_id"]),
        lambda x: x["labels"].update(component="api_secret"),
    ]:
        sample = _metric()
        mutate(sample)
        with pytest.raises(ValueError):
            MetricReference.validate(sample)


def test_metric_scope_label_consistency_and_valid_non_core_scope() -> None:
    mismatch = _metric()
    mismatch["labels"]["component"] = "tray_agent"
    with pytest.raises(ValueError):
        MetricReference.validate(mismatch)
    non_core = _metric()
    non_core["labels"]["component"] = "tray_agent"
    non_core["scope"]["component"] = "tray_agent"
    MetricReference.validate(non_core)


def _trace() -> dict[str, Any]:
    return {
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "component": "core_host",
        "operation_code": "OBSERVATION_INGEST",
        "started_at_utc": "2030-01-01T00:00:00Z",
        "ended_at_utc": "2030-01-01T00:00:01Z",
        "safe_attributes": {"category": "COMPONENT_STATUS", "result_code": "ACCEPTED"},
        "correlation_reference": {"kind": "LOCAL", "value": "request.1"},
    }


def test_trace_validator_and_adversarial_inputs() -> None:
    TraceReference.validate(_trace(), _all_contracts())
    for mutate in [
        lambda x: x.update(ended_at_utc="2029-12-31T23:59:59Z"),
        lambda x: x.update(operation_code="AUTHORIZE_ORDER"),
        lambda x: x["safe_attributes"].update(extra="x"),
        lambda x: x["safe_attributes"].update(result_code="some narrative result"),
        lambda x: x.update(environment="PROD"),
        lambda x: x.update(source_instance_id="run_bad"),
        lambda x: x.update(correlation_reference="bad"),
    ]:
        span = _trace()
        mutate(span)
        with pytest.raises(ValueError):
            TraceReference.validate(span, _all_contracts())


def test_nested_upstream_evidence_is_closed_canonical_and_document_bound() -> None:
    scope = {
        k: IDS[k]
        for k in (
            "exchange_account_id",
            "instrument_id",
            "execution_route_id",
            "market_data_route_id",
            "portfolio_id",
        )
    }
    valid = _evidence("TESTNET", scope)
    assert isinstance(valid, UpstreamReadinessEvidence)
    mutations = []
    for section in valid.REQUIRED:
        changed = deepcopy(valid.records)
        target = changed[section]
        if section in {
            "accounts_by_id",
            "instruments_by_id",
            "market_data_routes_by_id",
            "execution_routes_by_id",
            "risk_by_scope",
            "reconciliation_by_scope",
        }:
            if target:
                next(iter(target.values()))["unexpected"] = "x"
            else:
                continue
        elif target is not None:
            target["unexpected"] = "x"
        else:
            continue
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(ValueError):
            UpstreamReadinessEvidence(changed)
    for field, value in (
        ("runtime_session_id", "run_forged"),
        ("device_installation_id", "dev_forged"),
        ("state_store_identity_fingerprint_sha256", "ABC"),
        ("process_lock_owned", 1),
    ):
        changed = deepcopy(valid.records)
        changed["runtime_context"][field] = value
        with pytest.raises(ValueError):
            UpstreamReadinessEvidence(changed)
    fabricated_records = deepcopy(valid.records)
    fabricated_records["product_capabilities_evidence"] = _fabricated_m04_wrapper()
    fabricated = UpstreamReadinessEvidence(fabricated_records)
    assert fabricated.records["product_capabilities_evidence"] is not None
    categories = [
        "ADAPTER_STATUS",
        "MARKET_DATA_FRESHNESS",
        "EXECUTION_PATH_HEALTH",
        "SECURITY_RISK_GATE_HEALTH",
        "PERSISTENCE_HEALTH",
        "RECONCILIATION_HEALTH",
    ]
    _, testnet_oracle, _ = _readiness("TESTNET_PRIVATE_EXECUTION", "TESTNET", scope, categories)
    assert (
        testnet_oracle.evaluate(
            "TESTNET_PRIVATE_EXECUTION", "TESTNET", scope, "2030-01-01T00:00:01Z", fabricated
        )
        == "BLOCKED"
    )
    proof_fields = (
        "stage_evidence_id",
        "predecessor_stage_evidence_id",
        "validation_context_id",
        "document_fingerprint",
        "signed_payload_hash",
        "capability_set_hash",
    )
    for index, field in enumerate(proof_fields, 7):
        random_hex = deepcopy(fabricated_records)
        random_hex["product_capabilities_evidence"][field] = f"{index:x}" * 64
        diagnostic = UpstreamReadinessEvidence(random_hex)
        assert diagnostic.records["product_capabilities_evidence"][field] == f"{index:x}" * 64
    scalar = deepcopy(valid.records)
    scalar.pop("product_capabilities_evidence")
    scalar["runtime_context"]["product_capabilities_trust_state"] = "VALID"
    with pytest.raises(ValueError):
        UpstreamReadinessEvidence(scalar)
    wrong_key = deepcopy(valid.records)
    record = wrong_key["accounts_by_id"].pop(IDS["exchange_account_id"])
    wrong_key["accounts_by_id"][f"xacc_00000000-0000-7000-8000-000000000002"] = record
    with pytest.raises(ValueError):
        UpstreamReadinessEvidence(wrong_key)


def test_historical_replay_precedes_clock_and_conflict_for_known_sequence() -> None:
    ref = ObservationReference(_all_contracts(), [_policy("COMPONENT_STATUS", horizon=30, skew=20)])
    first = _observation()
    ref.ingest(first, "2030-01-01T00:00:00Z")
    second = _observation(
        observation_id="local.second",
        source_sequence=2,
        source_event_at_utc="2030-01-01T00:00:10Z",
        observed_at_utc="2030-01-01T00:00:10Z",
        ingested_at_utc="2030-01-01T00:00:10Z",
        expires_at_utc="2030-01-01T00:00:40Z",
    )
    ref.ingest(second, "2030-01-01T00:00:10Z")
    replay = ref.ingest(deepcopy(first), "2030-01-01T00:00:10Z")
    assert (
        replay["acceptance"]["replayed"]
        and replay["acceptance"]["observation"] == first
        and replay["current"]["observation"] == second
    )
    conflict = deepcopy(first)
    conflict["condition"] = "BLOCKED"
    with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
        ref.ingest(conflict, "2030-01-01T00:00:10Z")
    unseen = _observation(
        observation_id="local.unseen",
        source_sequence=3,
        observed_at_utc="2030-01-01T00:00:09Z",
        ingested_at_utc="2030-01-01T00:00:09Z",
        expires_at_utc="2030-01-01T00:00:39Z",
    )
    with pytest.raises(ValueError, match="CLOCK_REGRESSION"):
        ref.ingest(unseen, "2030-01-01T00:00:10Z")
    assert next(iter(ref.current.values()))["observation"] == second


def test_every_selector_family_executes_and_mutations_fail_closed() -> None:
    ref = _store(["COMPONENT_STATUS"])
    oracle = ReadinessReference(ref)
    scopes = {
        "PAPER_OPERATION": (
            "PAPER",
            {k: IDS[k] for k in ("exchange_account_id", "instrument_id", "execution_route_id")},
        ),
        "TESTNET_PRIVATE_EXECUTION": (
            "TESTNET",
            {
                k: IDS[k]
                for k in (
                    "exchange_account_id",
                    "instrument_id",
                    "execution_route_id",
                    "market_data_route_id",
                    "portfolio_id",
                )
            },
        ),
        "LIVE_EXECUTION": (
            "LIVE",
            {
                k: IDS[k]
                for k in (
                    "exchange_account_id",
                    "instrument_id",
                    "execution_route_id",
                    "market_data_route_id",
                    "portfolio_id",
                )
            },
        ),
        "MARKET_DATA_CONSUMPTION": (
            "TESTNET",
            {k: IDS[k] for k in ("market_data_route_id", "instrument_id")},
        ),
        "CORE_STARTUP": (
            None,
            {
                "device_installation_id": IDS["device_installation_id"],
                "state_store_identity_fingerprint_sha256": "a" * 64,
            },
        ),
        "PERSISTENCE_MUTATION": (
            None,
            {
                "device_installation_id": IDS["device_installation_id"],
                "state_store_identity_fingerprint_sha256": "a" * 64,
            },
        ),
        "RECONCILIATION_RECOVERY": (
            "TESTNET",
            {k: IDS[k] for k in ("exchange_account_id", "portfolio_id")},
        ),
    }
    outcomes = {}
    for row_id, (environment, scope) in scopes.items():
        evidence = _evidence(environment or "PAPER", scope)
        outcomes[row_id] = oracle.resolve_capability_authority(
            oracle.rows[row_id], environment, scope, ref.contracts, evidence
        )
    assert outcomes == {
        "PAPER_OPERATION": True,
        "TESTNET_PRIVATE_EXECUTION": False,
        "LIVE_EXECUTION": False,
        "MARKET_DATA_CONSUMPTION": True,
        "CORE_STARTUP": True,
        "PERSISTENCE_MUTATION": True,
        "RECONCILIATION_RECOVERY": True,
    }
    original = MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
    for index, row in enumerate(original):
        mutated = deepcopy(original)
        mutated[index]["capability_authority"]["selector"] = {"unknown": True}
        MACHINE["health_readiness_model"]["readiness_dependency_matrix"] = mutated
        try:
            with pytest.raises(ContractInconsistent):
                ReadinessReference(ref)
        finally:
            MACHINE["health_readiness_model"]["readiness_dependency_matrix"] = original


def test_every_category_scope_field_contract_is_machine_resolvable() -> None:
    contracts = _all_contracts()
    for category in MACHINE["observability_model"]["category_scope_registry"].values():
        for field, contract in category["field_contracts"].items():
            assert contract["milestone"] in {*contracts, "M0.12"}
            if contract["milestone"] in {"M0.5", "M0.6"}:
                upstream = _json_pointer(contracts[contract["milestone"]], contract["pointer"])
                assert upstream["id_field"] == field
                assert upstream["id_prefix"] == contract["identity_or_type"]
            elif contract["milestone"] == "M0.2":
                entity_name = contract["pointer"].split("=")[-1].rstrip("]")
                upstream = next(
                    x
                    for x in contracts["M0.2"]["entity_kinds"]
                    if x["canonical_name"] == entity_name
                )
                assert upstream["id_field"] == field
                assert upstream["id_prefix"] == contract["identity_or_type"]
            else:
                assert contract["identity_or_type"] in {"enum", "safe_handle", "lowercase_hex_64"}


def test_every_capability_authority_and_scope_binding_is_executable() -> None:
    ref = _store(["COMPONENT_STATUS"])
    oracle = ReadinessReference(ref)
    for row in oracle.rows.values():
        authority = row["capability_authority"]
        assert (
            _json_pointer(ref.contracts[authority["milestone"]], authority["pointer"]) is not None
        )
        for dependency in row["required_observations"]:
            assert (
                dependency["scope_binding"]["mode"]
                in MACHINE["health_readiness_model"]["dependency_scope_binding_modes"]
            )
    mutated = deepcopy(MACHINE["health_readiness_model"]["readiness_dependency_matrix"])
    mutated[0]["required_observations"][0]["scope_binding"] = {
        "mode": "MATCH_REQUEST_SCOPE_FIELDS",
        "fields": [],
    }
    original = MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
    MACHINE["health_readiness_model"]["readiness_dependency_matrix"] = mutated
    try:
        with pytest.raises(ContractInconsistent):
            ReadinessReference(ref)
    finally:
        MACHINE["health_readiness_model"]["readiness_dependency_matrix"] = original


def test_every_manifest_pointer_resolves_and_each_binding_family_mutation_fails() -> None:
    contracts = _all_contracts()
    ObservationReference(contracts, [_policy("COMPONENT_STATUS")])
    for binding in MACHINE["health_readiness_model"]["upstream_binding_manifest"]:
        for pointer in binding["pointers"]:
            mutated = deepcopy(contracts)
            parent, _, leaf = pointer.rpartition("/")
            target = _json_pointer(mutated[binding["milestone"]], parent or "/")
            if isinstance(target, dict) and leaf in target:
                target[leaf] = {"redteam": "changed"}
            else:
                mutated[binding["milestone"]] = {"redteam": "changed"}
            with pytest.raises(ContractInconsistent):
                ObservationReference(mutated, [_policy("COMPONENT_STATUS")])


@pytest.mark.parametrize(
    "drifted",
    ["UNKNOWN", "BROKEN", "IMPLEMENTED", None, {"unexpected": True}],
)
def test_m04_reachability_drift_fails_before_fabricated_wrapper_can_authorize(
    drifted: Any,
) -> None:
    scope = {
        key: IDS[key]
        for key in (
            "exchange_account_id",
            "instrument_id",
            "execution_route_id",
            "market_data_route_id",
            "portfolio_id",
        )
    }
    fabricated_records = deepcopy(_evidence("TESTNET", scope).records)
    fabricated_records["product_capabilities_evidence"] = _fabricated_m04_wrapper()
    fabricated = UpstreamReadinessEvidence(fabricated_records)
    assert fabricated.records["product_capabilities_evidence"] is not None
    contracts = _all_contracts()
    contracts["M0.4"]["signature_validation_pipeline"][
        "m0_4_cryptographic_signature_verification"
    ] = drifted
    with pytest.raises(ContractInconsistent):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")])


def test_m04_future_availability_drift_is_bound_and_fails_closed() -> None:
    contracts = _all_contracts()
    contracts["M0.4"]["signature_validation_pipeline"]["stage_result_registry"][
        "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED"
    ]["availability"] = "CURRENT_REACHABLE"
    with pytest.raises(ContractInconsistent):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")])


def test_every_declared_m04_reachability_pointer_is_manifest_bound() -> None:
    authority = MACHINE["health_readiness_model"]["product_capabilities_authority"]
    declared = {
        value for key, value in authority["reachability_source"].items() if key.endswith("pointer")
    }
    manifest = next(
        item
        for item in MACHINE["health_readiness_model"]["upstream_binding_manifest"]
        if item["milestone"] == "M0.4"
    )
    assert declared <= set(manifest["pointers"])
    assert authority["current_reachability_rule"] == {
        "pipeline_value_must_equal": "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4",
        "accepted_result_availability_must_equal": "FUTURE_ONLY",
        "derived_current_valid_reachable": False,
        "unexpected_value": "CONTRACT_INCONSISTENT_NEVER_REACHABLE",
        "negative_sentinel_comparison_forbidden": True,
    }
    assert authority["diagnostic_wrapper_in_positive_authorization_expression"] is False


def test_live_market_data_is_blocked_despite_canonical_looking_ready_route_and_ok_telemetry() -> (
    None
):
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    observations, oracle, evidence = _readiness(
        "MARKET_DATA_CONSUMPTION",
        "LIVE",
        scope,
        ["ADAPTER_STATUS", "MARKET_DATA_FRESHNESS"],
    )
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    assert route["route_readiness"]["readiness_state"] == "READY"
    live = next(
        item
        for item in observations.contracts["M0.4"]["execution_environments"]
        if item["environment_id"] == "LIVE"
    )
    assert live["endpoint_classes"] == ["PUBLIC_LIVE", "PRIVATE_LIVE"]
    assert (
        oracle.evaluate("MARKET_DATA_CONSUMPTION", "LIVE", scope, "2030-01-01T00:00:01Z", evidence)
        == "BLOCKED"
    )


def test_paper_and_testnet_market_data_remain_source_policy_and_venue_bound() -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    for environment in ("PAPER", "TESTNET"):
        _, oracle, evidence = _readiness(
            "MARKET_DATA_CONSUMPTION",
            environment,
            scope,
            ["ADAPTER_STATUS", "MARKET_DATA_FRESHNESS"],
        )
        assert (
            oracle.evaluate(
                "MARKET_DATA_CONSUMPTION",
                environment,
                scope,
                "2030-01-01T00:00:01Z",
                evidence,
            )
            == "OK"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exchange_id", "missing_venue"),
        ("environment", "PAPER"),
        ("instrument_id", f"instr_00000000-0000-7000-8000-000000000002"),
        ("adapter_family_id", "wrong_adapter_family"),
        ("workspace_id", f"ws_00000000-0000-7000-8000-000000000002"),
    ],
)
def test_market_data_route_scope_or_venue_mismatch_blocks(field: str, value: str) -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    _, oracle, evidence = _readiness(
        "MARKET_DATA_CONSUMPTION",
        "TESTNET",
        scope,
        ["ADAPTER_STATUS", "MARKET_DATA_FRESHNESS"],
    )
    evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]][field] = value
    assert (
        oracle.evaluate(
            "MARKET_DATA_CONSUMPTION", "TESTNET", scope, "2030-01-01T00:00:01Z", evidence
        )
        == "BLOCKED"
    )


def test_live_endpoint_policy_and_exchange_registry_drift_are_rejected() -> None:
    endpoint_drift = _all_contracts()
    endpoint_drift["M0.4"]["endpoint_policy"][
        "live_endpoint_resolution_current_edition_allowed"
    ] = True
    with pytest.raises(ContractInconsistent):
        ObservationReference(endpoint_drift, [_policy("COMPONENT_STATUS")])

    registry_drift = _all_contracts()
    registry_drift["M0.5"]["exchange_registry_contract"]["entries"][0]["status"] = "DISABLED"
    with pytest.raises(ContractInconsistent):
        ObservationReference(registry_drift, [_policy("COMPONENT_STATUS")])


def test_current_live_market_data_authority_sources_are_all_manifest_bound() -> None:
    contract = MACHINE["health_readiness_model"]["live_market_data_authority"]
    declared = {
        (
            contract["current_endpoint_policy"]["milestone"],
            contract["current_endpoint_policy"]["pointer"],
        ),
        (contract["route_contract"]["milestone"], contract["route_contract"]["entity_pointer"]),
        (contract["route_contract"]["milestone"], contract["route_contract"]["readiness_pointer"]),
        (
            contract["route_contract"]["milestone"],
            contract["route_contract"]["endpoint_registry_pointer"],
        ),
        (contract["venue_registry"]["milestone"], contract["venue_registry"]["pointer"]),
    }
    covered = {
        (binding["milestone"], pointer)
        for binding in MACHINE["health_readiness_model"]["upstream_binding_manifest"]
        for pointer in binding["pointers"]
    }
    assert declared <= covered
    assert contract["endpoint_classes_are_descriptive_only"] is True
    assert contract["current_enabled_live_venue_reachable"] is False
    assert not any(
        "LIVE" in entry["supported_environments"] and entry["status"] == "ENABLED"
        for entry in _load("exchange_accounts_and_instruments.json")["exchange_registry_contract"][
            "entries"
        ]
    )


def _m06_route_check() -> tuple[dict[str, Any], dict[str, Any]]:
    row = next(
        item
        for item in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        if item["readiness_id"] == "PAPER_OPERATION"
    )
    check = next(
        item for item in row["required_upstream_checks"] if item["check_id"] == "m06_route_ready"
    )
    return row, check


def test_minimal_caller_ready_execution_route_cannot_authorize_paper() -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    records = deepcopy(_evidence("PAPER", scope).records)
    records["execution_routes_by_id"][IDS["execution_route_id"]] = {
        "execution_route_id": IDS["execution_route_id"],
        "route_kind": "EXECUTION",
        "environment": "PAPER",
        "readiness_state": "READY",
        "instrument_id": IDS["instrument_id"],
        "exchange_account_id": IDS["exchange_account_id"],
    }
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


def test_source_shaped_paper_execution_route_and_core_readiness_are_positive() -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    observations, oracle, evidence = _readiness(
        "PAPER_OPERATION", "PAPER", scope, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    row, check = _m06_route_check()
    assert (
        evidence.derive(check, row, "PAPER", scope, observations.contracts, "2030-01-01T00:00:01Z")
        is True
    )
    assert (
        oracle.evaluate("PAPER_OPERATION", "PAPER", scope, "2030-01-01T00:00:01Z", evidence) == "OK"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint_class", "PAPER_PUBLIC_DATA"),
        ("exchange_id", "missing_venue"),
        ("workspace_id", f"ws_00000000-0000-7000-8000-000000000002"),
        ("environment", "TESTNET"),
        ("market_type", "MARGIN"),
        ("adapter_family_id", "wrong_adapter"),
        ("exchange_account_id", f"xacc_00000000-0000-7000-8000-000000000002"),
        ("supported_instrument_types", ["OPTION"]),
        ("route_status", "DISABLED"),
        ("route_capability_ceiling", ["READ_MARKET_DATA"]),
        ("authorization_dependencies", []),
    ],
)
def test_execution_route_source_graph_mismatch_blocks(field: str, value: Any) -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    evidence = _evidence("PAPER", scope)
    evidence.records["execution_routes_by_id"][IDS["execution_route_id"]][field] = value
    row, check = _m06_route_check()
    assert (
        evidence.derive(check, row, "PAPER", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


def test_execution_route_readiness_is_separate_core_owned_projection() -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    evidence = _evidence("PAPER", scope)
    route = evidence.records["execution_routes_by_id"][IDS["execution_route_id"]]
    row, check = _m06_route_check()
    route["route_readiness"]["readiness_state"] = "NOT_READY"
    assert (
        evidence.derive(check, row, "PAPER", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )
    route["route_readiness"]["readiness_state"] = "READY"
    route["route_readiness"]["sequence_state"] = "GAP"
    assert (
        evidence.derive(check, row, "PAPER", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


def test_testnet_execution_route_validation_is_independent_of_current_capability_denial() -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    evidence = _evidence("TESTNET", scope)
    row, check = _m06_route_check()
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is True
    )
    evidence.records["execution_routes_by_id"][IDS["execution_route_id"]]["endpoint_class"] = (
        "TESTNET_PUBLIC_DATA"
    )
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


def test_execution_route_authority_pointers_are_manifest_bound() -> None:
    contract = MACHINE["health_readiness_model"]["execution_route_authority"]
    declared = {
        (contract["route_definition"]["milestone"], contract["route_definition"]["entity_pointer"]),
        (
            contract["route_definition"]["milestone"],
            contract["route_definition"]["contract_pointer"],
        ),
        (contract["route_readiness"]["milestone"], contract["route_readiness"]["entity_pointer"]),
        (
            contract["route_readiness"]["milestone"],
            contract["route_readiness"]["contract_pointer"],
        ),
        (contract["venue_pointer"]["milestone"], contract["venue_pointer"]["pointer"]),
        *(("M0.6", pointer) for pointer in contract["policy_pointers"]),
    }
    covered = {
        (binding["milestone"], pointer)
        for binding in MACHINE["health_readiness_model"]["upstream_binding_manifest"]
        for pointer in binding["pointers"]
    }
    assert declared <= covered
    assert contract["explicit_route_kind_dispatch"] == ["MARKET_DATA", "EXECUTION"]
    assert contract["unknown_or_unhandled_route_kind"] == "BLOCKED_NEVER_GENERIC_SUCCESS"
    assert contract["caller_ready_is_authority"] is False


def test_unknown_route_kind_has_no_generic_success_path() -> None:
    scope = {
        key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")
    }
    records = deepcopy(_evidence("PAPER", scope).records)
    records["execution_routes_by_id"][IDS["execution_route_id"]]["route_kind"] = "UNKNOWN"
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


def _route_scope() -> dict[str, str]:
    return {key: IDS[key] for key in ("exchange_account_id", "instrument_id", "execution_route_id")}


@pytest.mark.parametrize("sequence", ["CONTIGUOUS", "GAP"])
def test_execution_readiness_requires_not_applicable_sequence(sequence: str) -> None:
    scope = _route_scope()
    evidence = _evidence("PAPER", scope)
    evidence.records["execution_routes_by_id"][IDS["execution_route_id"]]["route_readiness"][
        "sequence_state"
    ] = sequence
    row, check = _m06_route_check()
    assert (
        evidence.derive(check, row, "PAPER", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


@pytest.mark.parametrize("sequence", ["CURRENT", "UNKNOWN"])
def test_noncanonical_route_readiness_sequence_is_invalid_evidence(sequence: str) -> None:
    scope = _route_scope()
    records = deepcopy(_evidence("PAPER", scope).records)
    records["execution_routes_by_id"][IDS["execution_route_id"]]["route_readiness"][
        "sequence_state"
    ] = sequence
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


@pytest.mark.parametrize(
    ("observed_at", "expected"),
    [
        ("2030-01-01T00:00:00.001Z", True),
        ("2030-01-01T00:00:00Z", True),
        ("2029-12-31T23:59:59.999Z", False),
        ("2030-01-01T00:00:30.001Z", False),
    ],
)
def test_execution_readiness_freshness_boundary_and_future(
    observed_at: str, expected: bool
) -> None:
    scope = _route_scope()
    evidence = _evidence("PAPER", scope)
    evidence.records["execution_routes_by_id"][IDS["execution_route_id"]]["route_readiness"][
        "observed_at"
    ] = observed_at
    row, check = _m06_route_check()
    assert (
        evidence.derive(check, row, "PAPER", scope, _all_contracts(), "2030-01-01T00:00:30Z")
        is expected
    )


def test_stale_execution_readiness_blocks_end_to_end_paper() -> None:
    scope = _route_scope()
    _, oracle, evidence = _readiness(
        "PAPER_OPERATION", "PAPER", scope, ["COMPONENT_STATUS", "PERSISTENCE_HEALTH"]
    )
    evidence.records["execution_routes_by_id"][IDS["execution_route_id"]]["route_readiness"][
        "observed_at"
    ] = "2029-12-31T23:59:00Z"
    assert (
        oracle.evaluate("PAPER_OPERATION", "PAPER", scope, "2030-01-01T00:00:01Z", evidence)
        == "BLOCKED"
    )


@pytest.mark.parametrize(
    ("sequence", "expected"), [("CONTIGUOUS", True), ("GAP", False), ("NOT_APPLICABLE", False)]
)
def test_market_data_readiness_kind_specific_sequence(sequence: str, expected: bool) -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    evidence = _evidence("TESTNET", scope)
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route["route_readiness"]["sequence_state"] = sequence
    row = next(
        item
        for item in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        if item["readiness_id"] == "MARKET_DATA_CONSUMPTION"
    )
    check = next(
        item for item in row["required_upstream_checks"] if item["check_id"] == "m06_route_ready"
    )
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is expected
    )


@pytest.mark.parametrize("observed_at", ["2029-12-31T23:59:00Z", "2030-01-01T00:00:01.001Z"])
def test_market_data_stale_or_future_core_readiness_blocks(observed_at: str) -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    evidence = _evidence("TESTNET", scope)
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route["route_readiness"]["observed_at"] = observed_at
    row = next(
        item
        for item in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        if item["readiness_id"] == "MARKET_DATA_CONSUMPTION"
    )
    check = next(
        item for item in row["required_upstream_checks"] if item["check_id"] == "m06_route_ready"
    )
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route_capability_ceiling", "PLACE_ORDERS"),
        ("supported_instrument_types", "SPOT_PAIR"),
        ("authorization_dependencies", "PRODUCT_CAPABILITIES"),
        ("route_capability_ceiling", ("PLACE_ORDERS",)),
        ("supported_instrument_types", None),
    ],
)
def test_execution_route_arrays_are_strict_lists_before_membership(field: str, value: Any) -> None:
    scope = _route_scope()
    records = deepcopy(_evidence("PAPER", scope).records)
    records["execution_routes_by_id"][IDS["execution_route_id"]][field] = value
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


def test_market_route_local_ready_without_core_readiness_is_invalid() -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    records = deepcopy(_evidence("TESTNET", scope).records)
    route = records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route.pop("route_readiness")
    route["readiness_state"] = "READY"
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


@pytest.mark.parametrize("section", ["market_data_routes_by_id", "execution_routes_by_id"])
def test_route_readiness_orphan_or_wrong_kind_is_rejected(section: str) -> None:
    scope = {
        key: IDS[key]
        for key in (
            "exchange_account_id",
            "instrument_id",
            "execution_route_id",
            "market_data_route_id",
        )
    }
    records = deepcopy(_evidence("TESTNET", scope).records)
    route = next(iter(records[section].values()))
    route["route_readiness"]["route_id"] = (
        f"xroute_00000000-0000-7000-8000-000000000002"
        if section == "execution_routes_by_id"
        else f"mdr_00000000-0000-7000-8000-000000000002"
    )
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)
    route["route_readiness"]["route_id"] = route[
        "execution_route_id" if section == "execution_routes_by_id" else "market_data_route_id"
    ]
    route["route_readiness"]["route_kind"] = (
        "MARKET_DATA" if section == "execution_routes_by_id" else "EXECUTION"
    )
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_EVIDENCE"):
        UpstreamReadinessEvidence(records)


def test_route_readiness_policy_pointers_are_manifest_bound_and_drift_protected() -> None:
    policy = MACHINE["health_readiness_model"]["route_readiness_temporal_authority"]
    declared = {
        (
            policy["route_readiness_schema"]["milestone"],
            policy["route_readiness_schema"]["pointer"],
        ),
        (
            policy["route_readiness_contract"]["milestone"],
            policy["route_readiness_contract"]["pointer"],
        ),
        (policy["execution_max_age"]["milestone"], policy["execution_max_age"]["pointer"]),
        *((item["milestone"], item["pointer"]) for item in policy["array_registry_pointers"]),
    }
    covered = {
        (binding["milestone"], pointer)
        for binding in MACHINE["health_readiness_model"]["upstream_binding_manifest"]
        for pointer in binding["pointers"]
    }
    assert declared <= covered
    contracts = _all_contracts()
    contracts["M0.6"]["execution_route_contract"]["execution_readiness_max_age_seconds"] += 1
    with pytest.raises(ContractInconsistent):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")])


def _market_data_row_check() -> tuple[dict[str, Any], dict[str, Any]]:
    row = next(
        item
        for item in MACHINE["health_readiness_model"]["readiness_dependency_matrix"]
        if item["readiness_id"] == "MARKET_DATA_CONSUMPTION"
    )
    return row, next(
        item for item in row["required_upstream_checks"] if item["check_id"] == "m06_route_ready"
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"endpoint_class": "TESTNET_PUBLIC_DATA", "data_scope": "PRIVATE"}, False),
        (
            {
                "endpoint_class": "TESTNET_PUBLIC_DATA",
                "data_scope": "PUBLIC",
                "channel_types": ["PRIVATE_ORDERS"],
            },
            False,
        ),
        (
            {
                "endpoint_class": "TESTNET_PRIVATE_DATA",
                "data_scope": "PUBLIC",
                "channel_types": ["TRADES"],
            },
            False,
        ),
        (
            {
                "endpoint_class": "TESTNET_PUBLIC_DATA",
                "data_scope": "PUBLIC",
                "channel_types": ["TRADES", "ORDER_BOOK"],
            },
            True,
        ),
    ],
)
def test_market_data_endpoint_access_cross_field_rules(
    changes: dict[str, Any], expected: bool
) -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    evidence = _evidence("TESTNET", scope)
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route.update(changes)
    row, check = _market_data_row_check()
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is expected
    )


def test_market_data_access_mismatch_blocks_end_to_end_despite_fresh_ok_telemetry() -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    _, oracle, evidence = _readiness(
        "MARKET_DATA_CONSUMPTION",
        "TESTNET",
        scope,
        ["ADAPTER_STATUS", "MARKET_DATA_FRESHNESS"],
    )
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route["data_scope"] = "PRIVATE"
    assert (
        oracle.evaluate(
            "MARKET_DATA_CONSUMPTION", "TESTNET", scope, "2030-01-01T00:00:01Z", evidence
        )
        == "BLOCKED"
    )
    route["data_scope"] = "PUBLIC"
    route["channel_types"] = ["PRIVATE_BALANCES"]
    assert (
        oracle.evaluate(
            "MARKET_DATA_CONSUMPTION", "TESTNET", scope, "2030-01-01T00:00:01Z", evidence
        )
        == "BLOCKED"
    )


def test_every_market_data_instrument_id_must_resolve_to_the_exact_route_graph() -> None:
    scope = {key: IDS[key] for key in ("market_data_route_id", "instrument_id")}
    evidence = _evidence("TESTNET", scope)
    route = evidence.records["market_data_routes_by_id"][IDS["market_data_route_id"]]
    route["instrument_ids"].append("instr_00000000-0000-7000-8000-000000000002")
    row, check = _market_data_row_check()
    assert (
        evidence.derive(check, row, "TESTNET", scope, _all_contracts(), "2030-01-01T00:00:01Z")
        is False
    )


def test_live_market_data_schema_reference_is_current_and_self_consistent() -> None:
    health = MACHINE["health_readiness_model"]
    route_contract = health["live_market_data_authority"]["route_contract"]
    schema = health["upstream_evidence_contract"]["nested_section_schemas"][
        "market_data_routes_by_id"
    ]
    assert route_contract["joined_projection_schema_ref"].endswith(
        "/nested_section_schemas/market_data_routes_by_id"
    )
    assert "required_projection_fields" not in route_contract
    assert "readiness_state" not in schema["required_fields"]
    assert "route_readiness" in schema["required_fields"]
    assert (
        health["route_readiness_temporal_authority"]["persisted_route_readiness_state_forbidden"]
        is True
    )


def test_market_data_route_access_contract_is_manifest_bound_and_drift_protected() -> None:
    health = MACHINE["health_readiness_model"]
    authority = health["market_data_route_access_authority"]
    manifest = next(
        item for item in health["upstream_binding_manifest"] if item["milestone"] == "M0.6"
    )
    assert authority["contract_pointer"] in manifest["pointers"]
    assert authority["endpoint_registry_pointer"] in manifest["pointers"]
    contracts = _all_contracts()
    contracts["M0.6"]["market_data_route_contract"]["endpoint_access_policy"] = "WEAKENED"
    with pytest.raises(ContractInconsistent):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")])
    contracts = _all_contracts()
    contracts["M0.6"]["market_data_route_contract"]["private_channels_require_private_scope"] = (
        False
    )
    with pytest.raises(ContractInconsistent):
        ObservationReference(contracts, [_policy("COMPONENT_STATUS")])


# S9D-C2 gate oracle. Privileged mutations deliberately have no local proof model.
M010 = _load("identity_device_authentication_and_secrets.json")


class AlertAuthorizationBlocked(RuntimeError):
    pass


def _frozen_m010_authorize_alert_action(operation: str) -> None:
    """Interpret the closed frozen operation registry; unknown operations fail closed."""
    registry = M010["operation_policy_registry"]
    ownership = {
        operation
        for operations in M010["operation_ownership"].values()
        if isinstance(operations, list)
        for operation in operations
    }
    if operation not in registry or operation not in ownership:
        raise AlertAuthorizationBlocked("OPERATION_UNSUPPORTED")
    raise AssertionError("alert operation unexpectedly entered frozen M0.10")


def _blocked_alert_mutation(state: dict[str, Any], operation: str) -> None:
    before = deepcopy(state)
    with pytest.raises(AlertAuthorizationBlocked, match="OPERATION_UNSUPPORTED"):
        _frozen_m010_authorize_alert_action(operation)
    assert state == before


def test_s9d_c25_status_is_honestly_blocked_and_updater_stays_not_started() -> None:
    assert MACHINE["status"] == "IN_PROGRESS_S9D_C25_BLOCKED_CATALOG_RUNTIME_ACCEPTANCE"
    assert MACHINE["contract_identity"] == {
        "contract_id": "M0.12-audit-observability-alerts-updater",
        "version": "1.46.0",
        "phase": "S9D_C25_BLOCKED_CATALOG_RUNTIME_ACCEPTANCE",
        "machine_source_of_truth": True,
        "markdown_is_projection_only": True,
    }
    release = MACHINE["alert_model"]["release_update_disposition"]
    assert release["status"] == "FUTURE_SOURCE_NOT_CURRENT_AUTHORITY"
    assert release["updater_state_machines"] == "OPEN"


def test_s9d_c3_consumes_m010_without_a_local_proof_schema() -> None:
    authority = MACHINE["alert_model"]["executable_authority"]
    assert authority["plain_authorize_executes_mutation"] is False
    assert "validate_downstream_authorized_mutation" in authority["authorization_boundary"]
    assert authority["updater"] == "NOT_STARTED"
    assert authority["manual_resolution_policy"]["DOMAIN_EXECUTION_FAILURE"] == "DENIED"


def test_no_local_fake_authentication_proof_or_accepted_action_remains() -> None:
    policy = MACHINE["alert_model"]["operator_action_policy"]
    assert len(policy["accepted_operations"]) == 4
    assert policy["authority"].startswith("canonical M0.10")
    assert "authorization_reference" not in json.dumps(MACHINE["alert_model"])


def test_domain_execution_fact_resolution_is_open_and_self_resolution_forbidden() -> None:
    policy = next(
        item
        for item in MACHINE["alert_model"]["alert_type_registry"]
        if item["alert_type"] == "DOMAIN_EXECUTION_FAILURE"
    )
    frozen_events = _load("commands_events_order_lifecycle_and_idempotency.json")["event_contract"][
        "event_types"
    ]
    sources = {
        "ORDER_REJECTED",
        "ORDER_EXTERNAL_OUTCOME_UNKNOWN",
        "IDEMPOTENCY_CONFLICT",
    }
    assert sources <= set(frozen_events)
    assert set(policy["source_selector"]["event_types"]) == sources
    assert policy["resolution_policy"].startswith(
        "OPEN_NO_UNAMBIGUOUS_CORRECTIVE_SUCCESSOR_IN_FROZEN_M0.7"
    )
    assert "original or unrelated same-scope event REJECT" in policy["resolution_policy"]
    assert (
        "OPEN:"
        in MACHINE["alert_model"]["resolution_contract"]["typed_paths"]["DOMAIN_EXECUTION_FAILURE"]
    )


def test_dedup_registry_scope_declarations_are_locally_corrected() -> None:
    by_type = {item["alert_type"]: item for item in MACHINE["alert_model"]["alert_type_registry"]}
    assert by_type["KILL_SWITCH_ACTIVE"]["dedup_policy"]["components"] == [
        "alert_type",
        "environment",
        "scope_type",
        "scope_id",
        "M09_KILL_SWITCH",
    ]
    assert by_type["PERSISTENCE_RECOVERY_REQUIRED"]["dedup_policy"]["components"] == [
        "alert_type",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "M11_RECOVERY",
    ]
    assert by_type["ALERT_DELIVERY_SUBSYSTEM_FAILURE"]["dedup_policy"]["components"] == [
        "alert_type",
        "environment",
        "component",
        "condition_code",
        "source_instance_id",
    ]


def test_c4_defect_status_distinguishes_executable_and_upstream_open_work() -> None:
    statuses = MACHINE["alert_model"]["local_defect_status"]
    assert set(statuses) == {
        "domain_fact_corrective_resolution",
        "source_membership_current_authority",
        "canonical_ids_time",
        "semantic_alert_validation",
        "dedup_registry_executable_parity",
        "currentness_ordering",
        "security_manual_resolution",
        "atomic_audit",
        "clear_suppression",
        "internal_delivery_condition",
        "multi_source_observations",
        "escalation",
        "delivery_attempts",
        "semantic_restore",
        "source_selectors",
    }
    assert statuses["domain_fact_corrective_resolution"].startswith("OPEN_SOURCE_AUTHORITY")
    assert statuses["source_selectors"].startswith("PARTIAL_OPEN")
    assert all(
        value.startswith("EXECUTABLE")
        for key, value in statuses.items()
        if key not in {"domain_fact_corrective_resolution", "source_selectors"}
    )
    assert not MACHINE["contract_identity"]["phase"].endswith("CLOSED")


def test_markdown_remains_exact_machine_projection_after_c2_gate() -> None:
    assert MARKDOWN_PATH.read_text(encoding="utf-8") == _render_markdown(MACHINE)


def test_s9d_c19_records_executable_adapter_boundary() -> None:
    disposition = MACHINE["alert_model"]["executable_authority"][
        "s9d_c18_source_authority_disposition"
    ]
    integrated = "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
    assert disposition["status"] == integrated
    assert "consume_effective_current" in disposition["production_api"]
    assert "ObservationAuthority" in disposition["production_types"]
    assert disposition["alertstore_adapter"].startswith("IMPLEMENTED")
    assert disposition["market_data_current_condition"] == integrated
    assert disposition["execution_route_condition"] == integrated
    assert "pure executable oracle" in disposition["executable_oracle"]
    assert disposition["membership_authority"].startswith("EXECUTABLE")
    assert disposition["currentness_and_expiry_fence"].startswith("EXECUTABLE")
    assert disposition["durable_historical_lookup"].startswith("EXECUTABLE")
    statuses = {
        alert_type: status
        for alert_type, (_, status) in PRODUCTION_SOURCE_RESOLUTION_POLICIES.items()
    }
    assert statuses["MARKET_DATA_CURRENT_CONDITION"] == integrated
    assert statuses["EXECUTION_ROUTE_CONDITION"] == integrated
    assert statuses["KILL_SWITCH_ACTIVE"] == (
        "M09_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
    )
    assert all(
        status == "OPEN_SOURCE_AUTHORITY"
        for alert_type, status in statuses.items()
        if alert_type
        not in {
            "MARKET_DATA_CURRENT_CONDITION",
            "EXECUTION_ROUTE_CONDITION",
            "KILL_SWITCH_ACTIVE",
        }
    )


def test_s9d_c20_adapter_status_and_production_projection_have_machine_parity() -> None:
    integrated = "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
    statuses = []

    def collect(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "s9d_adapter_integration":
                    statuses.append(nested)
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(MACHINE)
    assert statuses and set(statuses) == {integrated}

    frozen = MACHINE["alert_model"]["executable_authority"]["s9c_adapter_projection"]
    for category, runtime in S9C_PRODUCTION_PROJECTION.items():
        machine = frozen[category]
        assert machine["alert_type"] == runtime["alert_type"]
        assert machine["source_family"] == "OBSERVATION_CONDITION"
        assert machine["fact_type"] == category
        assert machine["severity"] == dict(runtime["severity"])
        assert machine["healthy_resolution_supported"] is True
        assert machine["resolution_policy_id"] == S9C_RESOLUTION_POLICY_ID
        assert machine["scope_fields"] == list(runtime["scope_fields"])


def test_s9d_c25_records_honest_m08_reconciliation_authority_blocker() -> None:
    alerting = MACHINE["alert_model"]["executable_authority"]
    disposition = alerting["s9d_c25_m08_reconciliation_authority_disposition"]
    assert disposition["status"] == "S9D_C25_BLOCKED_M08_EXECUTABLE_RECONCILIATION_AUTHORITY"
    assert disposition["match_authority"].startswith("UNAVAILABLE")
    assert disposition["internal_accounting_authority"] == (
        "M0.8_INTERNAL_ACCOUNTING_AUTHORITY_BLOCKED_UPSTREAM_SOURCE_MEMBERSHIP"
    )
    assert disposition["observed_balance_authority"] == "AVAILABLE"
    assert disposition["observed_balance_authority_boundary"]["implementation"].endswith(
        "CoreAcceptedObservedBalanceFactProjection"
    )
    assert disposition["reconciliation_result_authority"] == "MISSING"
    assert len(disposition["missing_upstream_dependencies"]) == 6
    assert disposition["c26"] == "NOT_STARTED"
    assert disposition["reconciliation_divergence_alertstore_adapter"] == "NOT_STARTED"
    assert disposition["reconciliation_divergence_status"] == "OPEN_SOURCE_AUTHORITY"
    assert disposition["source_producer_authenticity"] == "OPEN"
    assert disposition["s9d"] == "OPEN"
    assert disposition["updater"] == "NOT_STARTED"
    assert disposition["m010"] == "CLOSED"
    assert disposition["c24_r1"] == "CLOSED"
    policies = alerting["production_source_resolution_policies"]
    assert policies["RECONCILIATION_DIVERGENCE"]["executable_status"] == "OPEN_SOURCE_AUTHORITY"
    assert policies["KILL_SWITCH_ACTIVE"]["executable_status"] == (
        "M09_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
    )
    for alert_type in ("MARKET_DATA_CURRENT_CONDITION", "EXECUTION_ROUTE_CONDITION"):
        assert policies[alert_type]["executable_status"] == (
            "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
        )
