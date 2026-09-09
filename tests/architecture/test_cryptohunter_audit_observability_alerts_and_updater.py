"""Executable foundation checks for the canonical M0.12 contract."""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata

import pytest
from copy import deepcopy
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
    assert MACHINE["status"] == "IN_PROGRESS_AUDIT_JOURNAL_CLOSED"
    assert MACHINE["contract_identity"] == {
        "contract_id": "M0.12-audit-observability-alerts-updater",
        "version": "0.2.5",
        "phase": "S9B_C5_AUDIT_JOURNAL_CONTRACT",
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
