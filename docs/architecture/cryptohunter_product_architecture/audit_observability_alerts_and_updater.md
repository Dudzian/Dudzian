# M0.12 — Audit, observability, alerts and updater

> Deterministyczna projekcja kanonicznego JSON. Nie jest niezależnym źródłem semantyki.

## `schema_version`

```json
"cryptohunter.audit_observability_alerts_and_updater.v1"
```

## `m0_element`

```json
"M0.12"
```

## `status`

```json
"IN_PROGRESS_S9D_C24_R1_M09_HISTORICAL_CURRENTNESS_OTHER_SOURCE_AUTHORITIES_OPEN"
```

## `contract_identity`

```json
{
  "contract_id": "M0.12-audit-observability-alerts-updater",
  "version": "1.29.1",
  "phase": "S9D_C24_R1_M09_HISTORICAL_CURRENTNESS_CLOSED",
  "machine_source_of_truth": true,
  "markdown_is_projection_only": true
}
```

## `upstream_dependencies`

```json
[
  {
    "milestone": "M0.2",
    "artifact": "canonical_domain_vocabulary.json",
    "binding": "durable identities, environment-bound entities and runtime-session identity"
  },
  {
    "milestone": "M0.3",
    "artifact": "process_topology_and_lifecycle.json",
    "binding": "CoreHost authority, process roles, readiness, single instance, restart and recovery coordination"
  },
  {
    "milestone": "M0.4",
    "artifact": "environment_and_product_capabilities.json",
    "binding": "environments, ProductCapabilities, readiness gates and current LIVE denial"
  },
  {
    "milestone": "M0.5",
    "artifact": "exchange_accounts_and_instruments.json",
    "binding": "exchange, account and instrument identity used by observations and audit scope"
  },
  {
    "milestone": "M0.6",
    "artifact": "strategy_market_data_and_execution_routing.json",
    "binding": "route identity, freshness and adapter readiness facts"
  },
  {
    "milestone": "M0.7",
    "artifact": "commands_events_order_lifecycle_and_idempotency.json",
    "binding": "command, domain-event, order and execution transition provenance"
  },
  {
    "milestone": "M0.8",
    "artifact": "ledger_portfolio_capital_and_pnl.json",
    "binding": "economic facts and reconciliation authority"
  },
  {
    "milestone": "M0.9",
    "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
    "binding": "risk, kill-switch and execution-lease authority and fail-close behavior"
  },
  {
    "milestone": "M0.10",
    "artifact": "identity_device_authentication_and_secrets.json",
    "binding": "identity, authentication, operator authorization and secret-boundary authority"
  },
  {
    "milestone": "M0.11",
    "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
    "binding": "StateStore schema, forward-only migration, backup, recovery and rollback constraints"
  }
]
```

## `canonical_vocabulary`

```json
{
  "record_types": [
    "AuditEvent",
    "DomainEventReference",
    "StructuredLogRecord",
    "MetricSample",
    "TraceSpan",
    "HealthObservation",
    "ComponentStatusObservation",
    "Alert",
    "AlertLifecycleRecord",
    "ReleaseManifest",
    "ArtifactManifest",
    "ArtifactVerificationResult",
    "UpdatePlan",
    "UpdateAttempt",
    "RollbackPlan"
  ],
  "health_dimensions": [
    "LIVENESS",
    "HEALTH",
    "READINESS"
  ],
  "condition_states": [
    "UNKNOWN",
    "OK",
    "DEGRADED",
    "BLOCKED"
  ],
  "alert_lifecycle_states": [
    "RAISED",
    "ACKNOWLEDGED",
    "RESOLVED"
  ],
  "alert_severities": [
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL"
  ],
  "alert_categories": [
    "OPERATIONS",
    "MARKET_DATA",
    "EXECUTION",
    "PERSISTENCE",
    "RECONCILIATION",
    "SECURITY",
    "RISK",
    "RESOURCE",
    "RELEASE_UPDATE"
  ],
  "update_phases": [
    "DISCOVERY",
    "DOWNLOAD",
    "VERIFICATION",
    "STAGING",
    "AUTHORIZATION",
    "INSTALLATION",
    "RESTART",
    "POST_INSTALL_VERIFICATION",
    "ROLLBACK",
    "FAILED_UPDATE_RECOVERY"
  ],
  "failure_actions": [
    "OBSERVE_ONLY",
    "DEGRADE",
    "BLOCK_CAPABILITY",
    "TRIGGER_RECONCILIATION",
    "REQUIRE_OPERATOR_ACTION",
    "INTERACT_WITH_KILL_SWITCH",
    "RESTART_OR_RECOVER",
    "BLOCK_RELEASE_UPDATE"
  ],
  "audit_event_upstream_binding": {
    "semantic_owner": "M0.2 canonical_domain_vocabulary.json",
    "entity_pointer": "/entity_kinds entry selected where canonical_name == AuditEvent",
    "bound_fields": [
      "canonical_name",
      "id_field",
      "id_prefix",
      "purpose",
      "parent",
      "persistence",
      "audit_event_categories",
      "optional_references",
      "relationships",
      "identifier_policy"
    ],
    "identity_redefinition_by_M0.12": false
  },
  "identity_classification": {
    "A_EXACT_UPSTREAM_CANONICAL_ENTITY_ID": [
      {
        "field": "audit_event_id",
        "source": "canonical_domain_vocabulary.json AuditEvent entity; format, prefix and parent remain M0.2-owned"
      },
      {
        "field": "runtime_session_id",
        "source": "canonical_domain_vocabulary.json RuntimeSession; optional AuditEvent reference under M0.2 rules"
      }
    ],
    "B_M012_RECORD_OR_ATTEMPT_LOCAL_IDENTIFIER": [
      {
        "field": "contract_id",
        "status": "M0.12 contract metadata identifier; not a durable domain entity"
      },
      {
        "field": "observation_id",
        "status": "M0.12-local correlation handle; not an M0.2 persistent entity"
      },
      {
        "field": "log_record_id",
        "status": "optional M0.12-local diagnostic handle; not durable authority"
      },
      {
        "field": "alert_id",
        "status": "M0.12-local logical durable-record identity; alrt_ plus canonical lowercase UUIDv7; CoreHost generated only; not M0.2 identity"
      }
    ],
    "C_UNRESOLVED_IDENTITY_TO_CLOSE_LATER": [
      {
        "field": "release_id",
        "status": "release-domain local/unresolved; exact identity semantics deferred"
      },
      {
        "field": "build_id",
        "status": "release-domain local/unresolved; exact identity semantics deferred"
      },
      {
        "field": "artifact_id",
        "status": "release-domain local/unresolved; exact identity semantics deferred"
      },
      {
        "field": "update_attempt_id",
        "status": "M0.12 attempt-local/unresolved; exact identity semantics deferred"
      },
      {
        "field": "signing_key_id",
        "status": "trust-policy reference/unresolved; not an M0.2 entity"
      },
      {
        "field": "correlation_id",
        "status": "cross-contract correlation reference consumed when available; exact general identity policy not closed by S9A-C1"
      }
    ],
    "new_M02_style_prefixes_declared": false
  }
}
```

## `ownership_boundaries`

```json
[
  {
    "surface": "DURABLE_AUDIT_EVIDENCE",
    "owner": "M0.2 owns canonical AuditEvent identity/schema; M0.12 owns append policy; trusted writer is lifecycle-phase dependent",
    "authority_source": "canonical M0.2 AuditEvent plus an append decision from the phase-appropriate privileged trusted path",
    "durability": "DURABLE_APPEND_ONLY",
    "ordering": "DeviceInstallation-scoped durable monotonic append sequence; environment is optional scope only",
    "identity": "audit_event_id only; canonical format, prefix and DeviceInstallation parent remain M0.2-owned",
    "retention": "policy-versioned; security, execution and economic minimum retention may not be shortened by UI",
    "rebuildability": "not replaceable by logs or metrics; projections rebuild from retained records",
    "trust_boundary": "phase-aware trusted AuditEvent append boundary; not CoreHost-only",
    "failure_policy": "required AuditEvent append failure blocks the associated privileged or economic transition before acknowledgement where atomic audit is required"
  },
  {
    "surface": "OPERATIONAL_TELEMETRY",
    "owner": "emitting component under Core policy",
    "authority_source": "current observation of an authoritative component",
    "durability": "BEST_EFFORT_OR_BOUNDED_DURABLE",
    "ordering": "timestamp and source sequence; gaps permitted and explicit",
    "identity": "observation_id plus component and runtime session",
    "retention": "bounded operational policy",
    "rebuildability": "usually recomputable; loss must not invent health",
    "trust_boundary": "untrusted until source and freshness are validated",
    "failure_policy": "loss yields UNKNOWN or DEGRADED; never opens capability"
  },
  {
    "surface": "METRICS",
    "owner": "metrics pipeline",
    "authority_source": "aggregated observations",
    "durability": "BOUNDED",
    "ordering": "time-series ordering",
    "identity": "metric name and bounded labels",
    "retention": "operational policy",
    "rebuildability": "not guaranteed",
    "trust_boundary": "non-authoritative observation",
    "failure_policy": "observe or degrade monitoring only unless canonical safety evidence is also missing"
  },
  {
    "surface": "STRUCTURED_LOGS",
    "owner": "emitting component",
    "authority_source": "diagnostic statement",
    "durability": "BEST_EFFORT",
    "ordering": "source-local timestamp and sequence where available",
    "identity": "log_record_id optional; correlation_id required for privileged paths",
    "retention": "redaction-aware operational policy",
    "rebuildability": "not authoritative and not guaranteed",
    "trust_boundary": "diagnostic boundary; secrets forbidden",
    "failure_policy": "logging failure never converts a denied action to allowed"
  },
  {
    "surface": "HEALTH_READINESS_OBSERVATIONS",
    "owner": "CoreHost health aggregator",
    "authority_source": "fresh validated component observations plus upstream lifecycle/capability authorities",
    "durability": "CURRENT_PROJECTION_WITH_OPTIONAL_HISTORY",
    "ordering": "observed_at, expires_at and source sequence",
    "identity": "component, dimension, scope and environment",
    "retention": "bounded history",
    "rebuildability": "rebuildable from current probes and canonical facts",
    "trust_boundary": "observation cannot mint readiness authority",
    "failure_policy": "missing or stale safety evidence maps to UNKNOWN/BLOCKED and fails closed"
  },
  {
    "surface": "OPERATOR_AND_SECURITY_RISK_ALERTS",
    "owner": "CoreHost alert lifecycle service",
    "authority_source": "trusted fact or validated observation reference",
    "durability": "DURABLE_LIFECYCLE_FOR_REQUIRED_CATEGORIES",
    "ordering": "per-alert revision and raised/resolved timestamps",
    "identity": "stable alert_id and deduplication_key",
    "retention": "category policy; lifecycle history retained",
    "rebuildability": "condition alerts may be regenerated but acknowledgement history is not inferred",
    "trust_boundary": "UI and delivery channels are projection/transport only",
    "failure_policy": "delivery failure is observable; safety action is decided independently"
  },
  {
    "surface": "RELEASE_UPDATE_METADATA",
    "owner": "trusted release publisher; locally verified by privileged update coordinator",
    "authority_source": "authenticated ReleaseManifest plus local compatibility evaluation",
    "durability": "verified manifest and decisions durable",
    "ordering": "release identity and channel sequence; anti-rollback policy",
    "identity": "release_id, product_version, build_id, channel and platform",
    "retention": "current, staged, installed and rollback evidence retained",
    "rebuildability": "remote metadata is rediscoverable but local decisions and attempts are durable",
    "trust_boundary": "network/download boundary is untrusted",
    "failure_policy": "missing, ambiguous, expired, revoked or invalid metadata blocks update"
  },
  {
    "surface": "UPDATE_AND_ROLLBACK_AUTHORITY",
    "owner": "Bootstrapper coordinating with the single authoritative CoreHost lifecycle",
    "authority_source": "verified artifact, compatible UpdatePlan and required operator authorization",
    "durability": "durable phase journal and recovery markers",
    "ordering": "single serialized installation attempt",
    "identity": "update_attempt_id bound to release, artifact, installation and pre-state",
    "retention": "through post-install verification and recovery horizon",
    "rebuildability": "attempt authority cannot be reconstructed from UI state",
    "trust_boundary": "privileged installer boundary outside UI",
    "failure_policy": "fail closed, preserve recoverable prior artifact where compatible, otherwise recovery-required"
  },
  {
    "surface": "ARTIFACT_AUTHENTICITY",
    "owner": "privileged local artifact verifier",
    "authority_source": "exact bytes, digest, trusted signing identity, signature policy and manifest binding",
    "durability": "verification result bound to exact artifact bytes",
    "ordering": "verification after download and before eligibility",
    "identity": "artifact_id, digest, size, platform and release_id",
    "retention": "through install/rollback evidence horizon",
    "rebuildability": "re-verifiable from exact retained bytes and trust roots",
    "trust_boundary": "signing trust roots are build/protected policy; remote metadata cannot replace them",
    "failure_policy": "any mismatch blocks staging eligibility and installation"
  },
  {
    "surface": "UI_PROJECTIONS",
    "owner": "DesktopShell/TrayAgent presentation",
    "authority_source": "Core-confirmed read models",
    "durability": "cache only",
    "ordering": "projection revision and staleness",
    "identity": "upstream record/observation/alert/update identifiers",
    "retention": "presentation policy",
    "rebuildability": "fully rebuildable",
    "trust_boundary": "untrusted client input",
    "failure_policy": "stale/unknown displayed explicitly; UI cannot acknowledge success or install locally"
  }
]
```

## `trust_boundaries`

```json
[
  {
    "boundary": "AUDIT_EVENT_APPEND",
    "trusted_owner": "M0.12 append policy using the lifecycle-phase writer authorized by frozen M0.3/M0.10 boundaries",
    "untrusted_inputs": [
      "caller claims",
      "UI events",
      "logs",
      "metrics"
    ],
    "rule": "validate canonical AuditEvent identity and phase writer; append is atomic with or precedes acknowledgement where required; PRE_CORE does not require a running CoreHost"
  },
  {
    "boundary": "OBSERVATION_INGRESS",
    "trusted_owner": "CoreHost health aggregator",
    "untrusted_inputs": [
      "adapter self-report",
      "remote endpoints",
      "cached UI state"
    ],
    "rule": "validate identity, scope, environment, sequence and freshness; observations never grant authority"
  },
  {
    "boundary": "ALERT_LIFECYCLE",
    "trusted_owner": "CoreHost alert lifecycle service",
    "untrusted_inputs": [
      "notification delivery receipt",
      "UI acknowledgement request"
    ],
    "rule": "acknowledgement records operator awareness only; resolution requires current trusted fact/observation for condition alerts"
  },
  {
    "boundary": "RELEASE_DISCOVERY_AND_DOWNLOAD",
    "trusted_owner": "privileged local artifact verifier",
    "untrusted_inputs": [
      "remote manifest",
      "download URL",
      "downloaded bytes",
      "UI selection"
    ],
    "rule": "download produces only a candidate; exact authenticity and compatibility verification precede install eligibility"
  },
  {
    "boundary": "INSTALLATION",
    "trusted_owner": "Bootstrapper update coordinator",
    "untrusted_inputs": [
      "UI apply command",
      "artifact filename",
      "remote channel label"
    ],
    "rule": "serialized authorization, safe shutdown, install journal, restart and post-install verification are mandatory"
  }
]
```

## `audit_model`

```json
{
  "distinctions": {
    "AuditEvent": "canonical M0.2 durable evidence that a phase-appropriate trusted writer recorded an audit-required privileged, security, execution or economic transition",
    "DomainEventReference": "reference to an immutable M0.7 domain event; the event remains owned by its upstream domain contract and is not copied as audit-store authority",
    "StructuredLogRecord": "best-effort diagnostic narrative",
    "MetricSample": "aggregated numeric observation",
    "TraceSpan": "timing and causal diagnostic observation",
    "HealthObservation": "freshness-bounded statement about condition",
    "UI_notification": "projection/delivery only"
  },
  "required_properties": [
    "canonical M0.2 AuditEvent identity/schema",
    "privileged trusted writer appropriate to lifecycle phase",
    "durable",
    "immutable and append-only",
    "identity-bound",
    "device-bound",
    "environment-bound where applicable",
    "ordered",
    "tamper-evident",
    "recovery-compatible",
    "secret-redacted"
  ],
  "domain_event_boundary": [
    "a domain event proves its upstream state transition under M0.7",
    "an AuditEvent proves the accountable actor/request/result and may reference but never redefine the domain event",
    "one fact may require both records, atomically linked by correlation/causation",
    "audit storage may share a physical transaction with StateStore but is a distinct logical contract"
  ],
  "required_transition_classes": [
    "privileged operation",
    "identity/security mutation",
    "capability/environment decision",
    "execution/order/fill transition",
    "ledger/economic transition",
    "risk/kill-switch/lease transition",
    "persistence migration/restore/recovery",
    "update authorization/install/rollback"
  ],
  "runtime_session_binding": {
    "upstream_rule": "AuditEvent may exist without RuntimeSession",
    "PRE_CORE": "runtime_session_id is optional and must not be required",
    "CORE_RUNTIME": "runtime_session_id is recorded when the AuditEvent is created while Core is running, according to M0.2 semantics",
    "identity_parent": "DeviceInstallation remains the canonical parent regardless of RuntimeSession presence"
  },
  "writer_authority": {
    "semantic_identity_schema_owner": "M0.2 canonical AuditEvent",
    "append_policy_owner": "M0.12",
    "phases": [
      {
        "phase": "PRE_CORE",
        "trusted_append_path": "privileged product path backed by the existing M0.3 external product provisioning boundary and/or the applicable M0.10 transition owner",
        "allowed_events": [
          "bootstrap/setup",
          "login/PIN/biometric",
          "licensing",
          "API-key management",
          "update",
          "device rebind"
        ],
        "runtime_session_required": false,
        "constraints": [
          "Bootstrapper may transport protected references but cannot mint, accept or elevate bootstrap/security authority",
          "append proves the event only and cannot create the underlying transition authority"
        ]
      },
      {
        "phase": "CORE_RUNTIME",
        "trusted_append_path": "CoreHost",
        "allowed_events": [
          "Core-accepted, denied or completed runtime, security, execution, economic, risk and recovery transitions"
        ],
        "runtime_session_required": true,
        "constraints": [
          "CoreHost remains sole mutable trading-state authority",
          "AuditEvent does not replace the upstream domain fact"
        ]
      },
      {
        "phase": "MAINTENANCE_UPDATE_RECOVERY",
        "trusted_append_path": "Bootstrapper only as the M0.3 maintenance/update participant consuming an already-authorized protected handoff; CoreHost or the applicable M0.10/M0.11 owner records its own transitions",
        "allowed_events": [
          "maintenance/update attempt observations owned by Bootstrapper",
          "Core/recovery transitions owned by their existing authority"
        ],
        "runtime_session_required": false,
        "constraints": [
          "no trading-state authority",
          "no execution authority",
          "no security-policy authority",
          "no StateStore mutation authority outside frozen contracts",
          "no self-granted maintenance authorization"
        ]
      }
    ],
    "direct_writer_non_authorities": [
      "DesktopShell",
      "TrayAgent",
      "raw caller",
      "structured log",
      "metric",
      "UI notification"
    ]
  }
}
```

## `observability_model`

```json
{
  "categories": [
    "STRUCTURED_LOGS",
    "METRICS",
    "TRACES",
    "COMPONENT_STATUS",
    "ADAPTER_STATUS",
    "MARKET_DATA_FRESHNESS",
    "EXECUTION_PATH_HEALTH",
    "PERSISTENCE_HEALTH",
    "RECONCILIATION_HEALTH",
    "SECURITY_RISK_GATE_HEALTH",
    "RESOURCE_RUNTIME_HEALTH",
    "UPDATE_RELEASE_STATE"
  ],
  "observation_rules": [
    "every observation identifies source, component, environment/scope where applicable, observed_at, freshness/expiry and confidence",
    "cardinality and secret-redaction policies apply before export",
    "telemetry outage cannot be interpreted as OK",
    "observation may describe but never mutate upstream authority",
    "cross-environment aggregation is explicitly non-authoritative"
  ],
  "projection_rule": "dashboards, diagnostic bundles, notifications and UI are lossy rebuildable projections",
  "category_contracts": [
    {
      "category": "STRUCTURED_LOGS",
      "owner": "emitting component",
      "source_authority_reference": "diagnostic statement",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "event-local; no readiness freshness",
      "retention_class": "BEST_EFFORT",
      "cardinality_policy": "bounded fields; no user labels",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": false,
      "contributes_to_readiness": false,
      "scope_schema_ref": "observability_model.category_scope_registry.STRUCTURED_LOGS"
    },
    {
      "category": "METRICS",
      "owner": "metrics pipeline",
      "source_authority_reference": "numeric aggregate of validated observations",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "sample timestamp; category policy",
      "retention_class": "BOUNDED",
      "cardinality_policy": "registry labels only; raw IDs forbidden",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": false,
      "contributes_to_readiness": false,
      "scope_schema_ref": "observability_model.category_scope_registry.METRICS"
    },
    {
      "category": "TRACES",
      "owner": "emitting component",
      "source_authority_reference": "diagnostic timing and causality",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "span-local; never readiness evidence",
      "retention_class": "BOUNDED",
      "cardinality_policy": "bounded attributes; raw IDs forbidden",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": false,
      "contributes_to_readiness": false,
      "scope_schema_ref": "observability_model.category_scope_registry.TRACES"
    },
    {
      "category": "COMPONENT_STATUS",
      "owner": "CoreHost health aggregator",
      "source_authority_reference": "M0.3 lifecycle and validated component checks",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per scope",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.COMPONENT_STATUS"
    },
    {
      "category": "ADAPTER_STATUS",
      "owner": "CoreHost adapter owner",
      "source_authority_reference": "M0.6 route/adaptor validation",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per route/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.ADAPTER_STATUS"
    },
    {
      "category": "MARKET_DATA_FRESHNESS",
      "owner": "CoreHost market-data owner",
      "source_authority_reference": "M0.6 route readiness and source data currency",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "M0.6-derived source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per route/instrument/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.MARKET_DATA_FRESHNESS"
    },
    {
      "category": "EXECUTION_PATH_HEALTH",
      "owner": "CoreHost execution owner",
      "source_authority_reference": "M0.5 operability + M0.6 route + M0.9 authority",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per execution scope/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.EXECUTION_PATH_HEALTH"
    },
    {
      "category": "PERSISTENCE_HEALTH",
      "owner": "CoreHost StateStore owner",
      "source_authority_reference": "M0.11 accepted persistence/recovery facts",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per StateStore/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.PERSISTENCE_HEALTH"
    },
    {
      "category": "RECONCILIATION_HEALTH",
      "owner": "CoreHost reconciliation owner",
      "source_authority_reference": "M0.8 reconciliation and M0.11 recovery facts",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per reconciliation scope/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.RECONCILIATION_HEALTH"
    },
    {
      "category": "SECURITY_RISK_GATE_HEALTH",
      "owner": "CoreHost safety owner",
      "source_authority_reference": "M0.9 risk/kill-switch/lease and M0.10 security facts",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per safety scope/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": true,
      "scope_schema_ref": "observability_model.category_scope_registry.SECURITY_RISK_GATE_HEALTH"
    },
    {
      "category": "RESOURCE_RUNTIME_HEALTH",
      "owner": "owning process",
      "source_authority_reference": "M0.3 process lifecycle plus bounded resource probes",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "source/category policy required",
      "retention_class": "CURRENT_PLUS_BOUNDED_HISTORY",
      "cardinality_policy": "one current key per resource class/session",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": true,
      "contributes_to_readiness": false,
      "scope_schema_ref": "observability_model.category_scope_registry.RESOURCE_RUNTIME_HEALTH"
    },
    {
      "category": "UPDATE_RELEASE_STATE",
      "owner": "future privileged update coordinator",
      "source_authority_reference": "future S9D updater authority; diagnostic only in S9C",
      "required_scope": "SEE scope_schema_ref (legacy human summary only; non-authoritative)",
      "freshness_semantics": "not readiness evidence in S9C",
      "retention_class": "BOUNDED",
      "cardinality_policy": "one current key per candidate; untrusted metadata bounded",
      "secret_policy": "M0.10-bound deny raw secrets; allow safe IDs, fingerprints, reason codes, counts, enums and bounded diagnostics",
      "contributes_to_health": false,
      "contributes_to_readiness": false,
      "scope_schema_ref": "observability_model.category_scope_registry.UPDATE_RELEASE_STATE"
    }
  ],
  "observation_envelope": {
    "closed": true,
    "additional_fields": "REJECT",
    "fields": [
      {
        "name": "observation_id",
        "type": "non-empty string",
        "nullable": false,
        "semantics": "M0.12-local correlation/deduplication handle; non-authoritative; not an M0.2 durable entity identity"
      },
      {
        "name": "category",
        "type": "enum",
        "nullable": false,
        "registry": "observability_model.categories"
      },
      {
        "name": "source_component",
        "type": "enum reference",
        "nullable": false,
        "registry": "M0.3 process_roles.name or category-authorized internal component registry"
      },
      {
        "name": "source_instance_id",
        "type": "string",
        "nullable": true,
        "semantics": "runtime_session_id for CoreHost when applicable; otherwise source-local restart/session identity"
      },
      {
        "name": "environment",
        "type": "enum reference",
        "nullable": true,
        "registry": "M0.4 execution_environments.environment_id; required by category/scope policy"
      },
      {
        "name": "scope",
        "type": "closed object of canonical references",
        "nullable": false,
        "semantics": "category-authorized M0.2/M0.5/M0.6/M0.8/M0.9/M0.11 references; no caller-defined keys"
      },
      {
        "name": "source_event_at_utc",
        "type": "RFC3339 UTC string",
        "nullable": true,
        "semantics": "upstream fact time; not receipt time and not ordering authority alone"
      },
      {
        "name": "observed_at_utc",
        "type": "RFC3339 UTC string",
        "nullable": false,
        "semantics": "time probe formed the observation"
      },
      {
        "name": "ingested_at_utc",
        "type": "RFC3339 UTC string",
        "nullable": false,
        "semantics": "trusted ingress receipt time"
      },
      {
        "name": "expires_at_utc",
        "type": "RFC3339 UTC string",
        "nullable": false,
        "semantics": "exclusive validity horizon selected by versioned source/category policy"
      },
      {
        "name": "freshness_policy_id",
        "type": "registered string",
        "nullable": false,
        "semantics": "versioned source/category policy; no global timeout"
      },
      {
        "name": "source_sequence",
        "type": "non-negative integer",
        "nullable": true,
        "semantics": "monotonic only within source_component + source_instance_id"
      },
      {
        "name": "condition",
        "type": "enum",
        "nullable": false,
        "registry": "canonical_vocabulary.condition_states"
      },
      {
        "name": "reason_code",
        "type": "registered string",
        "nullable": false,
        "semantics": "safe machine-readable diagnostic"
      },
      {
        "name": "value",
        "type": "category-closed scalar/object",
        "nullable": true,
        "semantics": "bounded redacted diagnostic; never authority"
      },
      {
        "name": "source_quality",
        "type": "enum",
        "nullable": false,
        "registry": [
          "DIRECT",
          "DERIVED",
          "CACHED"
        ],
        "semantics": "confidence metadata; cannot promote authority"
      },
      {
        "name": "correlation_reference",
        "type": "safe reference",
        "nullable": true,
        "semantics": "correlation only; cannot prove authorization or business outcome"
      }
    ]
  },
  "current_projection": {
    "key_fields": [
      "category",
      "source_component",
      "source_instance_id",
      "environment",
      "canonicalized_scope"
    ],
    "observation_id_is_key": false,
    "replacement": "higher source_sequence in same source session replaces; without sequence, strictly later observed_at_utc replaces after skew validation",
    "history": "optional bounded and lossy",
    "authority": false
  },
  "freshness_and_clock": {
    "freshness_states": [
      "FRESH",
      "STALE"
    ],
    "fresh": "explicit now < expires_at_utc and all timestamp/policy checks pass",
    "expiry": "now >= expires_at_utc derives STALE; displayed condition becomes UNKNOWN with reason OBSERVATION_EXPIRED; prior OK is never retained",
    "policy": "horizon is registered and versioned per source/category; no arbitrary global timeout",
    "ordering": "source sequence within source session is primary when present; wall clock alone never proves global ordering",
    "validation": {
      "future_timestamp": "reject observed/source event time beyond policy skew bound",
      "clock_moved_backwards": "reject older observed_at in same key unless a new source_instance_id establishes restart",
      "older_source_timestamp": "without source_sequence reject SOURCE_EVENT_REGRESSION; increasing sequence may establish legal later sample with older source-event time and surfaces source ordering diagnostics",
      "expires_before_or_equal_observed": "reject",
      "unreasonable_expiry": "reject if horizon differs from freshness_policy_id",
      "ingested_before_observed": "allowed only within policy skew bound; otherwise reject",
      "older_observed_same_current_key": "reject CLOCK_REGRESSION regardless of increasing source_sequence; only new source_instance_id creates a new key/context"
    }
  },
  "source_sequence_contract": {
    "scope": [
      "source_component",
      "source_instance_id"
    ],
    "global_sequence": false,
    "duplicate_same_sequence_same_content": "idempotent replay; current projection unchanged",
    "duplicate_same_sequence_conflicting_content": "reject CONTRACT_INCONSISTENT",
    "gap": "accept and surface SEQUENCE_GAP/DEGRADED confidence; never synthesize missing facts",
    "regression": "reject",
    "restart_reset": "requires new non-null source_instance_id; sequence may restart only in new session",
    "validation_order_for_sequenced_observation": [
      "lookup exact source component + source instance + sequence",
      "exact content => historical replay return without current-clock validation or mutation",
      "conflicting content => DUPLICATE_SEQUENCE_CONFLICT",
      "only unseen sequence => regression, current clock, source-event, gap and replacement validation"
    ]
  },
  "structured_log_schema": {
    "required": [
      "timestamp_utc",
      "level",
      "component",
      "event_code",
      "safe_fields"
    ],
    "optional": [
      "source_instance_id",
      "environment",
      "correlation_reference",
      "narrative_message"
    ],
    "levels": [
      "DEBUG",
      "INFO",
      "WARNING",
      "ERROR",
      "CRITICAL"
    ],
    "safe_fields": "closed event-code schema; arbitrary dictionaries rejected",
    "authority": false,
    "narrative_message": "diagnostic only; never parsed back into state",
    "delivery": "best effort",
    "additional_fields": "REJECT",
    "event_registry": {
      "COMPONENT_PROBE": {
        "required_safe_fields": [
          "condition"
        ],
        "optional_safe_fields": [
          "reason_code"
        ],
        "field_domains": {
          "condition": [
            "UNKNOWN",
            "OK",
            "DEGRADED",
            "BLOCKED"
          ],
          "reason_code": "SAFE_CODE"
        }
      }
    }
  },
  "metric_schema": {
    "required": [
      "metric_name",
      "value",
      "unit",
      "labels",
      "scope",
      "timestamp_utc"
    ],
    "metric_name": "closed versioned registry",
    "value_types": [
      "INTEGER",
      "DECIMAL"
    ],
    "unit": "registered per metric",
    "labels": "closed per-metric keys and values",
    "cardinality": "finite registered dimensions only; user-controlled labels and raw entity IDs forbidden",
    "authority": false,
    "additional_fields": "REJECT",
    "registry": {
      "component_probe_total": {
        "unit": "count",
        "label_schema": {
          "component": [
            "core_host",
            "tray_agent",
            "desktop_shell",
            "bootstrapper"
          ],
          "condition": [
            "UNKNOWN",
            "OK",
            "DEGRADED",
            "BLOCKED"
          ]
        },
        "scope_schema_ref": "COMPONENT_STATUS"
      }
    },
    "value_type": "finite integer or decimal; bool forbidden",
    "scope_label_consistency": {
      "component_probe_total": "labels.component MUST equal scope.component; scope is validated through category_scope_registry[scope_schema_ref], not hardcoded to core_host"
    }
  },
  "trace_schema": {
    "purpose": "diagnostic timing and causality",
    "required": [
      "trace_id",
      "span_id",
      "component",
      "operation_code",
      "started_at_utc",
      "ended_at_utc",
      "safe_attributes"
    ],
    "optional": [
      "parent_span_id",
      "correlation_reference",
      "causation_reference",
      "source_instance_id",
      "environment"
    ],
    "prohibitions": [
      "prove authorization",
      "prove order acceptance",
      "replace AuditEvent",
      "replace M0.7 event"
    ],
    "authority": false,
    "additional_fields": "REJECT",
    "operation_registry": {
      "OBSERVATION_INGEST": {
        "required_attributes": [
          "category"
        ],
        "optional_attributes": [
          "result_code"
        ],
        "attribute_domains": {
          "category": [
            "STRUCTURED_LOGS",
            "METRICS",
            "TRACES",
            "COMPONENT_STATUS",
            "ADAPTER_STATUS",
            "MARKET_DATA_FRESHNESS",
            "EXECUTION_PATH_HEALTH",
            "PERSISTENCE_HEALTH",
            "RECONCILIATION_HEALTH",
            "SECURITY_RISK_GATE_HEALTH",
            "RESOURCE_RUNTIME_HEALTH",
            "UPDATE_RELEASE_STATE"
          ],
          "result_code": "SAFE_CODE"
        }
      }
    },
    "id_syntax": "16..64 lowercase hex",
    "parent_relation": "parent_span_id absent or valid span id different from span_id"
  },
  "redaction_and_export": {
    "upstream": "M0.10 identity_device_authentication_and_secrets.json",
    "forbidden_raw": [
      "password",
      "PIN",
      "API key",
      "API secret",
      "token",
      "private key",
      "biometric template",
      "biometric assertion",
      "protected bootstrap material"
    ],
    "allowed": [
      "safe canonical IDs",
      "fingerprints",
      "reason codes",
      "counts",
      "state enums",
      "bounded diagnostics"
    ],
    "applies_to": [
      "observations",
      "structured logs",
      "metrics",
      "traces",
      "diagnostic bundles"
    ],
    "secret_like_input": "reject before storage/export",
    "diagnostic_bundle": "lossy redacted bounded-history projection with cardinality limits; never backup of business or audit state"
  },
  "source_component_registry": {
    "authority": "M0.3 process_roles.name only",
    "values": [
      "core_host",
      "tray_agent",
      "desktop_shell",
      "bootstrapper"
    ],
    "internal_component_registry": null
  },
  "source_instance_contract": {
    "CORE_HOST_RUNTIME_SESSION": {
      "discriminator": "source_component == core_host",
      "field": "source_instance_id",
      "source": {
        "milestone": "M0.2",
        "pointer": "/entity_kinds[canonical_name=RuntimeSession]",
        "id_field": "runtime_session_id",
        "id_prefix": "run",
        "syntax": "canonical lowercase UUIDv7"
      }
    },
    "NON_CORE_LOCAL_SESSION": {
      "discriminator": "source_component != core_host",
      "type": "M0.12-local non-authoritative source-session handle",
      "syntax": "1..96 SAFE_HANDLE characters [A-Za-z0-9][A-Za-z0-9._:-]*",
      "restart_semantics": "new handle required for sequence reset",
      "durable_domain_entity": false,
      "claims_M02_prefix": false
    }
  },
  "category_scope_registry": {
    "STRUCTURED_LOGS": {
      "required_keys": [],
      "optional_keys": [
        "component"
      ],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "OPTIONAL",
      "field_contracts": {
        "component": {
          "milestone": "M0.3",
          "pointer": "/process_roles/name",
          "identity_or_type": "enum"
        }
      },
      "allowed_source_components": [
        "core_host",
        "tray_agent",
        "desktop_shell",
        "bootstrapper"
      ],
      "source_instance_policy": "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_OPTIONAL"
    },
    "METRICS": {
      "required_keys": [],
      "optional_keys": [
        "component",
        "resource_class"
      ],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "OPTIONAL",
      "field_contracts": {
        "component": {
          "milestone": "M0.3",
          "pointer": "/process_roles/name",
          "identity_or_type": "enum"
        },
        "resource_class": {
          "milestone": "M0.12",
          "pointer": "/observability_model/resource_class_registry",
          "identity_or_type": "enum"
        }
      },
      "allowed_source_components": [
        "core_host",
        "tray_agent",
        "desktop_shell"
      ],
      "source_instance_policy": "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_OPTIONAL"
    },
    "TRACES": {
      "required_keys": [],
      "optional_keys": [
        "component"
      ],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "OPTIONAL",
      "field_contracts": {
        "component": {
          "milestone": "M0.3",
          "pointer": "/process_roles/name",
          "identity_or_type": "enum"
        }
      },
      "allowed_source_components": [
        "core_host",
        "tray_agent",
        "desktop_shell",
        "bootstrapper"
      ],
      "source_instance_policy": "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_OPTIONAL"
    },
    "COMPONENT_STATUS": {
      "required_keys": [
        "component"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "OPTIONAL",
      "field_contracts": {
        "component": {
          "milestone": "M0.3",
          "pointer": "/process_roles/name",
          "identity_or_type": "enum"
        }
      },
      "allowed_source_components": [
        "core_host",
        "tray_agent",
        "desktop_shell",
        "bootstrapper"
      ],
      "source_instance_policy": "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_REQUIRED"
    },
    "ADAPTER_STATUS": {
      "required_keys": [],
      "optional_keys": [
        "exchange_account_id",
        "market_data_route_id",
        "execution_route_id"
      ],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "REQUIRED",
      "field_contracts": {
        "exchange_account_id": {
          "milestone": "M0.5",
          "pointer": "/exchange_account_contract",
          "identity_or_type": "xacc"
        },
        "market_data_route_id": {
          "milestone": "M0.6",
          "pointer": "/entity_registry/MarketDataRoute",
          "identity_or_type": "mdr"
        },
        "execution_route_id": {
          "milestone": "M0.6",
          "pointer": "/entity_registry/ExecutionRoute",
          "identity_or_type": "xroute"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST",
      "at_least_one_of": [
        "exchange_account_id",
        "market_data_route_id",
        "execution_route_id"
      ]
    },
    "MARKET_DATA_FRESHNESS": {
      "required_keys": [
        "market_data_route_id",
        "instrument_id"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "REQUIRED",
      "field_contracts": {
        "market_data_route_id": {
          "milestone": "M0.6",
          "pointer": "/entity_registry/MarketDataRoute",
          "identity_or_type": "mdr"
        },
        "instrument_id": {
          "milestone": "M0.5",
          "pointer": "/instrument_contract",
          "identity_or_type": "instr"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST"
    },
    "EXECUTION_PATH_HEALTH": {
      "required_keys": [
        "exchange_account_id",
        "instrument_id",
        "execution_route_id"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "REQUIRED",
      "field_contracts": {
        "exchange_account_id": {
          "milestone": "M0.5",
          "pointer": "/exchange_account_contract",
          "identity_or_type": "xacc"
        },
        "instrument_id": {
          "milestone": "M0.5",
          "pointer": "/instrument_contract",
          "identity_or_type": "instr"
        },
        "execution_route_id": {
          "milestone": "M0.6",
          "pointer": "/entity_registry/ExecutionRoute",
          "identity_or_type": "xroute"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST"
    },
    "PERSISTENCE_HEALTH": {
      "required_keys": [
        "device_installation_id",
        "state_store_identity_fingerprint_sha256"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "FORBIDDEN",
      "field_contracts": {
        "device_installation_id": {
          "milestone": "M0.2",
          "pointer": "/entity_kinds[canonical_name=DeviceInstallation]",
          "identity_or_type": "dev"
        },
        "state_store_identity_fingerprint_sha256": {
          "milestone": "M0.11",
          "pointer": "/state_store_identity_contract",
          "identity_or_type": "lowercase_hex_64"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST"
    },
    "RECONCILIATION_HEALTH": {
      "required_keys": [
        "exchange_account_id",
        "portfolio_id"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "REQUIRED",
      "field_contracts": {
        "exchange_account_id": {
          "milestone": "M0.5",
          "pointer": "/exchange_account_contract",
          "identity_or_type": "xacc"
        },
        "portfolio_id": {
          "milestone": "M0.2",
          "pointer": "/entity_kinds[canonical_name=Portfolio]",
          "identity_or_type": "port"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST"
    },
    "SECURITY_RISK_GATE_HEALTH": {
      "required_keys": [
        "exchange_account_id",
        "instrument_id",
        "execution_route_id"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "REQUIRED",
      "field_contracts": {
        "exchange_account_id": {
          "milestone": "M0.5",
          "pointer": "/exchange_account_contract",
          "identity_or_type": "xacc"
        },
        "instrument_id": {
          "milestone": "M0.5",
          "pointer": "/instrument_contract",
          "identity_or_type": "instr"
        },
        "execution_route_id": {
          "milestone": "M0.6",
          "pointer": "/entity_registry/ExecutionRoute",
          "identity_or_type": "xroute"
        }
      },
      "allowed_source_components": [
        "core_host"
      ],
      "source_instance_policy": "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST"
    },
    "RESOURCE_RUNTIME_HEALTH": {
      "required_keys": [
        "component",
        "resource_class"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "OPTIONAL",
      "field_contracts": {
        "component": {
          "milestone": "M0.3",
          "pointer": "/process_roles/name",
          "identity_or_type": "enum"
        },
        "resource_class": {
          "milestone": "M0.12",
          "pointer": "/observability_model/resource_class_registry",
          "identity_or_type": "enum"
        }
      },
      "allowed_source_components": [
        "core_host",
        "tray_agent",
        "desktop_shell",
        "bootstrapper"
      ],
      "source_instance_policy": "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_REQUIRED"
    },
    "UPDATE_RELEASE_STATE": {
      "required_keys": [
        "release_candidate_reference"
      ],
      "optional_keys": [],
      "forbidden_keys_policy": "ALL_KEYS_NOT_IN_REQUIRED_OR_OPTIONAL",
      "environment_requirement": "FORBIDDEN",
      "field_contracts": {
        "release_candidate_reference": {
          "milestone": "M0.12",
          "pointer": "future S9D local candidate reference",
          "identity_or_type": "safe_handle"
        }
      },
      "allowed_source_components": [
        "bootstrapper"
      ],
      "source_instance_policy": "LOCAL_SESSION_REQUIRED"
    }
  },
  "resource_class_registry": [
    "CPU",
    "MEMORY",
    "DISK",
    "IPC",
    "THREAD_PROGRESS"
  ],
  "category_value_schemas": {
    "STRUCTURED_LOGS": {
      "allowed_keys": [
        "emitted_count"
      ],
      "required_keys": [],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "METRICS": {
      "allowed_keys": [
        "sample_count"
      ],
      "required_keys": [],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "TRACES": {
      "allowed_keys": [
        "span_count"
      ],
      "required_keys": [],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "COMPONENT_STATUS": {
      "allowed_keys": [
        "progress_counter"
      ],
      "required_keys": [
        "progress_counter"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "ADAPTER_STATUS": {
      "allowed_keys": [
        "transport_state"
      ],
      "required_keys": [
        "transport_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "MARKET_DATA_FRESHNESS": {
      "allowed_keys": [
        "last_data_at_utc",
        "sequence_state"
      ],
      "required_keys": [
        "last_data_at_utc",
        "sequence_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "EXECUTION_PATH_HEALTH": {
      "allowed_keys": [
        "path_state"
      ],
      "required_keys": [
        "path_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "PERSISTENCE_HEALTH": {
      "allowed_keys": [
        "integrity_state",
        "recovery_required"
      ],
      "required_keys": [
        "integrity_state",
        "recovery_required"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "RECONCILIATION_HEALTH": {
      "allowed_keys": [
        "reconciliation_state"
      ],
      "required_keys": [
        "reconciliation_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "SECURITY_RISK_GATE_HEALTH": {
      "allowed_keys": [
        "risk_allowed",
        "kill_switch_inactive",
        "lease_state"
      ],
      "required_keys": [
        "risk_allowed",
        "kill_switch_inactive",
        "lease_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "RESOURCE_RUNTIME_HEALTH": {
      "allowed_keys": [
        "usage_ratio"
      ],
      "required_keys": [
        "usage_ratio"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    },
    "UPDATE_RELEASE_STATE": {
      "allowed_keys": [
        "candidate_state"
      ],
      "required_keys": [
        "candidate_state"
      ],
      "additional_keys": "REJECT",
      "max_serialized_bytes": 2048,
      "scalar_types": [
        "string",
        "integer",
        "decimal",
        "boolean",
        "null"
      ],
      "nested_objects": false,
      "safe_value_policy": "SAFE_SCALAR_NO_SECRETS"
    }
  },
  "safe_field_contract": {
    "safe_handle": {
      "min_length": 1,
      "max_length": 128,
      "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]*$",
      "control_characters": "REJECT"
    },
    "safe_code": {
      "min_length": 1,
      "max_length": 64,
      "pattern": "^[A-Z][A-Z0-9_]*$"
    },
    "canonical_uuidv7_id": {
      "pattern": "^<upstream-prefix>_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    },
    "correlation_reference": {
      "closed_variants": {
        "LOCAL": {
          "exact_keys": [
            "kind",
            "value"
          ],
          "kind": "LOCAL",
          "value_schema": "safe_handle",
          "authority": false
        },
        "CANONICAL": {
          "exact_keys": [
            "kind",
            "entity",
            "value"
          ],
          "kind": "CANONICAL",
          "entity_registry": [
            "RuntimeSession",
            "ExchangeAccount",
            "Instrument",
            "MarketDataRoute",
            "ExecutionRoute"
          ],
          "value_schema": "selected upstream identity"
        }
      }
    },
    "secret_rejection": "schema-first: closed keys/types/enums/identities; then shared recursive prohibited-secret classifier as defense in depth"
  },
  "freshness_policy_contract": {
    "closed": true,
    "required_fields": [
      "policy_id",
      "version",
      "category",
      "source_class",
      "validity_horizon_seconds",
      "allowed_observed_future_skew_seconds",
      "allowed_source_event_future_skew_seconds",
      "allowed_ingest_before_observed_skew_seconds"
    ],
    "additional_fields": "REJECT",
    "field_contracts": {
      "policy_id": "SAFE_CODE",
      "version": "positive non-boolean integer",
      "category": "observability categories enum",
      "source_class": "source_component registry or registered class",
      "validity_horizon_seconds": "positive non-boolean integer",
      "allowed_observed_future_skew_seconds": "non-negative non-boolean integer",
      "allowed_source_event_future_skew_seconds": "non-negative non-boolean integer",
      "allowed_ingest_before_observed_skew_seconds": "non-negative non-boolean integer"
    },
    "selection_key": [
      "category",
      "source_class",
      "policy_id",
      "version"
    ],
    "deployment_values": "injected registered configuration; no global duration",
    "expiry_derivation": "expires_at_utc == observed_at_utc + validity_horizon_seconds"
  },
  "sequence_result_contract": {
    "accepted_fields": [
      "observation",
      "replayed",
      "sequence_gap",
      "effective_condition",
      "effective_reason_codes"
    ],
    "gap": "sequence > prior + 1 => accepted, sequence_gap=true, effective condition from sequence_gap_condition_composition[raw condition], append SEQUENCE_GAP; never weaken UNKNOWN/BLOCKED",
    "old_exact_replay": "lookup precedes current-clock checks; exact historical acceptance returns replayed=true and explicit current remains newer; no mutation"
  },
  "sequence_gap_condition_composition": {
    "OK": "DEGRADED",
    "DEGRADED": "DEGRADED",
    "UNKNOWN": "UNKNOWN",
    "BLOCKED": "BLOCKED",
    "invariant": "diagnostic uncertainty may preserve or restrict; never make a condition less restrictive"
  },
  "source_instance_policy_interpreter": {
    "RUNTIME_SESSION_REQUIRED_FOR_CORE_HOST": {
      "core_host": "CANONICAL_RUNTIME_SESSION_REQUIRED",
      "non_core": "FORBIDDEN_BY_CATEGORY_SOURCE_POLICY"
    },
    "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_REQUIRED": {
      "core_host": "CANONICAL_RUNTIME_SESSION_REQUIRED",
      "non_core": "LOCAL_SAFE_SESSION_REQUIRED"
    },
    "RUNTIME_SESSION_FOR_CORE_HOST_ELSE_LOCAL_SESSION_OPTIONAL": {
      "core_host": "CANONICAL_RUNTIME_SESSION_REQUIRED",
      "non_core": "LOCAL_SAFE_SESSION_OPTIONAL"
    },
    "LOCAL_SESSION_REQUIRED": {
      "core_host": "FORBIDDEN_BY_CATEGORY_SOURCE_POLICY",
      "non_core": "LOCAL_SAFE_SESSION_REQUIRED"
    }
  },
  "executable_authority": {
    "phase": "S9C_C14",
    "production_module": "bot_core.observability.authority",
    "production_types": [
      "ObservationAuthority",
      "FrozenEnvironmentRegistryBinding",
      "ObservationAuthorityCarrier",
      "InMemoryObservationAuthorityCarrier",
      "AtomicObservationAuthorityState",
      "CanonicalObservation",
      "ObservationKey",
      "AcceptedObservation",
      "EffectiveCurrentObservation",
      "FreshnessPolicy"
    ],
    "production_apis": [
      "ObservationAuthority.compose",
      "ObservationAuthority.resolve_historical_acceptance",
      "ObservationAuthority.resolve_current",
      "ObservationAuthority.consume_effective_current",
      "owner composition return capability.publish (non-exported writer type)"
    ],
    "executable_categories": [
      "MARKET_DATA_FRESHNESS",
      "EXECUTION_PATH_HEALTH"
    ],
    "oracle_only_categories": [
      "STRUCTURED_LOGS",
      "METRICS",
      "TRACES",
      "COMPONENT_STATUS",
      "ADAPTER_STATUS",
      "PERSISTENCE_HEALTH",
      "RECONCILIATION_HEALTH",
      "SECURITY_RISK_GATE_HEALTH",
      "RESOURCE_RUNTIME_HEALTH",
      "UPDATE_RELEASE_STATE"
    ],
    "membership_authority": "EXECUTABLE_CARRIER_OWNED_ATOMIC_MEMBERSHIP",
    "effective_current_authority": "EXECUTABLE_EXACT_KEY_QUERY_WITH_QUERY_TIME_EXPIRY_PROJECTION",
    "historical_lookup": "EXECUTABLE_ACCEPTANCE_ID_LOOKUP_AFTER_SUPERSESSION_AND_RESTART",
    "currentness_fence": "EXECUTABLE_CARRIER_WIDE_AUTHORITY_FENCE_HELD_THROUGH_DOWNSTREAM_CONSUMER_PUBLICATION",
    "source_producer_authenticity": "OPEN_SOURCE_PRODUCER_AUTHENTICITY",
    "physical_durable_m1_carrier": "OPEN_IN_MEMORY_REFERENCE_CARRIER_ONLY",
    "s9d_adapter_integration": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN",
    "retention_contract": {
      "presentation_retention": "CURRENT_PLUS_BOUNDED_HISTORY",
      "authority_provenance_retention": "RETAIN_WHILE_DURABLE_DOWNSTREAM_PROVENANCE_MAY_BE_RESTORED",
      "resolution": "NO_CONFLICT_PRESENTATION_RETENTION_DOES_NOT_DELETE_REFERENCED_AUTHORITY_MEMBERSHIP"
    },
    "writer_capability_ownership": "NON_EXPORTED_CAPABILITY_MINTED_ONLY_BY_OBSERVATION_AUTHORITY_COMPOSE",
    "restore_projection_validation": "EXACT_RECONSTRUCTION_OF_CURRENT_REPLAY_LAST_SEQUENCE_AND_ACCEPTANCE_ID_FROM_ACCEPTED_HISTORY",
    "c1_differential_parity": "EXECUTABLE_CATEGORIES_DIFFERENTIALLY_TESTED_AGAINST_INDEPENDENT_OBSERVATION_REFERENCE",
    "c1_lexical_parity": "EXACT_SAFE_HANDLE_128_AND_SAFE_CODE_64",
    "canonical_environment_authority": "COMPILE_TIME_PINNED_PROJECTION_WITH_ARCHITECTURE_PARITY_TO_FROZEN_M0.4",
    "historical_freshness_policy_binding": "IMMUTABLE_FULL_POLICY_SNAPSHOT_VERSION_AND_SHA256_FINGERPRINT",
    "historical_policy_restore": "INDEPENDENT_OF_CURRENT_ACTIVE_POLICY_VERSION",
    "historical_freshness_policy_validation": "SAME_CANONICAL_VALIDATOR_AS_LIVE_CONFIGURATION_POLICY_ID_EXACTLY_BOUND_TO_OBSERVATION",
    "restore_transition_validation": "EXACT_REPLAY_OF_LIVE_CLOCK_SOURCE_EVENT_SEQUENCE_TRANSITION_RULES",
    "canonical_environment_binding": "IMMUTABLE_OWNED_FROZENSET_PINNED_TO_FROZEN_M0.4",
    "restore_primitive_shape_validation": "EXACT_TYPES_NO_BOOL_INT_ALIASING",
    "sequence_replay_content_identity": "CANONICAL_JSON_FINGERPRINT_EXACT_SCALAR_TYPES_MAPPING_ORDER_INDEPENDENT_HISTORICAL_POLICY",
    "sequence_replay_validation_order": "PRE_REPLAY_CANONICAL_PAYLOAD_THEN_KNOWN_SEQUENCE_FINGERPRINT_THEN_UNSEEN_TEMPORAL_FRESHNESS_TRANSITION",
    "observation_ingress_primitive_validation": "EXACT_TYPES_BEFORE_HASH_BASED_LOOKUP_NO_RAW_HASHABILITY_EXCEPTIONS",
    "canonical_json_scalar_validation": "STRICT_JSON_FINITE_NUMBERS_ONLY_ALLOW_NAN_FALSE",
    "canonical_timestamp_validation": "STRICT_RFC3339_UTC_LEXICAL_GATE_NO_PYTHON_ISO8601_SUPERSET",
    "historical_acceptance_identity": "REVISION_ACCEPTED_AT_CONTENT_POLICY_BOUND",
    "numeric_representability": "VALUE_SOURCE_SEQUENCE_AND_POLICY_NUMBERS_FAIL_CLOSED_BEFORE_CANONICAL_SERIALIZATION_OR_TIMEDELTA_LIMITS"
  },
  "s9d_adapter_integration": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
}
```

## `health_readiness_model`

```json
{
  "dimensions": {
    "LIVENESS": "process responds or makes progress; does not imply safe operation",
    "HEALTH": "component condition across relevant checks",
    "READINESS": "specific capability may be attempted only when all upstream lifecycle/capability gates independently authorize it"
  },
  "condition_states": {
    "UNKNOWN": "no current trusted evidence; neither OK nor DEGRADED; required safety dependency fails closed",
    "OK": "fresh check satisfies its declared dependency only; never grants upstream authority",
    "DEGRADED": "fresh reduced-quality condition; blocks only capabilities whose matrix dependency maps DEGRADED to BLOCKED",
    "BLOCKED": "fresh explicit condition forbids affected capability"
  },
  "composition": [
    "never collapse dimensions into one healthy boolean",
    "readiness is per capability, environment and scope",
    "M0.3 startup readiness and process lifecycle are inputs, not replaced",
    "M0.4 ProductCapabilities and environment_readiness remain authority",
    "UNKNOWN safety evidence is BLOCKED for the affected capability",
    "telemetry cannot promote readiness"
  ],
  "composition_rules": {
    "liveness": "response/progress evidence only; does not imply safe, healthy, or ready",
    "health": "component/dependency condition composed from required and optional checks; required UNKNOWN/STALE gives UNKNOWN, required BLOCKED gives BLOCKED, relevant degradation gives DEGRADED; capability context is not inferred",
    "readiness": "capability/environment/scope projection only; upstream authority AND every required fresh dependency",
    "required_checks": "all evaluated; missing/stale never positive",
    "optional_checks": "may degrade component health but do not block a capability unless that capability matrix marks them required",
    "not_worst_enum": "component health and each capability readiness are independently composed"
  },
  "observation_ingestion": {
    "steps": [
      "validate closed envelope and category source/scope policy",
      "validate environment and canonical references",
      "reject secret-like content",
      "validate timestamps and versioned horizon",
      "apply session-scoped sequence rules",
      "idempotently accept exact replay or replace deterministic current key",
      "derive expiry using explicit now",
      "recompose component health and capability readiness"
    ],
    "reject_codes": [
      "UNKNOWN_CATEGORY",
      "WRONG_SOURCE",
      "WRONG_SCOPE",
      "ILLEGAL_ENVIRONMENT",
      "MALFORMED_TIMESTAMP",
      "FUTURE_TIMESTAMP",
      "INVALID_EXPIRY",
      "UNKNOWN_FRESHNESS_POLICY",
      "SEQUENCE_REGRESSION",
      "DUPLICATE_SEQUENCE_CONFLICT",
      "SECRET_CONTENT",
      "UNKNOWN_CONDITION"
    ]
  },
  "readiness_dependency_matrix": [
    {
      "readiness_id": "PAPER_OPERATION",
      "capability_authority": {
        "milestone": "M0.4",
        "pointer": "/ProductCapabilities/current_edition_capability_policy",
        "selector": {
          "capability": "PAPER_LOCAL_SIMULATION",
          "environment": "PAPER"
        },
        "selector_family": "CAPABILITY_ENVIRONMENT"
      },
      "allowed_environments": [
        "PAPER"
      ],
      "scope_schema": {
        "required_keys": [
          "exchange_account_id",
          "instrument_id",
          "execution_route_id"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m03_ready",
          "milestone": "M0.3",
          "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m04_capability_allowed",
          "milestone": "M0.4",
          "pointer": "/ProductCapabilities/current_edition_capability_policy",
          "scope_binding": "CAPABILITY_SELECTOR"
        },
        {
          "check_id": "m05_account_operable",
          "milestone": "M0.5",
          "pointer": "/current_edition_account_operability_policy",
          "scope_binding": "REQUEST_ACCOUNT"
        },
        {
          "check_id": "m05_instrument_operable",
          "milestone": "M0.5",
          "pointer": "/instrument_contract/metadata_vs_tradability",
          "scope_binding": "REQUEST_INSTRUMENT"
        },
        {
          "check_id": "m06_route_ready",
          "milestone": "M0.6",
          "pointer": "/route_readiness_contract",
          "scope_binding": "REQUEST_ROUTES"
        },
        {
          "check_id": "m11_persistence_accepted",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "COMPONENT_STATUS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "ALLOW",
          "scope_binding": {
            "mode": "FIXED_SCOPE",
            "fixed_scope": {
              "component": "core_host"
            }
          }
        },
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "TESTNET_PRIVATE_EXECUTION",
      "capability_authority": {
        "milestone": "M0.4",
        "pointer": "/ProductCapabilities/current_edition_capability_policy",
        "selector": {
          "capability": "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
          "environment": "TESTNET"
        },
        "selector_family": "CAPABILITY_ENVIRONMENT"
      },
      "allowed_environments": [
        "TESTNET"
      ],
      "scope_schema": {
        "required_keys": [
          "exchange_account_id",
          "instrument_id",
          "execution_route_id",
          "market_data_route_id",
          "portfolio_id"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m03_ready",
          "milestone": "M0.3",
          "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m04_capability_allowed",
          "milestone": "M0.4",
          "pointer": "/ProductCapabilities/current_edition_capability_policy",
          "scope_binding": "CAPABILITY_SELECTOR"
        },
        {
          "check_id": "m05_account_operable",
          "milestone": "M0.5",
          "pointer": "/current_edition_account_operability_policy",
          "scope_binding": "REQUEST_ACCOUNT"
        },
        {
          "check_id": "m05_instrument_operable",
          "milestone": "M0.5",
          "pointer": "/instrument_contract/metadata_vs_tradability",
          "scope_binding": "REQUEST_INSTRUMENT"
        },
        {
          "check_id": "m06_route_ready",
          "milestone": "M0.6",
          "pointer": "/route_readiness_contract",
          "scope_binding": "REQUEST_ROUTES"
        },
        {
          "check_id": "m09_risk_allowed",
          "milestone": "M0.9",
          "pointer": "/risk_decision_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m09_kill_switch_inactive",
          "milestone": "M0.9",
          "pointer": "/kill_switch_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m09_execution_lease_valid",
          "milestone": "M0.9",
          "pointer": "/execution_lease_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m11_persistence_accepted",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "ADAPTER_STATUS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "market_data_route_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "MARKET_DATA_FRESHNESS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "market_data_route_id",
              "instrument_id"
            ]
          }
        },
        {
          "category": "EXECUTION_PATH_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "instrument_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "SECURITY_RISK_GATE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "instrument_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        },
        {
          "category": "RECONCILIATION_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "portfolio_id"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "LIVE_EXECUTION",
      "capability_authority": {
        "milestone": "M0.4",
        "pointer": "/ProductCapabilities/current_edition_capability_policy",
        "selector": {
          "field": "live_allowed_in_current_edition",
          "equals": true,
          "environment": "LIVE"
        },
        "selector_family": "FIELD_EQUALS_ENVIRONMENT"
      },
      "allowed_environments": [
        "LIVE"
      ],
      "scope_schema": {
        "required_keys": [
          "exchange_account_id",
          "instrument_id",
          "execution_route_id",
          "market_data_route_id",
          "portfolio_id"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m03_ready",
          "milestone": "M0.3",
          "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m04_capability_allowed",
          "milestone": "M0.4",
          "pointer": "/ProductCapabilities/current_edition_capability_policy",
          "scope_binding": "CAPABILITY_SELECTOR"
        },
        {
          "check_id": "m05_account_operable",
          "milestone": "M0.5",
          "pointer": "/current_edition_account_operability_policy",
          "scope_binding": "REQUEST_ACCOUNT"
        },
        {
          "check_id": "m05_instrument_operable",
          "milestone": "M0.5",
          "pointer": "/instrument_contract/metadata_vs_tradability",
          "scope_binding": "REQUEST_INSTRUMENT"
        },
        {
          "check_id": "m06_route_ready",
          "milestone": "M0.6",
          "pointer": "/route_readiness_contract",
          "scope_binding": "REQUEST_ROUTES"
        },
        {
          "check_id": "m09_risk_allowed",
          "milestone": "M0.9",
          "pointer": "/risk_decision_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m09_kill_switch_inactive",
          "milestone": "M0.9",
          "pointer": "/kill_switch_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m09_execution_lease_valid",
          "milestone": "M0.9",
          "pointer": "/execution_lease_contract",
          "scope_binding": "REQUEST_EXECUTION_SCOPE"
        },
        {
          "check_id": "m11_persistence_accepted",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "ADAPTER_STATUS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "market_data_route_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "MARKET_DATA_FRESHNESS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "market_data_route_id",
              "instrument_id"
            ]
          }
        },
        {
          "category": "EXECUTION_PATH_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "instrument_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "SECURITY_RISK_GATE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "instrument_id",
              "execution_route_id"
            ]
          }
        },
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "MARKET_DATA_CONSUMPTION",
      "capability_authority": {
        "milestone": "M0.6",
        "pointer": "/route_readiness_contract",
        "selector": {
          "route_kind": "MARKET_DATA"
        },
        "selector_family": "ROUTE_KIND"
      },
      "allowed_environments": [
        "PAPER",
        "TESTNET",
        "LIVE"
      ],
      "scope_schema": {
        "required_keys": [
          "market_data_route_id",
          "instrument_id"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m04_capability_allowed",
          "milestone": "M0.4",
          "pointer": "/ProductCapabilities/current_edition_capability_policy",
          "scope_binding": "CAPABILITY_SELECTOR"
        },
        {
          "check_id": "m05_instrument_operable",
          "milestone": "M0.5",
          "pointer": "/instrument_contract/metadata_vs_tradability",
          "scope_binding": "REQUEST_INSTRUMENT"
        },
        {
          "check_id": "m06_route_ready",
          "milestone": "M0.6",
          "pointer": "/route_readiness_contract",
          "scope_binding": "REQUEST_ROUTES"
        }
      ],
      "required_observations": [
        {
          "category": "ADAPTER_STATUS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "market_data_route_id"
            ]
          }
        },
        {
          "category": "MARKET_DATA_FRESHNESS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "market_data_route_id",
              "instrument_id"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "CORE_STARTUP",
      "capability_authority": {
        "milestone": "M0.3",
        "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
        "selector": {
          "state": "READY"
        },
        "selector_family": "STARTUP_STATE"
      },
      "allowed_environments": [
        "NONE"
      ],
      "scope_schema": {
        "required_keys": [
          "device_installation_id",
          "state_store_identity_fingerprint_sha256"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m03_ready",
          "milestone": "M0.3",
          "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m03_single_instance",
          "milestone": "M0.3",
          "pointer": "/single_instance_policy",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m11_persistence_accepted",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "COMPONENT_STATUS",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "ALLOW",
          "scope_binding": {
            "mode": "FIXED_SCOPE",
            "fixed_scope": {
              "component": "core_host"
            }
          }
        },
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "PERSISTENCE_MUTATION",
      "capability_authority": {
        "milestone": "M0.11",
        "pointer": "/transaction_protocol",
        "selector": {
          "accepted_state_required": true
        },
        "selector_family": "ACCEPTED_STATE_REQUIRED"
      },
      "allowed_environments": [
        "NONE",
        "PAPER",
        "TESTNET"
      ],
      "scope_schema": {
        "required_keys": [
          "device_installation_id",
          "state_store_identity_fingerprint_sha256"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m03_single_instance",
          "milestone": "M0.3",
          "pointer": "/single_instance_policy",
          "scope_binding": "CURRENT_RUNTIME_CONTEXT"
        },
        {
          "check_id": "m11_persistence_accepted",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    },
    {
      "readiness_id": "RECONCILIATION_RECOVERY",
      "capability_authority": {
        "milestone": "M0.8",
        "pointer": "/reconciliation_protocol",
        "selector": {
          "attempt_only": true
        },
        "selector_family": "ATTEMPT_ONLY"
      },
      "allowed_environments": [
        "PAPER",
        "TESTNET"
      ],
      "scope_schema": {
        "required_keys": [
          "exchange_account_id",
          "portfolio_id"
        ],
        "additional_keys": "REJECT"
      },
      "required_upstream_checks": [
        {
          "check_id": "m08_reconciliation_authorized",
          "milestone": "M0.8",
          "pointer": "/reconciliation_protocol",
          "scope_binding": "REQUEST_RECONCILIATION_SCOPE"
        },
        {
          "check_id": "m11_recovery_attempt_allowed",
          "milestone": "M0.11",
          "pointer": "/startup_recovery_model",
          "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
        }
      ],
      "required_observations": [
        {
          "category": "RECONCILIATION_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "MATCH_REQUEST_SCOPE_FIELDS",
            "fields": [
              "exchange_account_id",
              "portfolio_id"
            ]
          }
        },
        {
          "category": "PERSISTENCE_HEALTH",
          "freshness": "FRESH_REQUIRED",
          "unknown_or_stale": "BLOCKED",
          "degraded": "BLOCKED",
          "scope_binding": {
            "mode": "CURRENT_INSTALLATION_STATESTORE",
            "context_fields": [
              "device_installation_id",
              "state_store_identity_fingerprint_sha256"
            ]
          }
        }
      ],
      "result_states": [
        "OK",
        "BLOCKED"
      ]
    }
  ],
  "authority_intersection": {
    "formula": "EffectiveReadiness(capability, environment, scope) = upstream_authority(capability, environment, scope) INTERSECT all matrix-required observations fresh and satisfying their capability-specific mapping",
    "upper_bound": "M0.12 readiness <= upstream authority",
    "positive_result": "OK means may attempt; it is not command acceptance, authentication, lease grant, kill-switch clear, persistence acceptance or update eligibility",
    "unknown_safety": "UNKNOWN or missing evidence for execution, risk, lease, persistence, security or market-data freshness => BLOCKED",
    "telemetry_outage": "missing safety observations => UNKNOWN/BLOCKED; loss of purely diagnostic logs/metrics/traces only degrades monitoring health"
  },
  "upstream_binding_manifest": [
    {
      "milestone": "M0.2",
      "artifact": "canonical_domain_vocabulary.json",
      "pointers": [
        "/entity_kinds"
      ]
    },
    {
      "milestone": "M0.3",
      "artifact": "process_topology_and_lifecycle.json",
      "pointers": [
        "/process_roles",
        "/single_instance_policy",
        "/startup_sequence",
        "/corehost_runtime_session_and_readiness_contract/startup_readiness"
      ]
    },
    {
      "milestone": "M0.4",
      "artifact": "environment_and_product_capabilities.json",
      "pointers": [
        "/execution_environments",
        "/ProductCapabilities/current_edition_capability_policy",
        "/environment_readiness",
        "/signature_validation_pipeline/m0_4_cryptographic_signature_verification",
        "/signature_validation_pipeline/stage_result_registry/CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED/availability",
        "/fail_closed_fallbacks/SAFE_LOCAL_ONLY",
        "/endpoint_policy/live_endpoint_resolution_current_edition_allowed",
        "/endpoint_policy/testnet_allowed_endpoint_classes",
        "/endpoint_policy/paper_execution_endpoint_class"
      ]
    },
    {
      "milestone": "M0.5",
      "artifact": "exchange_accounts_and_instruments.json",
      "pointers": [
        "/exchange_account_contract",
        "/instrument_contract",
        "/current_edition_account_operability_policy",
        "/exchange_registry_contract/entries",
        "/instrument_type_registry"
      ]
    },
    {
      "milestone": "M0.6",
      "artifact": "strategy_market_data_and_execution_routing.json",
      "pointers": [
        "/entity_registry/MarketDataRoute",
        "/entity_registry/ExecutionRoute",
        "/route_readiness_contract",
        "/current_edition_execution_pair_policy",
        "/endpoint_class_registry",
        "/execution_route_contract",
        "/environment_policy",
        "/authorization_dependencies_by_environment",
        "/authorization_dependency_policy",
        "/entity_registry/RouteReadiness",
        "/execution_route_contract/execution_readiness_max_age_seconds",
        "/record_schemas/MarketDataRoute",
        "/record_schemas/ExecutionRoute",
        "/record_schemas/RouteReadiness",
        "/array_enum_registries",
        "/market_data_route_contract"
      ]
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointers": [
        "/reconciliation_protocol"
      ]
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointers": [
        "/risk_decision_contract",
        "/kill_switch_contract",
        "/execution_lease_contract",
        "/lease_validation_conditions"
      ]
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointers": [
        "/secret_reference_policy",
        "/audit_safe_payload"
      ]
    },
    {
      "milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "pointers": [
        "/state_store_identity_contract",
        "/startup_recovery_model",
        "/transaction_protocol",
        "/failure_registry"
      ]
    }
  ],
  "audit_path_health": {
    "contract_level": "S9B logical journal rules are closed and may be a required readiness dependency",
    "condition_code": "AUDIT_EVIDENCE_PATH_AVAILABLE",
    "physical_deployment": "not production-ready until future M1 physical durable carrier is architecture-approved and implemented",
    "failure": "required durable append path unavailable blocks affected privileged/economic capability; a healthy log sink is insufficient"
  },
  "ui_projection": {
    "allowed": [
      "OK",
      "DEGRADED",
      "UNKNOWN",
      "BLOCKED",
      "stale",
      "last observation",
      "reason"
    ],
    "forbidden": [
      "override readiness",
      "mark dependency OK",
      "clear safety block",
      "acknowledge canonical success",
      "mint upstream authority"
    ],
    "implementation_in_S9C": false
  },
  "derived_freshness_state": {
    "STALE": "derived freshness status, not a fifth condition enum; projection condition is UNKNOWN and required safety dependency is BLOCKED"
  },
  "upstream_evidence_contract": {
    "input_type": "UpstreamReadinessEvidence",
    "caller_boolean_map_allowed": false,
    "required_sections": [
      "runtime_context",
      "product_capabilities_evidence",
      "accounts_by_id",
      "instruments_by_id",
      "market_data_routes_by_id",
      "execution_routes_by_id",
      "risk_by_scope",
      "persistence_context",
      "reconciliation_by_scope"
    ],
    "derivation_flow": [
      "validate structured evidence shape",
      "bind evidence to requested/canonical context",
      "resolve capability authority from injected frozen contract",
      "derive internal facts",
      "apply matrix"
    ],
    "check_registry": {
      "m03_ready": {
        "check_id": "m03_ready",
        "milestone": "M0.3",
        "pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness",
        "scope_binding": "CURRENT_RUNTIME_CONTEXT"
      },
      "m03_single_instance": {
        "check_id": "m03_single_instance",
        "milestone": "M0.3",
        "pointer": "/single_instance_policy",
        "scope_binding": "CURRENT_RUNTIME_CONTEXT"
      },
      "m04_capability_allowed": {
        "check_id": "m04_capability_allowed",
        "milestone": "M0.4",
        "pointer": "/ProductCapabilities/current_edition_capability_policy",
        "scope_binding": "CAPABILITY_SELECTOR"
      },
      "m05_account_operable": {
        "check_id": "m05_account_operable",
        "milestone": "M0.5",
        "pointer": "/current_edition_account_operability_policy",
        "scope_binding": "REQUEST_ACCOUNT"
      },
      "m05_instrument_operable": {
        "check_id": "m05_instrument_operable",
        "milestone": "M0.5",
        "pointer": "/instrument_contract/metadata_vs_tradability",
        "scope_binding": "REQUEST_INSTRUMENT"
      },
      "m06_route_ready": {
        "check_id": "m06_route_ready",
        "milestone": "M0.6",
        "pointer": "/route_readiness_contract",
        "scope_binding": "REQUEST_ROUTES"
      },
      "m08_reconciliation_authorized": {
        "check_id": "m08_reconciliation_authorized",
        "milestone": "M0.8",
        "pointer": "/reconciliation_protocol",
        "scope_binding": "REQUEST_RECONCILIATION_SCOPE"
      },
      "m09_risk_allowed": {
        "check_id": "m09_risk_allowed",
        "milestone": "M0.9",
        "pointer": "/risk_decision_contract",
        "scope_binding": "REQUEST_EXECUTION_SCOPE"
      },
      "m09_kill_switch_inactive": {
        "check_id": "m09_kill_switch_inactive",
        "milestone": "M0.9",
        "pointer": "/kill_switch_contract",
        "scope_binding": "REQUEST_EXECUTION_SCOPE"
      },
      "m09_execution_lease_valid": {
        "check_id": "m09_execution_lease_valid",
        "milestone": "M0.9",
        "pointer": "/execution_lease_contract",
        "scope_binding": "REQUEST_EXECUTION_SCOPE"
      },
      "m11_persistence_accepted": {
        "check_id": "m11_persistence_accepted",
        "milestone": "M0.11",
        "pointer": "/startup_recovery_model",
        "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
      },
      "m11_recovery_attempt_allowed": {
        "check_id": "m11_recovery_attempt_allowed",
        "milestone": "M0.11",
        "pointer": "/startup_recovery_model",
        "scope_binding": "CURRENT_INSTALLATION_STATESTORE"
      }
    },
    "nested_section_schemas": {
      "runtime_context": {
        "shape": "CLOSED_RECORD",
        "required_fields": [
          "runtime_session_id",
          "device_installation_id",
          "state_store_identity_fingerprint_sha256",
          "startup_readiness_state",
          "process_lock_owned"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "runtime_session_id": "M0.2 RuntimeSession canonical ID",
          "device_installation_id": "M0.2 DeviceInstallation canonical ID",
          "state_store_identity_fingerprint_sha256": "M0.11 lowercase hex64",
          "startup_readiness_state": "M0.3 startup readiness enum",
          "process_lock_owned": "exact boolean"
        },
        "provenance": "accepted current M0.3/M0.11 runtime context; consumed, not minted"
      },
      "product_capabilities_evidence": {
        "shape": "NULL_OR_NON_AUTHORITATIVE_DIAGNOSTIC_REFERENCE",
        "required_fields": [
          "carrier_kind",
          "evidence_schema_version",
          "stage_id",
          "stage_result",
          "stage_evidence_id",
          "predecessor_stage_evidence_id",
          "validation_context_id",
          "document_fingerprint",
          "capabilities_id",
          "edition_id",
          "signed_payload_hash",
          "capability_set_hash",
          "payload_schema_version",
          "signature_schema_version",
          "complete_stage_evidence",
          "same_validation_context_id",
          "same_document_fingerprint",
          "issuer_attestation_verified"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "carrier_kind": "CORE_ACCEPTED_M04_VALIDATED_SNAPSHOT_REFERENCE",
          "all fingerprint/evidence id fields": "lowercase hex64 except schema/identity fields",
          "booleans": "exact boolean"
        },
        "canonical_sources": [
          "M0.4#/validation_context_contract",
          "M0.4#/stage_evidence_chain_contract",
          "M0.4#/ProductCapabilities/validated_snapshot_rules"
        ],
        "provenance": "optional non-authoritative diagnostic mirror of future-only M0.4 model; ordinary dict, hashes, IDs, and booleans never establish accepted upstream object identity or current readiness",
        "authority_by_itself": false,
        "current_positive_authority": false,
        "future_only_fields_are_diagnostic_mirrors": true
      },
      "accounts_by_id": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_type": "M0.5 ExchangeAccount ID",
        "key_equals_field": "exchange_account_id",
        "required_fields": [
          "exchange_account_id",
          "workspace_id",
          "exchange_id",
          "environment",
          "market_type",
          "lifecycle_state",
          "connection_state",
          "execution_authorization"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "exchange_account_id": "M0.5 canonical ID",
          "environment": "M0.4 enum",
          "states": "M0.5 registries",
          "workspace_id": "M0.2 Workspace canonical ID",
          "exchange_id": "M0.5 exchange registry identity",
          "market_type": "M0.5 market type registry"
        },
        "provenance": "accepted current M0.5 account projection with exact workspace/exchange/environment/market binding"
      },
      "instruments_by_id": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_type": "M0.5 Instrument ID",
        "key_equals_field": "instrument_id",
        "required_fields": [
          "instrument_id",
          "workspace_id",
          "exchange_id",
          "environment",
          "market_type",
          "instrument_type",
          "source_adapter_family_id",
          "trading_status"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "instrument_id": "M0.5 canonical ID",
          "environment": "M0.4 enum",
          "trading_status": "M0.5 registry",
          "workspace_id": "M0.2 Workspace canonical ID",
          "exchange_id": "M0.5 exchange registry identity",
          "market_type": "M0.5 market type registry",
          "instrument_type": "M0.5 instrument type registry",
          "source_adapter_family_id": "exact M0.5 trusted instrument projection adapter family"
        },
        "provenance": "accepted current M0.5 instrument projection with exact workspace/exchange/environment/market binding"
      },
      "market_data_routes_by_id": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_type": "M0.6 MarketDataRoute ID",
        "key_equals_field": "market_data_route_id",
        "required_fields": [
          "market_data_route_id",
          "route_kind",
          "workspace_id",
          "exchange_id",
          "environment",
          "market_type",
          "adapter_family_id",
          "endpoint_class",
          "data_scope",
          "instrument_ids",
          "channel_types",
          "snapshot_stream_semantics",
          "sequence_policy",
          "freshness_policy",
          "reconnect_policy",
          "route_status",
          "instrument_id",
          "route_readiness"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "market_data_route_id": "M0.6 canonical ID",
          "route_kind": "MARKET_DATA joined discriminator",
          "source_route_fields": "exact M0.6 MarketDataRoute closed schema",
          "instrument_id": "joined requested M0.5 Instrument reference contained in instrument_ids",
          "route_readiness": "exact Core-owned non-durable M0.6 RouteReadiness projection"
        },
        "provenance": "Joined projection: exact source-shaped persisted M0.6 MarketDataRoute plus separately sourced Core-owned RouteReadiness and exact M0.5 instrument/venue graph; persisted readiness_state is forbidden."
      },
      "execution_routes_by_id": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_type": "M0.6 ExecutionRoute ID",
        "key_equals_field": "execution_route_id",
        "required_fields": [
          "execution_route_id",
          "route_kind",
          "workspace_id",
          "exchange_account_id",
          "exchange_id",
          "environment",
          "market_type",
          "adapter_family_id",
          "endpoint_class",
          "supported_instrument_types",
          "route_status",
          "route_capability_ceiling",
          "authorization_dependencies",
          "instrument_id",
          "route_readiness"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "execution_route_id": "M0.6 canonical ID",
          "route_kind": "EXECUTION",
          "workspace_id": "M0.2 canonical ID",
          "exchange_account_id": "M0.5 canonical ID",
          "exchange_id": "M0.5 exchange registry identity",
          "environment": "M0.4 enum",
          "market_type": "M0.5 market type registry",
          "adapter_family_id": "exact enabled venue binding",
          "endpoint_class": "M0.6 endpoint class allowing EXECUTION",
          "supported_instrument_types": "closed list of M0.5 instrument types",
          "route_status": "ACTIVE required",
          "route_capability_ceiling": "must contain PLACE_ORDERS",
          "authorization_dependencies": "exact M0.6 environment dependency set",
          "instrument_id": "joined requested M0.5 Instrument reference",
          "route_readiness": "exact Core-owned non-durable M0.6 RouteReadiness projection",
          "array_fields": "exact unique non-empty lists validated against frozen registries before membership"
        },
        "provenance": "Joined projection: source-shaped persisted M0.6 ExecutionRoute plus separately sourced Core-owned RouteReadiness and exact M0.5 account/instrument/venue graph; caller READY alone has no authority."
      },
      "risk_by_scope": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_construction": [
          "environment",
          "exchange_account_id",
          "instrument_id",
          "execution_route_id"
        ],
        "key_separator": ":",
        "required_fields": [
          "environment",
          "exchange_account_id",
          "instrument_id",
          "execution_route_id",
          "risk_allowed",
          "kill_switch_inactive",
          "execution_lease_valid"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "scope identities": "canonical upstream",
          "outcomes": "exact boolean"
        },
        "provenance": "accepted M0.9 risk/kill-switch/lease projections; key must equal ordered contained scope"
      },
      "persistence_context": {
        "shape": "CLOSED_RECORD",
        "required_fields": [
          "device_installation_id",
          "state_store_identity_fingerprint_sha256",
          "accepted",
          "recovery_attempt_allowed"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "device_installation_id": "M0.2 canonical ID",
          "state_store_identity_fingerprint_sha256": "M0.11 lowercase hex64",
          "outcomes": "exact boolean"
        },
        "provenance": "accepted current M0.11 recovery disposition"
      },
      "reconciliation_by_scope": {
        "shape": "MAP_OF_CLOSED_RECORDS",
        "key_construction": [
          "environment",
          "exchange_account_id",
          "portfolio_id"
        ],
        "key_separator": ":",
        "required_fields": [
          "environment",
          "exchange_account_id",
          "portfolio_id",
          "attempt_allowed"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_types": {
          "scope identities": "canonical upstream",
          "attempt_allowed": "exact boolean"
        },
        "provenance": "accepted M0.8 reconciliation projection; key must equal ordered contained scope"
      }
    },
    "product_capabilities_evidence_consumption": {
      "current_runtime": "None is canonical because M0.4 VALID is unreachable",
      "ordinary_mapping": "may be structurally classified as FUTURE_CONTRACT_MODEL_ONLY but authority=false",
      "stage_evidence_ids": "never trusted by syntax; no M0.12 recomputation substitutes for M0.4 issuer attestation",
      "raw_document_fields": "not consumed for authority",
      "future_fixture": "not accepted as current authority",
      "positive_authority_requirement": "opaque actual M0.4 validated result from a future reachable trusted boundary; unavailable today",
      "permanent_non_authority_invariant": "presence or structural validity MUST NEVER participate in any positive capability authorization expression"
    }
  },
  "dependency_scope_binding_modes": {
    "MATCH_REQUEST_SCOPE_FIELDS": "all declared fields must exist in request and observation and equal exactly; empty fields forbidden as CONTRACT_INCONSISTENT",
    "FIXED_SCOPE": "observation scope must exactly contain declared fixed_scope values",
    "CURRENT_RUNTIME_CONTEXT": "observation must bind declared fields from validated runtime context; empty context forbidden",
    "CURRENT_INSTALLATION_STATESTORE": "observation must exactly match validated current device_installation_id and state_store_identity_fingerprint_sha256"
  },
  "capability_selector_registry": {
    "CAPABILITY_ENVIRONMENT": {
      "exact_keys": [
        "capability",
        "environment"
      ],
      "compatible_milestone": "M0.4",
      "compatible_pointer": "/ProductCapabilities/current_edition_capability_policy",
      "legal_capabilities": [
        "PAPER_LOCAL_SIMULATION",
        "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"
      ],
      "semantics": "PAPER derives SAFE_LOCAL_ONLY local simulation; TESTNET conditional permission additionally requires currently reachable M0.4 VALID, which is unavailable; diagnostic/future wrapper has no authority"
    },
    "FIELD_EQUALS_ENVIRONMENT": {
      "exact_keys": [
        "field",
        "equals",
        "environment"
      ],
      "compatible_milestone": "M0.4",
      "legal_field": "live_allowed_in_current_edition",
      "legal_equals": true,
      "legal_environment": "LIVE"
    },
    "ROUTE_KIND": {
      "exact_keys": [
        "route_kind"
      ],
      "compatible_milestone": "M0.6",
      "legal_route_kind": "MARKET_DATA",
      "semantics": "exact requested MarketDataRoute record, environment, instrument and READY state"
    },
    "STARTUP_STATE": {
      "exact_keys": [
        "state"
      ],
      "compatible_milestone": "M0.3",
      "legal_state": "READY"
    },
    "ACCEPTED_STATE_REQUIRED": {
      "exact_keys": [
        "accepted_state_required"
      ],
      "compatible_milestone": "M0.11",
      "legal_value": true
    },
    "ATTEMPT_ONLY": {
      "exact_keys": [
        "attempt_only"
      ],
      "compatible_milestone": "M0.8",
      "legal_value": true
    }
  },
  "product_capabilities_authority": {
    "frozen_owner": "M0.4 environment_and_product_capabilities.json",
    "reachability_source": {
      "pipeline_pointer": "/signature_validation_pipeline/m0_4_cryptographic_signature_verification",
      "current_value": "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4",
      "future_accepted_result_availability_pointer": "/signature_validation_pipeline/stage_result_registry/CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED/availability",
      "future_value": "FUTURE_ONLY"
    },
    "current_reachability": "VALIDATED_SNAPSHOT_AND_TRUST_STATE_VALID_UNREACHABLE_IN_CURRENT_M0_4",
    "current_paper": {
      "authority_path": "M0.4 SAFE_LOCAL_ONLY fallback",
      "requires_valid_snapshot": false,
      "local_simulation_only": true,
      "private_exchange_side_effects": false
    },
    "current_testnet_private": "BLOCKED_WITHOUT_REACHABLE_VALID_M04_EVIDENCE",
    "current_live": "BLOCKED_BY_CURRENT_EDITION",
    "future_contract_model": "M0.4 closure-local future fixture proves future semantics only; never current authority",
    "M012_mints_authority": false,
    "current_reachability_rule": {
      "pipeline_value_must_equal": "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4",
      "accepted_result_availability_must_equal": "FUTURE_ONLY",
      "derived_current_valid_reachable": false,
      "unexpected_value": "CONTRACT_INCONSISTENT_NEVER_REACHABLE",
      "negative_sentinel_comparison_forbidden": true
    },
    "diagnostic_wrapper_in_positive_authorization_expression": false,
    "future_handoff_policy": "Any upstream M0.4 change fails closed until M0.12 is explicitly revised to bind an actual trusted accepted-output primitive; disappearance of the current sentinel never enables TESTNET automatically."
  },
  "live_market_data_authority": {
    "current_endpoint_policy": {
      "milestone": "M0.4",
      "pointer": "/endpoint_policy/live_endpoint_resolution_current_edition_allowed",
      "required_current_value": false
    },
    "endpoint_classes_are_descriptive_only": true,
    "route_contract": {
      "milestone": "M0.6",
      "entity_pointer": "/entity_registry/MarketDataRoute",
      "readiness_pointer": "/route_readiness_contract",
      "endpoint_registry_pointer": "/endpoint_class_registry",
      "joined_projection_schema_ref": "/health_readiness_model/upstream_evidence_contract/nested_section_schemas/market_data_routes_by_id",
      "market_data_route_contract_pointer": "/market_data_route_contract"
    },
    "venue_registry": {
      "milestone": "M0.5",
      "pointer": "/exchange_registry_contract/entries",
      "requirements": [
        "exact exchange_id entry exists",
        "status == ENABLED",
        "environment in supported_environments",
        "market_type in supported_market_types",
        "adapter_family_id exact match"
      ]
    },
    "current_enabled_live_venue_reachable": false,
    "caller_ready_is_authority": false,
    "current_results": {
      "PAPER": "SOURCE_POLICY_AND_ROUTE_GATES",
      "TESTNET": "SOURCE_POLICY_AND_ROUTE_GATES",
      "LIVE": "BLOCKED"
    },
    "future_handoff_policy": "LIVE market data requires a coordinated explicit M0.4 endpoint-policy, M0.5 enabled LIVE venue, M0.6 route-proof, and M0.12 binding update; enum presence, endpoint_classes, or caller READY never enables it automatically."
  },
  "execution_route_authority": {
    "explicit_route_kind_dispatch": [
      "MARKET_DATA",
      "EXECUTION"
    ],
    "unknown_or_unhandled_route_kind": "BLOCKED_NEVER_GENERIC_SUCCESS",
    "route_definition": {
      "milestone": "M0.6",
      "entity_pointer": "/entity_registry/ExecutionRoute",
      "contract_pointer": "/execution_route_contract"
    },
    "route_readiness": {
      "milestone": "M0.6",
      "entity_pointer": "/entity_registry/RouteReadiness",
      "contract_pointer": "/route_readiness_contract",
      "source": "Core-owned route_readiness_by_id only"
    },
    "policy_pointers": [
      "/endpoint_class_registry",
      "/environment_policy",
      "/authorization_dependencies_by_environment",
      "/authorization_dependency_policy",
      "/current_edition_execution_pair_policy"
    ],
    "venue_pointer": {
      "milestone": "M0.5",
      "pointer": "/exchange_registry_contract/entries"
    },
    "caller_ready_is_authority": false,
    "required_graph_equalities": [
      "route.workspace_id == account.workspace_id == instrument.workspace_id",
      "route.exchange_id == account.exchange_id == instrument.exchange_id",
      "route.environment == account.environment == instrument.environment == request environment",
      "route.market_type == account.market_type == instrument.market_type",
      "route.exchange_account_id == requested account",
      "instrument.instrument_type in route.supported_instrument_types"
    ]
  },
  "route_readiness_temporal_authority": {
    "validation_time_source": "explicit readiness evaluation now argument; one trusted logical RFC3339 UTC instant per evaluation; no system-clock reads",
    "route_readiness_schema": {
      "milestone": "M0.6",
      "pointer": "/record_schemas/RouteReadiness"
    },
    "route_readiness_contract": {
      "milestone": "M0.6",
      "pointer": "/route_readiness_contract"
    },
    "execution_max_age": {
      "milestone": "M0.6",
      "pointer": "/execution_route_contract/execution_readiness_max_age_seconds",
      "boundary": "age_seconds <= max_age_seconds"
    },
    "market_data_max_age": {
      "source": "exact source-shaped MarketDataRoute.freshness_policy.max_age_seconds",
      "boundary": "age_seconds <= max_age_seconds"
    },
    "sequence_state_by_kind": {
      "MARKET_DATA": "CONTIGUOUS",
      "EXECUTION": "NOT_APPLICABLE"
    },
    "future_observed_at": "BLOCKED",
    "stale_ready": "BLOCKED",
    "persisted_route_readiness_state_forbidden": true,
    "strict_array_shape": "exact unique non-empty list; strings, tuples, mappings, scalars, booleans and null rejected before membership",
    "array_registry_pointers": [
      {
        "milestone": "M0.5",
        "pointer": "/instrument_type_registry"
      },
      {
        "milestone": "M0.6",
        "pointer": "/array_enum_registries"
      }
    ]
  },
  "market_data_route_access_authority": {
    "milestone": "M0.6",
    "contract_pointer": "/market_data_route_contract",
    "endpoint_registry_pointer": "/endpoint_class_registry",
    "cross_field_rules": [
      "endpoint.environment == route.environment",
      "MARKET_DATA in endpoint.allowed_route_kinds",
      "endpoint.access_scope == route.data_scope",
      "PUBLIC data_scope forbids every PRIVATE_* channel"
    ],
    "private_channel_prefix": "PRIVATE_",
    "classification": "trusted-context graph mismatch makes route not source-ready",
    "all_instrument_ids": "each canonical ID must resolve to an accepted instrument projection with exact workspace/exchange/environment/market_type and source_adapter_family_id graph",
    "schema_source": "upstream_evidence_contract.nested_section_schemas.market_data_routes_by_id",
    "top_level_readiness_state_forbidden": true,
    "joined_route_readiness_required": true,
    "current_private_disposition": "BLOCKED_WITHOUT_ACCOUNT_CREDENTIAL_AUTHORITY_NOT_PRESENT_IN_MARKET_DATA_CONSUMPTION_SCOPE"
  }
}
```

## `alert_model`

```json
{
  "authority": {
    "owner": "CoreHost alert lifecycle service",
    "record_class": "M0.12-local logical durable record",
    "non_authorities": [
      "UI",
      "TrayAgent",
      "DesktopShell",
      "notification dispatcher",
      "delivery channels"
    ],
    "alert_is_authority": false,
    "forbidden_effects": [
      "mint ProductCapabilities",
      "change readiness",
      "authorize execution or authentication",
      "mint ExecutionLease",
      "change kill switch or risk decision",
      "accept persistence or reconciliation",
      "authorize update",
      "alter upstream facts"
    ]
  },
  "identity": {
    "syntax": "alrt_<UUIDv7 lowercase canonical 8-4-4-4-12 hexadecimal>",
    "regex": "^alrt_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    "classification": "M0.12-local logical durable-record identity",
    "generator": "CoreHost alert lifecycle service only",
    "properties": [
      "stable across redelivery",
      "unique per lifecycle instance",
      "immutable",
      "not caller/UI selected",
      "not message-derived",
      "not upstream canonical entity ID"
    ],
    "m02_entity": false,
    "collision": "regenerate before durable acceptance; never overwrite existing alert_id"
  },
  "source_family_registry": [
    "OBSERVATION_CONDITION",
    "UPSTREAM_STATE_CONDITION",
    "DOMAIN_EVENT_FACT",
    "AUDIT_EVENT_FACT",
    "SYSTEM_INTERNAL_CONDITION",
    "RELEASE_UPDATE_RESERVED"
  ],
  "source_reference_union": {
    "discriminator": "variant",
    "variants": {
      "OBSERVATION": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "category",
          "current_key",
          "source_component",
          "source_instance_id",
          "environment",
          "scope",
          "observed_at_utc",
          "sequence"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "OBSERVATION"
          },
          "authority": "accepted effective-current member of injected S9C ObservationReference only"
        }
      },
      "DOMAIN_EVENT": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "audit_event_id",
          "event_type",
          "event_fingerprint_sha256",
          "aggregate_version",
          "environment",
          "scope"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "DOMAIN_EVENT"
          },
          "authority": "recomputed exact M0.7 envelope plus accepted aggregate membership/order"
        }
      },
      "AUDIT_EVENT": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "audit_event_id",
          "content_fingerprint_sha256",
          "sequence",
          "chain_fingerprint_sha256",
          "action",
          "outcome",
          "environment",
          "scope"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "AUDIT_EVENT"
          },
          "authority": "accepted S9B journal membership, content recomputation, sequence and chain continuity"
        }
      },
      "M08_RECONCILIATION": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "outcome",
          "scope",
          "as_of_utc",
          "source_fingerprint_sha256"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "M08_RECONCILIATION"
          },
          "authority": "accepted M0.8 observed balance projection and exact reconciliation result"
        }
      },
      "M09_KILL_SWITCH": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "scope",
          "environment",
          "state",
          "source_revision",
          "effective_at_utc",
          "generation",
          "record_fingerprint_sha256",
          "accepted_membership_id"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "M09_KILL_SWITCH"
          },
          "authority": "accepted content binding plus current designation; fingerprint alone is integrity only"
        }
      },
      "M09_RISK_DECISION": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "decision",
          "decision_fingerprint_sha256",
          "scope",
          "evaluated_at_utc"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "M09_RISK_DECISION"
          },
          "authority": "exact member of Core accepted_decisions registry; fingerprint alone is insufficient"
        }
      },
      "M11_RECOVERY": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "scope",
          "result",
          "generation",
          "observed_at_utc",
          "evidence_fingerprint_sha256"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "M11_RECOVERY"
          },
          "authority": "fresh current M0.11 verified StateStore observation and recovery result"
        }
      },
      "SYSTEM_INTERNAL": {
        "type": "closed_object",
        "required_fields": [
          "variant",
          "condition_code",
          "component",
          "source_instance_id",
          "environment",
          "scope",
          "observed_at_utc",
          "sequence"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "variant": {
            "const": "SYSTEM_INTERNAL"
          },
          "authority": "Core-owned accepted internal condition registry only; caller/UI/delivery cannot mint"
        }
      }
    },
    "unknown_variant": "REJECT",
    "additional_fields": "REJECT",
    "authority_rule": "shape/hash never establishes accepted/current membership"
  },
  "scope_schema": {
    "type": "closed_discriminated_union",
    "discriminator": "kind",
    "variants": {
      "OBSERVATION_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "fields"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "OBSERVATION_SCOPE"
          },
          "fields": {
            "type": "exact S9C category scope"
          }
        }
      },
      "KILL_SWITCH_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "scope_type",
          "scope_id"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "KILL_SWITCH_SCOPE"
          },
          "scope_type": {
            "upstream": "M0.9 /scope_hierarchy/applicable_order"
          },
          "scope_id": {
            "upstream": "M0.9 /scope_hierarchy/scope_id_policy"
          }
        }
      },
      "RISK_DECISION_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "command_id",
          "command_request_fingerprint_sha256",
          "order_id",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "exchange_id",
          "instrument_id",
          "execution_route_id"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "RISK_DECISION_SCOPE"
          }
        }
      },
      "RECONCILIATION_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "asset_reference",
          "source_id"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "RECONCILIATION_SCOPE"
          }
        }
      },
      "STATE_STORE_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "device_installation_id",
          "state_store_identity_fingerprint_sha256"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "STATE_STORE_SCOPE"
          }
        }
      },
      "DOMAIN_EVENT_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "order_id",
          "command_id",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "exchange_id",
          "instrument_id",
          "execution_route_id"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "DOMAIN_EVENT_SCOPE"
          }
        }
      },
      "AUDIT_EVENT_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "device_installation_id",
          "operator_id",
          "runtime_session_id"
        ],
        "optional_fields": [
          "workspace_id",
          "exchange_account_id"
        ],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "AUDIT_EVENT_SCOPE"
          }
        }
      },
      "INTERNAL_SCOPE": {
        "type": "closed_object",
        "required_fields": [
          "kind",
          "component"
        ],
        "optional_fields": [],
        "additional_fields": "REJECT",
        "field_schemas": {
          "kind": {
            "const": "INTERNAL_SCOPE"
          }
        }
      }
    },
    "unknown_variant": "REJECT",
    "rule": "exact source-owned scope; no global workspace requirement"
  },
  "alert_type_registry": [
    {
      "alert_type": "MARKET_DATA_CURRENT_CONDITION",
      "category": "MARKET_DATA",
      "source_family": "OBSERVATION_CONDITION",
      "source_selector": {
        "variant": "OBSERVATION",
        "categories": [
          "ADAPTER_STATUS",
          "MARKET_DATA_FRESHNESS"
        ]
      },
      "scope_dimensions": [
        "workspace_id",
        "exchange_account_id",
        "instrument_id",
        "market_data_route_id"
      ],
      "severity_policy": {
        "UNKNOWN": "ERROR",
        "DEGRADED": "WARNING",
        "BLOCKED": "ERROR"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "source category",
          "exact observation scope",
          "source_component",
          "source_instance_id"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "FRESH_EFFECTIVE_CURRENT_OBSERVATION_OK_EXACT_SCOPE",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "EXECUTION_ROUTE_CONDITION",
      "category": "EXECUTION",
      "source_family": "OBSERVATION_CONDITION",
      "source_selector": {
        "variant": "OBSERVATION",
        "categories": [
          "EXECUTION_CONNECTIVITY",
          "EXECUTION_ROUTE_STATUS"
        ]
      },
      "scope_dimensions": [
        "workspace_id",
        "exchange_account_id",
        "instrument_id",
        "execution_route_id"
      ],
      "severity_policy": {
        "UNKNOWN": "ERROR",
        "DEGRADED": "ERROR",
        "BLOCKED": "CRITICAL"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "source category",
          "exact observation scope",
          "source_component",
          "source_instance_id"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "FRESH_EFFECTIVE_CURRENT_OBSERVATION_OK_EXACT_SCOPE",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "KILL_SWITCH_ACTIVE",
      "category": "RISK",
      "source_family": "UPSTREAM_STATE_CONDITION",
      "source_selector": {
        "variant": "M09_KILL_SWITCH",
        "state": "ACTIVE"
      },
      "scope_dimensions": [
        "scope_type",
        "scope_id"
      ],
      "severity_policy": {
        "ACTIVE": "CRITICAL"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "scope_type",
          "scope_id",
          "M09_KILL_SWITCH"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "accepted-current newer/equal-generation M0.9 INACTIVE exact scope/environment",
      "suppression_policy": "NO_ORDINARY_DELIVERY_SUPPRESSION",
      "escalation_policy": "CRITICAL",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "RISK_DECISION_DENIED",
      "category": "RISK",
      "source_family": "UPSTREAM_STATE_CONDITION",
      "source_selector": {
        "variant": "M09_RISK_DECISION",
        "decisions": [
          "DENY",
          "INCOMPLETE"
        ]
      },
      "scope_dimensions": [
        "command_id",
        "command_request_fingerprint_sha256",
        "order_id",
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "exchange_id",
        "instrument_id",
        "execution_route_id"
      ],
      "severity_policy": {
        "DENY": "ERROR",
        "INCOMPLETE": "ERROR"
      },
      "lifecycle_class": "FACT_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "instrument_id",
          "execution_route_id",
          "exact_source_fact_identity"
        ],
        "window": "EXACT_IMMUTABLE_FACT_IDENTITY"
      },
      "resolution_policy": "newer accepted M0.9 ALLOW exact same command/request and execution scope",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "RECONCILIATION_DIVERGENCE",
      "category": "RECONCILIATION",
      "source_family": "UPSTREAM_STATE_CONDITION",
      "source_selector": {
        "variant": "M08_RECONCILIATION",
        "outcomes": [
          "DRIFT",
          "MISSING_INTERNAL_FACT",
          "MISSING_EXTERNAL_FACT",
          "UNMAPPED_ASSET",
          "UNSUPPORTED"
        ]
      },
      "scope_dimensions": [
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "asset_reference",
        "source_id"
      ],
      "severity_policy": {
        "DRIFT": "ERROR",
        "MISSING_INTERNAL_FACT": "CRITICAL",
        "MISSING_EXTERNAL_FACT": "ERROR",
        "UNMAPPED_ASSET": "ERROR",
        "UNSUPPORTED": "WARNING"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "source_condition_discriminator"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "ACCEPTED_M0.8_RECONCILIATION_MATCH_EXACT_SCOPE",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "PERSISTENCE_RECOVERY_REQUIRED",
      "category": "PERSISTENCE",
      "source_family": "UPSTREAM_STATE_CONDITION",
      "source_selector": {
        "variant": "M11_RECOVERY",
        "results": [
          "RECOVERY_REQUIRED",
          "RESTORE_REQUIRED",
          "PERSISTENCE_BLOCKED"
        ]
      },
      "scope_dimensions": [
        "device_installation_id",
        "state_store_identity_fingerprint_sha256"
      ],
      "severity_policy": {
        "RECOVERY_REQUIRED": "CRITICAL",
        "RESTORE_REQUIRED": "CRITICAL",
        "PERSISTENCE_BLOCKED": "CRITICAL"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "device_installation_id",
          "state_store_identity_fingerprint_sha256",
          "M11_RECOVERY"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "ACCEPTED_M0.11_RECOVERY_COMPLETION_EXACT_STORE",
      "suppression_policy": "NO_ORDINARY_DELIVERY_SUPPRESSION",
      "escalation_policy": "CRITICAL",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "SECURITY_PRIVILEGED_FAILURE",
      "category": "SECURITY",
      "source_family": "AUDIT_EVENT_FACT",
      "source_selector": {
        "variant": "AUDIT_EVENT",
        "qualified_actions": [
          "TRUST_DEVICE",
          "REVOKE_DEVICE",
          "SETUP_PIN",
          "CHANGE_PIN",
          "RESET_PIN",
          "ROTATE_SECRET_REFERENCE",
          "REBIND_SECRET_REFERENCE",
          "ACTIVATE_CREDENTIAL_PROFILE",
          "DEACTIVATE_CREDENTIAL_PROFILE",
          "CHANGE_RISK_POLICY",
          "CHANGE_KILL_SWITCH",
          "CHANGE_PRODUCT_CAPABILITIES",
          "GRANT_LIVE_ACCESS",
          "SUSPEND_LIVE_ACCESS",
          "REVOKE_LIVE_ACCESS"
        ],
        "outcomes": [
          "DENIED",
          "FAILED"
        ],
        "requires_exact_actor_session_device_scope": true
      },
      "scope_dimensions": [
        "workspace_id",
        "device_installation_id"
      ],
      "severity_policy": {
        "DENIED": "ERROR",
        "FAILED": "CRITICAL"
      },
      "lifecycle_class": "FACT_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "workspace_id",
          "device_installation_id",
          "exact_source_fact_identity"
        ],
        "window": "EXACT_IMMUTABLE_FACT_IDENTITY"
      },
      "resolution_policy": "BLOCKED_MANUAL_REVIEW_REQUIRES_UNREPRESENTABLE_M0.10_OPERATION",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "SECURITY",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "DOMAIN_EXECUTION_FAILURE",
      "category": "EXECUTION",
      "source_family": "DOMAIN_EVENT_FACT",
      "source_selector": {
        "variant": "DOMAIN_EVENT",
        "event_types": [
          "ORDER_REJECTED",
          "ORDER_EXTERNAL_OUTCOME_UNKNOWN",
          "IDEMPOTENCY_CONFLICT"
        ]
      },
      "scope_dimensions": [
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "instrument_id",
        "execution_route_id"
      ],
      "severity_policy": {
        "ORDER_REJECTED": "WARNING",
        "ORDER_EXTERNAL_OUTCOME_UNKNOWN": "CRITICAL",
        "IDEMPOTENCY_CONFLICT": "ERROR"
      },
      "lifecycle_class": "FACT_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "instrument_id",
          "execution_route_id",
          "exact_source_fact_identity"
        ],
        "window": "EXACT_IMMUTABLE_FACT_IDENTITY"
      },
      "resolution_policy": "OPEN_NO_UNAMBIGUOUS_CORRECTIVE_SUCCESSOR_IN_FROZEN_M0.7; original or unrelated same-scope event REJECT; no manual resolution until M0.10 gate closes",
      "suppression_policy": "TIMED_OPERATOR",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    },
    {
      "alert_type": "ALERT_DELIVERY_SUBSYSTEM_FAILURE",
      "category": "OPERATIONS",
      "source_family": "SYSTEM_INTERNAL_CONDITION",
      "source_selector": {
        "variant": "SYSTEM_INTERNAL",
        "condition_code": "ALERT_DELIVERY_FAILURE"
      },
      "scope_dimensions": [
        "workspace_id"
      ],
      "severity_policy": {
        "DEGRADED": "WARNING",
        "BLOCKED": "ERROR"
      },
      "lifecycle_class": "CONDITION_ALERT",
      "dedup_policy": {
        "components": [
          "alert_type",
          "environment",
          "component",
          "condition_code",
          "source_instance_id"
        ],
        "window": "ACTIVE_LIFECYCLE"
      },
      "resolution_policy": "FRESH_INTERNAL_DELIVERY_HEALTH_OK",
      "suppression_policy": "NO_SELF_DELIVERY_ALERT",
      "escalation_policy": "STANDARD",
      "retention_class": "DURABLE_ALERT_HISTORY",
      "audit_policy": {
        "raise_redelivery": "SOURCE_EVIDENCE_SUFFICIENT",
        "acknowledge": "AUDIT_REQUIRED_ATOMIC",
        "suppression_change": "AUDIT_REQUIRED_ATOMIC",
        "resolution": "SOURCE_EVIDENCE_REQUIRED"
      },
      "operator_visibility_policy": "CRITICAL_ALWAYS_VISIBLE_OTHER_ACTIVE_VISIBLE"
    }
  ],
  "release_update_disposition": {
    "category": "RELEASE_UPDATE",
    "source_family": "RELEASE_UPDATE_RESERVED",
    "status": "FUTURE_SOURCE_NOT_CURRENT_AUTHORITY",
    "alert_types": [],
    "updater_state_machines": "OPEN"
  },
  "alert_schema": {
    "type": "closed_object",
    "required_fields": [
      "alert_id",
      "alert_type",
      "category",
      "severity",
      "environment",
      "scope",
      "source_reference",
      "raised_at",
      "last_seen_at",
      "occurrence_count",
      "lifecycle",
      "acknowledgement",
      "resolution",
      "deduplication_key",
      "suppression",
      "escalation",
      "operator_visibility",
      "revision"
    ],
    "optional_fields": [],
    "additional_fields": "REJECT",
    "field_schemas": {
      "alert_id": {
        "type": "canonical alrt UUIDv7"
      },
      "alert_type": {
        "registry": "/alert_model/alert_type_registry"
      },
      "category": {
        "derived": true
      },
      "severity": {
        "derived": true
      },
      "environment": {
        "type": "nullable upstream environment"
      },
      "scope": {
        "pointer": "/alert_model/scope_schema"
      },
      "source_reference": {
        "pointer": "/alert_model/source_reference_union"
      },
      "raised_at": {
        "type": "canonical UTC timestamp"
      },
      "last_seen_at": {
        "type": "canonical UTC timestamp"
      },
      "occurrence_count": {
        "type": "positive_non_bool_integer"
      },
      "lifecycle": {
        "enum": [
          "RAISED",
          "ACKNOWLEDGED",
          "RESOLVED"
        ]
      },
      "acknowledgement": {
        "type": "null_or_Acknowledgement"
      },
      "resolution": {
        "type": "null_or_Resolution"
      },
      "deduplication_key": {
        "type": "dk1 lowercase SHA256"
      },
      "suppression": {
        "type": "null_or_Suppression"
      },
      "escalation": {
        "type": "Escalation"
      },
      "operator_visibility": {
        "type": "OperatorVisibility"
      },
      "revision": {
        "type": "positive_non_bool_integer"
      }
    }
  },
  "nested_schemas": {
    "acknowledgement": {
      "type": "closed_object",
      "required_fields": [
        "operator_id",
        "acknowledged_at",
        "reason_code",
        "revision"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "resolution": {
      "type": "closed_object",
      "required_fields": [
        "resolved_at",
        "resolution_policy",
        "resolution_source_reference",
        "resolved_code",
        "revision"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "suppression": {
      "type": "closed_object",
      "required_fields": [
        "suppressed",
        "scope",
        "reason_code",
        "authorized_by",
        "starts_at",
        "expires_at",
        "policy_id",
        "revision"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "escalation": {
      "type": "closed_object",
      "required_fields": [
        "level",
        "routes",
        "evaluated_at",
        "policy_version",
        "delivery_failure_count"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "operator_visibility": {
      "type": "closed_object",
      "required_fields": [
        "visible",
        "attention_required",
        "persistence_class",
        "mandatory_routes"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    }
  },
  "lifecycle": {
    "states": [
      "RAISED",
      "ACKNOWLEDGED",
      "RESOLVED"
    ],
    "classes": [
      "CONDITION_ALERT",
      "FACT_ALERT"
    ],
    "allowed_transitions": [
      [
        "RAISED",
        "RESOLVED"
      ]
    ],
    "terminal": [
      "RESOLVED"
    ],
    "recurrence": "new alert_id; prior history remains terminal",
    "orthogonal_dimensions": [
      "suppression",
      "escalation",
      "delivery"
    ],
    "RAISED": "active and not acknowledged",
    "ACKNOWLEDGED": "operator awareness recorded; source fault remains active",
    "RESOLVED": "trusted resolution evidence recorded; history retained",
    "blocked_transitions": [
      [
        "RAISED",
        "ACKNOWLEDGED"
      ],
      [
        "ACKNOWLEDGED",
        "RESOLVED via operator manual review"
      ]
    ]
  },
  "deduplication": {
    "key_encoding": "dk1:SHA-256(lowercase hex of canonical UTF-8 JSON array of registry-declared components)",
    "caller_supplied": false,
    "message_fields_forbidden": true,
    "active_condition": "same key updates same active alert",
    "fact": "same exact fact identity is replay",
    "resolved_condition_recurrence": "new alert_id with same semantic key",
    "index": "rebuildable deduplication_key -> active alert_id; history is authority",
    "time_window_for_identity": false,
    "condition_discriminator": "registry source selector discriminator: observation category + source identity, or typed upstream condition family; never alert_type alone"
  },
  "redelivery": {
    "mutates": [
      "revision += 1",
      "last_seen_at=max accepted source time",
      "occurrence_count += 1",
      "source_reference if policy permits",
      "severity may only escalate"
    ],
    "preserves": [
      "alert_id",
      "raised_at",
      "ACKNOWLEDGED lifecycle"
    ],
    "severity_downgrade": "FORBIDDEN",
    "source_order": "accepted upstream currentness/order first; out-of-order is replay/no-op and cannot replace source_reference",
    "last_seen_at": "max(previous accepted source timestamp, incoming accepted source timestamp)",
    "history": "every accepted non-replay has full HistoryEntry",
    "executable_fields": [
      "raised_at_utc",
      "last_seen_at_utc",
      "occurrence_count"
    ],
    "replay": "exact accepted historical evidence returns current Alert without increment even after freshness window",
    "new_failing": "revision +1, occurrence_count +1, source last_seen max, raised_at unchanged, ACK/lifecycle preserved"
  },
  "operator_action_policy": {
    "accepted_operations": [
      "M0.12/ALERT_ACKNOWLEDGE",
      "M0.12/ALERT_SET_SUPPRESSION",
      "M0.12/ALERT_CLEAR_SUPPRESSION",
      "M0.12/ALERT_MANUAL_FACT_RESOLUTION"
    ],
    "authority": "canonical M0.10 declarations and validate_downstream_authorized_mutation only",
    "cas": "expected_alert_revision must equal externally designated current revision at commit",
    "failure": "controlled denial/conflict leaves alert history, current designation and audit outbox unchanged"
  },
  "suppression_contract": {
    "dimension": "delivery only",
    "ordinary_manual": "time bounded; expires_at required and later than starts_at",
    "permanent": "NOT_SUPPORTED_IN_S9D",
    "expiry": "Core-derived unsuppress mutation; lifecycle/source/severity unchanged",
    "critical": "active CRITICAL always visible on IN_APP and TRAY_PERSISTENT; repeated noise may be throttled",
    "cannot_change": [
      "lifecycle",
      "severity",
      "source",
      "readiness",
      "kill switch",
      "risk",
      "execution",
      "acknowledgement",
      "resolution"
    ],
    "time_validation": "starts_at == explicit now_utc; UTC offset zero; expires_at > starts_at; now monotonic; equality is expired",
    "expiry_mutation": "Core-derived, revisioned, durable UNSUPPRESS_EXPIRED with history/replay; other alert fields invariant",
    "executable_delivery_policy": {
      "WARNING": "all delivery and escalation routes suppressible",
      "ERROR": "all delivery and escalation routes suppressible",
      "CRITICAL": "IN_APP and TRAY_PERSISTENT mandatory; other routes suppressible",
      "expiry_or_clear": "normal canonical delivery resumes",
      "non_effects": [
        "fact",
        "resolution",
        "severity",
        "source currentness",
        "safety actions"
      ]
    }
  },
  "escalation_contract": {
    "policy_version": "S9D-1",
    "authority_effect": false,
    "levels": {
      "INFO": {
        "after_seconds": null,
        "routes": [
          "IN_APP"
        ]
      },
      "WARNING": {
        "after_seconds": 900,
        "routes": [
          "IN_APP",
          "LOCAL_OS_NOTIFICATION"
        ]
      },
      "ERROR": {
        "after_seconds": 300,
        "routes": [
          "IN_APP",
          "LOCAL_OS_NOTIFICATION",
          "TRAY_PERSISTENT"
        ]
      },
      "CRITICAL": {
        "after_seconds": 0,
        "routes": [
          "IN_APP",
          "TRAY_PERSISTENT",
          "OPERATOR_ATTENTION_REQUIRED"
        ]
      }
    },
    "acknowledgement_effect": "may stop timed repeat delivery but never lower severity/visibility floor",
    "delivery_failure_effect": "increment failure count and re-evaluate route; no safety mutation",
    "boundary": "elapsed threshold inclusive: WARNING 900, ERROR 300, CRITICAL 0; before false, at/after true",
    "executable_boundary": "requires unresolved failing fact, FAILED delivery, failure threshold, elapsed threshold, exact alert/delivery/escalation revisions, monotonic time and resolved route; produces typed ESCALATION history edge",
    "route_execution": "each escalation atomically creates exactly one EscalationRouteIntent per effective canonical route; every route requires a configured executor and exact canonical destination; missing executor fails closed",
    "suppression_interaction": {
      "WARNING": "does not advance while suppressed",
      "ERROR": "does not advance while suppressed",
      "CRITICAL": "advances only mandatory IN_APP and TRAY_PERSISTENT routes; failure counters do not change during escalation"
    }
  },
  "delivery_contract": {
    "classes": [
      "IN_APP",
      "LOCAL_OS_NOTIFICATION",
      "TRAY_PERSISTENT",
      "OPERATOR_ATTENTION_REQUIRED"
    ],
    "attempt_schema": {
      "type": "closed_object",
      "executable_record": "DeliveryAttempt",
      "required_fields": [
        "alert_id",
        "expected_alert_revision",
        "expected_delivery_revision",
        "attempt_id",
        "channel",
        "destination",
        "outcome",
        "attempted_at_utc",
        "retry_count",
        "escalation_level",
        "escalation_route_intent_id",
        "content_fingerprint_sha256"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {},
      "authority": "stable semantic attempt context excludes outcome and attempted_at_utc; adapter-owned first result supplies immutable outcome/time and trusted historical membership"
    },
    "receipt_is_acknowledgement": false,
    "history_durability": "ATOMIC_ALERTSTORE_CARRIER",
    "lifecycle_restore_dependency": true,
    "failure_semantics": [
      "does not resolve",
      "does not change source/readiness/safety",
      "does not block canonical safety action"
    ],
    "recursion": {
      "dedicated_type": "ALERT_DELIVERY_SUBSYSTEM_FAILURE",
      "single_active_per_environment_component": "dedup key",
      "delivery_of_dedicated_alert_failure_creates_alert": false
    },
    "separate_from_alert_record": false,
    "canonical_runtime_routes": [
      "IN_APP",
      "LOCAL_OS_NOTIFICATION",
      "TRAY_PERSISTENT",
      "OPERATOR_ATTENTION_REQUIRED"
    ],
    "SYSTEM_INTERNAL": "source-reference variant only; not a runtime delivery class or transport substitute",
    "result_provenance": "DeliveryAdapter historical membership plus immutable canonical route destination; self-hash alone is insufficient",
    "replay": "attempt_id binds original persisted request context; exact retry is no-op even after later alert mutations",
    "suppression": "suppressible attempts fail SUPPRESSED_DELIVERY before adapter execution; CRITICAL mandatory visibility routes remain executable",
    "escalation_execution_api": "execute_escalation_route(intent_id, now_utc); intent is execution authority and caller cannot select route attempt_id",
    "route_intent_identity": "deterministic hash of every semantic field including attempt_id; attempt links back by escalation_route_intent_id",
    "aggregate_state": "DeliveryState channel and destination are the actual last attempted canonical route and its canonical owner",
    "resolved": "new delivery and unconsumed old route intents reject ALERT_RESOLVED",
    "adapter_contract": "attempt_idempotent(attempt_id, immutable context) durably returns the same historical result without repeating external effect; conflicting context rejects",
    "preflight": "all deterministic revision/time/resolution/suppression/route/destination/replay checks precede adapter execution",
    "carrier_failure": "adapter historical idempotent result survives failed result-carrier commit and exact retry records it once"
  },
  "resolution_contract": {
    "condition": "fresh trusted effective-current exact-scope non-alert source evidence only",
    "stale_missing_wrong_scope": "REJECT",
    "manual_fixed_boolean": "REJECT",
    "fact": "exact alert-type designated corrective evidence only; age is irrelevant",
    "resolved_terminal": true,
    "typed_paths": {
      "MARKET_DATA_CURRENT_CONDITION": "S9C effective-current OK exact category/key/source/environment/scope and not expired",
      "EXECUTION_ROUTE_CONDITION": "S9C effective-current OK exact category/key/source/environment/scope and not expired",
      "KILL_SWITCH_ACTIVE": "current accepted M0.9 INACTIVE exact scope/environment and generation >= alert source",
      "RECONCILIATION_DIVERGENCE": "accepted M0.8 MATCH exact complete reconciliation key",
      "PERSISTENCE_RECOVERY_REQUIRED": "current accepted M0.11 COMPLETED exact device/store identity",
      "RISK_DECISION_DENIED": "accepted M0.9 ALLOW exact command/request/execution scope",
      "DOMAIN_EXECUTION_FAILURE": "OPEN: frozen M0.7 defines facts and transition graph but no single corrective successor mapping for ORDER_REJECTED, ORDER_EXTERNAL_OUTCOME_UNKNOWN, or IDEMPOTENCY_CONFLICT; self/unrelated event rejected",
      "SECURITY_PRIVILEGED_FAILURE": "manual resolution only for exact closed-policy types; M0.10 downstream mutation validation required"
    }
  },
  "durability": {
    "physical_carrier": "AtomicAlertAuthorityState is the executable in-memory combined publication carrier; production physical carrier remains OPEN_PHYSICAL_CARRIER_M1",
    "authoritative_units": [
      "one AtomicAlertAuthorityState combined publication",
      "AlertStoreSnapshot: accepted revisions, current designations, typed history, dedup, AuditObligation, OperatorReplayEntry, delivery state",
      "external trusted committed HistoricalAuthorizationDecision membership owned by the same combined carrier state",
      "trusted HistoricalSourceDecision membership atomically published with each source edge"
    ],
    "history_entry_schema": {
      "type": "closed_object",
      "executable_record": "MutationHistoryEntry",
      "required_fields": [
        "mutation_id",
        "alert_id",
        "pre_revision",
        "post_revision",
        "mutation_type",
        "before_fingerprint",
        "after_fingerprint",
        "timestamp_utc",
        "source_evidence_reference",
        "audit_obligation_id",
        "delivery_attempt_id",
        "causation_id",
        "correlation_id"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "revision": "alert revision starts at 1 per incident; store_revision equals committed mutation-history length; every accepted mutation increments both transaction edge count and store revision exactly once",
    "atomic_commit": "Operator mutations atomically publish AlertStoreSnapshot plus HistoricalAuthorizationDecision membership; source mutations atomically publish AlertStoreSnapshot plus HistoricalSourceDecision membership through one carrier state replacement. Ordinary carrier commits cannot mint either provenance class.",
    "restore": "validate complete semantic history and indexes without repair; corruption blocks restore",
    "canonicalization": "NFC UTF-8 JSON, sorted keys, separators comma/colon; SHA-256 lowercase; record fingerprint excludes no semantic current fields",
    "replay_entry_schema": {
      "type": "closed_object",
      "required_fields": [
        "replay_id",
        "account_id",
        "operator_id",
        "device_installation_id",
        "environment",
        "operation",
        "declared_intent",
        "alert_id",
        "expected_alert_revision",
        "mutation",
        "causation_id",
        "correlation_id",
        "result_revision",
        "audit_obligation_id"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    },
    "record_schema_scope": "DeliveryAttempt, MutationHistoryEntry, OperatorReplayEntry, and HistoricalSourceDecision schemas are exact executable dataclass schemas; other listed authority records are semantic runtime records described by invariants rather than separate exact closed-object schemas.",
    "historical_source_decision_schema": {
      "type": "closed_object",
      "executable_record": "HistoricalSourceDecision",
      "required_fields": [
        "decision_id",
        "evidence_reference",
        "evidence_ids",
        "transaction_time_utc",
        "result",
        "severity",
        "resolution_policy_id",
        "source_fence"
      ],
      "optional_fields": [],
      "additional_fields": "REJECT",
      "field_schemas": {}
    }
  },
  "restore_validation": [
    "unique alert_id",
    "one active alert per dedup key",
    "contiguous positive revisions from 1",
    "legal forward lifecycle only",
    "resolved absent from active index",
    "record equals final history projection",
    "acknowledgement has matching history/authorization",
    "suppression has matching authorization/history and valid expiry",
    "resolution has allowed source evidence/history",
    "replay mutation identity has one fingerprint",
    "index exactly equals rebuilt active history",
    "exact genesis semantics",
    "history/revision bijection and sealed mutation_id",
    "source evidence reference exact identity binding",
    "store_revision equals mutation history length",
    "sealed delivery attempt replay uniqueness and exact DeliveryState projection",
    "exact per-severity escalation threshold and routes",
    "delivery destination from immutable canonical route resolver and result membership from adapter authority",
    "audit actor and action matched to durable HistoricalAuthorizationDecision membership, never restart-ephemeral M0.10 proof membership",
    "escalation route-intent exact set, uniqueness, ownership and edge binding",
    "suppression-to-delivery policy parity",
    "exact per-source generation/revision fence vectors",
    "historical authorization decision membership independent of ephemeral M0.10 proof lifetime",
    "source fence exact equality to historical source fact",
    "canonical unique positive source fence shape",
    "route intent full semantic identity including attempt_id",
    "one-to-one escalation route intent and DeliveryAttempt link",
    "actual last route equals aggregate DeliveryState channel/destination",
    "resolved alerts cannot consume stale route intents",
    "operator replay exact semantic identity, edge and audit bijection",
    "redelivery raised/last-seen/occurrence exact source semantics",
    "operator environment equals authoritative alert environment",
    "route attempt uses current exact intent and aggregate channel/destination",
    "delivery adapter durable idempotency provenance",
    "source selector runtime shape and complete canonical registry identity",
    "closed canonical source-resolution policy identity independent of selector booleans",
    "source last_seen chronology by parsed UTC datetime, never lexical stamp order",
    "OperatorReplayEntry exact alert/pre/post edge equality with AuditObligation and MutationHistoryEntry",
    "successor current_at_utc equals its exact MutationHistoryEntry timestamp",
    "operator edge, AuditObligation, HistoricalAuthorizationDecision and successor exact transaction-time equality",
    "manual resolution exact RESOLVED lifecycle and MANUAL_CLOSED_POLICY mode",
    "combined carrier state forbids operator snapshot without matching committed historical decision",
    "historical decision-like data in ordinary AlertStoreSnapshot creates no authority",
    "exact AuditObligation historical-decision IDs equal combined carrier membership keys",
    "one HistoricalAuthorizationDecision per operator edge with no missing, orphan, or duplicate reference",
    "every decision mapping key equals a recomputed canonical decision_id",
    "restore snapshot and historical membership originate from one pinned AtomicAlertAuthorityState",
    "ordinary AlertStoreCarrier has no combined historical-membership mint API",
    "no module-global publication credential exists"
  ],
  "crash_matrix": [
    {
      "operation": "raise",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "redelivery",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "acknowledge",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "suppress",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "unsuppress_or_expiry",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "escalate",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    },
    {
      "operation": "resolve",
      "before_atomic_commit": "no visible mutation",
      "after_atomic_commit": "complete mutation visible; replay idempotent"
    }
  ],
  "audit_integration": {
    "required_operations": [
      "ACKNOWLEDGE",
      "SET_SUPPRESSION",
      "CLEAR_SUPPRESSION",
      "AUTHORIZED_MANUAL_FACT_RESOLUTION"
    ],
    "raise_redelivery": "no additional event when authoritative source evidence already identifies fact; type audit policy applies",
    "failure": "S9B required append failure => entire privileged mutation rejected/rolled back",
    "not_substitute_for": [
      "AuditEvent",
      "M0.7 DomainEvent",
      "M0.8 ledger evidence",
      "M0.9 risk evidence"
    ],
    "source_audit_event": "selected fact only; does not replace alert lifecycle",
    "blocked_operations": [
      "ACKNOWLEDGE",
      "SET_SUPPRESSION",
      "CLEAR_SUPPRESSION",
      "AUTHORIZED_MANUAL_FACT_RESOLUTION"
    ],
    "blocked_behavior": "reject before audit append or Alert mutation; atomic append design cannot close before M0.10 authorizes the operation"
  },
  "freshness": "reuse S9C effective current observation; raw stored OK after expiry is not evidence; missing/stale remains active or maps UNKNOWN per type, never resolves",
  "ui_projection": {
    "fields": [
      "lifecycle",
      "severity",
      "scope",
      "source_family",
      "acknowledgement",
      "suppression",
      "escalation",
      "operator_visibility",
      "source_staleness",
      "revision"
    ],
    "intents": [
      "ACKNOWLEDGE",
      "SET_SUPPRESSION"
    ],
    "rule": "client projection only; response becomes canonical only after accepted Core mutation; no local authority"
  },
  "redaction": {
    "reuse": [
      "M0.10 audit_safe_payload",
      "S9C safe_field_contract"
    ],
    "forbidden": [
      "PIN",
      "password",
      "raw token",
      "API key",
      "API secret",
      "private key",
      "biometric material",
      "raw credential"
    ],
    "human_text": "non-authoritative rendered projection from alert_type plus safe structured fields; excluded from identity and dedup"
  },
  "upstream_binding_manifest": [
    {
      "milestone": "M0.2",
      "artifact": "canonical_domain_vocabulary.json",
      "pointer": "/entity_kinds",
      "sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.2",
      "artifact": "canonical_domain_vocabulary.json",
      "pointer": "/identifier_policy",
      "sha256": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.7",
      "artifact": "commands_events_order_lifecycle_and_idempotency.json",
      "pointer": "/event_contract/event_types",
      "sha256": "5e62e1c2761c383c51160d3db9eff8a378ad9fa250b0764cefc2a95505e85621",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.7",
      "artifact": "commands_events_order_lifecycle_and_idempotency.json",
      "pointer": "/event_contract/event_schema_registry",
      "sha256": "1a00bea752efca1cc06740eb144b1d6c72162c47bead09a72e5406fe8e1bfa85",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.7",
      "artifact": "commands_events_order_lifecycle_and_idempotency.json",
      "pointer": "/event_contract/envelope_schema",
      "sha256": "66dd37a084874295ace4568900f465009db148c9542bf5aa960b7e54ba89d18d",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.7",
      "artifact": "commands_events_order_lifecycle_and_idempotency.json",
      "pointer": "/event_contract/fingerprint",
      "sha256": "567355f92de30f538d3be322b5ca1a19c69b36f6a3972092df557a4d89df38a1",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.7",
      "artifact": "commands_events_order_lifecycle_and_idempotency.json",
      "pointer": "/event_contract/ordering",
      "sha256": "cb5b41881ba32e072b63946aa5c1d825b019cd87b507e0dc8706b85ec007c3f7",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointer": "/reconciliation_protocol/key",
      "sha256": "7c4fff75fdb07046e62e62b8d2a74b9b6586f6c7de1e47a43d81dbf81597b484",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointer": "/reconciliation_protocol/outcomes",
      "sha256": "94f5c932d55d28eef34f96c9e9a73111cd33e3cd96297e3359728cd4057831c2",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointer": "/reconciliation_protocol/observed_fact_fields",
      "sha256": "6856418e9b7e448a8496c3732bfa979a2dc40b38406c39b6d21686552eb23a63",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointer": "/reconciliation_protocol/source_authority",
      "sha256": "4bcc55786bf05c57fa68accf394b4ad4d5e869f68c841faea8f32d5623ebc6e3",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.8",
      "artifact": "ledger_portfolio_capital_and_pnl.json",
      "pointer": "/reconciliation_protocol/outcome_rules",
      "sha256": "77a5d37be640c97f89a57c20e619864fc6ce99d726ca54fcde717dfc158d2ead",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/scope_hierarchy/applicable_order",
      "sha256": "ab56fb8ab2838bd20c746f08ef8096917a68e6115fc42ff1d29c20942d086d78",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/scope_hierarchy/scope_id_policy",
      "sha256": "cd8d10a76ee06ee180fb59204f6433b5a552ea0681f0616a8278cec214b3e4b3",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/kill_switch_contract/record_fields",
      "sha256": "bc0a68a7622e475f334b9c741436793ca2ddd0ace1c93b489307215df87ca313",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/kill_switch_contract/transition",
      "sha256": "4f55cf0e72fe591845fe4c895283eb62979728849dc7cf460fb6f9147cfaeb95",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/kill_switch_contract/authority",
      "sha256": "bbdbbe69ecd1c0c34e202022ffe017692dc267d9abbd403b4a31d9dac38e042a",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/kill_switch_contract/generation_validation",
      "sha256": "ee892ff0aed9f8a2e7d05729135de41699e64ff96ef8a83dafddc43fd1cc3a5b",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/risk_decision_contract/required_fields",
      "sha256": "7eccd54c17ee9def7a4f151ec684fd8623f36299a1e6f5dfa1c3716c7a46d92a",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/risk_decision_contract/decisions",
      "sha256": "c5fca7951b15ead9804fdc2c1e89d7d07b936ca9141aa58b38ae8e280997958b",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/risk_decision_contract/decision_fingerprint",
      "sha256": "4281b97a278a492c6c81b314b4088ed4b655be0e0cf0e8c1b4d32d2cb76546d4",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/core_authority_registries/accepted_decisions",
      "sha256": "7d32d17c1af31ab0eb6c62c942d59e1f9f758504f723cbc4a11ecec93fcef3ca",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.9",
      "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
      "pointer": "/core_authority_registries/accepted_content",
      "sha256": "116ed03c401a4cdd27aaa071954d46312a297bb9c13988f2decb07a5d086f0f2",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/authority/public_authorization_inputs",
      "sha256": "af01ca12a48bac50e4a80d6c35c671ede8bd678584663ec0a5ad74714251a533",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/authority/public_authorization_forbidden_inputs",
      "sha256": "6e765224ab28e1c050e1e235ac29e2fc44be2c2aef0b6dd6f64256b1574b9c4c",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/operation_policy_registry",
      "sha256": "f3cf802bebe7dd5ce70e88ee25bbddce442e5f6f9d6582d55193e2cfe24f8f59",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/proof_policy",
      "sha256": "811e80629e97f9afe16be1636927066577c39d1fe5211067bbaee52a88d30c97",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/executable_boundary_schemas/AuthenticationProof",
      "sha256": "d41cb6a5735327c67b506d68ba6ee7339dbe5b24fc5b7806560e1bcb1f7ba750",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.10",
      "artifact": "identity_device_authentication_and_secrets.json",
      "pointer": "/executable_boundary_schemas/CoreIssuedAuthenticationProofBinding",
      "sha256": "5c2adf1beee44f074f34d01e2c3ce6043775fffaf77dadb2bedf56a3a3ea2050",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "pointer": "/state_store_identity_contract",
      "sha256": "8836af4304ee417a1185f8c011a355ea322fb13dde45ab8a63812fa8be5a0ecd",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "pointer": "/startup_recovery_model/trusted_observation_gates",
      "sha256": "ac1570a9ad78809e5c9cb1ab2ef8286606242b56431bbd5b3b2d16e528d5e80b",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "pointer": "/restore_contract/result",
      "sha256": "f0ab79f1f7a32bcb460b5c9cd825e1508e90d8472e013918dbc95e9e7c95d9ff",
      "drift": "ContractInconsistent"
    },
    {
      "milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "pointer": "/restore_contract/promotion_gates",
      "sha256": "2dcdec86df165754280e66b9754bec01bc006ba19092520aac5bf1c04387b81e",
      "drift": "ContractInconsistent"
    }
  ],
  "principles": [
    "stable identity survives redelivery",
    "deduplication groups equivalent active conditions without deleting history",
    "suppression affects delivery noise, never source fault or safety action",
    "escalation changes routing/visibility, not upstream authority",
    "condition resolution requires fresh trusted evidence",
    "fact alerts resolve only by explicit canonical policy",
    "alerts do not create ProductCapabilities, readiness, authentication, risk, execution, persistence or update authority"
  ],
  "local_defect_status": {
    "domain_fact_corrective_resolution": "OPEN_SOURCE_AUTHORITY; executable path fails closed",
    "source_membership_current_authority": "EXECUTABLE_REPLAY_BEFORE_FRESHNESS_AND_CLOSED_SELECTOR",
    "canonical_ids_time": "EXECUTABLE",
    "semantic_alert_validation": "EXECUTABLE_ENVIRONMENT_AND_REDELIVERY_PARITY",
    "dedup_registry_executable_parity": "EXECUTABLE",
    "currentness_ordering": "EXECUTABLE_SOURCE_STORE_CARRIER_LOCK_ORDER",
    "security_manual_resolution": "EXECUTABLE_CLOSED_POLICY",
    "atomic_audit": "EXECUTABLE_PINNED_COMBINED_READ_INTERNAL_PUBLISHER; OPEN_PHYSICAL_CARRIER_M1",
    "clear_suppression": "EXECUTABLE",
    "internal_delivery_condition": "EXECUTABLE_CANONICAL_ROUTES_SYSTEM_INTERNAL_SOURCE_ONLY",
    "multi_source_observations": "EXECUTABLE_REQUIRED_SET_VALIDATION",
    "escalation": "EXECUTABLE_CURRENT_INTENT_ONLY_PREFLIGHT",
    "delivery_attempts": "EXECUTABLE_LATER_TIME_CRASH_REPLAY_STABLE_FIRST_RESULT",
    "semantic_restore": "EXECUTABLE_DEFENSIVE_ATOMIC_PIN_AND_SCHEMA_PARITY",
    "source_selectors": "PARTIAL_OPEN_PRODUCTION_UPSTREAM_SOURCE_AUTHORITIES"
  },
  "executable_authority": {
    "implementation": "bot_core/alerts/store.py",
    "operator_operations": [
      "M0.12/ALERT_ACKNOWLEDGE",
      "M0.12/ALERT_SET_SUPPRESSION",
      "M0.12/ALERT_CLEAR_SUPPRESSION",
      "M0.12/ALERT_MANUAL_FACT_RESOLUTION"
    ],
    "authorization_boundary": "AuthorizationAuthority.validate_downstream_authorized_mutation(proof, request, now_utc, actual_target_scope, actual_mutation) under shared security lock, followed by exact AlertStore CAS and commit under the store lock",
    "plain_authorize_executes_mutation": false,
    "currentness": "accepted contiguous revisions plus an external current_designations map; alert content cannot self-designate current",
    "audit_coupling": "AtomicAlertAuthorityState is accepted only when AuditObligation decision references and carrier-owned HistoricalAuthorizationDecision membership are an exact one-to-one set; missing, orphan, duplicate, key-mismatched or identity-invalid decisions reject",
    "manual_resolution_policy": {
      "default": "DENIED",
      "allowed_alert_types": [
        "OPERATOR_WORKFLOW_REQUIRED"
      ],
      "DOMAIN_EXECUTION_FAILURE": "DENIED"
    },
    "delivery_owner": {
      "IN_APP": "core_host/in_app",
      "LOCAL_OS_NOTIFICATION": "desktop_shell/os_notification",
      "TRAY_PERSISTENT": "tray_agent/persistent",
      "OPERATOR_ATTENTION_REQUIRED": "core_host/operator_attention"
    },
    "restore": "validates live-policy parity for source resolution/redelivery severity and terminality, escalation preconditions, exact transition times, history/replay/audit/delivery provenance, and store_revision without repair",
    "updater": "NOT_STARTED",
    "source_authority": "structured SourceEvidenceSet validated for accepted/current membership, selector, freshness and complete required sources; self-hash is integrity only",
    "suppression": "Core-owned one-hour bounded expiry plus revisioned UNSUPPRESS_EXPIRED; permanent suppression rejected",
    "escalation": "closed severity policy: WARNING 900s, ERROR 300s, CRITICAL 0s, with exact canonical route tuple persisted in DeliveryState and history",
    "history": "sealed deterministic mutation_id; exact bijection between accepted revisions and typed MutationHistoryEntry edges",
    "source_linearization": "SourceEvidenceAuthority.consume_current holds currentness through construction and the sole carrier publication of source edge plus HistoricalSourceDecision; global order source -> store -> carrier",
    "condition_identity": "current key and dedup identity bind alert type, source family, environment, scope, fact type, condition key and source id; mixed conditions reject",
    "severity": "derived from accepted source evidence under closed selector minimum; monotonic for an active incident",
    "delivery_replay": "DeliveryAttempt binds the original external-effect alert/delivery revision, route, destination and attempt identity; a later DELIVERY_ATTEMPT finalization edge may reconcile the adapter-owned historical result onto the then-current Alert without rewriting intervening history",
    "source_ordering": "exact per-source (source_id, generation, revision) vector with component-wise advancement and generation reset support; exact references replay without mutation",
    "suppression_delivery": "executable SUPPRESSED_DELIVERY gate with CRITICAL mandatory visibility floor",
    "routing": "canonical delivery classes execute directly; SYSTEM_INTERNAL is only a source variant, never delivery transport",
    "route_intents": "sealed escalation transition is linked to exactly one executable EscalationRouteIntent per effective required route",
    "delivery_provenance": "canonical route destination plus DeliveryAdapter historical result membership; coherent resealing cannot create authority",
    "audit_provenance": "AuditObligation binds a durable historical decision; restore compares full actor/device/account/environment/operation/intent/target/mutation/causation/correlation without live proof membership",
    "historical_authorization": "fresh read views resolve carrier-owned membership but expose no public publication API; restore provenance uses the pinned combined-state mapping rather than a second live carrier projection",
    "source_fence_provenance": "HistoricalSourceDecision carries canonical evidence_ids and binds reference, transaction time, result, severity, policy identity, and ordered fence; fresh authorities resolve durable accepted evidence directly without prior validate_current caches",
    "source_set_identity": "semantic set canonicalized by source_id before evidence-reference derivation; reordered exact sets are identical replays",
    "source_replay": "exact evidence replay returns the current Alert projection after later metadata mutations and creates no edge",
    "route_intent_consumption": "execute_escalation_route(intent_id) derives route, destination and attempt_id exclusively from one persisted sealed intent; arbitrary caller attempt identities reject",
    "route_attempt_binding": "DeliveryAttempt carries exact escalation_route_intent_id; restore proves one-to-one intent execution and exact route/destination/alert/escalation/attempt fields",
    "resolved_delivery": "new ordinary or unconsumed escalation-route delivery rejects ALERT_RESOLVED",
    "operator_environment": "owner-side request.environment must equal authoritative Alert dedup environment before live authorization/provenance/effect",
    "operator_replay": "sealed OperatorReplayEntry atomically binds complete request/mutation meaning and exact alert/pre/post edge to the same AuditObligation and MutationHistoryEntry",
    "delivery_crash_safety": "DeliveryAdapter stable semantic identity excludes outcome and retry time; historical_result recovers the immutable first external result after unrelated commits without another external effect, while changed original context conflicts",
    "delivery_preflight": "historical_result presence distinguishes completed-effect finalization from first execution; absent history requires current unresolved/failing escalation, route, destination, suppression and expiry checks before adapter execution",
    "suppression_expiry_boundary": "delivery/escalation at expires_at materializes UNSUPPRESS_EXPIRED and requires retry at its successor revision",
    "source_replay_freshness": "historical accepted reference is checked for committed replay before current/fresh validation; replay returns current Alert",
    "redelivery": "raised_at_utc immutable; source-derived last_seen_at_utc monotonic; occurrence_count increments once per newer failing evidence and never for exact replay/operator/delivery",
    "selector_validation": "exact SourceSelector runtime types; non-empty canonical unique source tuple; supported environment/severity; complete selector identity rejects duplicates without replacement",
    "route_bypass": "ordinary request_delivery is rejected during ESCALATION_PENDING; only execute_escalation_route consumes exact current intent",
    "stale_route_preflight": "intent escalation revision, route, destination, unresolved state and suppression are validated before adapter call",
    "source_resolution_policy": "closed immutable Core/M0.12 policy registry binds selector identity, required sources, automatic healthy-resolution support and corrective policy identity; selector booleans cannot create authority",
    "source_time": "last_seen_at_utc is derived using parsed chronological UTC instants for runtime and restore",
    "production_source_resolution_policies": {
      "MARKET_DATA_CURRENT_CONDITION": {
        "canonical_corrective_authority": "S9C effective-current OK exact category/key/source/environment/scope and not expired",
        "executable_status": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
      },
      "EXECUTION_ROUTE_CONDITION": {
        "canonical_corrective_authority": "S9C effective-current OK exact category/key/source/environment/scope and not expired",
        "executable_status": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
      },
      "KILL_SWITCH_ACTIVE": {
        "canonical_corrective_authority": "current accepted M0.9 INACTIVE exact scope/environment and generation >= alert source",
        "executable_status": "M09_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
      },
      "RECONCILIATION_DIVERGENCE": {
        "canonical_corrective_authority": "accepted M0.8 MATCH exact complete reconciliation key",
        "executable_status": "OPEN_SOURCE_AUTHORITY"
      },
      "PERSISTENCE_RECOVERY_REQUIRED": {
        "canonical_corrective_authority": "current accepted M0.11 COMPLETED exact device/store identity",
        "executable_status": "OPEN_SOURCE_AUTHORITY"
      },
      "RISK_DECISION_DENIED": {
        "canonical_corrective_authority": "accepted M0.9 ALLOW exact command/request/execution scope",
        "executable_status": "OPEN_SOURCE_AUTHORITY"
      },
      "DOMAIN_EXECUTION_FAILURE": {
        "canonical_corrective_authority": "OPEN: frozen M0.7 defines facts and transition graph but no single corrective successor mapping for ORDER_REJECTED, ORDER_EXTERNAL_OUTCOME_UNKNOWN, or IDEMPOTENCY_CONFLICT; self/unrelated event rejected",
        "executable_status": "OPEN_SOURCE_AUTHORITY"
      },
      "SECURITY_PRIVILEGED_FAILURE": {
        "canonical_corrective_authority": "manual resolution only for exact closed-policy types; M0.10 downstream mutation validation required",
        "executable_status": "OPEN_SOURCE_AUTHORITY"
      }
    },
    "production_source_policy_status": "S9C_AND_M09_ADAPTERS_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN; OTHER_SOURCE_AUTHORITIES_OPEN",
    "m09_kill_switch_authority": {
      "executable_status": "AVAILABLE",
      "implementation": "bot_core/m09_kill_switch_authority.py",
      "membership_authority": "independent carrier-owned CoreAcceptedContentBinding membership plus accepted kill-switch context in durable carrier history; raw DTOs, records and fingerprints never self-enroll",
      "frozen_boundary_types": "CoreAcceptedContentBinding(membership_id, content_fingerprint_sha256); PrevalidatedKillSwitchContext(history, membership_id, context_fingerprint_sha256)",
      "restore_trust_chain": "revalidates Core membership, exact bound history content, accepted-authority membership reference, record fingerprints, context fingerprint, generation history and current projection",
      "currentness": "exact scope_type, scope_id and environment designation",
      "history_and_restore": "superseded membership retained and current projection reconstructed fail-closed from accepted history",
      "generation": "positive non-boolean integer strictly increasing without reuse or rollback per exact scope/environment",
      "carrier_fence": "consume_current holds the M0.9 carrier-wide fence through the downstream callback",
      "kill_switch_active_alertstore_adapter": "INTEGRATED_PRODUCTION_SOURCE_AUTHORITY",
      "accepted_transaction_time": "AcceptedKillSwitchAuthorityEntry.accepted_at_utc is explicit trusted carrier transaction metadata; never inferred from KillSwitchRecord.effective_at_utc",
      "transaction_time_ordering": "global nondecreasing for accepted entries; per exact scope/environment strictly increasing only for genuinely-new transitions; exact replay and overlap are exempt from per-scope advancement",
      "historical_currentness": "resolve_historical_current_record reconstructs exact per-scope current designation at transaction time under carrier fence; issuance and restore both require this proof plus effective_at_utc <= transaction time"
    },
    "synthetic_source_resolution_policies": {
      "TEST_MULTI_SOURCE": "NON_PRODUCTION_TEST_FIXTURE",
      "OPERATOR_WORKFLOW_REQUIRED": "NON_PRODUCTION_TEST_FIXTURE_FOR_CLOSED_MANUAL_LIFECYCLE"
    },
    "operator_transaction_publication": "trusted publisher closure exists only inside the AlertStore operator mutation runtime after composition; ordinary AlertStoreCarrier exposes no combined decision-mint method and no importable singleton authorizes publication",
    "combined_failure_semantics": "BEFORE_PUBLICATION and FORMER_POST_CARRIER_PRE_DECISION fault points fail before the sole replacement; HISTORICAL_TRANSACTION_COMMIT_FAILED also precedes publication; externally recoverable state is entirely before or entirely after",
    "combined_state_restore": "constructor/restore pins one AtomicAlertAuthorityState and verifies source history from HistoricalSourceDecision evidence_ids plus durable upstream accepted evidence; historical evidence need not remain current or fresh now",
    "atomic_read": "load_atomic_state acquires the carrier lock once and returns one immutable AtomicAlertAuthorityState; concurrent ordinary commits linearize wholly before or after that read and cannot create a torn restore projection",
    "immutable_atomic_pin": "load_atomic_state is defensively pinned: tuple record collections and nested tuple semantics require exact immutable shapes, while mappings are copied into owned MappingProxyType projections; carrier-retained mutable aliases cannot change restore validation",
    "schema_parity": "machine closed-object required_fields exactly equal dataclasses.fields for DeliveryAttempt, MutationHistoryEntry, OperatorReplayEntry, and HistoricalSourceDecision",
    "s9d_c18_source_authority_disposition": {
      "status": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN",
      "frozen_canonical_artifact": "audit_observability_alerts_and_updater.json observability_model",
      "executable_oracle": "tests/architecture/test_cryptohunter_audit_observability_alerts_and_updater.py S9C-C1 pure executable oracle",
      "production_api": "ObservationAuthority.resolve_historical_acceptance, ObservationAuthority.resolve_current, ObservationAuthority.consume_effective_current; non-exported owner-composed writer capability publish",
      "production_types": "bot_core.observability.authority ObservationAuthority, ObservationAuthorityCarrier, AtomicObservationAuthorityState, AcceptedObservation, ObservationKey",
      "membership_authority": "EXECUTABLE: carrier-owned accepted history; content fingerprints do not mint membership",
      "currentness_and_expiry_fence": "EXECUTABLE: carrier-wide authority fence serializes independent runtime views and is held through downstream consumer durable publication",
      "durable_historical_lookup": "EXECUTABLE: carrier state restores superseded accepted A independently of freshness/currentness and process caches",
      "failing_authority": "INTEGRATED_FOR_TWO_S9C_TYPED_PATHS",
      "alertstore_adapter": "IMPLEMENTED: exact acceptance membership and fenced currentness projection",
      "market_data_current_condition": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN",
      "execution_route_condition": "S9C_ADAPTER_INTEGRATED_SOURCE_PRODUCER_AUTHENTICITY_OPEN"
    },
    "s9c_alertstore_adapter": "REAL_ACCEPTED_OBSERVATION_MEMBERSHIP_ONLY",
    "s9c_alertstore_currentness": "SEMANTIC_WINNER_SELECTED_AND_HELD_UNDER_S9C_CARRIER_FENCE_THROUGH_ALERTSTORE_PUBLICATION",
    "s9c_historical_source_validation": "DURABLE_ACCEPTANCE_PLUS_EFFECTIVE_CURRENT_AT_TRANSACTION_TIME",
    "s9c_source_reference": "EXACT_ACCEPTANCE_ID_NO_PROCESS_LOCAL_REFERENCE_CACHE",
    "s9c_source_fence": "STABLE_SEMANTIC_SOURCE_ID_ADAPTER_EPOCH_AND_S9C_TRANSACTION_REVISION",
    "s9c_condition_identity": "STABLE_ACROSS_ACCEPTANCE_REVISION_AND_RUNTIME_SESSION",
    "cross_carrier_semantics": "SOURCE_FENCE_THROUGH_ATOMIC_ALERTSTORE_PUBLICATION_NOT_PHYSICAL_TWO_STORE_ATOMICITY",
    "s9c_exact_observation_currentness": "EXACT_OBSERVATION_KEY_INCLUDES_RUNTIME_SESSION",
    "s9d_s9c_semantic_currentness": "LATEST_TRANSACTION_REVISION_FOR_EXACT_CATEGORY_SOURCE_COMPONENT_ENVIRONMENT_SCOPE_ACROSS_RUNTIME_SESSIONS",
    "s9d_runtime_restart_semantics": "NEW_RUNTIME_SEMANTIC_ACCEPTANCE_SUPERSEDES_OLDER_RUNTIME_FOR_ALERT_AUTHORITY",
    "s9c_adapter_projection": {
      "MARKET_DATA_FRESHNESS": {
        "alert_type": "MARKET_DATA_CURRENT_CONDITION",
        "source_family": "OBSERVATION_CONDITION",
        "fact_type": "MARKET_DATA_FRESHNESS",
        "severity": {
          "UNKNOWN": "ERROR",
          "DEGRADED": "WARNING",
          "BLOCKED": "ERROR"
        },
        "healthy_resolution_supported": true,
        "resolution_policy_id": "S9D/S9C_EFFECTIVE_CURRENT_OK_EXACT_SCOPE_V1",
        "scope_fields": [
          "market_data_route_id",
          "instrument_id"
        ]
      },
      "EXECUTION_PATH_HEALTH": {
        "alert_type": "EXECUTION_ROUTE_CONDITION",
        "source_family": "OBSERVATION_CONDITION",
        "fact_type": "EXECUTION_PATH_HEALTH",
        "severity": {
          "UNKNOWN": "ERROR",
          "DEGRADED": "ERROR",
          "BLOCKED": "CRITICAL"
        },
        "healthy_resolution_supported": true,
        "resolution_policy_id": "S9D/S9C_EFFECTIVE_CURRENT_OK_EXACT_SCOPE_V1",
        "scope_fields": [
          "exchange_account_id",
          "instrument_id",
          "execution_route_id"
        ]
      }
    },
    "s9c_transaction_time_order": "GLOBAL_NONDECREASING_ACCEPTED_AT",
    "historical_provenance_nonretroactivity": "LATER_SAME_SEMANTIC_TRANSACTION_CANNOT_ENTER_AN_ALREADY_COMMITTED_HISTORICAL_INSTANT",
    "s9d_source_transaction_time": "NEW_SOURCE_EDGE_NOT_BEFORE_REFERENCED_S9C_ACCEPTANCE_OR_CURRENT_ALERT_REVISION",
    "s9c_semantic_transaction_time_order": "STRICTLY_INCREASING_ACCEPTED_AT_PER_OBSERVATION_SEMANTIC_KEY",
    "s9c_same_second_parallelism": "EQUAL_ACCEPTED_AT_ALLOWED_ONLY_FOR_DISTINCT_SEMANTIC_KEYS"
  }
}
```

## `updater_model`

```json
{
  "phase_rules": [
    {
      "phase": "DISCOVERY",
      "output": "untrusted release candidate",
      "authority_created": false
    },
    {
      "phase": "DOWNLOAD",
      "output": "untrusted artifact candidate",
      "authority_created": false
    },
    {
      "phase": "VERIFICATION",
      "output": "ArtifactVerificationResult bound to exact bytes and trusted policy",
      "authority_created": "authenticity and integrity eligibility only"
    },
    {
      "phase": "STAGING",
      "output": "isolated staged candidate",
      "authority_created": false
    },
    {
      "phase": "AUTHORIZATION",
      "output": "installation authorization bound to verified plan",
      "authority_created": "permission for one exact attempt only"
    },
    {
      "phase": "INSTALLATION",
      "output": "installed candidate pending verification",
      "authority_created": false
    },
    {
      "phase": "RESTART",
      "output": "M0.3-coordinated process restart",
      "authority_created": false
    },
    {
      "phase": "POST_INSTALL_VERIFICATION",
      "output": "accepted installed release or recovery decision",
      "authority_created": "installed release designation only after all checks"
    },
    {
      "phase": "ROLLBACK",
      "output": "compatible prior binary/restored state or denied rollback",
      "authority_created": "only under exact rollback plan"
    },
    {
      "phase": "FAILED_UPDATE_RECOVERY",
      "output": "resumable, rollback, restore or operator-required disposition",
      "authority_created": false
    }
  ],
  "trust_model": [
    "release authenticity is distinct from M0.10 user authentication",
    "artifact signature never substitutes for operator authorization",
    "PIN/biometric/authentication proof never substitutes for artifact authenticity",
    "UI and remote metadata cannot create update authority",
    "trust roots and channel policy are protected local/build policy",
    "verification is repeated if exact bytes or manifest binding changes"
  ],
  "lifecycle_binding": [
    "Bootstrapper coordinates update lifecycle",
    "CoreHost remains sole mutable product-state authority",
    "DesktopShell and TrayAgent are clients/projections",
    "single-instance ownership is preserved",
    "safe CoreHost quiesce/shutdown and restart use M0.3 intents",
    "no second trading or StateStore writer authority is introduced"
  ]
}
```

## `release_artifact_authority`

```json
{
  "release_identity_fields": [
    "release_id",
    "product_version",
    "build_id",
    "release_channel",
    "platform",
    "minimum_supported_version",
    "supported_upgrade_paths",
    "StateStore_schema_compatibility",
    "manifest_version"
  ],
  "artifact_identity_fields": [
    "artifact_id",
    "release_id",
    "platform",
    "artifact_type",
    "size_bytes",
    "digest_algorithm",
    "artifact_digest",
    "signature_algorithm",
    "signing_key_id",
    "signature",
    "download_provenance"
  ],
  "install_eligibility_requires": [
    "recognized release/channel/platform",
    "exact manifest schema",
    "trusted non-revoked signing identity",
    "signature over canonical manifest and exact artifact digest",
    "recomputed exact artifact digest and size",
    "supported source product version",
    "M0.11 StateStore schema compatibility",
    "successful isolated staging validation",
    "required operator authorization",
    "safe M0.3 lifecycle plan"
  ],
  "candidate_rule": "discovered or downloaded material remains a candidate and is never installable authority",
  "anti_rollback_rule": "older version selection requires explicit trusted rollback plan; version ordering alone never authorizes it"
}
```

## `failure_policy`

```json
[
  {
    "condition": "diagnostic export or non-safety metric loss",
    "actions": [
      "OBSERVE_ONLY",
      "DEGRADE"
    ]
  },
  {
    "condition": "market data stale or route/adaptor evidence missing",
    "actions": [
      "BLOCK_CAPABILITY",
      "REQUIRE_OPERATOR_ACTION"
    ]
  },
  {
    "condition": "execution or accounting divergence",
    "actions": [
      "BLOCK_CAPABILITY",
      "TRIGGER_RECONCILIATION",
      "REQUIRE_OPERATOR_ACTION"
    ]
  },
  {
    "condition": "risk/kill-switch/lease evidence missing or stale",
    "actions": [
      "BLOCK_CAPABILITY",
      "INTERACT_WITH_KILL_SWITCH",
      "REQUIRE_OPERATOR_ACTION"
    ]
  },
  {
    "condition": "StateStore integrity/recovery/readiness unknown",
    "actions": [
      "BLOCK_CAPABILITY",
      "RESTART_OR_RECOVER",
      "REQUIRE_OPERATOR_ACTION"
    ]
  },
  {
    "condition": "required audit evidence cannot be durably appended",
    "actions": [
      "BLOCK_CAPABILITY",
      "REQUIRE_OPERATOR_ACTION"
    ]
  },
  {
    "condition": "release authenticity, compatibility or authorization absent",
    "actions": [
      "BLOCK_RELEASE_UPDATE"
    ]
  },
  {
    "condition": "post-install verification failure",
    "actions": [
      "BLOCK_RELEASE_UPDATE",
      "RESTART_OR_RECOVER",
      "REQUIRE_OPERATOR_ACTION"
    ]
  }
]
```

## `rollback_persistence_rules`

```json
[
  "M0.11 migration policy remains FORWARD_ONLY",
  "binary rollback is allowed only when the prior binary explicitly supports the current StateStore physical schema and representation",
  "a pre-update backup is mandatory when the UpdatePlan can mutate StateStore",
  "backup creation does not itself authorize restore",
  "restore remains subject to M0.11 validation and M0.3 protected restore-freshness authority",
  "never downgrade StateStore schema merely to enable old binary startup",
  "if no prior binary can open the migrated StateStore, rollback is BLOCKED and recovery uses a forward-compatible repair/newer release or an independently authorized restore",
  "post-update verification must cover product build identity, StateStore compatibility, recovery completion and capability readiness before designation"
]
```

## `cross_contract_invariants`

```json
[
  "CoreHost remains the sole authority for mutable trading state, while canonical AuditEvent append authority is phase-aware",
  "environments equal the upstream M0.4 execution environment registry",
  "process-role names are consumed from M0.3 rather than redefined",
  "audit references M0.7 events but does not replace event-store/domain ownership",
  "M0.8 economic facts and reconciliation results remain accounting authority",
  "M0.9 risk policy, kill switch and ExecutionLease remain safety authority",
  "M0.10 identity/authentication/authorization remains distinct from signing authenticity",
  "M0.11 StateStore schema and migration registry remain compatibility authority",
  "alerts and observations never mint upstream authority",
  "UI is projection-only",
  "LIVE remains target-capable but blocked by current M0.4/M0.6 policy",
  "M0.12 introduces no alternate LIVE gate"
]
```

## `current_state_classification`

```json
[
  {
    "subsystem": "Python logging configuration and structured application logging",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/logging/app.py",
      "bot_core/logging/config.py"
    ],
    "reason": "useful redaction/formatting base, but diagnostic logs are not durable audit authority"
  },
  {
    "subsystem": "Metrics registry/exporters/SLO/observability service",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/observability/metrics.py",
      "bot_core/observability/exporters.py",
      "bot_core/observability/server.py",
      "bot_core/runtime/metrics_service.py",
      "config/observability/slo.yml"
    ],
    "reason": "useful telemetry pipeline requiring canonical identity, freshness and bounded-cardinality alignment"
  },
  {
    "subsystem": "Operational bundles, dashboard sync, hypercare and DR observations",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/observability/bundle.py",
      "bot_core/observability/dashboard_sync.py",
      "bot_core/observability/hypercare.py",
      "bot_core/observability/dr_failover.py"
    ],
    "reason": "valuable projections; must not establish authority"
  },
  {
    "subsystem": "Alert router/dispatcher/throttling and delivery channels",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/alerts/router.py",
      "bot_core/alerts/dispatcher.py",
      "bot_core/alerts/throttle.py",
      "bot_core/alerts/channels/"
    ],
    "reason": "delivery mechanics are reusable after separation from canonical alert identity/lifecycle"
  },
  {
    "subsystem": "Legacy audit loggers and paper audit journal",
    "classification": "REWRITE",
    "evidence": [
      "bot_core/ai/audit.py",
      "bot_core/auto_trader/audit.py",
      "bot_core/data/ohlcv/audit.py",
      "bot_core/reporting/audit.py",
      "bot_core/resilience/audit.py",
      "bot_core/runtime/paper_audit_journal.py",
      "bot_core/security/tls_audit.py",
      "bot_core/security/token_audit.py"
    ],
    "reason": "fragmented file/log-shaped evidence cannot be canonical cross-domain durable audit authority"
  },
  {
    "subsystem": "Exchange/feed and AI health helpers",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/exchanges/health.py",
      "bot_core/exchanges/health_checks.py",
      "bot_core/ai/health.py",
      "ui/backend/feed_health_tracker.py"
    ],
    "reason": "useful probes but current health vocabularies do not prove capability readiness"
  },
  {
    "subsystem": "Durable observation and restore/migration install helpers",
    "classification": "KEEP",
    "evidence": [
      "bot_core/persistence/durable_observation.py",
      "bot_core/persistence/restore_migration_install.py"
    ],
    "reason": "M0.11-aligned foundations remain upstream-owned and are consumed, not redefined"
  },
  {
    "subsystem": "Security update verification and differential update helpers",
    "classification": "ADAPT",
    "evidence": [
      "bot_core/security/update.py",
      "bot_core/security/update_bundle_utils.py",
      "bot_core/update/differential.py"
    ],
    "reason": "candidate verification mechanics require complete release identity, trust-root, compatibility and attempt authority"
  },
  {
    "subsystem": "C++ OfflineUpdateManager direct UI installation/rollback",
    "classification": "REWRITE",
    "evidence": [
      "ui/src/update/OfflineUpdateManager.cpp",
      "ui/src/update/OfflineUpdateManager.hpp"
    ],
    "reason": "UI-side apply and simulated rollback conflict with privileged Bootstrapper/Core lifecycle and M0.11 compatibility"
  },
  {
    "subsystem": "UI alerts, health, metrics, telemetry and notification surfaces",
    "classification": "ADAPT",
    "evidence": [
      "ui/src/models/AlertsModel.cpp",
      "ui/src/health/HealthStatusController.cpp",
      "ui/src/grpc/MetricsClient.cpp",
      "ui/src/telemetry/UiTelemetryReporter.cpp",
      "ui/qml/components/AlertCenterPanel.qml",
      "ui/qml/components/UpdateManagerPanel.qml"
    ],
    "reason": "presentation is reusable only as canonical Core projection/client; local acknowledgement/lifecycle authority must be rewritten"
  },
  {
    "subsystem": "PySide preview audit/observability/rollback read models",
    "classification": "SOURCE_ONLY",
    "evidence": [
      "ui/pyside_app/preview_audit_envelope_read_model.py",
      "ui/pyside_app/preview_observability_read_model.py",
      "ui/pyside_app/preview_rollback_read_model.py"
    ],
    "reason": "explicit preview/read-model material is not production authority"
  },
  {
    "subsystem": "Release, installer, updater, integrity and diagnostics scripts",
    "classification": "SOURCE_ONLY",
    "evidence": [
      "scripts/manage_release.py",
      "scripts/desktop_updater.py",
      "scripts/offline_update.py",
      "scripts/package_update.py",
      "scripts/check_update_integrity.py",
      "scripts/generate_diagnostics.py",
      "scripts/build_release_bundle.py"
    ],
    "reason": "operator/build tooling informs implementation but cannot be runtime product authority"
  },
  {
    "subsystem": "Packaging/release GitHub Actions",
    "classification": "SOURCE_ONLY",
    "evidence": [
      ".github/workflows/"
    ],
    "reason": "CI release evidence and transport are not local update/install authority"
  },
  {
    "subsystem": "Tests, fixtures, docs, runbooks and historical current-state inventory",
    "classification": "SOURCE_ONLY",
    "evidence": [
      "tests/",
      "docs/observability/",
      "docs/deploy/",
      "docs/deployment/",
      "docs/runbooks/",
      "docs/architecture/cryptohunter_product_architecture/current_state_inventory.json"
    ],
    "reason": "validation/source material; M0.1 snapshot remains immutable history"
  },
  {
    "subsystem": "Competing UI-local acknowledgement clearing or local update success as authority",
    "classification": "DELETE",
    "evidence": [
      "ui/src/models/AlertsModel.cpp",
      "ui/backend/update_controller.py"
    ],
    "reason": "any path treating local presentation state as canonical lifecycle/install success must not survive canonical runtime migration"
  }
]
```

## `non_goals`

```json
[
  "no M1 runtime implementation",
  "no production file changes",
  "no alteration of frozen M0.2-M0.11 contracts or executable oracles",
  "no selection of database, telemetry vendor, signing algorithm, installer technology or transport",
  "no activation of LIVE",
  "no claim that legacy audit/log/update code is canonical",
  "no S9D alert closure or updater closure in S9C",
  "no physical AuditEvent carrier implementation"
]
```

## `open_items`

```json
[
  "Production physical AlertStoreCarrier remains OPEN_PHYSICAL_CARRIER_M1; logical authorization provenance atomicity is executable, while production corrective-source authority integrations remain OPEN_SOURCE_AUTHORITY.",
  "architecture-approve and implement the physical durable AuditEvent journal/carrier relative to frozen M0.11; canonical audit readiness remains blocked until then",
  "implement versioned deployment-specific probe freshness budgets within the closed source/category policy (no architecture semantic gap)",
  "implement the S9D logical alert carrier physically in M1 without changing its authority semantics",
  "select release-channel registry, manifest canonicalization, signing algorithms, trust-root provisioning, rotation and revocation",
  "close UpdatePlan/UpdateAttempt/rollback state machines and crash matrix",
  "define exact operator authorization policy per update/rollback class using M0.10 primitives",
  "define product-version and StateStore compatibility matrix including forward-repair paths",
  "close pre-update backup and post-install verification evidence schemas",
  "map every individual legacy implementation path to the M1 migration backlog after M0.12 closure"
]
```

## `audit_journal_contract`

```json
{
  "upstream_binding": {
    "selection": "consume the single M0.2 entity where canonical_name == AuditEvent; no copied identity/category registry is authoritative",
    "required_exact_keys": [
      "canonical_name",
      "id_field",
      "id_prefix",
      "parent",
      "persistence",
      "audit_event_categories",
      "optional_references",
      "relationships"
    ],
    "identifier_policy_pointer": "canonical_domain_vocabulary.json#/identifier_policy",
    "drift_disposition": "CONTRACT_INCONSISTENT_FAIL_CLOSED"
  },
  "schema": {
    "fields": [
      "audit_event_id",
      "category",
      "event_type",
      "payload_family",
      "occurred_at_utc",
      "device_installation_id",
      "runtime_session_id",
      "operator_id",
      "environment",
      "workspace_id",
      "exchange_account_id",
      "order_id",
      "ledger_entry_id",
      "correlation_id",
      "causation_id",
      "writer_phase",
      "outcome",
      "reason_code",
      "safe_payload",
      "m07_event_envelope",
      "content_fingerprint_sha256",
      "sequence",
      "previous_chain_fingerprint_sha256",
      "chain_fingerprint_sha256"
    ],
    "field_contracts": {
      "audit_event_id": {
        "owner": "M0.2",
        "source": "/entity_kinds/AuditEvent + /identifier_policy",
        "type": "evt-prefixed UUIDv7",
        "required": "always",
        "nullable": false,
        "meaning": "sole durable event identity"
      },
      "category": {
        "owner": "M0.2",
        "source": "/entity_kinds/AuditEvent/audit_event_categories",
        "type": "upstream closed enum",
        "required": "always",
        "nullable": false,
        "meaning": "retention and audit classification"
      },
      "event_type": {
        "owner": "M0.7 for CORE_DOMAIN_EVENT; M0.12 otherwise",
        "source": "M0.7 /event_contract/event_types or M0.12 /audit_journal_contract/payload_families",
        "type": "closed enum selected by payload_family",
        "required": "always",
        "nullable": false,
        "meaning": "exact event schema discriminator"
      },
      "payload_family": {
        "owner": "M0.12",
        "source": "/audit_journal_contract/payload_families",
        "type": "closed enum",
        "required": "always",
        "nullable": false,
        "meaning": "selects the authoritative closed envelope/payload schema"
      },
      "occurred_at_utc": {
        "owner": "M0.7 when CORE_DOMAIN_EVENT; M0.12 otherwise",
        "source": "M0.7 /event_contract/envelope_schema or M0.12 timestamp policy",
        "type": "RFC3339 UTC timestamp",
        "required": "always",
        "nullable": false,
        "meaning": "observation time; never append order"
      },
      "device_installation_id": {
        "owner": "M0.2",
        "source": "consumed from canonical_domain_vocabulary.json /entity_kinds entry DeviceInstallation",
        "type": "canonical DeviceInstallation ID; prefix resolved from upstream entity, currently dev",
        "required": "always",
        "nullable": false,
        "meaning": "parent and primary journal domain"
      },
      "runtime_session_id": {
        "owner": "M0.2 identity; M0.12 contextual policy",
        "source": "consumed from canonical_domain_vocabulary.json /entity_kinds entry RuntimeSession plus AuditEvent optional_references",
        "type": "canonical RuntimeSession ID; prefix resolved from upstream entity, currently run",
        "required": "CORE_RUNTIME only",
        "nullable": true,
        "meaning": "runtime context, not global parent"
      },
      "operator_id": {
        "owner": "M0.2",
        "source": "AuditEvent optional_references",
        "type": "canonical id prefix op",
        "required": "when an authenticated operator is actor",
        "nullable": true,
        "meaning": "accountable operator reference"
      },
      "environment": {
        "owner": "M0.4",
        "source": "execution environment registry",
        "type": "PAPER|TESTNET|LIVE",
        "required": "when event has trading environment scope",
        "nullable": true,
        "meaning": "optional event scope, not journal identity"
      },
      "workspace_id": {
        "owner": "M0.2",
        "source": "AuditEvent optional_references",
        "type": "canonical id prefix ws",
        "required": "when workspace scoped",
        "nullable": true,
        "meaning": "workspace reference"
      },
      "exchange_account_id": {
        "owner": "M0.2",
        "source": "AuditEvent optional_references",
        "type": "canonical id prefix xacc",
        "required": "when account scoped",
        "nullable": true,
        "meaning": "exchange-account reference"
      },
      "order_id": {
        "owner": "M0.2/M0.7",
        "source": "AuditEvent optional_references; M0.7 envelope",
        "type": "canonical id prefix ord",
        "required": "CORE_DOMAIN_EVENT",
        "nullable": true,
        "meaning": "order aggregate reference"
      },
      "ledger_entry_id": {
        "owner": "M0.2/M0.8",
        "source": "AuditEvent optional_references and relationship",
        "type": "canonical id prefix led",
        "required": "when ledger fact referenced",
        "nullable": true,
        "meaning": "economic fact reference"
      },
      "correlation_id": {
        "owner": "M0.7 when available; M0.12 contextual correlation otherwise",
        "source": "M0.7 /event_contract/envelope_schema; M0.3 causation/correlation when available",
        "type": "canonical correlation reference",
        "required": "CORE_DOMAIN_EVENT; otherwise when available",
        "nullable": true,
        "meaning": "cross-event workflow correlation"
      },
      "causation_id": {
        "owner": "M0.7 when available; originating contract otherwise",
        "source": "M0.7 /event_contract/envelope_schema; M0.3 when available",
        "type": "canonical cause reference",
        "required": "contextual",
        "nullable": true,
        "meaning": "immediate causal fact reference"
      },
      "writer_phase": {
        "owner": "M0.12 constrained by M0.3/M0.10/M0.11",
        "source": "/audit_model/writer_authority/phases",
        "type": "PRE_CORE|CORE_RUNTIME|MAINTENANCE_UPDATE_RECOVERY",
        "required": "always",
        "nullable": false,
        "meaning": "selects trusted writer policy"
      },
      "outcome": {
        "owner": "M0.12",
        "source": "/audit_journal_contract/outcomes",
        "type": "INTENT|ACCEPTED|DENIED|COMPLETED|FAILED|RECOVERY_REQUIRED",
        "required": "always",
        "nullable": false,
        "meaning": "recorded transition disposition"
      },
      "reason_code": {
        "owner": "originating upstream contract or M0.12 event schema",
        "source": "/audit_journal_contract/safe_code_type or exact originating upstream registry",
        "type": "SAFE_CODE when present",
        "required": "contextual; required for DENIED/FAILED/RECOVERY_REQUIRED",
        "nullable": true,
        "meaning": "safe denial/failure reason, never raw exception"
      },
      "safe_payload": {
        "owner": "M0.7 for CORE_DOMAIN_EVENT; family owner otherwise",
        "source": "closed event registry",
        "type": "closed object",
        "required": "always",
        "nullable": false,
        "meaning": "exact non-secret authoritative evidence"
      },
      "content_fingerprint_sha256": {
        "owner": "M0.12",
        "source": "/audit_journal_contract/integrity",
        "type": "64 lowercase hex",
        "required": "always",
        "nullable": false,
        "meaning": "content integrity only"
      },
      "sequence": {
        "owner": "M0.12 journal serializer",
        "source": "/audit_journal_contract/ordering",
        "type": "positive integer",
        "required": "accepted append",
        "nullable": false,
        "meaning": "primary DeviceInstallation append order"
      },
      "previous_chain_fingerprint_sha256": {
        "owner": "M0.12 journal serializer",
        "source": "/audit_journal_contract/integrity",
        "type": "64 lowercase hex",
        "required": "accepted append",
        "nullable": false,
        "meaning": "predecessor proof or genesis constant"
      },
      "chain_fingerprint_sha256": {
        "owner": "M0.12 journal serializer",
        "source": "/audit_journal_contract/integrity",
        "type": "64 lowercase hex",
        "required": "accepted append",
        "nullable": false,
        "meaning": "chain continuity proof 1:1 with audit_event_id"
      },
      "m07_event_envelope": {
        "owner": "M0.7",
        "source": "commands_events_order_lifecycle_and_idempotency.json#/event_contract/envelope_schema",
        "type": "exact closed M0.7 envelope",
        "required": "CORE_DOMAIN_EVENT only; null otherwise",
        "nullable": true,
        "meaning": "complete immutable upstream event fact; outer audit_event_id must equal nested audit_event_id"
      }
    },
    "additional_properties": false,
    "proof_fields_excluded_from_content": [
      "content_fingerprint_sha256",
      "sequence",
      "previous_chain_fingerprint_sha256",
      "chain_fingerprint_sha256"
    ]
  },
  "payload_families": {
    "CORE_DOMAIN_EVENT": {
      "owner": "M0.7",
      "event_registry_pointer": "commands_events_order_lifecycle_and_idempotency.json#/event_contract/event_schema_registry",
      "envelope_pointer": "commands_events_order_lifecycle_and_idempotency.json#/event_contract/envelope_schema",
      "rule": "M0.7 event envelope is consumed exactly; M0.12 adds device/runtime/writer/outcome and journal proof without redefining aggregate ordering",
      "representation_shape": "EXACT_NESTED_FROZEN_ENVELOPE",
      "identity_invariant": "outer audit_event_id == m07_event_envelope.audit_event_id",
      "outer_reference_invariant": "every non-null overlapping outer field equals the nested M0.7 field",
      "m07_fingerprint_rule": "validate and recompute nested event_fingerprint_sha256 before M0.12 content fingerprint",
      "object_semantics": "Python type is exactly dict and key set equals frozen envelope_schema.fields; insertion order is irrelevant; canonical sorted-key JSON owns fingerprint semantics"
    },
    "PRE_CORE_SECURITY_EVENT": {
      "owner": "M0.12 constrained by M0.3/M0.10",
      "allowed_event_types": [
        "AUTHENTICATION_DECIDED",
        "AUTHORIZATION_DECIDED",
        "SECURITY_MUTATION_RECORDED",
        "BOOTSTRAP_TRANSITION"
      ],
      "runtime_session_policy": "must be absent"
    },
    "MAINTENANCE_UPDATE_EVENT": {
      "owner": "M0.12 constrained by M0.3/M0.10/M0.11",
      "allowed_event_types": [
        "UPDATE_TRANSITION",
        "PERSISTENCE_TRANSITION"
      ],
      "runtime_session_policy": "optional only when existing Core context is referenced"
    },
    "RECOVERY_EVENT": {
      "owner": "M0.12 constrained by M0.3/M0.11",
      "allowed_event_types": [
        "RECOVERY_TRANSITION",
        "PERSISTENCE_TRANSITION"
      ],
      "runtime_session_policy": "optional"
    },
    "CORE_CONTROL_EVENT": {
      "owner": "M0.12 consuming M0.4/M0.8/M0.9/M0.10",
      "allowed_event_types": [
        "AUTHENTICATION_DECIDED",
        "AUTHORIZATION_DECIDED",
        "SECURITY_MUTATION_RECORDED",
        "CONFIGURATION_CHANGED",
        "LIVE_ACTIVATION_DECIDED",
        "RISK_CONTROL_TRANSITION",
        "ECONOMIC_CORRECTION_RECORDED",
        "PERSISTENCE_TRANSITION"
      ],
      "runtime_session_policy": "required"
    }
  },
  "non_m07_event_schema_registry": {
    "AUTHENTICATION_DECIDED": {
      "required_fields": [
        "method",
        "result_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "method": {
          "type": "enum",
          "source": "identity_device_authentication_and_secrets.json#/registries/factors"
        },
        "result_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "authentication"
    },
    "AUTHORIZATION_DECIDED": {
      "required_fields": [
        "operation_code",
        "decision_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "operation_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "decision_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "authorization"
    },
    "SECURITY_MUTATION_RECORDED": {
      "required_fields": [
        "mutation_code",
        "target_reference",
        "revision"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "mutation_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "target_reference": "canonical_id",
        "revision": "positive_integer"
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "device_management"
    },
    "CONFIGURATION_CHANGED": {
      "required_fields": [
        "configuration_key_code",
        "revision"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "configuration_key_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "revision": "positive_integer"
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "configuration"
    },
    "LIVE_ACTIVATION_DECIDED": {
      "required_fields": [
        "decision_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "decision_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "live_activation"
    },
    "RISK_CONTROL_TRANSITION": {
      "required_fields": [
        "control_code",
        "state_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "control_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "state_code": {
          "type": "enum",
          "source": "risk_hierarchy_kill_switch_and_execution_lease.json#/kill_switch_contract/states"
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "risk"
    },
    "ECONOMIC_CORRECTION_RECORDED": {
      "required_fields": [
        "correction_code",
        "source_fingerprint_sha256"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "correction_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "source_fingerprint_sha256": "sha256_hex"
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "trading"
    },
    "PERSISTENCE_TRANSITION": {
      "required_fields": [
        "operation_code",
        "first_sequence",
        "last_sequence",
        "checkpoint_fingerprint_sha256",
        "retention_policy_version",
        "result_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "operation_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "first_sequence": {
          "type": "positive_integer"
        },
        "last_sequence": {
          "type": "positive_integer"
        },
        "checkpoint_fingerprint_sha256": {
          "type": "sha256_hex"
        },
        "retention_policy_version": {
          "type": "constant",
          "value": "AUDIT_RETENTION_V1"
        },
        "result_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "recovery"
    },
    "UPDATE_TRANSITION": {
      "required_fields": [
        "update_attempt_reference",
        "phase_code",
        "artifact_fingerprint_sha256"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "update_attempt_reference": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "phase_code": {
          "type": "enum",
          "source": "audit_observability_alerts_and_updater.json#/canonical_vocabulary/update_phases"
        },
        "artifact_fingerprint_sha256": "sha256_hex"
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "update"
    },
    "BOOTSTRAP_TRANSITION": {
      "required_fields": [
        "step_code",
        "result_code"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "step_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "result_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        }
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "security"
    },
    "RECOVERY_TRANSITION": {
      "required_fields": [
        "recovery_code",
        "trusted_prefix_sequence"
      ],
      "nullable_fields": [],
      "field_schemas": {
        "recovery_code": {
          "type": "safe_code",
          "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
          "normalization": "NFC",
          "max_length": 64
        },
        "trusted_prefix_sequence": "non_negative_integer"
      },
      "additional_properties": false,
      "forbidden_secret_scan": true,
      "exact_category": "recovery"
    }
  },
  "outcomes": [
    "INTENT",
    "ACCEPTED",
    "DENIED",
    "COMPLETED",
    "FAILED",
    "RECOVERY_REQUIRED"
  ],
  "secret_policy": {
    "forbidden_names": [
      "password",
      "pin",
      "pin_verifier_secret",
      "biometric_template",
      "biometric_assertion",
      "api_key",
      "api_secret",
      "access_token",
      "refresh_token",
      "credential_plaintext",
      "bootstrap_secret",
      "private_signing_key"
    ],
    "recursive_name_match": true,
    "raw_exceptions_and_caller_blobs_forbidden": true,
    "allowed_evidence": [
      "canonical IDs",
      "reference IDs",
      "fingerprints",
      "generation/revision",
      "safe reason codes",
      "algorithm/key IDs",
      "boolean/result state"
    ],
    "rule": "encryption never permits secret material in AuditEvent"
  },
  "canonicalization": {
    "reuse": "M0.7 and M0.11 canonical NFC/UTF-8 JSON semantics: recursively NFC-normalize strings and keys, reject duplicate/non-string keys, booleans-as-integers, floats and non-finite numbers, emit UTF-8 JSON with lexicographically sorted keys and no insignificant whitespace",
    "content_domain_separator": "CryptoHunter/M0.12/AuditEventContent/v1\\x00",
    "chain_domain_separator": "CryptoHunter/M0.12/AuditEventChain/v1\\x00",
    "hash": "SHA-256 lowercase hexadecimal"
  },
  "ordering": {
    "primary_domain": "device_installation_id",
    "reason": "AuditEvent parent is DeviceInstallation and PRE_CORE/security/update events need no environment",
    "sequence": "durably serialized positive integer; genesis=1; every new accepted event is prior+1; no process-local counter",
    "environment": "optional event scope only",
    "secondary_axes": [
      "M0.7 per-order aggregate_version",
      "M0.7 correlation_id/causation_id",
      "M0.8 ledger ordering",
      "occurred_at_utc observation time",
      "M0.7 aggregate_version is independent from M0.12 sequence and may differ"
    ],
    "clock_rule": "timestamp regression is legal and cannot alter accepted sequence"
  },
  "integrity": {
    "content_fingerprint": "SHA-256(domain separator || canonical JSON of every immutable event field excluding four proof fields)",
    "genesis_previous_chain_fingerprint_sha256": "64 zero characters",
    "chain_fingerprint": "SHA-256(chain domain separator || UTF-8 decimal sequence || 0x00 || previous chain fingerprint bytes-as-lowercase-hex || 0x00 || content fingerprint bytes-as-lowercase-hex)",
    "proof_identity": "proof metadata is carried 1:1 by audit_event_id and has no separate durable identity",
    "verification": "recompute content and chain; require exact sequence and predecessor; verify retained checkpoint anchor before suffix"
  },
  "append_protocol": {
    "serializer_authority": "one durable compare-and-append authority per DeviceInstallation shared across PRE_CORE, CoreHost and authorized Bootstrapper handoff; implementations may differ but a Python/process lock is insufficient",
    "new_append": "validate upstream binding, identity/parent, phase writer, closed schema/scope/secrets and fingerprints; reserve exactly tail+1 and predecessor; durably accept before success acknowledgement",
    "same_id_same_content": "REPLAY idempotent success only when caller-supplied proof exact-matches stored original proof; return stored original proof; append zero records",
    "same_id_different_content": "IDENTITY_CONFLICT fail closed",
    "same_sequence_different_event": "SEQUENCE_CONFLICT fail closed",
    "same_payload_different_id": "append as distinct identity; no similarity dedupe",
    "immutability": "no accepted field may be updated; correction/resolution is a new causally referenced AuditEvent"
  },
  "required_transition_matrix": [
    {
      "transition_class": "authentication",
      "writer_phase": "PRE_CORE or CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "decision evidence before acknowledgement",
      "authority_owner": "M0.10 authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "authorization",
      "writer_phase": "PRE_CORE or CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "decision evidence before acknowledgement",
      "authority_owner": "M0.10 authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "device/security mutation",
      "writer_phase": "PRE_CORE or CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "intent then completion; shared-boundary atomic where legal",
      "authority_owner": "M0.10 authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "credential management",
      "writer_phase": "PRE_CORE or CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "intent then completion; never secret material",
      "authority_owner": "M0.10 authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "live activation decisions",
      "writer_phase": "CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "decision before acknowledgement",
      "authority_owner": "M0.4/CoreHost authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "configuration changes",
      "writer_phase": "CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "intent then completion",
      "authority_owner": "configuration authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "execution/order/fill",
      "writer_phase": "CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "M0.7 fact and audit proof in legal shared boundary or pending recovery",
      "authority_owner": "M0.7/CoreHost authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "risk/kill switch/lease",
      "writer_phase": "CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "decision/transition evidence before acknowledgement",
      "authority_owner": "M0.9/CoreHost authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "economic/ledger correction",
      "writer_phase": "CORE_RUNTIME",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "M0.8 fact and audit proof in legal shared boundary or pending recovery",
      "authority_owner": "M0.8/CoreHost authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "migration",
      "writer_phase": "MAINTENANCE_UPDATE_RECOVERY",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "intent before; completion after verified commit",
      "authority_owner": "M0.11 migration authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "backup/restore/recovery",
      "writer_phase": "MAINTENANCE_UPDATE_RECOVERY",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "request/denial/acceptance before action; completion after verification",
      "authority_owner": "M0.11 plus M0.3 protected freshness authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "update authorization/install/rollback",
      "writer_phase": "MAINTENANCE_UPDATE_RECOVERY",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "authorization/intent before; completion after verification",
      "authority_owner": "M0.3 authorized Bootstrapper handoff",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    },
    {
      "transition_class": "bootstrap/setup",
      "writer_phase": "PRE_CORE",
      "audit_requirement": "AUDIT_REQUIRED",
      "append_ordering": "decision evidence before acknowledgement",
      "authority_owner": "M0.3 external provisioning/M0.10 authority",
      "failure_disposition": "NO_SUCCESS_ACKNOWLEDGEMENT; BLOCK_OR_RECOVERY_REQUIRED"
    }
  ],
  "non_audit_examples": [
    "debug log line",
    "metric sample",
    "trace span",
    "periodic successful health probe",
    "UI notification delivery"
  ],
  "failure_and_recovery": {
    "append_before_transition_fails": "transition does not execute and is not acknowledged",
    "transition_fails_after_intent": "append FAILED outcome as a new event; if unavailable retain recovery-required state and no success",
    "transition_commits_completion_append_fails": "transition remains unacknowledged; affected capability fails closed; durable recovery-required condition must be established at earliest legal append point",
    "crash_between": "on restart verify trusted prefix and reconcile authoritative state; append RECOVERY_REQUIRED then completion/failure; never infer missing success",
    "atomicity": "only claim atomicity when mutation and journal share a legal durable transaction; otherwise explicit PENDING/UNACKNOWLEDGED/RECOVERY_REQUIRED protocol",
    "clean_prefix_recovery": "resume only after full prefix verification and authoritative-state reconciliation"
  },
  "physical_carrier": {
    "decision": "OPEN_FOR_M1_IMPLEMENTATION",
    "audit_result": "M0.11 StateStore v2 registry has no standalone AuditEvent representation; existing representations do not legally carry the complete journal; frozen M0.11 neither specifies nor authorizes a concrete external carrier",
    "contract_scope": "M0.12 closes the implementation-neutral logical journal only; physical integration requires a future architecture decision without mutating M0.11",
    "forbidden": [
      "add AuditEvent to PERSISTENCE_RECORD_REGISTRY",
      "StateStore schema v3",
      "audit_events table",
      "select a storage vendor"
    ],
    "blocker": "Durable deployment cannot claim canonical audit readiness until a physical carrier satisfying this contract is architecture-approved and implemented"
  },
  "backup_restore": {
    "independence": "StateStore backup/restore never establishes audit authority and never erases, truncates, resurrects or rolls back newer journal events",
    "backup": "capture journal version, DeviceInstallation domain, sealed checkpoint/tail proof and StateStore binding; backup is evidence candidate only",
    "restore": "compare restored checkpoint with current journal; current newer valid extension survives; equal prefix is allowed; divergent/forked/unknown prefix fails closed",
    "bootstrap_events": [
      "RESTORE_REQUESTED",
      "RESTORE_DENIED",
      "RESTORE_ACCEPTED",
      "RESTORE_COMPLETED",
      "RECOVERY_REQUIRED",
      "RECOVERY_COMPLETED"
    ],
    "bootstrap_rule": "request/denial/acceptance append at maintenance authority before StateStore replacement; completion after restored StateStore verification; journal remains independently appendable so no restore/audit cycle"
  },
  "retention": {
    "policy_owner": "versioned protected product policy accepted through authorized configuration/update path; UI/operator cannot directly shorten",
    "duration_rule": "deployment durations remain open; no arbitrary calendar duration is asserted",
    "retroactivity": "new policy applies prospectively; it may extend existing minima but cannot retroactively shorten an already accepted event minimum",
    "classes": {
      "SHORT_OPERATIONAL": "bounded non-security runtime evidence; pruning allowed after policy minimum",
      "PRODUCT_HISTORY": "configuration/licensing/device history; pruning only by sealed segments",
      "SECURITY_HISTORY": "authentication/authorization/credential/security history; pruning only by sealed segments",
      "ECONOMIC_HISTORY": "trading/risk/ledger history; pruning only by sealed segments",
      "RELEASE_HISTORY": "migration/recovery/update/bootstrap history; pruning only by sealed segments"
    },
    "category_matrix": [
      {
        "category": "authentication",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "authorization",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "configuration",
        "retention_class": "PRODUCT_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "credential_management",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "device_management",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "licensing",
        "retention_class": "PRODUCT_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "live_activation",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "runtime",
        "retention_class": "SHORT_OPERATIONAL",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "trading",
        "retention_class": "ECONOMIC_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "risk",
        "retention_class": "ECONOMIC_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "recovery",
        "retention_class": "RELEASE_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "update",
        "retention_class": "RELEASE_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      },
      {
        "category": "security",
        "retention_class": "SECURITY_HISTORY",
        "minimum_durability": "DURABLE_APPEND_ONLY_UNTIL_VERSIONED_POLICY_MINIMUM_AND_VALID_SEAL",
        "pruning_allowed": true,
        "export_allowed": true,
        "operator_or_ui_may_shorten": false
      }
    ],
    "export": "allowed only as integrity-verifiable secret-redacted evidence; export never deletes authority",
    "policy_version": "AUDIT_RETENTION_V1"
  },
  "pruning": {
    "rule": "never row/time cutoff deletion",
    "sealed_checkpoint": {
      "fields": [
        "device_installation_id",
        "first_sequence",
        "last_sequence",
        "predecessor_chain_fingerprint_sha256",
        "terminal_chain_fingerprint_sha256",
        "segment_content_fingerprint_sha256",
        "journal_version",
        "retention_policy_version",
        "compact_replay_fingerprint_sha256"
      ],
      "identity": "no new domain ID; deterministic proof for a contiguous sequence range",
      "requirements": [
        "verify full segment before seal",
        "durably retain checkpoint before removing event bodies",
        "preserve predecessor and terminal anchors",
        "next retained event must bind terminal anchor",
        "pruning itself emits audit evidence"
      ],
      "field_contracts": {
        "device_installation_id": "canonical M0.2 DeviceInstallation ID; exact journal and every body domain",
        "first_sequence": "positive non-bool integer; previous checkpoint last+1 or genesis 1",
        "last_sequence": "positive non-bool integer >= first_sequence; exact compact/body count",
        "predecessor_chain_fingerprint_sha256": "64 lowercase hex; zero at genesis, otherwise previous checkpoint terminal",
        "terminal_chain_fingerprint_sha256": "64 lowercase hex; exact last compact/body chain proof",
        "segment_content_fingerprint_sha256": "64 lowercase hex over canonical full bodies, established before deletion and retained as historical seal",
        "journal_version": "exact audit_journal_contract.version; unknown versions fail closed",
        "retention_policy_version": "exact audit_journal_contract.retention.policy_version authorizing this prune",
        "compact_replay_fingerprint_sha256": "64 lowercase hex over ordered compact replay metadata for this range"
      }
    },
    "executable_semantics": "checkpoint chain validates every declared field; seal only fully validated contiguous active bodies; bind compact replay metadata; remove bodies from all full-body containers; require next retained PERSISTENCE_TRANSITION audit evidence; active suffix verification starts from latest retained terminal anchor",
    "successive_segment_model": {
      "model": "CHECKPOINT_CHAIN",
      "first": "genesis segment starts at 1 with zero predecessor",
      "next": "first_sequence equals previous checkpoint last_sequence + 1 and predecessor equals previous terminal fingerprint",
      "body_requirement": "each new segment uses only current active bodies plus prior retained checkpoint anchor; deleted earlier bodies are never required",
      "retained_authority": [
        "ordered checkpoint chain",
        "compact replay metadata keyed 1:1 by audit_event_id",
        "active suffix"
      ],
      "overlap_or_gap": "fail closed"
    },
    "compact_replay_metadata": {
      "fields": [
        "audit_event_id",
        "content_fingerprint_sha256",
        "sequence",
        "previous_chain_fingerprint_sha256",
        "chain_fingerprint_sha256",
        "pruning_evidence_binding"
      ],
      "identity": "no new domain identity; mapping key and audit_event_id are the existing canonical AuditEvent identity",
      "forbidden": [
        "safe_payload",
        "m07_event_envelope",
        "any other full AuditEvent field"
      ],
      "integrity": "each segment checkpoint compact_replay_fingerprint_sha256 is SHA-256 of canonical ordered compact metadata; sequence/predecessor/chain continuity is independently recomputed",
      "pruning_evidence_binding": {
        "ordinary_event": null,
        "AUDIT_SEGMENT_PRUNED": {
          "fields": [
            "operation_code",
            "first_sequence",
            "last_sequence",
            "checkpoint_fingerprint_sha256",
            "retention_policy_version",
            "result_code"
          ],
          "source": "exact validated safe_payload subset; no actor/runtime/full payload retained"
        },
        "identity": "representation metadata under existing audit_event_id; no ID or prefix",
        "integrity": "included in checkpoint compact_replay_fingerprint_sha256",
        "validation": "None or exact closed six-field object; positive non-bool range, lowercase SHA-256 checkpoint fingerprint, accepted policy constant and exact operation/result codes"
      },
      "index_invariants": [
        "mapping key equals contained audit_event_id",
        "key is canonical M0.2 AuditEvent ID",
        "exact compact field set",
        "active and compact indexes are disjoint",
        "all retained AuditEvent IDs are globally unique",
        "every compact record belongs to exactly one committed checkpoint range",
        "no compact records when checkpoint chain is empty",
        "compact sequences are globally unique",
        "compact records equal the disjoint union authenticated by ordered checkpoint ranges"
      ]
    },
    "audit_evidence": {
      "event_type": "PERSISTENCE_TRANSITION",
      "payload_family": "MAINTENANCE_UPDATE_EVENT",
      "exact_category": "recovery",
      "operation_codes": [
        "AUDIT_SEGMENT_PRUNED"
      ],
      "ordering": [
        "fully validate eligible segment bodies",
        "durably retain checkpoint and compact replay metadata",
        "remove eligible full bodies from every full-body container",
        "append AUDIT_SEGMENT_PRUNED evidence as the next active AuditEvent; it is ineligible for the operation it describes"
      ],
      "payload_fields": [
        "operation_code",
        "first_sequence",
        "last_sequence",
        "checkpoint_fingerprint_sha256",
        "retention_policy_version",
        "result_code"
      ],
      "absence": "until required evidence append succeeds journal is RECOVERY_REQUIRED and ordinary append/success acknowledgement is blocked",
      "uniqueness": "exactly one successful evidence AuditEvent per checkpoint fingerprint/range; same audit_event_id is replay, different audit_event_id is ambiguous conflict",
      "retained_verification": "every committed checkpoint must resolve exactly one active full or compact pruning_evidence_binding with exact canonical checkpoint fingerprint, range, policy and COMPLETED result"
    },
    "partial_failure": {
      "checkpoint_durable_body_removal_failed": "retain checkpoint plus bodies, mark RECOVERY_REQUIRED, do not claim PRUNED; retry/reconcile idempotently",
      "body_removal_succeeded_evidence_append_failed": "retain checkpoint and compact index, mark RECOVERY_REQUIRED, block ordinary append and pruning success acknowledgement until exact pruning evidence is appended",
      "silent_success_forbidden": true,
      "retry": "exact pending checkpoint retry is idempotently eligible; different checkpoint/range/hash is rejected; no manual recovery reset"
    },
    "pending_recovery_state_machine": {
      "NONE": "no incomplete pruning; ordinary append, restore and next prune may proceed",
      "CHECKPOINT_DURABLE_BODIES_RETAINED": "pending descriptor retains exact checkpoint/fingerprint/range/policy/expected evidence plus expected next position; bodies and active_by_id remain, committed chain and compact index do not advance; only exact checkpoint retry may proceed",
      "BODIES_REMOVED_EVIDENCE_PENDING": "checkpoint promoted, compact metadata retained and bodies removed; only exact expected PERSISTENCE_TRANSITION at exact next sequence/predecessor may append",
      "recovery_required": "derived as pending_pruning != null; no public/manual boolean setter",
      "completion": "exact evidence APPENDED or exact same-ID REPLAY after durable append clears pending; unrelated evidence never clears it",
      "restore": "any pending state blocks restore and cannot be used as completed anchor",
      "next_prune": "blocked until NONE"
    },
    "expected_evidence_descriptor": {
      "fields": [
        "checkpoint",
        "checkpoint_fingerprint_sha256",
        "first_sequence",
        "last_sequence",
        "retention_policy_version",
        "expected_payload",
        "expected_sequence",
        "expected_predecessor",
        "phase"
      ],
      "identity": "no durable domain ID",
      "payload_derivation": "AUDIT_SEGMENT_PRUNED + exact checkpoint range + SHA-256(canonical checkpoint) + checkpoint policy + COMPLETED"
    },
    "checkpoint_evidence_graph": {
      "checkpoint_fingerprint": "SHA-256 over canonical NFC/UTF-8 sorted-key JSON of the complete checkpoint",
      "required_relation": "each committed checkpoint resolves exactly one active full evidence or retained compact pruning binding whose payload equals deterministic expected payload",
      "latest": "validate full active AuditEvent, journal position/chain and exact payload",
      "older": "when full evidence is later pruned, its minimal binding remains inside compact metadata and compact segment fingerprint",
      "mutation": "changing checkpoint including segment_content_fingerprint changes checkpoint fingerprint and breaks retained evidence relation",
      "missing_or_duplicate": "fail closed / RECOVERY_REQUIRED",
      "bijection": "completed state requires exact 1:1 relation: every successful evidence maps exactly one checkpoint and every checkpoint maps exactly one evidence",
      "no_orphans": "active or compact successful pruning evidence matching zero or multiple expected checkpoint payloads fails closed",
      "pending_latest_exception": "only a real BODIES_REMOVED_EVIDENCE_PENDING descriptor equal to the latest committed checkpoint may permit that exact latest relation to have 0 or 1 matches; every prior relation remains exactly 1 and all actual evidence must map exactly once"
    }
  },
  "corruption": {
    "conditions": [
      "invalid content fingerprint",
      "broken chain",
      "duplicate sequence",
      "unknown predecessor",
      "sequence gap",
      "truncated active segment",
      "malformed canonical event",
      "unsupported journal version",
      "fork or divergent restore",
      "identity-index key mismatch",
      "active/index divergence",
      "compact/index divergence",
      "duplicate AuditEvent identity",
      "checkpoint-evidence graph missing or ambiguous",
      "active/compact identity overlap",
      "orphan compact record outside checkpoint coverage",
      "duplicate compact append sequence",
      "compact history without checkpoint",
      "orphan successful pruning evidence",
      "pending latest-evidence exception without exact pending descriptor"
    ],
    "disposition": "fail closed affected privileged/safety and audit authority operations; preserve all evidence and surface recovery condition; never silently delete orphan metadata or repair, rebuild or rekey evidence/indexes"
  },
  "version": "cryptohunter.audit-journal.v1",
  "upstream_executable_bindings": {
    "M02": {
      "entities": {
        "AuditEvent": {
          "id_field": "audit_event_id",
          "id_prefix": "evt"
        },
        "DeviceInstallation": {
          "id_field": "device_installation_id",
          "id_prefix": "dev"
        },
        "RuntimeSession": {
          "id_field": "runtime_session_id",
          "id_prefix": "run"
        },
        "OperatorIdentity": {
          "id_field": "operator_id",
          "id_prefix": "op"
        },
        "Workspace": {
          "id_field": "workspace_id",
          "id_prefix": "ws"
        },
        "ExchangeAccount": {
          "id_field": "exchange_account_id",
          "id_prefix": "xacc"
        },
        "Order": {
          "id_field": "order_id",
          "id_prefix": "ord"
        },
        "LedgerEntry": {
          "id_field": "ledger_entry_id",
          "id_prefix": "led"
        }
      },
      "audit_event": {
        "id_prefix": "evt",
        "parent": "DeviceInstallation",
        "persistence": true,
        "audit_event_categories": [
          "authentication",
          "authorization",
          "configuration",
          "credential_management",
          "device_management",
          "licensing",
          "live_activation",
          "runtime",
          "trading",
          "risk",
          "recovery",
          "update",
          "security"
        ],
        "optional_references": [
          "runtime_session_id",
          "operator_id",
          "workspace_id",
          "exchange_account_id",
          "order_id",
          "ledger_entry_id"
        ],
        "relationships": [
          [
            "AuditEvent",
            "LedgerEntry",
            "optional_reference"
          ],
          [
            "DeviceInstallation",
            "AuditEvent",
            "one_to_many"
          ],
          [
            "RuntimeSession",
            "AuditEvent",
            "optional_runtime_reference"
          ]
        ]
      },
      "identifier_policy": {
        "persistent_id_format": "<prefix>_<uuidv7>",
        "regex": "^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        "uuid_version": "uuidv7",
        "rules": [
          "lowercase",
          "stable unique entity prefix",
          "no usernames/exchanges/symbols/secrets",
          "immutable after rename",
          "display names stored separately",
          "safe for JSON SQLite Protobuf logs API",
          "legacy strings migrated by explicit mapping",
          "never derive durable ID from name"
        ]
      }
    },
    "M03": {
      "bootstrapper_mode": "transport_and_discovery_only",
      "audit_boundary": {
        "uses": "canonical M0.2 AuditEvent; M0.7 causation/correlation when available",
        "ordering": "RuntimeSession is created after process-lock ownership and may exist before bootstrap validation; no artificial pre-RuntimeSession event requirement",
        "allowed": [
          "account_id",
          "device_installation_id",
          "intended_operator_id",
          "claim/reference fingerprint",
          "bootstrap generation",
          "reason code",
          "causation_id",
          "correlation_id"
        ],
        "forbidden": [
          "raw bootstrap secret",
          "PIN",
          "biometric material",
          "API credentials",
          "protected provisioning payload"
        ]
      },
      "maintenance_authorization": {
        "may_consume_maintenance_authorization": true,
        "may_apply_runtime_mutations": false,
        "may_request_runtime_mutations": false
      }
    },
    "M04": {
      "execution_environments": [
        "PAPER",
        "TESTNET",
        "LIVE"
      ]
    },
    "M07": {
      "envelope_schema": {
        "fields": [
          "audit_event_id",
          "event_type",
          "order_id",
          "aggregate_version",
          "correlation_id",
          "causation_id",
          "command_id",
          "environment",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "exchange_id",
          "instrument_id",
          "execution_route_id",
          "occurred_at_utc",
          "safe_payload",
          "event_fingerprint_sha256"
        ],
        "nullable_fields": [
          "causation_id",
          "command_id"
        ],
        "field_schemas": {
          "audit_event_id": {
            "type": "id",
            "prefix": "evt"
          },
          "event_type": {
            "type": "enum",
            "registry": "event_types"
          },
          "order_id": {
            "type": "id",
            "prefix": "ord"
          },
          "aggregate_version": {
            "type": "positive_integer"
          },
          "correlation_id": {
            "type": "id",
            "prefix": "corr"
          },
          "causation_id": {
            "type": "id",
            "prefix": "cause"
          },
          "command_id": {
            "type": "id",
            "prefix": "cmd"
          },
          "environment": {
            "type": "enum",
            "values": [
              "PAPER",
              "TESTNET",
              "LIVE"
            ]
          },
          "workspace_id": {
            "type": "id",
            "prefix": "ws"
          },
          "portfolio_id": {
            "type": "id",
            "prefix": "port"
          },
          "exchange_account_id": {
            "type": "id",
            "prefix": "xacc"
          },
          "exchange_id": {
            "type": "non_empty_string"
          },
          "instrument_id": {
            "type": "id",
            "prefix": "instr"
          },
          "execution_route_id": {
            "type": "id",
            "prefix": "xroute"
          },
          "occurred_at_utc": {
            "type": "timestamp"
          },
          "safe_payload": {
            "type": "event_safe_payload"
          },
          "event_fingerprint_sha256": {
            "type": "sha256_hex"
          }
        },
        "fingerprint_excluded_fields": [
          "event_fingerprint_sha256"
        ],
        "scope_fields": [
          "environment",
          "workspace_id",
          "portfolio_id",
          "exchange_account_id",
          "exchange_id",
          "instrument_id",
          "execution_route_id"
        ]
      },
      "event_types": [
        "ORDER_PLANNED",
        "ORDER_DISPATCHED",
        "ORDER_ACKNOWLEDGED",
        "ORDER_REJECTED",
        "ORDER_PARTIALLY_FILLED",
        "ORDER_FILLED",
        "ORDER_CANCEL_REQUESTED",
        "ORDER_CANCEL_CONFIRMED",
        "ORDER_CANCEL_REJECTED",
        "ORDER_REPLACE_REQUESTED",
        "ORDER_REPLACE_CONFIRMED",
        "ORDER_REPLACE_REJECTED",
        "ORDER_EXPIRED",
        "ORDER_EXTERNAL_OUTCOME_UNKNOWN",
        "ORDER_RECONCILIATION_OBSERVED",
        "COMMAND_ACCEPTED",
        "COMMAND_REJECTED",
        "COMMAND_REPLAYED",
        "IDEMPOTENCY_CONFLICT",
        "EVENT_REPLAY_IGNORED",
        "EVENT_REJECTED"
      ],
      "event_schema_registry": {
        "ORDER_PLANNED": {
          "safe_payload_fields": [
            "side",
            "order_type",
            "quantity"
          ],
          "field_schemas": {
            "side": {
              "type": "string"
            },
            "order_type": {
              "type": "string"
            },
            "quantity": {
              "type": "decimal"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_DISPATCHED": {
          "safe_payload_fields": [
            "client_order_id"
          ],
          "field_schemas": {
            "client_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_ACKNOWLEDGED": {
          "safe_payload_fields": [
            "venue_order_id"
          ],
          "field_schemas": {
            "venue_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_REJECTED": {
          "safe_payload_fields": [
            "reason_code"
          ],
          "field_schemas": {
            "reason_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_PARTIALLY_FILLED": {
          "safe_payload_fields": [
            "fill_id",
            "venue_trade_id",
            "cumulative_executed_quantity"
          ],
          "field_schemas": {
            "fill_id": {
              "type": "id",
              "prefix": "fill"
            },
            "venue_trade_id": {
              "type": "string"
            },
            "cumulative_executed_quantity": {
              "type": "decimal"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_FILLED": {
          "safe_payload_fields": [
            "fill_id",
            "venue_trade_id",
            "cumulative_executed_quantity"
          ],
          "field_schemas": {
            "fill_id": {
              "type": "id",
              "prefix": "fill"
            },
            "venue_trade_id": {
              "type": "string"
            },
            "cumulative_executed_quantity": {
              "type": "decimal"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_CANCEL_REQUESTED": {
          "safe_payload_fields": [
            "reason_code"
          ],
          "field_schemas": {
            "reason_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_CANCEL_CONFIRMED": {
          "safe_payload_fields": [
            "venue_order_id"
          ],
          "field_schemas": {
            "venue_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_CANCEL_REJECTED": {
          "safe_payload_fields": [
            "reason_code"
          ],
          "field_schemas": {
            "reason_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_REPLACE_REQUESTED": {
          "safe_payload_fields": [
            "replacement_order_id"
          ],
          "field_schemas": {
            "replacement_order_id": {
              "type": "id",
              "prefix": "ord"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_REPLACE_CONFIRMED": {
          "safe_payload_fields": [
            "replacement_order_id",
            "venue_order_id"
          ],
          "field_schemas": {
            "replacement_order_id": {
              "type": "id",
              "prefix": "ord"
            },
            "venue_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_REPLACE_REJECTED": {
          "safe_payload_fields": [
            "reason_code"
          ],
          "field_schemas": {
            "reason_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_EXPIRED": {
          "safe_payload_fields": [
            "venue_order_id"
          ],
          "field_schemas": {
            "venue_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_EXTERNAL_OUTCOME_UNKNOWN": {
          "safe_payload_fields": [
            "operation_type",
            "client_order_id"
          ],
          "field_schemas": {
            "operation_type": {
              "type": "string"
            },
            "client_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "ORDER_RECONCILIATION_OBSERVED": {
          "safe_payload_fields": [
            "trusted_fact_kind",
            "venue_order_id"
          ],
          "field_schemas": {
            "trusted_fact_kind": {
              "type": "enum",
              "values": [
                "ACKNOWLEDGED",
                "REJECTED",
                "PARTIAL_FILL",
                "FULL_FILL",
                "CANCEL_CONFIRMED",
                "REPLACE_CONFIRMED",
                "EXPIRED"
              ]
            },
            "venue_order_id": {
              "type": "string"
            }
          },
          "nullable_fields": [
            "venue_order_id"
          ],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "COMMAND_ACCEPTED": {
          "safe_payload_fields": [
            "command_id",
            "operation_type"
          ],
          "field_schemas": {
            "command_id": {
              "type": "id",
              "prefix": "cmd"
            },
            "operation_type": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "COMMAND_REJECTED": {
          "safe_payload_fields": [
            "command_id",
            "denial_code"
          ],
          "field_schemas": {
            "command_id": {
              "type": "id",
              "prefix": "cmd"
            },
            "denial_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "COMMAND_REPLAYED": {
          "safe_payload_fields": [
            "command_id"
          ],
          "field_schemas": {
            "command_id": {
              "type": "id",
              "prefix": "cmd"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "IDEMPOTENCY_CONFLICT": {
          "safe_payload_fields": [
            "command_id"
          ],
          "field_schemas": {
            "command_id": {
              "type": "id",
              "prefix": "cmd"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "EVENT_REPLAY_IGNORED": {
          "safe_payload_fields": [
            "replayed_audit_event_id"
          ],
          "field_schemas": {
            "replayed_audit_event_id": {
              "type": "id",
              "prefix": "evt"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        },
        "EVENT_REJECTED": {
          "safe_payload_fields": [
            "rejected_audit_event_id",
            "reason_code"
          ],
          "field_schemas": {
            "rejected_audit_event_id": {
              "type": "id",
              "prefix": "evt"
            },
            "reason_code": {
              "type": "string"
            }
          },
          "nullable_fields": [],
          "required_scope": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id"
          ]
        }
      },
      "fingerprint": "SHA-256 over canonical NFC/UTF-8 JSON of the complete immutable event envelope except exact event_fingerprint_sha256 field",
      "identity_policy": {
        "canonical_source": "canonical_domain_vocabulary.json /entity_kinds",
        "durable_ids": {
          "OrderIntent": {
            "field": "order_intent_id",
            "prefix": "oint"
          },
          "Order": {
            "field": "order_id",
            "prefix": "ord"
          },
          "Fill": {
            "field": "fill_id",
            "prefix": "fill"
          },
          "AuditEvent": {
            "field": "audit_event_id",
            "prefix": "evt"
          }
        },
        "command_identity": {
          "field": "command_id",
          "format": "cmd_<lowercase canonical UUIDv7>",
          "classification": "immutable idempotency identity, not aggregate identity",
          "scope": [
            "workspace_id",
            "environment",
            "exchange_account_id",
            "command_id"
          ],
          "order_id_is_command_id": false
        },
        "external_ids": [
          "venue_order_id",
          "venue_trade_id",
          "client_order_id"
        ],
        "external_ids_are_not_durable_ids": true
      }
    },
    "M11": {
      "current_state_store_schema_version": 2,
      "physical_schema_registry_entries": [
        {
          "state_store_schema_version": 1,
          "sqlite_schema_fingerprint_sha256": "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"
        },
        {
          "state_store_schema_version": 2,
          "sqlite_schema_fingerprint_sha256": "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"
        }
      ],
      "rollback_policy": "FORWARD_ONLY"
    }
  },
  "safe_code_type": {
    "type": "SAFE_CODE",
    "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
    "normalization": "NFC",
    "minimum_length": 1,
    "maximum_length": 64,
    "purpose": "bounded non-narrative code only; not an open business-value registry"
  },
  "retained_state_authority": {
    "precondition": "ordinary append, replay, restore decision and next seal/prune proceed only from a verified current retained journal state",
    "full_verifier": [
      "checkpoint chain and compact segment integrity",
      "exact checkpoint-to-pruning-evidence graph",
      "active suffix content and chain",
      "active and compact identity-index exactness",
      "active/compact disjointness",
      "global audit_event_id uniqueness"
    ],
    "pending_exception": "exact latest-checkpoint recovery evidence uses the pending descriptor gate and verifies all prior completed history plus latest checkpoint/compact integrity while allowing only its one missing evidence relation",
    "implementation_neutrality": "reference oracle may eagerly reverify; production may use an equivalent validated durable tail/index/checkpoint proof or generation-bound verification state and need not rescan O(N)",
    "repair": "verification detects and never silently rebuilds, rekeys or repairs an index",
    "compact_coverage": "all compact entries map exactly once to ordered committed checkpoint ranges; sequences are unique and exactly cover their authenticated ranges; zero checkpoints requires zero compact entries",
    "evidence_graph": "classification-based exact checkpoint/evidence bijection; no count-based substitution and no orphan active/compact pruning evidence",
    "authority_failure": "orphan compact/evidence blocks verify, append/replay, restore and seal/prune before identity lookup or mutation"
  }
}
```

## `downstream_operation_declarations`

```json
[
  {
    "owner_milestone": "M0.12",
    "operation": "M0.12/ALERT_ACKNOWLEDGE",
    "factor_policy": "PIN",
    "freshness_seconds": 60,
    "authorization_scope": "alert_lifecycle",
    "environments": [
      "PAPER",
      "TESTNET",
      "LIVE"
    ],
    "target_scope_contract": [
      "alert_id",
      "expected_alert_revision",
      "alert_scope"
    ],
    "mutation_binding_contract": [
      "intent",
      "value"
    ],
    "declared_intent": "ACKNOWLEDGE",
    "definition_revision": 1
  },
  {
    "owner_milestone": "M0.12",
    "operation": "M0.12/ALERT_SET_SUPPRESSION",
    "factor_policy": "PIN",
    "freshness_seconds": 60,
    "authorization_scope": "alert_lifecycle",
    "environments": [
      "PAPER",
      "TESTNET",
      "LIVE"
    ],
    "target_scope_contract": [
      "alert_id",
      "expected_alert_revision",
      "alert_scope"
    ],
    "mutation_binding_contract": [
      "intent",
      "value"
    ],
    "declared_intent": "SET_SUPPRESSION",
    "definition_revision": 1
  },
  {
    "owner_milestone": "M0.12",
    "operation": "M0.12/ALERT_CLEAR_SUPPRESSION",
    "factor_policy": "PIN",
    "freshness_seconds": 60,
    "authorization_scope": "alert_lifecycle",
    "environments": [
      "PAPER",
      "TESTNET",
      "LIVE"
    ],
    "target_scope_contract": [
      "alert_id",
      "expected_alert_revision",
      "alert_scope"
    ],
    "mutation_binding_contract": [
      "intent",
      "value"
    ],
    "declared_intent": "CLEAR_SUPPRESSION",
    "definition_revision": 1
  },
  {
    "owner_milestone": "M0.12",
    "operation": "M0.12/ALERT_MANUAL_FACT_RESOLUTION",
    "factor_policy": "PIN",
    "freshness_seconds": 60,
    "authorization_scope": "alert_lifecycle",
    "environments": [
      "PAPER",
      "TESTNET",
      "LIVE"
    ],
    "target_scope_contract": [
      "alert_id",
      "expected_alert_revision",
      "alert_scope"
    ],
    "mutation_binding_contract": [
      "intent",
      "value"
    ],
    "declared_intent": "MANUAL_FACT_RESOLUTION",
    "definition_revision": 1
  }
]
```

## `current_downstream_operation_definition_revisions`

```json
{
  "M0.12/ALERT_ACKNOWLEDGE": 1,
  "M0.12/ALERT_SET_SUPPRESSION": 1,
  "M0.12/ALERT_CLEAR_SUPPRESSION": 1,
  "M0.12/ALERT_MANUAL_FACT_RESOLUTION": 1
}
```
