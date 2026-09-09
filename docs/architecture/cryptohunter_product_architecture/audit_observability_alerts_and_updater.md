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
"IN_PROGRESS_FOUNDATION"
```

## `contract_identity`

```json
{
  "contract_id": "M0.12-audit-observability-alerts-updater",
  "version": "0.1.0",
  "phase": "S9A_CONTRACT_FOUNDATION",
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
      "optional_references"
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
      }
    ],
    "C_UNRESOLVED_IDENTITY_TO_CLOSE_LATER": [
      {
        "field": "alert_id",
        "status": "M0.12-local/unresolved; exact identity semantics deferred"
      },
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
    "ordering": "per-environment monotonic sequence with causal references",
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
  "projection_rule": "dashboards, diagnostic bundles, notifications and UI are lossy rebuildable projections"
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
    "UNKNOWN": "no current trusted evidence",
    "OK": "all required current checks pass",
    "DEGRADED": "service operates with reduced quality but no forbidden capability is opened",
    "BLOCKED": "specific capability must not proceed"
  },
  "composition": [
    "never collapse dimensions into one healthy boolean",
    "readiness is per capability, environment and scope",
    "M0.3 startup readiness and process lifecycle are inputs, not replaced",
    "M0.4 ProductCapabilities and environment_readiness remain authority",
    "UNKNOWN safety evidence is BLOCKED for the affected capability",
    "telemetry cannot promote readiness"
  ]
}
```

## `alert_model`

```json
{
  "required_fields": [
    "alert_id",
    "category",
    "severity",
    "environment",
    "scope",
    "source_reference",
    "raised_at",
    "lifecycle",
    "acknowledgement",
    "resolution",
    "deduplication_key",
    "suppression",
    "escalation",
    "operator_visibility",
    "revision"
  ],
  "lifecycle": {
    "RAISED": "active and not acknowledged",
    "ACKNOWLEDGED": "operator awareness recorded; source fault remains active",
    "RESOLVED": "trusted resolution evidence recorded; history retained"
  },
  "principles": [
    "stable identity survives redelivery",
    "deduplication groups equivalent active conditions without deleting history",
    "suppression affects delivery noise, never source fault or safety action",
    "escalation changes routing/visibility, not upstream authority",
    "condition resolution requires fresh trusted evidence",
    "fact alerts resolve only by explicit canonical policy",
    "alerts do not create ProductCapabilities, readiness, authentication, risk, execution, persistence or update authority"
  ]
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
    "reason": "usable presentation code after projection-only and non-boolean readiness alignment"
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
  "no complete field-level schemas for every M0.12 record in S9A"
]
```

## `open_items`

```json
[
  "close exact AuditEvent and append transaction schemas, ordering domains, tamper-evidence chain and retention matrix",
  "decide the physical durable AuditEvent journal/carrier relative to the frozen M0.11 PersistenceRecord registry without adding a registry entry or StateStore schema v3 in S9A-C1",
  "close alert category/severity policies, deduplication windows, suppression authorization, escalation routes and durable schemas",
  "close probe-specific freshness budgets and capability-scoped health composition",
  "select release-channel registry, manifest canonicalization, signing algorithms, trust-root provisioning, rotation and revocation",
  "close UpdatePlan/UpdateAttempt/rollback state machines and crash matrix",
  "define exact operator authorization policy per update/rollback class using M0.10 primitives",
  "define product-version and StateStore compatibility matrix including forward-repair paths",
  "close pre-update backup and post-install verification evidence schemas",
  "define privacy/redaction/export policy and telemetry retention/cardinality budgets",
  "map every individual legacy implementation path to the M1 migration backlog after M0.12 closure"
]
```
