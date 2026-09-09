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
"IN_PROGRESS_AUDIT_JOURNAL_CLOSED"
```

## `contract_identity`

```json
{
  "contract_id": "M0.12-audit-observability-alerts-updater",
  "version": "0.2.5",
  "phase": "S9B_C5_AUDIT_JOURNAL_CONTRACT",
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
  "no observability/health, alert or updater closure in S9B"
]
```

## `open_items`

```json
[
  "architecture-approve and implement the physical durable AuditEvent journal/carrier relative to frozen M0.11; canonical audit readiness remains blocked until then",
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
