# Roadmap — OPS-ALERT-01: Telegram Alerts & Independent Watchdog

## Status and scheduling

**PLANNED — post-Stage 9 cross-cutting workstream, not implemented or qualified by this document.**

Begin only after Stage 9 closure. Engineering may be scheduled during Stage 10/platform stabilization, but this workstream **does not** change or certify Stage 9, Windows Stage 10, the frozen Windows Stage 0–14 contracts, M0–M6, legal enrollment, external Root Proof authority, or production readiness. Its end-to-end acceptance is required **before any claim that CryptoHunter can operate unattended**, especially for a guarded LIVE canary. It is not a new numbered Stage 10/11/12.

## Existing building blocks to reuse and verify

- `bot_core/alerts/channels/telegram.py` — Telegram Bot API transport.
- `bot_core/runtime/observability.py` — environment-dependent channel construction and alert routing; an offline profile skips network channels.
- `scripts/send_alert.py` and `scripts/get_telegram_chat_id.py` — manual diagnostics; not a substitute for runtime integration.
- `config/core.yaml` — sample `telegram:primary` wiring and a **placeholder** chat ID; production secrets must be injected outside the repository.
- `docs/audit/paper_trading_log.md` — historical successful Telegram smoke, **not** current Stage 9 or production evidence.

First action: inventory actual producers/consumers, runtime lifecycles, current Telegram delivery/health tests, the active environment, persisted incident records and operator configuration. Record KEEP / ADAPT / REWRITE decisions; never assume a configured channel is online.

## OPS-ALERT-1 — Delivery and incident contract

- Define incident identity, severity (`INFO`, `WARNING`, `CRITICAL`), source, environment, UTC time, state transitions (`OPEN`, `ACKNOWLEDGED`, `RECOVERED`), redacted context and deterministic correlation.
- Map critical causes: process crash, missing heartbeat, host unreachable, stale exchange/market data, risk-gate failure, adapter/API outage, persistence/reconciliation error, credential/secret expiry, and notification-channel degradation.
- Register telemetry producers explicitly; prevent noisy duplicates and alert storms. Critical events must remain visible despite throttling policy.
- Ensure failed sends, retries, acknowledgements and recovery are retained in an auditable incident log; do not treat enqueued messages as delivered.

## OPS-ALERT-2 — Independent heartbeat and failure detection

- Emit a monotonic-sequence or otherwise replay-resistant, authenticated heartbeat from the actual running Core, with runtime instance ID, freshness, mode and non-sensitive health summary.
- Establish configurable publish/expiry thresholds, clock-skew handling and startup grace; distinguish `STARTING`, `HEALTHY`, `DEGRADED`, `MISSING` and `RECOVERED`.
- Run the watchdog **outside the monitored process**. Detect full-host shutdown and lost local connectivity by placing a watcher in a **separate failure domain** (for example a remote host/service). A local-only timer cannot report a dead machine.
- Implement idempotent incident transitions, durable state across restarts and alert after bounded missed heartbeats. Avoid false recovery from a single stale/replayed event.
- Explicitly document dependency on Internet, Telegram availability and the independent watcher's own health.

## OPS-ALERT-3 — Telegram transport hardening

- Use the existing `TelegramChannel` and secret manager; support the operator-owned private chat/channel and confirm bot permissions, never commit tokens or a real chat ID.
- Escape/sanitize Telegram formatting and limit message length and payload fields. Do not disclose API keys, positions/account identifiers, credentials, private data or sensitive diagnostic attachments.
- Handle API rejection, 401/403, 429/Retry-After, timeouts, DNS/network partitions and invalid chat IDs with bounded retries, backoff, safe audit and escalation/fallback behavior.
- Include an incident ID, concise actionable diagnosis, severity, timestamp, environment, recommended next step and a **recovery notification** once independently verified.
- A delivery failure must be observable by an independent channel/watchdog; Telegram cannot be its own only health monitor.

## OPS-ALERT-4 — Operations and safe-state policy

- Provide operator procedures for channel configuration, test delivery, incident acknowledgement, controlled silence, escalation, credential rotation, lost connectivity, recovery and watchdog restart.
- Define who can acknowledge/resolve critical incidents and how long an unattended environment may remain without verified monitoring. Do not invent an implicit trade-authorization policy.
- Monitoring must never bypass Risk Engine, ExecutionLease, exchange permissions, kill switch, release authority or signed environment gates.
- Telegram unavailability alone must **not** initiate unsafe liquidation or permit unbounded trading; design the environment-specific fail-safe separately and require explicit authorization for LIVE behavior.

## OPS-ALERT-5 — Acceptance evidence (future execution; not run in this roadmap PR)

- Prove real Telegram delivery from the configured runtime and the independent watchdog to the operator-owned destination, with timestamped/audited delivery outcome.
- Execute bounded synthetic failures: crashed Core process, terminated host/unreachable network, missed/late/replayed heartbeat, invalid token/chat ID, Telegram 403/429/timeout, watcher restart, duplicated alert, and recovery.
- Confirm alarm freshness within explicitly approved SLO, deduplication, escalation after missing acknowledgement, correct `RECOVERED` transition and retained evidence. Determine SLO values during implementation; do not claim unmeasured latency.
- Verify real remote-watcher separation for host failures and recovery after disconnection; local-only tests do not prove host-outage alerting.
- Cover PAPER/TESTNET before considering any guarded LIVE rollout. Preserve all Stage 9/10, production and manually approved qualification boundaries.

## Definition of done

- [ ] Existing Telegram/alert integrations audited and wiring verified for the selected runtime.
- [ ] Secure credentials and explicit environment/channel enablement in place.
- [ ] External watchdog operational across a separate failure domain.
- [ ] End-to-end synthetic incident and Telegram delivery evidence archived.
- [ ] Alert noise, missing-heartbeat, outage, retry, acknowledgement and recovery paths qualified.
- [ ] Operator runbook, recovery/fallback plan and monitoring ownership approved.
- [ ] Independent risk/production authorization gates unchanged and qualified separately.

**Current state:** backlog entry only. No code changes, tests, host actions, notifications, production approvals or Stage 9/10 status changes are implied by this documentation update.
