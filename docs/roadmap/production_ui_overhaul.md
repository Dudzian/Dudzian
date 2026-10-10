# Roadmap — Production UI Overhaul & Core Integration

**Status: PLANNED — M4.2/M4.3, mandatory before the first beta.**
**Scope:** the canonical Windows PySide6/QML DesktopShell, backed by actual CoreHost runtime and typed commands/read models. Existing static/local preview screens and alternate legacy desktop stacks are not evidence of working product UI. Do not replace them with a second round of clickable mockups.

## UI-1 — End-to-end inventory and redesign

- Inventory every QML view/control/route/interaction in `ui/pyside_app` and relevant older UI paths; classify KEEP/ADAPT/REWRITE/DELETE against the canonical architecture and beta user tasks.
- Define a clear navigation and information architecture with one consistent operational home, device/session/account/environment context, and no dead-end buttons.
- Replace local-only preview data wherever a beta-critical function should display actual runtime state. Preserve preview/testing only as explicitly labeled isolated modes.
- Build incrementally as running vertical slices: Core contract → read model → UI component → command → feedback/error/reconnect → E2E proof; do not deliver static Figma/screenshots/fixtures as completion.

## UI-2 — Exchanges & Accounts (first-class working settings)

A dedicated **Settings → Exchanges & Accounts** surface must support multiple accounts and subaccounts, including separate PAPER, TESTNET and LIVE profiles for each qualified venue where available.

- Venue picker derived only from the supported build-time Exchange Registry; **Zonda must never appear**.
- Account display label, exact supported market type, environment, portfolio and route assignments, status, account identity, balance/open-order/permission/readiness views.
- Masked credential entry (API key, secret and exchange-specific passphrase), secure write-only handoff to Core-managed OS secret store; opaque `secure-store://` metadata in read models. No plaintext persistence in QML, logs, YAML, analytics, crash dumps or exported diagnostics.
- Connection check (public/private), account identity verification, observed scope check, credential replacement/rotation/revoke, withdrawal-permission rejection, IP allowlist guidance when supported and immediate halt on auth/reconciliation mismatch.
- No TESTNET/LIVE cross-use or hidden fallback; unavailable sandboxes clearly disabled; no UI toggle can override ProductCapabilities, Risk Engine, ExecutionLease or legal enrollment.
- Users can configure supported accounts without touching source/config files or CLI; availability and activation state must be explained in the UI.

## UI-3 — Real operational product (no mock-only controls)

Implement full beta-scope interactive panels for: operational Dashboard/Command Center, trading and orders/fills, positions/portfolio/equity/PnL, risk limits and kill switch, strategies and routing, market scanner/market data, paper/backtesting, alerts/audit/diagnostics, updates/rollback and settings.

Every visible beta-critical control must be a real command, action or explicit disabled state with reason. Never silently simulate a save, connection, order, live activation or update. Status indicators must reflect real Core state with timestamps/provenance, not QML-local booleans.

## UI-4 — Interaction quality and safety

- Global loading, empty, stale, disconnected, reconnecting, degraded, blocked, permission denied, invalid input, retry and fatal error states; accessible keyboard/focus support, readable layouts and Windows DPI/resolution handling.
- Per-venue and aggregate status/health and clear separation of PAPER/TESTNET/LIVE. No simultaneous account ambiguity in any order/strategy action.
- Model safe cancellation, confirmation for sensitive actions, asynchronous progress and completion/error receipts, and persistence of user settings across restarts.
- Do not grant execution authority through display state or UI controls. Core remains owner of validation, credentials, portfolio capital, routes and licensing.

## UI-5 — Qualification and beta exit

- Native PySide6/QML integration tests, UI ↔ Core command/read-model tests, Windows packaged EXE/MSI smoke and replay/reconnect scenarios.
- End-to-end operator journey, **without CLI**: first launch → add PAPER account → simulate order → observe fill/risk/PnL → add independent qualified TESTNET accounts → verify credentials privately → route independent orders safely → inspect audit → reconnect/restart and recover exact state.
- Every visible beta-critical button/field is implemented and testable; all preview-only controls are either clearly segregated from the operator workflow or removed.
- UX inspection is performed on the running compiled product, never screenshot approval alone.
- Any beta-critical mock, dead button, fake API credential form, wrong-account routing, leaked secret or missing risk/permission feedback is a PB BLOCKER/HIGH finding.

**Sequence:** UI-1 inventory → UI-2 account settings and UI-3 real runtime vertical slices (can proceed together once canonical commands exist) → UI-4 quality → UI-5 native E2E → PB repository-wide final audit → beta.
