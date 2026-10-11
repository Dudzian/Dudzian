# Roadmap — Post-Beta Internationalization & Localization (I18N)

**Status: PLANNED — implementation starts after the first beta; no release or readiness claim.**
**Product:** CryptoHunter's canonical Windows PySide6/QML DesktopShell and user-facing notifications/dialogs/tray surfaces.
**Owner boundary:** presentation-only localization; Core, risk, execution, credentials, entitlement, audit event identities and signed payloads remain authoritative and locale-independent.
**Canonical milestone reference:** [Product Delivery Roadmap — POST-BETA-I18N](../architecture/cryptohunter_product_architecture/product_delivery_roadmap.md).

## Scope boundary and sequencing

- **Before first beta (M4.2):** make the rebuilt, genuinely interactive UI translation-ready while touching its components. Use Qt-compatible extraction hooks (for example `qsTr()`/Python translation functions or equivalent stable keys), avoid unextractable hard-coded text in newly developed surfaces, separate machine-readable Core codes from their display messages and keep translation catalogs out of trusted Core data. This is a limited architectural seam, **not** a requirement to ship multiple languages, a flag selector or full localization test coverage for beta.
- **After beta starts:** implement I18N-1 through I18N-6 in a separate reviewed set of changes, based on actual operational feedback. The language feature can ship independently of Stage 11/12 and must not silently modify M5/M6 or PB exit criteria.
- Initial proposed locales are **Polish (`pl-PL`)** and **English** (exact language/region tag selected during catalog review). Additional locales (e.g. Icelandic, German, Spanish) are candidates, not commitments or implemented status.
- One application and one runtime; locale resources are versioned catalogs, not forked products. Where offline distribution is required, approved catalogs are included in signed/reproducible release artifacts.

## I18N-1 — Inventory and translation infrastructure

- Inventory visible text in current canonical QML/PySide6 pages, navigation, settings, onboarding, tooltips, loading/empty/degraded/error states, notifications, tray, dialogs, confirmation windows, strategy explanations and operator safety warnings.
- Define a shared locale/message-key policy for QML and Python and a translation extraction/update workflow. Handle context, pluralization, positional/named parameters and long text without concatenating translatable fragments.
- Keep API and Core domain/status codes stable; map them to translated explanations exclusively at the UI boundary. Establish catalog validation, ownership and review for safety-critical terminology.
- No localization-dependent branch may grant permissions or change trading state.

## I18N-2 — Language catalogs and quality

- Prepare and review candidate Polish and English catalogs for the supported UI; determine one exact canonical fallback locale and locale tags before release.
- Cover operator-critical labels and warnings consistently: buy/sell, limit/market orders, quantity, fees, stop loss, risk, liquidation, cancel, PAPER/TESTNET/LIVE and kill switch.
- Track missing, stale, duplicate or invalid translations and catalog completeness. A candidate language does not ship unless safety-critical translations are reviewed and full supported-surface coverage requirements are met.
- Subsequent languages are independently versioned/tested; support native language names and avoid associating an entire language with a single country flag.

## I18N-3 — Runtime language control

- A compact, discoverable header/Settings selector shows a flag or locale icon **together with** native language name/code; click/tap opens the available-language picker. Do not rely on endless flag-only cycling when several languages are available.
- Selection applies immediately without UI/Core restart. Live-retranslate existing views, Python dialogs, window titles, menus, tooltips, tray entries, notification templates and subsequently opened surfaces.
- Preserve navigation position, in-progress forms, unsaved edits, asynchronous callbacks and operational command state. Language changes must not resend trading commands, duplicate orders, alter account/environment routing or unlock blocked actions.
- Use Qt's supported translator/retranslation mechanisms consistently in QML and PySide6. Correctly update dynamic parameters and plural forms.

## I18N-4 — Persistence, fallback and secure boundaries

- Store the locale preference as ordinary per-user presentation configuration; restore it after restart and keep it independent from exchange accounts/credentials and Core execution state.
- Unsupported or corrupt catalogs fall back predictably to the approved canonical language. Never expose raw translation keys as security decisions; log translation failures without secrets.
- Locale choice must not alter identifiers, machine-readable logs, canonical JSON, cryptographic hashes/signatures, license/entitlement evidence, exchanges' API payloads, order parameters or risk thresholds.

## I18N-5 — Regional presentation

- Display dates, times, decimal/grouping separators, relative timestamps and user-facing currency formatting according to the selected locale where appropriate.
- Preserve exact numeric values, currency/asset codes, time standards in source data, machine parsers, order amounts and stored history. Any editable numeric field must parse and normalize safely with explicit validation; switching language cannot silently reinterpret submitted values.
- Plan RTL/layout mirroring separately if such locales are later approved; do not claim unsupported fonts, locales or rendering modes are implemented.

## I18N-6 — Acceptance and qualification

Required tests for **each shipped locale**:

1. Select locale in app header/Settings and verify immediate live retranslation without restarting, including already-open dialogs/tray and newly presented messages.
2. Save/restart/reconnect and verify retained choice; test missing/stale/corrupt catalogs and deterministic fallback.
3. Exercise plural forms, dynamic parameters, long strings, diacritics, keyboard/focus, accessible control names and supported Windows DPI/resolution profiles.
4. Use real Core-bound PAPER/TESTNET UI, including concurrent account/status updates; verify zero changes to Core command intents, order identity, execution semantics, active strategies and risk/kill-switch authority across repeated language changes.
5. Cover confirmations and errors for credentials, exchange routing, live activation denial, order cancellation, stale data, rejected orders and permission/connection problems. Safety-critical meaning must be independently checked by a reviewer competent in the shipped locale.
6. Run PySide6/QML native tests and packaged Windows executable smoke; check catalog completeness and reject unreviewed languages from release artifacts.

**Exit:** verified, persistent and accessible runtime language picker; catalog and fallback validation; full documented shipped-locale UI coverage; native test and negative safety evidence. There is no first-beta requirement to provide this capability.

## Out of scope

- No separate language-specific builds or forks.
- No translation or mutation of broker/exchange instrument names, signed security payloads, credentials, raw API fields or historical audit identifiers.
- No automated unreviewed translations on operator-critical safety screens.
- No new authority from UI state, no changes to Stage 9/10 gates, frozen contracts, current beta readiness markers or independent Stage 11/12 priorities.
