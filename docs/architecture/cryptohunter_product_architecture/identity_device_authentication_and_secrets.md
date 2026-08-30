# M0.10 — identity, device, PIN, biometrics and secrets

## Status i granica

M0.10 jest zamkniętym, implementacyjnie neutralnym kontraktem bezpieczeństwa. Nie implementuje runtime'u, persystencji, keychaina, kryptografii produkcyjnej, biometrii platformowej ani skutków giełdowych. Trwałość i atomowe odtwarzanie pozostają w M0.11.

## Authority

Właścicielem authority jest `CoreHost`. Publiczny model nie zawiera metod accept/register/seed/set-current; modułowo prywatny harness testowy wyłącznie reprezentuje authority już zaakceptowaną przez M0.3, platformę, M0.5 albo immutable Core product policy i nie jest product API. Integralność nie jest authority: fingerprint wykazuje wyłącznie integralność treści, a authority ustanawia wcześniejsze accepted/current członkostwo w rejestrze Core. UI, TrayAgent, DesktopShell, caller boolean, caller role, nominalny obiekt i self-hash są nie-authority. Identity, authentication i authorization są trzema odrębnymi granicami; poprawne uwierzytelnienie nie nadaje entitlementu do operacji.

## Bootstrap M0.3

M0.10 interpretuje bezpośrednio `/first_run_bootstrap_authority_contract` z finalnego M0.3 i rewaliduje genuine accepted/current transition o celu `INITIAL_SECURITY_ESTABLISHMENT_ONLY`. Ręcznie utworzony lub self-hashed `BootstrapTransitionResult` nie daje authority. Modułowo prywatny frozen `_M03TrustedConsumedBootstrapEvidence` zawiera jawne upstream claim, challenge, PRE, POST i consumed authority; `M03BootstrapAuthorityView` jest projekcją evidence, nigdy odwrotnie, i nie istnieje default/fallback tworzący brakującą wartość. Bridge wymaga pre-existing exact membership wiążącego pełny view, historical PRE, current POST, consumed authority oraz jawne consumed claim/challenge pochodzące z trusted M0.3 evidence. Shape, recomputed binding i poprawne fingerprinty nie wystarczają; purpose musi być dokładnie initial-only, identyfikatory canonical, a generation/revision dodatnie i non-bool. Transition przygotowuje i waliduje cały shadow POST, po czym publikuje komplet wraz z consumption albo zero zmian. Jednorazowy bootstrap może ustanowić pierwszą current `OperatorIdentity`, pierwszy current `TRUSTED` device, pierwszy verifier PIN oraz początkowe generacje security/session. Potem jest terminalnie fenced i nie autoryzuje drugiego urządzenia, zwykłych operacji, LIVE ani `ExecutionLease`.

## Identity i device trust

Authority bieżącego operatora wynika z prywatnego rejestru accepted projekcji oraz Core-owned mapy scope→accepted fingerprint, a nie z caller-created `ACTIVE` lub `is_current=True`. `DeviceInstallation` jest tożsamością urządzenia, nie jego zaufaniem. Device trust używa analogicznej accepted/current mapy; executable graph dopuszcza `ABSENT→TRUSTED`, `ENROLLED_UNTRUSTED→TRUSTED` i `TRUSTED→REVOKED`, odrzuca `TRUSTED→TRUSTED`, a `REVOKED`/`REPLACED` są terminalne dla tego DeviceInstallation ID. OperatorIdentity `REVOKED` jest terminalne dla tego operator ID. Identity revision, device trust revision, platform enrollment revision oraz identity/device security generations są monotonicznymi fencing epochs bez rollbacku; historyczne `TRUSTED` nie wystarcza, a `REVOKED`, `REPLACED` i zmiana rewizji/generacji unieważniają stare proofy.

## PIN i biometria

Raw PIN istnieje wyłącznie na wejściu porównania Core. Referencyjny deterministyczny verifier służy tylko testowi architektury i nie jest produkcyjnym KDF. Błąd zwiększa licznik, trzeci błąd ustala dokładny lockout, poprawny PIN po wygaśnięciu lockoutu zapisuje nowy current record z wyzerowanym licznikiem i pustym lockoutem, a change/reset podnosi `pin_revision`. Raw PIN nie jest serializowany, audytowany ani fingerprintowany. Caller nie może dopisać faktora `PIN`.

CryptoHunter nie przechowuje materiału biometrycznego. Faktor `BIOMETRIC` pochodzi wyłącznie z walidacji platform assertion: `SUCCESS`, dokładne account/device, current enrollment revision, Core-derived challenge i inclusive freshness. Challenge jest deterministycznie wyprowadzany z pełnego request context, enrollment oraz security/session generations; caller nie wybiera expectation i assertion nie może być replayowane między operacjami lub requestami. Bool `biometric_ok` i caller factor set są nie-authority. `AuthenticationProof.factor_set` jest wyłącznie wynikiem derived Core; polityka `PIN_AND_BIOMETRIC` wymaga obu dowodów.

## Proof, sesja i authorization

Pola `AuthenticationProof` w JSON i dataclass są identyczne i uporządkowane. Sam obiekt lub poprawny self-hash nie może się self-enrollować. Authority proofa stanowi lookup fingerprintu w prywatnym, pre-existing rejestrze `CoreIssuedAuthenticationProofBinding`; publiczne `authorize(proof, request, now)` nie przyjmuje bindingu, current state, registry ani entitlement bool. Core sprawdza exact registry key, `CoreHost`, wszystkie pola bindingu, a następnie current identity/device/PIN/enrollment/security/session oraz exact account, operator, device, environment, operation, scope, mutation, causation i correlation. Proof jest wydawany tylko przez Core flow, który rozwiązuje current registries, wykonuje PIN/platform evidence, interpretuje exact operation policy i sam wyprowadza `factor_set`.

Czas jest canonical UTC; każda publiczna ścieżka odrzuca naive datetime i offset inny niż zero bez wyjątku. Obowiązuje `issued_at <= now <= expires_at`, `expires_at > issued_at` oraz inclusive `now-issued_at <= freshness_seconds`. Każda M0.10-owned operacja wiąże canonical scope oraz bezpieczny actual-mutation descriptor z request i proof fingerprintami; raw PIN nie uczestniczy w descriptorze. LOCK_SESSION może wyprowadzić tylko LOCKED, LOGOUT_SESSION tylko LOGGED_OUT, a unlock ma osobną factor path, która przed faktorami wymaga exact canonical M0.10 scope fingerprint. PIN revision i session generation są także monotoniczne: PIN same-revision dopuszcza wyłącznie failed-attempt/lockout state updates bez zmiany credential content, a inny session content przy tej samej generation jest denied. Proof raz stale po advance któregokolwiek epoch nie może zostać wskrzeszony current-state rollbackiem. Wykonywalne lock/logout tworzą accepted/current stan z wyższą session generation; unlock wymaga current entitlementu i nowych PIN+platform evidence, także podnosi generację. Stare proofy są `PROOF_STALE`. Samo istnienie `RuntimeSession`, także legalne podczas M0.3 `SETUP_REQUIRED`, nie uwierzytelnia. Osobny accepted/current `OperationEntitlementProjection` wiąże operację i scope; auth success bez entitlementu kończy się `AUTHORIZATION_DENIED`.

## Sekrety

Sekrety pozostają poza domeną. M0.10 konsumuje dokładną gramatykę M0.5: prefix `secure-store://`, niepusty opaque locator, bez whitespace, `?`, `#`, `=` i markerów payloadu wskazanych przez M0.5. `keyring://` jest `SECRET_INVALID`. Metadata używa pluralnego `permitted_operations`. `validate_secret_use` rozwiązuje accepted metadata przez prywatną Core mapę current scope→fingerprint; caller-created current binding nie jest wejściem. Exact current `AVAILABLE` może zostać użyte, a old/rotated/revoked/replaced są fenced.

## LiveAccessGrant i LIVE

Actual device target, grant id, policy scope, target state i current→next revision muszą odtworzyć dokładny mutation fingerprint proofa przed zmianą stanu. `GRANT_LIVE_ACCESS`, `SUSPEND_LIVE_ACCESS` i `REVOKE_LIVE_ACCESS` są wykonywalnymi M0.10-owned transitions wymagającymi legalnego proofa i entitlementu. `TRUST_DEVICE`/`REVOKE_DEVICE`, PIN i session transitions również należą do M0.10; credential/profile, risk, kill-switch i ProductCapabilities mutations kończą się na authorized security request do upstream owner. Dla tych handoffów M0.10 wiąże opaque fingerprint, lecz upstream owner musi niezależnie wyprowadzić fingerprint actual mutation i wymagać równości przed zmianą. Current grant designation używa dokładnego klucza `(account_id, device_installation_id, policy_scope_fingerprint_sha256)`, niezależnego od historycznego grant ID. Grant identity i scope lifecycle są rozdzielone: unseen ID zaczyna jako ACTIVE revision 1, ten sam REVOKED ID jest terminalny i nigdy nie może być użyty ponownie; current ACTIVE/SUSPENDED scope blokuje drugi grant, natomiast current REVOKED scope może przełączyć designation na fresh distinct never-before-seen ID. Cała stara historia pozostaje immutable i fenced. Dla zajętego installation/policy scope drugi ACTIVE grant jest denied; grant ID nie może zmienić account/device/policy/operator parent. Security generation jest authorization epoch każdej projekcji, nie immutable identity parent: nowa projekcja wiąże current coherent identity/device/proof generation, a historyczna lub current projekcja ze starą generation jest stale. SUSPEND wymaga current ACTIVE predecessor, a REVOKE current ACTIVE albo SUSPENDED predecessor. Przed `GRANT_ACTIVE` validator zawsze rozwiązuje current `ACTIVE` OperatorIdentity oraz current `TRUSTED` DeviceTrust i wymaga, by obie security generations były dokładnie równe generation grantu. Revocation, replacement, untrusted designation albo generation drift terminalnie fence’ują użycie projekcji. Ponowna równość wartości nie cofa fence: terminalnych parentów nie można reaktywować, a generation nie może przejść 1→2→1 bez modyfikowania grant history. Security authority grantu wynika z lookupu accepted projekcji przez prywatną Core mapę current scope→fingerprint pełnej projekcji account/operator/device/policy/revision/generation; caller-created binding nie jest wejściem. Suspend/revoke fence'uje stare future-LIVE proofy. Grant nie jest `ProductCapabilities`, `RiskDecision` ani `ExecutionLease` i sam nie włącza LIVE. Current LIVE pozostaje denied przez M0.4/M0.6. Future LIVE jest wspieranym targetem dopiero po wszystkich gate'ach M0.4–M0.10. Nie ma fallbacku TESTNET→LIVE, a security success nie wydaje i nie zastępuje `ExecutionLease` ani walidatora M0.9.

## Audit i spójność kontraktu

Pure safe-payload projector przepuszcza tylko bezpieczne IDs, opaque references, fingerprints i reason codes; usuwa PIN, verifier, materiał biometryczny, API secret, private key, passphrase, bearer token i plaintext payload. Zamknięta taksonomia odróżnia zwykłe denial od `CONTRACT_INCONSISTENT`.

Canonical current designation jest wyłącznie mapą scope→accepted fingerprint; nie istnieją hybrydowe `Current*Binding` authority schemas. Authorization wymaga dokładnie `TRUSTED`; `ENROLLED_UNTRUSTED` nie autoryzuje. Unlock sam rewaliduje current identity/device/PIN/LOCKED session/exact entitlement oraz pełną generation coherence. Każda publiczna request path najpierw waliduje strukturę i failuje zamknięcie bez incidental exception. Security generation identity/device/PIN/session/entitlement/proof musi być spójna, inaczej Core zwraca `CONTRACT_INCONSISTENT`. `executable_boundary_schemas` jest dwukierunkowo exact z `M010_AUTHORITY_DATACLASSES`. Niezależny, głęboko immutable `EXPECTED_PROTOCOL` attestuje cały niezaufany JSON, exact top-level keys, wszystkie semantyczne roots, dependency pointery i canonical content fingerprints, w tym finalny bootstrap M0.3. M0.10 nie rości trwałości; persistence, migrations, backup i recovery należą do M0.11.

## Canonical Source-Closure Projection

The canonical JSON remains authoritative. This section is its deterministic projection for the source-closure rules.

### `canonical_integrity_fingerprint_policy`

```json
{
  "algorithm": "SHA-256",
  "input_shape": "EXACT_CLOSED_JSON_OBJECT_OF_EXPLICIT_SEMANTIC_INPUT_FIELDS",
  "json_canonicalization": {
    "sort_keys": true,
    "separators": [
      ",",
      ":"
    ],
    "ensure_ascii": false,
    "allow_nan": false
  },
  "encoding": "UTF-8",
  "array_policy": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER; NEVER_SORT_UNLESS_OWNING_CONTRACT_REQUIRES",
  "object_key_policy": "KEY_ORDER_NON_SEMANTIC; SORT_KEYS_CANONICALIZES",
  "number_policy": "NO_FLOAT_COERCION; VALIDATE_OWNING_FIELD_CONTRACT_BEFORE_HASHING",
  "decimal_string_policy": "VALIDATE_CANONICAL_OWNING_FIELD_CONTRACT; NO_FINGERPRINT_LAYER_NORMALIZATION",
  "timestamp_policy": "VALIDATE_CANONICAL_OWNING_TIMESTAMP_CONTRACT; NO_FINGERPRINT_LAYER_NORMALIZATION",
  "unicode_policy": "HASH_EXACT_VALIDATED_STRINGS; NO_HIDDEN_NFC_OR_NFD_TRANSFORMATION",
  "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
  "validation": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
  "authority_boundary": "INTEGRITY_ONLY; NEVER_CREATES_ACCEPTED_MEMBERSHIP, CURRENT_AUTHORITY, LIVE_READINESS, OR SELF_ENROLLMENT"
}
```

### `registries`

```json
{
  "identity_states": [
    "ACTIVE",
    "REVOKED"
  ],
  "device_trust_states": [
    "ENROLLED_UNTRUSTED",
    "TRUSTED",
    "REVOKED",
    "REPLACED"
  ],
  "session_states": [
    "LOCKED",
    "UNLOCKED",
    "LOGGED_OUT"
  ],
  "biometric_outcomes": [
    "SUCCESS",
    "FAILED",
    "CANCELLED",
    "UNAVAILABLE"
  ],
  "factors": [
    "PIN",
    "BIOMETRIC"
  ],
  "factor_policies": [
    "PIN",
    "BIOMETRIC",
    "PIN_AND_BIOMETRIC"
  ],
  "environments": [
    "PAPER",
    "TESTNET",
    "LIVE"
  ],
  "secret_kinds": [
    "API_KEY",
    "API_SECRET",
    "PASSPHRASE",
    "PRIVATE_KEY"
  ],
  "secret_states": [
    "AVAILABLE",
    "ROTATED",
    "REVOKED",
    "REPLACED"
  ],
  "grant_states": [
    "ACTIVE",
    "SUSPENDED",
    "REVOKED"
  ],
  "failure_codes": [
    "MALFORMED_UNTRUSTED_CONTEXT",
    "IDENTITY_INVALID",
    "IDENTITY_REVOKED",
    "DEVICE_NOT_TRUSTED",
    "DEVICE_REVOKED",
    "AUTHENTICATION_REQUIRED",
    "AUTHENTICATION_FAILED",
    "FACTOR_UNAVAILABLE",
    "PIN_LOCKED",
    "PROOF_EXPIRED",
    "PROOF_STALE",
    "AUTHORIZATION_DENIED",
    "SECRET_INVALID",
    "SECRET_UNAVAILABLE",
    "SECRET_REVOKED",
    "SECRET_STALE",
    "OPERATION_UNSUPPORTED",
    "CONTRACT_INCONSISTENT"
  ],
  "device_trust_transition_graph": {
    "TRUST_DEVICE": [
      "ABSENT->TRUSTED",
      "ENROLLED_UNTRUSTED->TRUSTED"
    ],
    "REVOKE_DEVICE": [
      "TRUSTED->REVOKED"
    ],
    "terminal_states": [
      "REVOKED",
      "REPLACED"
    ],
    "denied": [
      "TRUSTED->TRUSTED",
      "REVOKED->TRUSTED",
      "REPLACED->TRUSTED"
    ]
  },
  "identity_transition_rule": "REVOKED is terminal for the same OperatorIdentity; ACTIVE cannot be restored for that identity",
  "secret_use_operation_registry": [
    "PRIVATE_DATA",
    "ORDER_ENTRY"
  ]
}
```

### `executable_boundary_terminal_fingerprints`

```json
{
  "SessionSecurityState": {
    "field": "content_fingerprint_sha256",
    "algorithm": "SHA-256",
    "input_fields": [
      "account_id",
      "operator_id",
      "device_installation_id",
      "runtime_session_id",
      "state",
      "session_generation",
      "security_generation"
    ],
    "excluded_fields": [
      "content_fingerprint_sha256"
    ],
    "input_shape": "JSON_OBJECT",
    "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
    "canonicalization": {
      "sort_keys": true,
      "separators": [
        ",",
        ":"
      ],
      "ensure_ascii": false,
      "allow_nan": false
    },
    "encoding": "UTF-8",
    "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
    "unicode_normalization": "NONE",
    "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
    "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
    "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
    "exact_fields_source_pointer": "/executable_boundary_schemas/SessionSecurityState"
  },
  "SecretMetadataProjection": {
    "field": "content_fingerprint_sha256",
    "algorithm": "SHA-256",
    "input_fields": [
      "secret_reference",
      "secret_kind",
      "exchange_account_id",
      "credential_profile_id",
      "exchange_id",
      "environment",
      "permitted_operations",
      "secret_revision",
      "state"
    ],
    "excluded_fields": [
      "content_fingerprint_sha256"
    ],
    "input_shape": "JSON_OBJECT",
    "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
    "canonicalization": {
      "sort_keys": true,
      "separators": [
        ",",
        ":"
      ],
      "ensure_ascii": false,
      "allow_nan": false
    },
    "encoding": "UTF-8",
    "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
    "unicode_normalization": "NONE",
    "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
    "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
    "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
    "exact_fields_source_pointer": "/executable_boundary_schemas/SecretMetadataProjection"
  }
}
```

### `executable_boundary_schemas`

```json
{
  "OperatorIdentitySecurityProjection": [
    "account_id",
    "operator_id",
    "state",
    "identity_revision",
    "security_generation",
    "content_fingerprint_sha256"
  ],
  "DeviceTrustProjection": [
    "account_id",
    "device_installation_id",
    "state",
    "trust_revision",
    "security_generation",
    "platform_enrollment_revision",
    "content_fingerprint_sha256"
  ],
  "PinVerifierRecord": [
    "account_id",
    "operator_id",
    "device_installation_id",
    "algorithm_id",
    "parameter_policy_version",
    "salt_reference",
    "verifier",
    "pin_revision",
    "failed_attempts",
    "lockout_until_utc",
    "security_generation",
    "content_fingerprint_sha256"
  ],
  "PlatformBiometricAssertion": [
    "account_id",
    "device_installation_id",
    "platform_authenticator_source",
    "platform_enrollment_revision",
    "challenge_fingerprint_sha256",
    "outcome",
    "verified_at_utc",
    "expires_at_utc",
    "assertion_fingerprint_sha256"
  ],
  "AuthenticationProof": [
    "account_id",
    "operator_id",
    "device_installation_id",
    "factor_set",
    "issued_at_utc",
    "expires_at_utc",
    "identity_revision",
    "device_trust_revision",
    "pin_revision",
    "platform_enrollment_revision",
    "security_generation",
    "session_generation",
    "environment",
    "operation",
    "scope_fingerprint_sha256",
    "mutation_fingerprint_sha256",
    "causation_id",
    "correlation_id",
    "proof_fingerprint_sha256"
  ],
  "CoreIssuedAuthenticationProofBinding": [
    "proof_fingerprint_sha256",
    "complete_proof_content_fingerprint_sha256",
    "authority_source",
    "account_id",
    "operator_id",
    "device_installation_id",
    "identity_revision",
    "device_trust_revision",
    "pin_revision",
    "platform_enrollment_revision",
    "security_generation",
    "session_generation"
  ],
  "SessionSecurityState": {
    "exact_fields": [
      "account_id",
      "operator_id",
      "device_installation_id",
      "runtime_session_id",
      "state",
      "session_generation",
      "security_generation",
      "content_fingerprint_sha256"
    ],
    "terminal_fingerprint": {
      "field": "content_fingerprint_sha256",
      "algorithm": "SHA-256",
      "input_fields": [
        "account_id",
        "operator_id",
        "device_installation_id",
        "runtime_session_id",
        "state",
        "session_generation",
        "security_generation"
      ],
      "excluded_fields": [
        "content_fingerprint_sha256"
      ],
      "input_shape": "JSON_OBJECT",
      "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
      "canonicalization": {
        "sort_keys": true,
        "separators": [
          ",",
          ":"
        ],
        "ensure_ascii": false,
        "allow_nan": false
      },
      "encoding": "UTF-8",
      "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
      "unicode_normalization": "NONE",
      "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
      "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
      "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
      "exact_fields_source_pointer": "/executable_boundary_schemas/SessionSecurityState"
    }
  },
  "OperationEntitlementProjection": [
    "account_id",
    "operator_id",
    "operation",
    "environment",
    "authorization_scope",
    "entitlement_revision",
    "security_generation",
    "content_fingerprint_sha256"
  ],
  "SecretMetadataProjection": {
    "exact_fields": [
      "secret_reference",
      "secret_kind",
      "exchange_account_id",
      "credential_profile_id",
      "exchange_id",
      "environment",
      "permitted_operations",
      "secret_revision",
      "state",
      "content_fingerprint_sha256"
    ],
    "terminal_fingerprint": {
      "field": "content_fingerprint_sha256",
      "algorithm": "SHA-256",
      "input_fields": [
        "secret_reference",
        "secret_kind",
        "exchange_account_id",
        "credential_profile_id",
        "exchange_id",
        "environment",
        "permitted_operations",
        "secret_revision",
        "state"
      ],
      "excluded_fields": [
        "content_fingerprint_sha256"
      ],
      "input_shape": "JSON_OBJECT",
      "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
      "canonicalization": {
        "sort_keys": true,
        "separators": [
          ",",
          ":"
        ],
        "ensure_ascii": false,
        "allow_nan": false
      },
      "encoding": "UTF-8",
      "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
      "unicode_normalization": "NONE",
      "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
      "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
      "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
      "exact_fields_source_pointer": "/executable_boundary_schemas/SecretMetadataProjection"
    },
    "field_schemas": {
      "permitted_operations": {
        "type": "canonical_unique_array_of_enum",
        "items_source_pointer": "/registries/secret_use_operation_registry",
        "min_items": 1,
        "unique": true,
        "canonical_order": "REGISTRY_ORDER",
        "validator_behavior": "REJECT_NON_CANONICAL_ORDER_OR_DUPLICATES_NEVER_SORT_OR_DEDUPLICATE",
        "validate_before_terminal_fingerprint": true,
        "domain_separation": "SECRET_USE_OPERATIONS_NOT_ADMIN_OPERATION_POLICY_OR_M05_CREDENTIAL_PERMISSIONS",
        "authority_boundary": "MEMBERSHIP_IN_INTRINSIC_METADATA_DOES_NOT_ESTABLISH_CURRENT_ACCEPTED_SECRET_AUTHORITY"
      }
    }
  },
  "LiveAccessGrantSecurityProjection": [
    "live_access_grant_id",
    "account_id",
    "operator_id",
    "device_installation_id",
    "policy_scope_fingerprint_sha256",
    "state",
    "grant_revision",
    "security_generation",
    "content_fingerprint_sha256"
  ],
  "InitialSecurityState": [
    "account_id",
    "operator_id",
    "device_installation_id",
    "security_generation",
    "session_generation",
    "state",
    "bootstrap_claim_fingerprint_sha256",
    "content_fingerprint_sha256"
  ],
  "InitialSecurityEstablishmentResult": [
    "account_id",
    "operator_id",
    "device_installation_id",
    "security_generation",
    "session_generation",
    "bootstrap_claim_fingerprint_sha256",
    "result",
    "result_fingerprint_sha256"
  ],
  "CoreAcceptedPlatformBiometricAssertionBinding": [
    "assertion_fingerprint_sha256",
    "complete_assertion_content_fingerprint_sha256",
    "authority_source",
    "account_id",
    "device_installation_id",
    "platform_enrollment_revision",
    "challenge_fingerprint_sha256"
  ],
  "M03BootstrapAuthorityView": [
    "claim_fingerprint_sha256",
    "account_id",
    "device_installation_id",
    "operator_id",
    "bootstrap_generation",
    "bootstrap_revision",
    "purpose",
    "pre_state_fingerprint_sha256",
    "post_state_fingerprint_sha256",
    "consumed_authority_fingerprint_sha256",
    "consumed_claim_fingerprint_sha256",
    "consumed_challenge_fingerprint_sha256"
  ],
  "M03AcceptedBootstrapAuthorityBinding": [
    "view_fingerprint_sha256",
    "complete_view_content_fingerprint_sha256",
    "claim_fingerprint_sha256",
    "account_id",
    "device_installation_id",
    "operator_id",
    "bootstrap_generation",
    "bootstrap_revision",
    "purpose",
    "accepted_pre_fingerprint_sha256",
    "current_post_fingerprint_sha256",
    "consumed_authority_fingerprint_sha256",
    "consumed_claim_fingerprint_sha256",
    "consumed_challenge_fingerprint_sha256",
    "authority_source"
  ]
}
```
