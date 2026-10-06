# Produkcyjny initial successor LPPI authority key

Boundary `establish_installed_lppi_authority_from_artifacts(...)` w
`deployment/windows_production_lppi_authority.py` rozwiązuje installed Production
Trust oraz otwiera istniejący pre-enrollment identity przez `open_existing`.
Nie przyjmuje ścieżki durable state, backendu NCrypt/TBS, publicznego klucza,
trust roots, zegara ani callbacku podpisującego od callera.

Wymagane wejścia to exact canonical PDSA package, retained request i signature,
signed PDSA challenge, cztery publiczne artefakty TPM exchange, endorsement,
pre-enrollment custody evidence oraz locator retained AK. Locator AK staje się
zaufany dopiero po native ReadPublic i porównaniu z exact signed target projection.
Issuer databases oraz ActivateCredential secret pozostają po stronie issuera.

Client acceptance weryfikuje Production Trust/quorum/expiry, exact request i
exchange bindings oraz oba retained AK CertifyCreation proofs. Otwarty lokalnie
pre-enrollment key przechodzi aktualną CNG qualification. Nowy CSPRNG nonce wiąże
świeży PoP i live CertifyCreation z exact package/request/device; publiczny
fingerprint albo stary request signature nie ustanawia acceptance capability.

Successor używa wyłącznie nazwy
`CryptoHunter.Stage9.Production.LPPI.Authority.v1`, providera
`Microsoft Platform Crypto Provider`, machine scope, ECDSA P-256, signing-only
i zakazu private export. Ma inny SEC1 point i CNG Unique Name niż retained pre-key.
Custody wymaga actual TPMT_PUBLIC, recomputed Name, creation hash/ticket,
CertifyCreation oraz podpisu exact retained AK. Brak supported PCP creation
properties albo inny TPM public profile kończy się fail-closed.

Frozen contract i jego hash znajdują się w
`docs/architecture/cryptohunter_product_architecture/stage9_lppi_authority_key_binding_contract.json`
oraz odpowiadającym `stage9_lppi_authority_key_binding_freeze.json`. Parent contract
pozostaje byte-identical. Initial `generation=1`; replacement i retirement nie są
obsługiwane.

```text
pre-enrollment fingerprint = SHA256(SEC1 uncompressed, 65 bytes)
authority fingerprint      = SHA256(canonical raw TPMT_PUBLIC)
authority TPM Name         = 000b || SHA256(canonical raw TPMT_PUBLIC)

ABSENT → CREATION_RESERVED → CANDIDATE
→ CONTINUITY_SIGNATURE_VERIFIED → AUTHORITY_KEY_POP_VERIFIED
→ CUSTODY_EVIDENCE_VERIFIED → VERIFIED_CONTINUITY → ACTIVE
```

Rekord `State/LPPIAuthority/initial-authority-key.json` jest singletonem, objętym
exclusive lock i atomic replace po fsync. Installer-owned machine state musi
zachować istniejącą ochronę katalogu `State`; publiczny JSON nie jest authority.
Nie jest to authenticated rollback-proof storage ani Protected Freshness.

| Cut-point | Recovery |
| --- | --- |
| Rezerwacja, native creation jeszcze nie próbowano | Resume tej samej rezerwacji; brak authority przed gates. |
| Marker native attempt, klucza brak | Reconciliation required; brak ponownego mintu. |
| Niejednoznaczny create, klucz istnieje | Otwórz i zakwalifikuj exact fixed identity. |
| Klucz reopened, record nadal reserved | Zachowaj exact SEC1/Unique Name; dokończ custody/binding. |
| Continuity/PoP już retained | Zweryfikuj i użyj exact retained bytes oraz candidate. |
| Custody verified, ACTIVE jeszcze nie committed | Resume exact candidate; powtórz kompletne gates. |
| ACTIVE committed, response lost | Zwróć ten sam binding/identity po fresh acceptance i native requalification. |

`VerifiedActiveLPPIAuthorityKey` jest exact-type private-registry capability.
Każde consequential użycie porównuje immutable snapshot z bieżącym retained state,
weryfikuje pełne signature history, current trust/expiry, CNG identity i live
successor TPM proof, a następnie fresh successor PoP. Getter publicznego native
projection jest wyłącznie opisem identity, bez uprawnienia do podpisania lub ACTIVE.
`copy.copy`, subclass, `object.__new__` i skopiowany ACTIVE row nie ustanawiają authority.

Hosted/cross-platform ABI symulacja nie kwalifikuje fizycznego TPM. Dostępność
`PCP_KEY_CREATIONHASH`, `PCP_KEY_CREATIONTICKET`, persistent creation ticket oraz
exact ECDSA-SHA256 TPM scheme na reopened PCP key wymagają physical Windows
qualification. Inne codecs/schemes nie są zgadywane ani zastępowane synthetic proof.

`WINDOWS_NATIVE_LPPI_SUCCESSOR_KEY_QUALIFICATION=NOT_RUN`;
`PHYSICAL_TPM_LPPI_AUTHORITY_CUSTODY_QUALIFICATION=NOT_RUN`;
`LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED`.

Stage 9 pozostaje `IN_PROGRESS`, `production_provisioning_ready=false`, Windows
`NOT_READY`, Stage 10 `NOT_STARTED` i `BLOCKED_UNTIL_LEGAL_ENROLLMENT`.
`ProductionMembershipSignerUnavailable` zachowuje fail-closed. Następny blok
implementacyjny to `ProductionMembershipSigner` wraz z
`LPPIAuthenticatedProvisioningOperationBindingV1`; ten PR ich nie implementuje.
