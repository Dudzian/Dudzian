# Windows TPM activation bridge (TEST_ONLY)

## Granica zaufania i mapa ponownego użycia

Bridge jest konsumentem Stage 9. Nie zmienia `ReleasePolicyV1`, równań
`PolicyAuthorize`, layoutu NV ani ceremonii production.

**EXISTING TPM REUSE MAP**

| Potrzeba | Primitive Stage 9 |
|---|---|
| K_PSA | profil ECC P-256/SHA-256 z `validate_tpmt_public`; fizyczny obiekt jest deterministycznym, transient `TPM2_CreatePrimary` w Owner hierarchy |
| TPM ReadPublic | transport/codec `TbsTransport`, `command_packet`, `response_parameters` i parsery TPM2B z `windows_tpm_substrate_probe.py` |
| TPM Name | kanoniczne `nameAlg || SHA256(TPMT_PUBLIC)` sprawdzane przez istniejący parser Stage 9 i niezależnie przez bridge |
| EK/AK evidence | te same TBS/TPM2 codecs; standardowy publiczny ECC EK primary i transient AK primary, bez zmiany EK i bez ownership takeover |
| cleanup | istniejący `NvProbe.flush` (`TPM2_FlushContext`) oraz `TbsTransport.close` |
| canonical TPM public encoding | surowe, marshalled `TPMT_PUBLIC` (lowercase hex) walidowane przez `validate_tpmt_public`; brak licensing-only interpretacji pól TPM |

Lifecycle TEST_ONLY celowo nie tworzy persistent handles ani NV. Każde uruchomienie
odtwarza deterministyczne primaries z hierarchy seed i identycznych template'ów, więc
inwariant dwóch uruchomień to identyczne `K_PSA.Name` i `TPMT_PUBLIC`. Wszystkie
transient handles są usuwane przed zamknięciem kontekstu TBS. Żaden blob prywatny,
klucz PEM/PKCS#8 ani scalar nie jest eksportowany.

`evidence_id` jest SHA-256 kanonicznego body evidence bez pola `evidence_id`.
`ActivationRequestV1.tpm.evidence_reference` jest dokładnie tym identyfikatorem.
`device_id` jest skrótem kanonicznej publicznej tożsamości K_PSA/EK/AK, a trwały
`installation_id` jest generowany przez CSPRNG jeden raz w lokalnym stanie aplikacji.

## Fizyczna attestation — implementacja gotowa do live gate

Publiczne `TPMT_PUBLIC`, Name i digesty EK/AK są wyłącznie self-consistent public
projection. Nie są issuer-grade dowodem pochodzenia sprzętowego. Żaden typ obiektu
Pythona nie stanowi capability kryptograficznej. Consequential issuer boundary ponownie
parsuje kanoniczne bajty `TPMEnrollmentRequestV1`, `TPMEnrollmentChallengeV1` i
`TPMEnrollmentChallengeResponseV1`, sprawdza ich binding oraz pending single-use
challenge. Concrete issuer verifier sam weryfikuje standardowy łańcuch
`MakeCredential` / `TPM2_ActivateCredential`, `TPM2_CertifyCreation` oraz K_PSA PoP.
Sam self-consistent JSON jest odrzucany przy issuance.

Te same trzy kanoniczne artefakty są kontraktem transportowo neutralnym: offline są
plikami wymienianymi ręcznie, a przyszły transport online przeniesie identyczne bajty.
`TPMEnrollmentRequestV1` wiąże pełny SHA-256 kanonicznych bajtów oraz `request_id`
docelowego `CryptoHunterActivationRequestV1`; challenge wiąże request przechodnio.
Obecny `PendingChallengeStore` jest wyłącznie in-memory — trwały, fail-closed store dla
manualnego offline round-trip pozostaje `NOT_IMPLEMENTED`. Production verifier nie
przyjmuje callbacku od callera. Weryfikuje recovered credential przypisany do pending
challenge, podpis AK nad pełnym `TPMS_ATTEST`, creation Name/hash oraz podpis PoP K_PSA.
Challenge jest konsumowany dopiero po udanym podpisaniu pakietu. Osobny
`TestOnlyTPMAttestationVerifier` służy wyłącznie testom.

Fizyczny adapter implementuje policy session z `PolicyCommandCode(Sign)` i
`PolicyGetDigest`, `TPM2_Sign`, `TPM2_ActivateCredential` z endorsement policy oraz
`TPM2_CertifyCreation`. Wszystkie sesje i transient primaries są flushowane. Implementacja
przeszła testy syntetycznych odpowiedzi, ale rzeczywisty TPM nie został jeszcze uruchomiony;
dlatego status to `READY_FOR_LIVE_TEST`, a nie `DONE`.

## Polecenia live do wykonania przez operatora

Polecenia zakładają checkout w `C:\CryptoHunter`, istniejący zweryfikowany plik
TEST_ONLY ReleasePolicy z fixture oraz katalog publicznego wyniku. Production jest
blokowane przed otwarciem TBS komunikatem:
`STOP — PRODUCTION CEREMONY NOT COMPLETE FOR DEVICE ENROLLMENT.`

```powershell
# RUN 1
cd C:\CryptoHunter
python scripts\cryptohunter_activation_request.py physical-test --environment TEST_ONLY --output C:\CryptoHunter-Test-Activation --release-policy C:\CryptoHunter\tests\fixtures\windows_stage9_release_policy_v1_test_only.json --edition pro --feature core_bot

# RUN 2
cd C:\CryptoHunter
python scripts\cryptohunter_activation_request.py physical-test --environment TEST_ONLY --output C:\CryptoHunter-Test-Activation --release-policy C:\CryptoHunter\tests\fixtures\windows_stage9_release_policy_v1_test_only.json --edition pro --feature core_bot

# NEGATIVE TEST
cd C:\CryptoHunter
python scripts\cryptohunter_activation_request.py negative-test --environment TEST_ONLY --output C:\CryptoHunter-Test-Activation-Negative --release-policy C:\CryptoHunter\tests\fixtures\windows_stage9_release_policy_v1_test_only.json --edition pro --feature core_bot

# CLEANUP
cd C:\CryptoHunter
python scripts\cryptohunter_activation_request.py cleanup --environment TEST_ONLY --output C:\CryptoHunter-Test-Activation
```

Publiczny raport będzie znajdował się w
`C:\CryptoHunter-Test-Activation\physical-preflight.json`. Każdy katalog
`CryptoHunter-Activation-Request-<short-id>` zawiera canonical activation request,
publiczną projekcję TPM oraz trzy publiczne artefakty exchange. Plaintext credential
secret i materiał prywatny nie są eksportowane.

Status: **WINDOWS TPM ACTIVATION BRIDGE = READY_FOR_LIVE_TEST**. Formalny stan Stage 9 i
production pozostaje bez zmian.
