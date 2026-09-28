# Stage-9 canonical-vector physical TPM cross-check

Ten disposable runner porównuje fizycznie wytworzone przez TPM digesty z
`Stage9PolicyVectorV1` wygenerowanym z canonical fixture oraz zaakceptowanym
przez niezależny verifier. Używa wyłącznie jawnie oznaczonych,
deterministycznych kluczy `TEST_ONLY`; nie tworzy materiału produkcyjnego.

Na podniesionym fizycznym Windows 11 z TPM 2.0 uruchom:

```powershell
py -3.12 -m deployment.windows_stage9_canonical_tpm_crosscheck `
  --output dist/windows/stage9-canonical-vector-crosscheck-evidence.json
```

Runner wymaga wolnego dokładnego indeksu `0x018f0001`, ponieważ jego handle jest
częścią canonical NV Name. Definiuje go tymczasowo, sprawdza pre-write Name,
wykonuje pierwszy rzeczywisty recovery increment, sprawdza post-write Name,
odczytuje rzeczywistą wartość countera i wykonuje oddzielne trial oraz realne
normal-branch sessions.
Sprawdzane są: oba recovery approvedPolicy, normal approvedPolicy, oba digesty
po `PolicyAuthorize` i uporządkowany root `PolicyOR`. Każdy checkpoint wymaga
`TPM == canonical generator == independent verifier`; rozjazd kończy się
niezerowym kodem. Artefakty `dist/windows/` są lokalnym, celowo nieśledzonym
evidence i nie wolno commitować host-specific wyników.

Fixture generation `7` jest sprawdzana wyłącznie jako dokładny digest w
`TPM_SE_TRIAL` (`fixture_generation_mode=TRIAL_DIGEST_ONLY`). Runner nigdy nie
zakłada początkowej wartości fizycznego countera i nie próbuje inkrementować go
do fixture generation. Po bootstrap increment odczytuje realne `g`, tworzy
wyłącznie w pamięci `RUNTIME_TEST_ONLY / OBSERVED_PHYSICAL_GENERATION /
NOT_FOR_FREEZE` vector, wykonuje realny normal branch dla `PolicyNV(g)` i wymaga
końcowej wartości `g + 1`.

Realny bootstrap ładuje publiczny `K_RECOVERY_TEST`, a realny normal branch
ładuje publiczny `K_PSA_TEST` pod `TPM_RH_OWNER`, dlatego
`VerifySignature` wystawia non-NULL `TPM_ST_VERIFIED` ticket wymagany przez
realne `PolicyAuthorize`. Evidence zapisuje tag, hierarchy i rozmiar digestu
ticketu oraz hierarchy klucza. `TPM_RH_NULL` pozostaje celowo użyty wyłącznie
dla obu handles `StartAuthSession` oraz syntetycznego ticketu `TPM_SE_TRIAL`,
w którym TPM pomija weryfikację ticketu.

## Freeze produkcyjny pozostaje osobnym gate

Po PASS nie wolno promować kluczy testowych ani generować automatycznie kluczy
prywatnych. Ceremony/freeze musi osobno:

1. zatwierdzić publiczne trust anchors **Product Release Root** i regułę progu;
2. zatwierdzić **PDSA verification keys** oraz ich custody/rotation;
3. przyjąć publiczny produkcyjny `K_RECOVERY`, którego private material pozostaje
   poza disposable probe;
4. wygenerować per-device produkcyjny `K_PSA` w docelowym TPM;
5. podpisać i związać `PDSAEnrollmentPackageV1`;
6. przeprowadzić `ReleasePolicyV1` signature ceremony;
7. zamrozić byte-complete canonical vector i freeze manifest z digestami,
   signerami, wersjami i provenance;
8. określić fail-closed migration/version rules, rollback i ponowną ceremonię
   przy każdej zmianie materiału lub serializacji.

Prywatne Product Release Root oraz recovery authority nie mogą powstać ani
zostać użyte w tym runnerze.
