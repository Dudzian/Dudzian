# Stage 9 — ceremonia produkcyjnego root-of-trust

## Granica zaufania i bezpieczeństwa

Ten zestaw narzędzi obsługuje wyłącznie materiał publiczny. Klucze prywatne Product Release
Root, PDSA i `K_RECOVERY`, seedy, skalar prywatny, PIN-y oraz poświadczenia HSM nie mogą
trafić do repozytorium ani na host docelowy. Narzędzie nie tworzy kluczy i nie ma komendy
„wykonaj wszystko”. Podpisanie odbywa się poza repozytorium, a do kolejnych faz wracają
jedynie publiczne, odłączone podpisy.

> **BOOTSTRAP_ROOT_TRUST = EXTERNAL_OPERATOR/CUSTODY DECISION**

Spójność kryptograficzna nie odpowiada na pytanie, czy odebrany zestaw jest **właściwym**
produkcyjnym rootem. Dwóch operatorów musi porównać canonical key-set digest z odciskiem
przekazanym niezależnymi, zatwierdzonymi kanałami custody. Root nigdy nie podpisuje
oświadczenia „I am the trusted root” jako źródła zaufania do samego siebie.

Każdy plik wejściowy i wyjściowy jest canonical JSON. Zalecane jest uruchamianie kolejnych
podkomend `python -m deployment.windows_stage9_production_ceremony <faza> --input ...
--output ...` w kontrolowanym katalogu sesji. Narzędzie odmawia nadpisania pliku.

## Phase A — Root material intake

1. Odbierz trzy publiczne, surowe 32-bajtowe klucze Ed25519 i ich trzy unikalne ID.
2. Uporządkuj rekordy leksykalnie po ID i uruchom `prepare-root-anchor`.
3. Potwierdź próg 2 z 3, środowisko, purpose i digest wyliczony dokładnie przez
   `CryptoHunter.ProductReleaseRootKeySetV1`.
4. Operatorzy porównują digest out-of-band i zapisują decyzję custody poza artefaktem.

## Phase B — PDSA intake

Odbierz wyłącznie publiczne rekordy PDSA, sprawdź ich leksykalny porządek, unikalność ID i
kluczy oraz zatwierdzony próg. Bundle zachowuje rekordy (ID wraz z kluczem), dlatego nie ma
niejednoznaczności równoległych tablic. Digest bundle musi odpowiadać projekcji wpisywanej
do `ReleasePolicyV1`.

## Phase C — Recovery authority intake

Odbierz publiczny `TPMT_PUBLIC` K_RECOVERY wraz z ID. Parser bajtowy musi potwierdzić ECC,
NIST P-256, SHA-256 Name, ECDSA/SHA-256, atrybuty `00040040`, pusty authPolicy, NULL
symmetric/KDF, 32-bajtowe X/Y i brak końcowych bajtów. Porównaj SHA-256 i wyprowadzoną
Name z rejestrem custody. Pochodzenie produkcyjne to `PRODUCTION_PROVISIONED`; prywatny
skalar pozostaje offline.

## Phase D — Release construction

`prepare-release` łączy zweryfikowane bundle z zatwierdzonym stałym profilem K_PSA,
policy refs, szablonem NV, kolejnością gałęzi, jawnym numerem wersji i jawnymi `valid_from`
oraz `valid_until`. Wynik przechodzi istniejący schema validator i canonical serializer.
Wylicz deterministic `ceremony_id` i signing request. Podpisywane bajty to wyłącznie
`RELEASE_DOMAIN || SHA256(canonical payload)`, wskazane w `message_to_sign_hex`. Publiczny,
immutable `Stage9CeremonyContextV1` jest zawsze rekonstruowany z root digest, release payload,
wersji i środowiska; operator ani dokument requestu nie może nadać własnego ceremony ID.

## Phase E — Offline signatures

Przenieś signing request do signerów A i B (opcjonalnie C) kontrolowanym kanałem. Każdy
signer porównuje ceremony ID, digest i dokładne message bytes. Do repo wraca wyłącznie
`CryptoHunter.Stage9DetachedSignatureV1`. Import odrzuca inną sesję/payload, nieznanego,
powtórzonego lub odwołanego signera i niepoprawny podpis. `assemble-signed-release`
składa envelope dopiero po ważnym quorum 2 z 3.

## Phase F — Initial revocation

`prepare-initial-revocation` wymaga jawnego `effective_at` i tworzy jedynego następcę
przypiętego genesis: sequence 1, zero32 previous digest, puste listy odwołań i authority
`PRODUCT_RELEASE_ROOT_QUORUM`. Utwórz osobny request dla
`REVOCATION_DOMAIN || SHA256(canonical payload)`, zbierz drugie quorum root i uruchom
`assemble-signed-revocation`. Request revocation przyjmuje ten sam zweryfikowany context co
release, więc zgodność samych kluczy root nie wystarcza. Lokalny zegar nie jest wejściem
konstrukcji.

## Phase G — Verification

`verify-ceremony` uruchamia istniejące `verify_revocation_state(...)`, a następnie
`verify_signed_release_policy(...)`: sprawdza quorum, okno ważności, wspólną tożsamość
roota release/revocation/pin, PDSA, K_RECOVERY, schemat i canonical JSON. Błąd dowolnego
warunku oznacza kod wyjścia różny od zera oraz **FINAL FREEZE NOT PUBLISHED**.

## Phase H — Freeze

Po PASS `build-freeze-manifest` buduje istniejący `FreezeManifestV1` zawierający source
revision i hashe, po czym `verify-freeze-manifest` sprawdza go canonical production
verifierem. Faza CLI `build-audit-transcript` wyprowadza audit wyłącznie z kompletnego
`VerifiedStage9CeremonyV1`. Wynik przechowuje authority artifacts jako canonical immutable
bytes, a każda krytyczna faza odtwarza pełną weryfikację przed użyciem cached projection.
Publiczny `CryptoHunter.Stage9CeremonyAuditV1` zapisuje ID i wersję narzędzia, revision
pochodzący wyłącznie ze zweryfikowanego FreezeManifest, środowisko, publiczne
ID/digesty/Name, przyjętych signerów, progi, jawnie podane timestampy, digest manifestu i
końcowy status — nigdy sekrety.

`publish-final` publikuje root/PDSA/recovery bundles, oba payloady, oba signing requesty,
oba signed envelopes, freeze manifest i audit. Canonical
`CryptoHunter.Stage9CeremonyPackageManifestV1` wiąże digest każdego pliku oraz digesty
ceremony. Publikacja używa `output/.staging/`, a dopiero kompletny zweryfikowany zestaw jest
atomowo przenoszony do `output/final/<ceremony_id>/`. `verify-final-package` przelicza
wszystkie dowody niezależnie od revision checkoutu weryfikatora. Istniejący finalny katalog
nigdy nie jest nadpisywany; retry musi mieć nową, jawnie rozróżnioną sesję/revision.
Bundle PDSA jest związany z podpisanym release przez keys, key IDs, threshold i purpose;
bundle K_RECOVERY dodatkowo przez key ID, provenance, pełny TPMT_PUBLIC, Name i profil.

## Phase I — Approval gate

PASS tego zestawu oznacza jedynie gotowość ceremony. Dopiero po rzeczywistym offline
ceremony i niezależnym zatwierdzeniu można rozważyć `WindowsExternalProvisioningHandoff`
oraz integrację MSI. Ten etap nie uruchamia Stage 10 ani updatera.

Formalny stan po wdrożeniu samego toolingu:

```text
PRODUCTION_CEREMONY_TOOLING = READY
PRODUCTION_ROOT_MATERIAL = NOT_PROVISIONED
STAGE 9 = IN_PROGRESS
CAŁY BLOK WINDOWS 0–14 = 9/15 DONE = 60.0%
WINDOWS_PRODUCTION_READY = NOT_READY
STAGE10 = NOT_STARTED
```
