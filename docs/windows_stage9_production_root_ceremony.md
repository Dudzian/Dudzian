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

Powyższy opis stanu jest historycznym snapshotem wdrożenia samego toolingu. Bieżącym,
jedynym źródłem statusu jest `deployment/stage9_current_status.json`; nie należy aktualizować
historycznych, zahashowanych snapshotów w celu prezentacji bieżącego licznika.

## Canonical operator runbook — production ceremony

### Twarde warunki wejścia

Operator wpisuje zatwierdzony SHA do `$ReviewedRevision`. Wszystkie poniższe warunki muszą być
spełnione jednocześnie: exact commit został zreviewowany i zatwierdzony do wydania; working tree
jest czyste; A-01 i A-02 mają `PASS`; clean install i TPM activation bridge mają `LIVE_PASS`;
production authority material jest dostępny; public digest preflight ma `PASS`; a production
ceremony ma `NOT_STARTED`. Niespełnienie choć jednego warunku oznacza **ABORT WITHOUT MUTATION**.

Rozdział ról jest dokładnie zgodny z frozen architecture; Product Release Root i PDSA pozostają
**separate offline authorities** z odrębną custody:

* Product Release Root: **Ed25519, 3 independent holders/keys, threshold 2-of-3**.
* PDSA: **Ed25519, 3 independent holders/keys, threshold 2-of-3**.

Każda authority wymaga własnego minimalnego quorum dwóch niezależnych, autoryzowanych holderów.
Żaden pojedynczy holder nie rekonstruuje kompletnej authority 2-of-3. Frozen architecture nie
rozstrzyga, czy konkretna osoba może pełnić role w obu osobnych authorities; nie wolno z tego
runbooka wyprowadzać ani obowiązku, ani zakazu łączenia ról. Private shares obu authorities nigdy
nie trafiają do repozytorium, evidence ani logów. Product Release Root osobno podpisuje release i
initial revocation, a PDSA pozostaje przypięta przez release policy.

### Dokładne ścieżki (PowerShell 7)

```powershell
$Repo = "C:\Users\kamil\Documents\GitHub\Dudzian"
$Authority = "C:\CryptoHunter-Production-Authority"
$AuthorityPublic = "$Authority\public"
$Session = "$Authority\ceremony-session"
$Evidence = "$Authority\evidence"
$Result = "$Authority\ceremony-results"
$Expected = "$Repo\deployment\stage9_expected_public_authorities.json"
$Status = "$Repo\deployment\stage9_current_status.json"
$ReviewedRevision = "<40-HEX-APPROVED-COMMIT-SHA>"
```

Canonical public inputs are exactly `$AuthorityPublic\product_root_anchor_bundle.json`,
`$AuthorityPublic\pdsa_public_bundle.json`, and `$AuthorityPublic\recovery_public_bundle.json`.
Ceremony phase inputs and detached public signatures are under `$Session`; public preflight
evidence is `$Evidence\authority-preflight.json`; the atomic result is
`$Result\final\<ceremony_id>\`. Override is allowed only by explicitly changing these variables
before any command and recording the resolved paths in the operator record.

### Read-only verification and dry preflight

Run from an elevated PowerShell whose transcript destination is approved for **public-only** data.
These commands only read/hash/validate, except that the optional preflight command creates one new
public evidence file and refuses to overwrite it:

Public preflight wymaga pełnego profilu Product Release Root `PRODUCTION / PRODUCTION` oraz PDSA
`PRODUCTION`, Ed25519, dokładnie 2-of-3; zgodność samego digestu nie wystarcza. Wartość
`Product Release Root.environment` opisuje niezmienny profil authority i nie jest profilem ani
stanem cyklu życia ceremonii; etykiety ceremonii, takie jak `STAGE9_PRODUCTION_CEREMONY_V1`, są
odrębnym kontraktem.

```powershell
Set-Location $Repo
if ((git rev-parse HEAD).Trim() -ne $ReviewedRevision) { throw "ABORT: revision" }
if (git status --porcelain) { throw "ABORT: dirty tree" }
git show --no-patch --format=fuller $ReviewedRevision

python -m deployment.windows_stage9_ceremony_readiness public-preflight `
  --reviewed-revision $ReviewedRevision --authority-dir $AuthorityPublic `
  --expected-manifest $Expected --evidence-output "$Evidence\authority-preflight.json"

# Canonical fail-closed, read-only entry gate (no output artifact):
python -m deployment.windows_stage9_ceremony_readiness entry-gate `
  --reviewed-revision $ReviewedRevision --repo $Repo --authority-dir $AuthorityPublic

# Dry verification of already assembled ceremony inputs; choose a NEW output filename.
python -m deployment.windows_stage9_production_ceremony verify-ceremony `
  --input "$Session\verify-ceremony-input.json" `
  --output "$Session\verify-ceremony-dry-result.json"
```

Formalny gate zawsze wyprowadza manifest i status z odpowiednio
`$Repo\deployment\stage9_expected_public_authorities.json` oraz
`$Repo\deployment\stage9_current_status.json`. Zewnętrzny override nie może spełnić gate. Gate
potwierdza obecność obu plików w `$ReviewedRevision` i porównuje ich working-tree bytes z exact Git
objects tego commitu, niezależnie od kontroli clean tree.

The gate's only success token is `READY_FOR_PRODUCTION_CEREMONY`. `BLOCKED` plus explicit reason
codes or any non-zero exit is **ABORT WITHOUT MUTATION**. The reviewed revision passed to preflight
must equal HEAD and later the freeze/audit `source_revision`; changing checkout invalidates approval.

Preflight evidence schema is `CryptoHunter.Stage9AuthorityPreflightEvidenceV1`, version `1`.
Required fields are `schema`, `version`, `tool_version`, `git_revision`,
`expected_manifest_sha256`, `actual_public_projection_digests` (all four identities), `result`, and
`timestamp_utc`. Verify it independently before proceeding:

```powershell
python -c "import json,pathlib; p=pathlib.Path(r'$Evidence\authority-preflight.json'); d=json.loads(p.read_text()); assert d['schema']=='CryptoHunter.Stage9AuthorityPreflightEvidenceV1' and d['version']==1 and d['git_revision']=='$ReviewedRevision' and d['result']=='PASS'; print('PREFLIGHT_EVIDENCE_PASS')"
python -c "import hashlib,json,pathlib; from deployment.windows_stage9_policy_material import canonical_json_bytes; m=json.loads(pathlib.Path(r'$Expected').read_text()); e=json.loads(pathlib.Path(r'$Evidence\authority-preflight.json').read_text()); assert hashlib.sha256(canonical_json_bytes(m)).hexdigest()==e['expected_manifest_sha256']; print('EXPECTED_MANIFEST_DIGEST_PASS')"
```

### MUTATING / POINT OF NO RETURN

Do not run this section during readiness review. After all preflight checks pass, production
ceremony is the **final planned mutating step** before legal production enrollment and Stage 10
reboot/24h soak. The offline holders first create detached signatures using their custody tooling;
only public detached signature JSON returns to `$Session`. Execute the documented phase commands
from Phases D–H above, using fresh output paths. The final publication command is:

```powershell
# MUTATING / POINT OF NO RETURN — DO NOT RUN UNTIL FORMALLY AUTHORIZED
python -m deployment.windows_stage9_production_ceremony publish-final `
  --input "$Session\publish-final-input.json" --output $Result
```

Before this command, any failure is **ABORT WITHOUT MUTATION**. Once any mutation may have begun:
stop; preserve evidence and console/public logs; **do not delete or overwrite** artifacts; do not
blindly retry. Inspect `$Result\.staging`, `$Result\final`, all phase outputs and custody records.
Run only `verify-final-package` against a complete immutable final directory; an incomplete staging
directory requires incident/custody review and a newly authorized session, never reuse or repair in
place. The implementation refuses output overwrite and atomically promotes verified publication.

### Independent post-check

Do not trust the text “ceremony succeeded”. A second operator in a clean checkout rereads the
resulting public files and executes:

```powershell
$FinalPackage = "$Result\final\<ceremony_id>"

python -c "import json,pathlib,sys; i=json.loads(pathlib.Path(r'$Session\verify-final-package-input.json').read_text()); actual=pathlib.Path(i['package_path']).resolve(); expected=pathlib.Path(r'$FinalPackage').resolve(); sys.exit(f'FINAL_PACKAGE_PATH_MISMATCH: {actual} != {expected}') if actual != expected else print('FINAL_PACKAGE_PATH_BINDING_PASS')"
if ($LASTEXITCODE -ne 0) {
    throw "ABORT: FINAL_PACKAGE_PATH_BINDING_FAILED"
}
python -m deployment.windows_stage9_production_ceremony verify-final-package `
  --input "$Session\verify-final-package-input.json" `
  --output "$Evidence\independent-final-verification.json"
if ($LASTEXITCODE -ne 0) {
    throw "ABORT: VERIFY_FINAL_PACKAGE_FAILED"
}
python -m deployment.windows_stage9_ceremony_readiness final-package-public-preflight `
  --reviewed-revision $ReviewedRevision `
  --repo $Repo `
  --final-package-dir $FinalPackage
if ($LASTEXITCODE -ne 0) {
    throw "ABORT: FINAL_PACKAGE_PUBLIC_PREFLIGHT_FAILED"
}
python -c "import json,pathlib,sys; p=pathlib.Path(r'$FinalPackage'); a=json.loads((p/'ceremony_audit.json').read_text()); f=json.loads((p/'freeze_manifest.json').read_text()); valid=a.get('schema')=='CryptoHunter.Stage9CeremonyAuditV1' and a.get('source_revision')=='$ReviewedRevision' and f.get('artifact_source_revision')=='$ReviewedRevision'; sys.exit('REVISION_BINDING_FAILED') if not valid else print('REVISION_BINDING_PASS')"
if ($LASTEXITCODE -ne 0) {
    throw "ABORT: REVISION_BINDING_FAILED"
}
Write-Host "INDEPENDENT_POST_CHECK_PASS"
```

The verifier must recompute public trust material, package artifact digests, schema/version,
revision binding and ceremony evidence. Production-loader readiness is checked only after the
independent package verification, using the existing loader with the verified final package; this
does not grant legal enrollment and does not start Stage 10.

### No-copy, no-log and cleanup rules

Never copy private authority material into the repository, commit authority files, upload private
artifacts to CI, print private keys/seeds/TPM auth data, or screen-capture secret material into
evidence. Evidence contains public projections and digests only. Cleanup may remove only explicitly
identified temporary, non-authoritative sensitive intermediates after custody approval. Never
destroy source authority or evidence required for audit.

Historical tooling snapshot (retained unchanged as evidence):

```text
PRODUCTION_CEREMONY_TOOLING = READY
PRODUCTION_ROOT_MATERIAL = NOT_PROVISIONED
STAGE 9 = IN_PROGRESS
CAŁY BLOK WINDOWS 0–14 = 9/15 DONE = 60.0%
WINDOWS_PRODUCTION_READY = NOT_READY
STAGE10 = NOT_STARTED
```
