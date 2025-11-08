# Runbook: Aktualizacja i dystrybucja bundli OEM

Ten runbook opisuje standardową procedurę przygotowania i dystrybucji nowej
wersji bundla OEM z uwzględnieniem notarizacji, aktualizacji delta oraz
walidacji fingerprintu sprzętowego.

## 1. Przygotowanie

1. Zbuduj artefakty `core/dist/<platform>` i `ui/dist/<platform>`.
2. Zweryfikuj aktualność kluczy HMAC (`OEM_BUNDLE_HMAC_KEY`,
   `OEM_LICENSE_HMAC_KEY`, `OEM_DECISION_HMAC_KEY`).
3. Przygotuj plik konfiguracyjny pipeline’u (`deploy/packaging/pipeline.release.yaml`):
   ```yaml
   notarization:
     bundle_id: com.example.core
     profile: oem-notary
     dry_run: false
     log_path: var/dist/notary/core-${PLATFORM}.json
   delta_updates:
     bases:
       - var/releases/core-oem-${PREV_VERSION}-${PLATFORM}.tar.gz
     output_dir: var/dist/delta/${PLATFORM}
   fingerprint_validation:
     expected: ${EXPECTED_FP}
     hmac_key: env:OEM_BUNDLE_HMAC_KEY
     verify_local: true
   ```

## 2. Budowa bundla

```bash
python deploy/packaging/build_core_bundle.py \
  --platform macos \
  --version 2024.06 \
  --daemon core/dist/macos \
  --ui ui/dist/macos \
  --config core.yaml=config/core.yaml \
  --config env=.env.production \
  --resource scripts=scripts/install/macos \
  --signing-key-path secrets/oem_manifest.key \
  --pipeline-config deploy/packaging/pipeline.release.yaml
```

* Pipeline wykona weryfikację `fingerprint.expected.json`. W przypadku błędów
  proces zostanie przerwany.
* Wygenerowane archiwum zostanie zgłoszone do notarizacji (`xcrun notarytool`).
  Raport znajdziesz w `var/dist/notary/`.
* W katalogu `var/dist/delta/<platform>` pojawią się delty względem zadanych
  wersji bazowych. Udostępnij je razem z pełnym archiwum w katalogu release’u.

## 3. Kontrola jakości

1. Sprawdź `docs/deployment/oem_installation.md` i zaktualizuj sekcję „Wersja”
   oraz listę fingerprintów.
2. Zbadaj log z notarizacji – status `Accepted` jest wymagany przed publikacją.
3. Przejrzyj `delta.json` w każdej paczce delta; upewnij się, że lista plików jest
   zgodna z oczekiwaniami (brak plików konfiguracyjnych i sekretów).
4. Zapisz raport w `var/reports/deployment/core-oem-2024.06-${PLATFORM}.json`.

## 4. Publikacja

1. Skopiuj pełne archiwum oraz paczki delta do `var/releases/<platform>/`.
2. Podpisz manifest release’u (`manifest.json`, `manifest.sig`).
3. Zaktualizuj `docs/support/faq.md` o informacje o wersji i wymaganych krokach
   aktualizacyjnych.
4. Zgłoś release w decision logu (`docs/decision_journal.md`).

## 5. Eskalacja

* Problemy z notarizacją – eskaluj do zespołu bezpieczeństwa (L3) i użyj logu z
  `var/dist/notary/*`.
* Błędy delta – porównaj `delta.json` z manifestem bundla. Jeżeli plik został
  pominięty, dodaj go ręcznie lub wykonaj pełny rebuild.
* Niezgodność fingerprintu – uruchom `secrets/licensing/offline_portal.py status`
  na docelowym hostcie i zweryfikuj fingerprint z `fingerprint.expected.json`.
