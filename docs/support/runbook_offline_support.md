# Runbook operacyjny: secret migration / HWID recovery / break-glass

Dokument opisuje **faktyczne zachowanie kodu** dla operacji na sekretach i licencjach HWID.
Zakres obejmuje tylko mechanizmy zaimplementowane w:

- `bot_core.security.file_storage.EncryptedFileSecretStorage`,
- `bot_core.security.keyring_storage.KeyringSecretStorage`,
- `bot_core.security.license_store.LicenseStore`,
- `secrets/licensing/offline_portal.py`,
- `bot_core.runtime.stage6_preset_cli`.

---

## 1) Migracja sekretów – jak działa realnie

## 1.1 Wybór backendu sekretów

`create_default_secret_storage` wybiera backend po OS/środowisku:

- Windows/macOS -> `KeyringSecretStorage` (natywny keychain),
- Linux z GUI -> `KeyringSecretStorage`,
- Linux headless -> `EncryptedFileSecretStorage` i **wymagane hasło** (`headless_passphrase`).

Wniosek operacyjny: migracja między hostami Linux headless opiera się na pliku magazynu + passphrase; migracja z/do keychain jest odrębną operacją (brak automatycznej konwersji keychain->plik w tym module).
Dobór powyżej to **logika domyślna kodu**, a nie gwarancja gotowości backendu systemowego. Operacyjnie: jeśli backend natywny keyring nie jest dostępny/poprawnie zainicjalizowany (brak zależności/backendu), inicjalizacja może zakończyć się `SecretStorageError` i wymaga naprawy środowiska hosta przed migracją.


## 1.2 Migracja plikowa (praktyczna)

Do migracji wsadowej używaj `stage6_preset_cli` (nazwa `secrets.enc` jest przykładowa; suffix pliku nie oznacza wymuszonego zewnętrznego formatu/toolingu):

```bash
python -m bot_core.runtime.stage6_preset_cli \
  --preset <preset> \
  --core-config config/runtime.yaml \
  --secrets-input <secrets.json|yaml> \
  --secrets-output <secrets.enc> \
  --secret-passphrase-env BOT_CORE_UI_SECRET_PASSPHRASE
```

Warunki z kodu:

- jeśli podasz `--secrets-input`, musisz też podać `--secrets-output` (inaczej `parser.error`),
- możesz filtrować wpisy (`--secrets-include`, `--secrets-exclude`),
- `--secrets-preview` daje podgląd bez zapisu.

## 1.3 Minimalna walidacja po migracji

1. Sprawdź, czy docelowy plik magazynu istnieje i jest czytelny dla tego samego hasła.
2. Uruchom komponent korzystający z `SecretManager` i potwierdź brak `SecretStorageError`.
3. Nie traktuj samego istnienia pliku jako sukcesu: poprawność potwierdza dopiero odczyt sekretów.

---

## 2) Co się dzieje przy zmianie hardware/VM

## 2.1 License store (`secrets/license_store.json`)

`LicenseStore` szyfruje payload kluczem wyprowadzonym z fingerprintu. Przy innym fingerprint:

- odczyt kończy się `LicenseStoreDecryptionError`,
- komunikat wskazuje na możliwą zmianę sprzętu.

Dodatkowo `offline_portal recover` wymaga starego fingerprintu do odczytu i nowego do ponownego zaszyfrowania.

## 2.2 Keyring secrets (`KeyringSecretStorage`)

Rekord master-key zawiera `hwid_digest`. Przy różnicy digestów:

- ładowanie kończy się błędem `SecretStorageError` (`fingerprint mismatch`),
- brak automatycznego obejścia w kodzie.

To oznacza, że sama migracja VM/hosta może zablokować odczyt sekretów keychain.

**Response path (literalnie):** brak analogicznego flow recover jak `offline_portal recover` dla `KeyringSecretStorage`. Przy `fingerprint mismatch` wykonaj planowaną re-prowizję sekretów na nowym hoście (odtworzenie z właściwego źródła prawdy, np. bezpieczny eksport operatora/KMS/procedura onboardingowa), a nie próbę odszyfrowania starych rekordów keychain.

## 2.3 License status i anty-rollback

`LicenseService` podpisuje sekcję monotoniczną statusu (`var/security/license_status.json`) fingerprintem/sekretem powiązania.

Jeżeli podpis jest niepoprawny albo sekret zniknął:

- pojawia się `LicenseStateTamperedError`,
- ładowanie licencji zostaje zablokowane.

Jeżeli nowa licencja wygląda na starszą (sequence/issued_at/effective_date):

- pojawia się `LicenseRollbackDetectedError`.

---

## 3) Backup / restore

## 3.1 Backup sekretów (EncryptedFileSecretStorage)

Obsługiwany natywnie przez CLI (`stage6_preset_cli`):

```bash
python -m bot_core.runtime.stage6_preset_cli \
  --preset <preset> \
  --core-config config/runtime.yaml \
  --secrets-output <secrets.enc> \
  --secret-passphrase-env BOT_CORE_UI_SECRET_PASSPHRASE \
  --secrets-backup var/backups/secrets-vault.b64
```

Dostępne też `--secrets-backup-stdout` (snapshot na stdout).

## 3.2 Restore sekretów (EncryptedFileSecretStorage)

```bash
python -m bot_core.runtime.stage6_preset_cli \
  --preset <preset> \
  --core-config config/runtime.yaml \
  --secrets-output <secrets.enc> \
  --secret-passphrase-env BOT_CORE_UI_SECRET_PASSPHRASE \
  --secrets-recover-from var/backups/secrets-vault.b64
```

Kod używa `recover_from_backup(...)`, odtwarza plik i od razu inicjalizuje storage.

## 3.3 Backup przy `offline_portal recover`

`offline_portal recover` przed nadpisaniem `--output` tworzy automatycznie plik `<output>.bak` (kopię techniczną).

## 3.4 Czego **nie** obejmuje backup/restore

- brak natywnej procedury export/import dla systemowego keychain jako całości,
- backup zaszyfrowanego pliku magazynu (np. `secrets.enc`) bez właściwego hasła jest bezużyteczny,
- backup licencji nie zastępuje procesu podpisu OEM.

---

## 4) Ograniczenia HWID binding (bez interpretacji marketingowej)

1. Wiązanie jest **ścisłe**: brak tolerancji „prawie ten sam host” dla `LicenseStore` i `KeyringSecretStorage`.
2. Brak fingerprintu (błąd providera) = operacja szyfrowania/odszyfrowania lub walidacji może się zatrzymać błędem.
3. `offline_portal recover` nie „łamie” kryptografii – wymaga dostarczenia poprawnego starego fingerprintu.
4. Podmiana/utrata sekretu użytego do podpisu statusu licencji może dać `LicenseStateTamperedError`.
5. Ochrona anty-rollback blokuje cofanie starszych licencji nawet jeśli podpis jest poprawny.

---

## 5) Break-glass przy awarii (procedura minimalna)

Stosuj, gdy po migracji sprzętu/VM system nie czyta `license_store.json`.

0. **Prereq check (krótki):** potwierdź dostępny stary fingerprint (np. `backups/fingerprint.old`), dostępność `OEM_LICENSE_HMAC_KEY` do kroku verify oraz wykonane kopie `license_store.json` / `license_status.json` / `security_admin.log`.
1. **Zatrzymaj runtime** (żeby nie nadpisywać artefaktów statusu/logów).
2. **Zabezpiecz kopie**:
   - `secrets/license_store.json`,
   - `var/security/license_status.json`,
   - `logs/security_admin.log`.
3. **Wykonaj recover**:

```bash
python secrets/licensing/offline_portal.py recover \
  --store secrets/license_store.json \
  --output secrets/license_store.json \
  --old-fingerprint-file backups/fingerprint.old \
  --read-local-new \
  --report var/reports/licensing/recovery-$(date -u +%Y%m%dT%H%M%SZ).json
```

4. **Potwierdź stan**:

```bash
python secrets/licensing/offline_portal.py status \
  --store secrets/license_store.json \
  --read-local
```

5. **Zweryfikuj licencję wsadowo**:

```bash
python secrets/licensing/offline_portal.py verify \
  --store secrets/license_store.json \
  --license /media/usb/licence.json \
  --read-local \
  --hmac-key env:OEM_LICENSE_HMAC_KEY
```

6. **Uruchom runtime** i sprawdź brak błędów `LicenseStoreDecryptionError`, `LicenseStateTamperedError`, `fingerprint mismatch`.
7. **Dołącz raport** z `--report` + `.bak` do incydentu/audytu.

---

## 6) Czego nie wolno robić

1. Nie edytuj ręcznie zaszyfrowanych plików (`license_store.json`, plik magazynu sekretów np. `secrets.enc`) w celu „naprawy” – to kończy się błędami deserializacji/dekryptażu.
2. Nie uruchamiaj `recover` bez archiwizacji starego store i fingerprintu źródłowego.
3. Nie nadpisuj/rotuj passphrase bez świeżego backupu (`--secrets-backup` lub `export_backup`).
4. Nie usuwaj sekcji podpisu i monotonicznej z `license_status.json` – może to wywołać blokadę tamper/rollback.
5. Nie zakładaj, że migracja hosta zachowa dostęp do keychain; traktuj to jako osobny punkt planu migracji.
6. Nie mieszaj środowisk (prod/paper) przy migracji sekretów – `SecretManager` waliduje environment i odrzuca niespójność.

---

## Szybki decision matrix (operacyjny)

- `LicenseStoreDecryptionError` po zmianie hosta -> `offline_portal recover`.
- `fingerprint mismatch` z keyring -> planowana re-inicjalizacja sekretów na nowym hoście.
- potrzebny backup magazynu sekretów -> `stage6_preset_cli --secrets-backup`.
- potrzebny restore magazynu sekretów -> `stage6_preset_cli --secrets-recover-from`.
- podejrzenie cofnięcia licencji -> nie wyłączaj ochrony; eskaluj z `license_status.json` i `security_admin.log`.
