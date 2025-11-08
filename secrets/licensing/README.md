# Portal licencyjny / offline weryfikacja

Katalog `secrets/licensing` zawiera narzędzia do zarządzania magazynem licencji
OEM w środowiskach odłączonych od sieci. Skrypt `offline_portal.py` udostępnia
interfejs CLI, który pozwala na:

* wyświetlenie stanu magazynu licencji (`status`),
* weryfikację podpisu oraz fingerprintu w dostarczonej licencji (`verify`),
* odzyskanie licencji po zmianie sprzętu poprzez ponowne zaszyfrowanie magazynu
  (`recover`).

## Instalacja

1. Aktywuj środowisko Pythona używane w repozytorium (`pip install -r requirements.txt`).
2. Upewnij się, że posiadasz kopię magazynu licencji (`license_store.json`) oraz
   podpisane licencje (`*.json`).
3. Przygotuj fingerprint bieżącego hosta (np. `python -m bot_core.security.fingerprint fingerprint`).

## Użycie CLI

Skrypt obsługuje trzy polecenia. Każde z nich przyjmuje ścieżkę do magazynu
licencji (`--store`) oraz fingerprint (`--fingerprint`). Fingerprint można
przekazać jawnie, odczytać z pliku (`--fingerprint-file`) lub wymusić odczyt z
lokalnego modułu TPM (`--read-local`).

### Status magazynu

```bash
python secrets/licensing/offline_portal.py status \
  --store secrets/license_store.json \
  --fingerprint OEM-DEVICE-1234
```

Polecenie wypisze listę licencji, ich status (`provisioned`, `revoked`, itp.)
oraz informacje o fingerprintach i ewentualnych problemach z deszyfracją.

### Walidacja licencji

```bash
python secrets/licensing/offline_portal.py verify \
  --store secrets/license_store.json \
  --license docs/licensing/sample_license.json \
  --fingerprint OEM-DEVICE-1234 \
  --hmac-key env:OEM_LICENSE_HMAC_KEY
```

Sprawdzenie licencji obejmuje weryfikację podpisu HMAC (jeśli dostarczono klucz)
oraz dopasowanie fingerprintu do bieżącej maszyny. Wynik zostanie wypisany w
formacie JSON, dzięki czemu można łatwo zapisać raport w `audit/licensing/`.

### Odzyskiwanie licencji (zmiana sprzętu)

```bash
python secrets/licensing/offline_portal.py recover \
  --store backups/license_store.json \
  --output secrets/license_store.json \
  --old-fingerprint OEM-OLD-9999 \
  --new-fingerprint OEM-DEVICE-1234
```

Procedura polega na rozszyfrowaniu magazynu licencji z użyciem starego
fingerprintu (`--old-fingerprint` lub `--old-fingerprint-file`), a następnie na
ponownym zaszyfrowaniu danych z nowym fingerprintem (`--new-fingerprint`).
Skrypt zapisze kopię bezpieczeństwa poprzedniego magazynu (`*.bak`) oraz raport
z operacji (`--report`).

## Procedura odzyskiwania licencji

1. **Identyfikacja problemu** – ustal, czy magazyn licencji nie otwiera się z
   powodu zmiany sprzętu (`LicenseStoreDecryptionError`) czy braku fingerprintu.
2. **Zabezpieczenie danych** – wykonaj kopię plików `secrets/license_store.json`
   oraz ostatnich raportów z `audit/licensing/`.
3. **Pozyskanie fingerprintów**:
   * stary fingerprint (z logów L1/L2 lub z dokumentacji odbioru sprzętu),
   * nowy fingerprint – wygenerowany narzędziem `bot_core.security.fingerprint`.
4. **Odzyskanie licencji** – użyj komendy `recover` z odpowiednimi parametrami.
   Wygeneruj raport JSON (`--report var/reports/licensing/recovery-<data>.json`).
5. **Walidacja** – uruchom `status` oraz `verify`, aby upewnić się, że licencje
   zostały poprawnie przypisane do nowego fingerprintu. Zaktualizuj decision log
   i poinformuj operatorów o zakończeniu procedury.

## Artefakty pomocnicze

* `offline_portal.py` – główny skrypt CLI.
* `recovery_request.sample.json` – przykładowy wniosek o odzyskanie licencji,
  który można przekazać zespołowi L3 (zawiera fingerprinty, identyfikator
  licencji oraz podpis operatora).

Wszystkie wygenerowane raporty powinny być odkładane w `var/reports/licensing/`
i podpisywane kluczem HMAC (`OEM_LICENSE_HMAC_KEY`) zgodnie z procedurami
compliance.
