# Runbook: Wsparcie offline dla klientów OEM

Dokument opisuje działania operacyjne dla zespołów wsparcia (L1/L2) obsługujących
instalacje OEM w trybie offline.

## 1. Przygotowanie stanowiska

1. Pobierz najnowszy bundel (`var/releases/<platform>/core-oem-*.tar.gz`) oraz
   odpowiadające mu paczki delta.
2. Zweryfikuj fingerprint urządzenia docelowego: `python
   secrets/licensing/offline_portal.py status --store /media/usb/license_store.json --read-local`.
3. Przygotuj nośnik USB z:
   * pełnym bundlem,
   * paczkami delta,
   * raportem notarizacji (`var/dist/notary/*.json`),
   * checklistą instalacji (`docs/deployment/oem_installation.md`).

## 2. Aktualizacja klienta

1. Wykonaj backup aktualnego magazynu licencji (`cp secrets/license_store.json
   secrets/license_store.json.preupdate`).
2. Uruchom `offline_portal.py verify`, aby potwierdzić zgodność licencji:
   ```bash
   python secrets/licensing/offline_portal.py verify \
     --store secrets/license_store.json \
     --license /media/usb/licence.json \
     --read-local \
     --hmac-key env:OEM_LICENSE_HMAC_KEY
   ```
3. Rozpakuj bundel i uruchom instalator zgodnie z `docs/deployment/oem_installation.md`.
4. Jeśli dostępna jest paczka delta, zastosuj ją po pomyślnej instalacji:
   ```bash
   tar -xf /media/usb/core-oem-2024.06-linux.delta.tar.gz -C /opt/core-oem
   ```
5. Sprawdź logi (`var/log/core/`) oraz `fingerprint.expected.json`.

## 3. Odzyskiwanie licencji

1. Gdy magazyn licencji nie otwiera się na nowym sprzęcie, użyj komendy
   `offline_portal.py recover` i zapisz raport:
   ```bash
   python secrets/licensing/offline_portal.py recover \
     --store secrets/license_store.json \
     --output secrets/license_store.json \
     --old-fingerprint-file backups/fingerprint.old \
     --read-local-new \
     --report var/reports/licensing/recovery-$(date -u +%Y%m%dT%H%M%SZ).json
   ```
2. Zweryfikuj wynik (`status`, `verify`) i prześlij raport do zespołu L3.

## 4. Eskalacja i raportowanie

* Każde zgłoszenie zakończ notatką w `docs/support/articles/` (lub istniejącym
  tickecie) z linkami do raportów (`var/reports/licensing/`, `var/reports/deployment/`).
* Krytyczne błędy notarizacji przekazuj do zespołu bezpieczeństwa wraz z logiem
  z `var/dist/notary/`.
* Rozbieżności fingerprintu – eskaluj do zespołu sprzętowego po dostarczeniu
  logów z `offline_portal.py` i `bootstrap/verify_fingerprint.py`.
