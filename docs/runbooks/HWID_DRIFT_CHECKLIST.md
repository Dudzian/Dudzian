# HWID drift – checklista operatora

Procedura dla przypadków, gdy podpis licencji nie zgadza się z aktualnym HWID, ale zmiany mogą mieścić się w tolerancji.

## Przygotowanie
- Sprawdź, czy logi `logs/security_admin.log` zawierają wpisy `hwid_drift` z komponentami oznaczonymi jako `tolerated` lub `blocked`.
- Ustal, czy zmiana nastąpiła po wymianie sprzętu (NIC, dysk) czy wskazuje na migrację na inny host.
- Zweryfikuj ścieżkę raportu `reports/ci/licensing_drift/compatibility.json` z ostatniego przebiegu nightly.

## Kroki operatora
1. **Ocena ryzyka** – jeśli `blocked` zawiera `cpu` lub `tpm`, wstrzymaj uruchomienia runtime i przejdź do procedury rebind/appeal.
2. **Drift tolerowany** – przy zmianie `mac` lub `disk` potwierdź w zgłoszeniu serwisowym numer seryjny wymienionego elementu i zanotuj ID licencji.
3. **Regeneracja fingerprintu** – uruchom `python scripts/generate_hwid_drift_report.py --output /tmp/hwid-drift.json`, aby potwierdzić, że lokalny fingerprint mieści się w tolerancji.
4. **Rebind offline** – jeżeli drift jest krytyczny, pobierz nowy fingerprint (`fingerprint.json`) i przekaż OEM do ponownego podpisania licencji.
5. **Appeal offline** – gdy rebind nie jest możliwy, eskaluj do security z dowodami (`logs/security_admin.log`, `fingerprint.json`, raport CI) i oczekuj na decyzję.

## Walidacja końcowa
- Zweryfikuj, że `LicenseValidationResult` w UI nie zawiera nowych błędów po ponownym uruchomieniu.
- Zaktualizuj wpis w dzienniku operacyjnym (data, komponenty zmienione, wynik procedury).
- Załaduj najnowszy artefakt z `reports/ci/licensing_drift/` do systemu ticketowego jako dowód zgodności.
