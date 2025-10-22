# Runbook: Promotion to Live

Migracja środowiska z paper/testnet do produkcyjnego trybu LIVE wymaga potwierdzenia
spójności profili ryzyka, licencji oraz konfiguracji alertingu. Niniejszy dokument
prowadzi operatora przez procedurę "promotion to live" oraz opisuje narzędzie
`scripts/promotion_to_live.py`, które automatyzuje weryfikację podpisanych
checklist, wymaganych dokumentów i statusów licencji.

## 1. Wymagane artefakty

Przed rozpoczęciem procedury upewnij się, że są dostępne następujące dane:

- **Checklisty LIVE** – sekcja `environments.*.live_readiness` w `config/core.yaml`
  powinna zawierać podpisy (`signed_by`, `signature_path`) oraz pełną listę
  dokumentów wymaganych przez Compliance/Security.
- **Raport risk** – aktualny profil ryzyka (`risk_profiles`) oraz ewentualne
  korekty po testach paper.
- **Licencja OEM** – aktywna licencja z ważną listą odwołań (jeżeli w środowisku
  CI licencja nie jest dostępna, dopuszczalne jest wykonanie procedury ze
  znacznikiem `--skip-license`).
- **Alerting** – skonfigurowane kanały (`alert_channels`), throttling i backend
  audytu (FileAlertAuditLog) dla środowiska LIVE.

## 2. Automatyczna synchronizacja

Uruchom raport synchronizacji:

```bash
python scripts/promotion_to_live.py <environment> \
  --config config/core.yaml \
  --output var/audit/live/promotion_report.json \
  --pretty
```

Parametry dodatkowe:

- `--skip-license` – pomija walidację licencji (przydatne w CI lub na środowiskach
  developerskich bez dostępu do kluczy).
- `--pretty` – zapisuje raport w formacie JSON z wcięciami.

Raport zawiera sekcje:

- `risk_profile_details` – skrócone limity z profilu ryzyka.
- `license` – wynik walidacji licencji (`status`, ostrzeżenia i błędy).
- `alerting` – kanały, throttling i backend audytu.
- `live_readiness_checklist` – weryfikacja podpisów KYC/AML oraz kompletności
  dokumentów LIVE zgodnie z konfiguracją.
- `live_readiness_metadata` – surowe metadane checklisty (nazwy dokumentów,
  podpisy, sumy SHA-256).

Podczas generowania raportu skrypt potwierdza fizyczną obecność każdego pliku
zadeklarowanego w sekcji `live_readiness.documents`, weryfikuje sumy SHA-256 i
istnienie artefaktów podpisów. Brak pliku, rozbieżność skrótu lub brak podpisu
powoduje oznaczenie pozycji jako `blocked` z odpowiednią przyczyną.

## 3. Checklist manualna

1. **Porównanie profili ryzyka** – potwierdź, że limity w `risk_profiles` są
   zgodne z ustaleniami zespołu ryzyka (np. raporty drawdown z paper tradingu).
2. **Licencja i moduły** – zweryfikuj `license.status` w raporcie. W przypadku
   błędów (`status=error`) należy naprawić brakujące klucze lub odświeżyć
   revocation list.
3. **Alerting** – upewnij się, że wymagane kanały (np. `telegram:primary`,
   `sms:orange_local`) znajdują się w raporcie oraz że throttling nie blokuje
   kategorii krytycznych.
4. **Dokumenty LIVE** – dla każdej pozycji `live_checklist` sprawdź status. W
   przypadku `blocked` należy uzupełnić brakujące podpisy lub pliki.
5. **Archiwizacja** – zapisz raport i wszystkie potwierdzenia w
   `var/audit/live/<YYYYMMDD>/promotion/` wraz z sumami kontrolnymi.

## 4. Eskalacja

- Brak podpisów (`live_checklist.status = "blocked"`) – eskaluj do Compliance.
- Nieprawidłowa licencja – eskaluj do Security/OEM Support.
- Brak kanałów alertingu – eskaluj do SRE; migracja jest wstrzymana do czasu
  przywrócenia powiadomień.

## 5. Po migracji

1. Monitoruj pierwsze zlecenia przez 60 minut (telemetria + alerty).
2. Zaktualizuj `docs/runbooks/PAPER_TO_LIVE_RUNBOOK.md` o numer raportu oraz
   timestamp przejścia.
3. Zarchiwizuj logi (`alerts/`, `risk_decisions.jsonl`, `live_execution/`) w
   katalogu audytowym.

Wykonanie powyższych kroków gwarantuje, że proces przejścia na tryb LIVE jest
udokumentowany, podpisany i zgodny z wymaganiami compliance oraz zespołów
operacyjnych.
