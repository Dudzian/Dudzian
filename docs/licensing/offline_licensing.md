# Licencjonowanie offline

Ten moduł wprowadza obsługę licencji offline podpisanych kluczem Ed25519. Pakiet
licencyjny (`.lic`) zawiera dwa pola: `payload_b64` oraz `signature_b64`. Po
zdekodowaniu i weryfikacji podpisu generowane są *capabilities* opisujące dostępne
funkcje, limity i moduły w ramach aplikacji bot_core oraz towarzyszących launcherów
CLI.

## Kluczowe komponenty

| Ścieżka | Opis |
| --- | --- |
| `bot_core/security/capabilities.py` | Definicje struktur danych opisujących moduły, limity i statusy trial/maintenance. |
| `bot_core/security/license_service.py` | Ładowanie pakietu `.lic`, weryfikacja Ed25519, budowa `LicenseCapabilities` oraz zapis migawki do `var/security/license_status.json` i loga JSONL `logs/security_admin.log`. |
| `bot_core/security/clock.py` | Monotoniczny zegar zapisujący ostatnią datę użycia licencji (anti-rollback). |
| `bot_core/security/hwid.py` | Dostawca fingerprintu sprzętowego zgodnego z `bot_core.security.fingerprint`. |
| `bot_core/security/guards.py` | Runtime'owe strażniki do egzekwowania limitów i wymagań edycji/modułów. |
| `python scripts/generate_license.py` | CLI do podpisywania payloadów JSON i generowania pakietów `.lic`. |

## Integracja runtime

`bot_core/runtime/bootstrap.py` wczytuje licencję offline, jeżeli zdefiniowano
zmienne środowiskowe:

```bash
export BOT_CORE_LICENSE_PATH=/opt/dudzian/licenses/acme.lic
export BOT_CORE_LICENSE_PUBLIC_KEY=0123ABCD...
```

Po pozytywnej weryfikacji `LicenseValidationResult.capabilities` zawiera
przetworzone uprawnienia. Moduły i launchery mogą korzystać z
`CapabilityGuard`, aby egzekwować dostęp do giełd, strategii, czy limitów
instancji. `ClockService` utrwala ostatnią datę użycia licencji, dzięki czemu
cofnięcie zegara systemowego nie przywróci wygasłych uprawnień, a
`HwIdProvider` dopasowuje pole `hwid` z payloadu do lokalnego fingerprintu
hosta. Migawka licencji (edycja, moduły, status trial/maintenance, skrót
payloadu) jest zapisywana w `var/security/license_status.json`, a zdarzenie
z weryfikacji trafia jako wpis JSONL do `logs/security_admin.log`, co
ułatwia audyt działań operatorów OEM.

## Integracja z interfejsami GUI

*Trading GUI* (`bot_core.ui.trading`) prezentuje podsumowanie licencji w
nagłówku okna i automatycznie blokuje niedozwolone funkcje:

- combobox sieci usuwa opcję **Live**, jeśli licencja nie dopuszcza środowiska
  produkcyjnego lub edycji Pro;
- tryb **Futures** jest ukrywany, gdy moduł `futures` nie został aktywowany;
- przycisk **Start** pozostaje nieaktywny, jeśli runtime `auto_trader` jest
  niedostępny w licencji.
- status licencji i komunikaty (np. konieczność rozszerzenia modułu) są
  aktualizowane w czasie rzeczywistym i powiązane z błędami strażnika
  `CapabilityGuard`.

Panel administracyjny Qt prezentuje edycję, aktywne moduły, środowiska,
utrzymanie/trial oraz dane posiadacza licencji. Listy modułów i runtime są
pobierane z `LicenseCapabilities`, co pozwala operatorom OEM szybko
zidentyfikować brakujące rozszerzenia bez zaglądania do plików `.lic`.

Kontroler sesji handlowej rezerwuje slot licencyjny (`paper_controller` lub
`live_controller`) w momencie startu i zwalnia go po zatrzymaniu. W przypadku
braku uprawnień użytkownik otrzymuje komunikat z instrukcją kontaktu z opiekunem
licencji, zgodnie z wymaganiami OEM.

## Tolerancja dryfu fingerprintu

Podpisy `license.json` i `fingerprint.json` są porównywane z marginesem tolerancji:

- dryf **MAC** lub **dysku** powoduje ostrzeżenie (`degraded`), ale nie blokuje startu,
- dryf **CPU** lub **TPM** wymaga ponownego przypisania licencji (`rebind_required`).

Scenariusze dryfu są dokumentowane w raporcie `reports/ci/licensing_drift/compatibility.json`, generowanym nocnie, aby operatorzy mogli potwierdzić zgodność sprzętu z podpisem OEM.

## Obsługa alertów dryfu licencji

Począwszy od nightly joba `Licensing drift consolidation` (`.github/workflows/licensing-drift.yml`) raporty z katalogu
`reports/ci/licensing_drift/` są agregowane do zestawień JSON/CSV oraz eksportu metryk Prometheus (`licensing_drift.prom`).
Job zawsze uruchamia konsolidację (nawet przy niepowodzeniu testów), a w razie braku `compatibility.json` generuje puste
podsumowanie, dzięki czemu monitoring nie traci sygnału o ostatnim biegu.
Pliki są kopiowane do `reports/ci/licensing_drift/dashboard/`, skąd mogą być scrapowane przez textfile collector lub serwowane
statycznie do Grafany. W tym samym katalogu publikowane są także `licensing_drift_summary.parquet` i
`licensing_drift_summary.csv` na potrzeby paneli trendów oraz zasilenia BI.

Pole `diagnostics` w `licensing_drift_summary.json` zbiera komunikaty o brakujących lub uszkodzonych artefaktach (np. brak logu
pytest albo `compatibility.json`). Dzięki temu operator może szybko zrozumieć przyczynę pustych metryk bez wchodzenia w logi
workflowa.

Reguły alertów Prometheusa (`deploy/prometheus/rules/stage6_alerts.yml`) pokrywają progi operacyjne:

- **LicensingDriftRebindRequired (critical):** `sum(licensing_drift_status{status="rebind_required"}) > 0` – natychmiastowa eskalacja do OEM/SRE, blokada startu runtime do czasu ponownego podpisu.
- **LicensingDriftRebindBurst (critical):** `sum_over_time(licensing_drift_status{status="rebind_required"}[7d]) > 1` – powtarzane rebindy w tygodniu, eskalacja do L3 Security i OEM z wnioskiem o badanie przyczyn sprzętowych.
- **LicensingDriftDegradedSpike (warning):** `sum_over_time(licensing_drift_status{status="degraded"}[7d]) > 3` – zaplanuj serwis sprzętu zanim dryf przejdzie w rebind.
- **LicensingDriftDegradedErosion (critical):** `sum_over_time(licensing_drift_status{status="degraded"}[7d]) > 5` – eskalacja do SRE z planem wymiany komponentów.

Dashboard Grafany `deploy/grafana/provisioning/dashboards/licensing_drift.json` (UID `licensing-drift`) prezentuje trendy odrzuceń/degradacji oraz
aktualny status scenariuszy HWID. Procedura operacyjna:

1. L1/NOC obserwuje panele trendów (`Kumulacja odrzuceń w 7 dniach`, `Trend tolerowanych dryfów (7d)`) oraz staty rebind/degraded; przy alertach critical otwiera zgłoszenie OEM oraz eskaluje do SRE/Security.
2. Zweryfikuj `reports/ci/licensing_drift/licensing_drift_summary.parquet` lub JSON/CSV (kolumny `scenario`, `status`, `blocked`) i odnotuj wynik w decision logu z ID alertu.
3. L2 przygotowuje rebind: zbiera fingerprint, synchronizuje z OEM, po podpisaniu aktualizuje magazyn licencji oraz domyka alert w Grafanie.
4. W scenariuszu degraded monitoruj wzrost liczników w kolejnych runach; jeśli wskaźnik przekroczy próg critical, przełącz playbook na procedurę rebind i eskaluj do OEM/SRE.

## Procedura rebind offline

1. Zbierz nowy fingerprint (`fingerprint.json`) z hosta, na którym wystąpił błąd `rebind_required`.
2. Zweryfikuj raport dryfu (`reports/ci/licensing_drift/compatibility.json`), aby potwierdzić, że zmiana dotyczy krytycznych komponentów.
3. Przekaż fingerprint OEM wraz z ID licencji; po otrzymaniu nowego pakietu `.lic` zapisz go na hoście i zrestartuj runtime.
4. Zaktualizuj dziennik operacyjny wpisem JSONL w `logs/security_admin.log` (źródło `licensing`). Minimalny schemat: `{ "event": "rebind", "license_id": "...", "components": ["cpu", "tpm"], "signed_by": "...", "result": "validated", "timestamp": "..." }`.
5. Wyeksportuj migawkę `var/security/license_status.json` po rebindzie i dołącz do zgłoszenia audytowego w systemie ticketowym OEM.

## Procedura appeal offline

1. Jeśli rebind nie jest możliwy (np. brak dostępu do OEM), przygotuj pakiet dowodów: log `logs/security_admin.log`, ostatni fingerprint oraz raport CI dryfu.
2. Przekaż pakiet do zespołu bezpieczeństwa; decyzja o tymczasowym obejściu (np. whitelist komponentów) musi być udokumentowana i wprowadzona jako wpis w `logs/security_admin.log` z polami: `appeal`, ID licencji, przyczyna, akceptujący audytor.
3. Zarejestruj wyjątek w systemie ticketowym i odnotuj jego identyfikator w logu audytowym (przykład: `{ "event": "appeal", "license_id": "...", "reason": "tpm replacement", "auditor": "...", "ticket": "OEM-1234", "decision": "temporary_allow" }`).
4. Po wykonaniu rebindu offline zaktualizuj wpis audytowy o numer biletu oraz potwierdzenie, że wyjątek został zamknięty, a następnie dołącz nową migawkę `license_status.json` do tego samego zgłoszenia.

## Launcher AutoTrader (headless)

`bot_core.auto_trader.app` stanowi kanoniczny runtime, a
`bot_core.auto_trader.paper_app.PaperAutoTradeApp` ładuje `LicenseCapabilities`
przy uruchomieniu. Strażnik licencji blokuje start, gdy brakuje modułu
`auto_trader`, odpowiedniego środowiska (`paper/demo` lub `live`) albo
przekroczono limity instancji. Wymagane zachowania:

- tryb **Live** wymaga edycji co najmniej *Pro* oraz dostępu do środowiska `live`;
- środowiska futures (np. `binance_futures`, `kraken_futures`) wymagają modułu
  `futures`;
- przed uruchomieniem rezerwowane są sloty licencyjne `paper_controller`/`live_controller`
  oraz `bot`; przekroczenie limitów skutkuje `LicenseCapabilityError` i alertem
  `license_restriction`.

Launcher CLI (wykorzystujący `PaperAutoTradeApp.main()`) kończy działanie kodem
wyjścia różnym od zera przy naruszeniu restrykcji, a logi zawierają komunikat o
blokadzie licencyjnej. Wywołanie deleguje do runtime `bot_core.auto_trader.app`,
dlatego scenariusze licencyjne należy testować również na importerach
korzystających bezpośrednio z pakietu `bot_core`.

## Generator licencji

Przykład użycia CLI do podpisania licencji:

```bash
python scripts/generate_license.py payload.json <private_key_hex> -o acme.lic --verify <public_key_hex>
```

Po wygenerowaniu narzędzie zapisuje pakiet `.lic` i wyświetla sumę SHA-256
payloadu.
