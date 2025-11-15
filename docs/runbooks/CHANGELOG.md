# Changelog runbooków operacyjnych

## 2025-11-10 – Kontrola HWID/licencji dla modułu cloud
- **Zakres**: `proto/trading.proto`, `bot_core/cloud/**`, `config/cloud/server*.yaml`, README oraz runbooki live/paper/security`.
- **Zmiana**: serwer cloudowy wymaga teraz handshake'u `CloudAuthService.AuthorizeClient`. Poprawnie podpisany payload `{license_id, fingerprint, nonce}` (HMAC na sekrecie powiązanym z HWID) zwraca token `CloudSession`, który trzeba przesyłać w nagłówku `Authorization`. Nowa sekcja `security.allowed_clients` w `config/cloud/server.yaml` przechowuje allowlistę HWID/licencji wraz ze źródłami kluczy, a wszystkie próby autoryzacji trafiają do `logs/security_admin.log`.
- **Działanie dla zespołów**: dodajcie własne wpisy allowlisty (najlepiej przez `shared_secret_env`), przygotujcie instrukcje podpisywania payloadu dla operatorów UI i dopiszcie do runbooków krok „cloud auth audit” – podczas startu backendu za flagą należy zweryfikować wpisy w `logs/security_admin.log` i potwierdzić, że tylko uprawnione HWID-y otrzymały token.

## 2025-11-06 – Cloud runtime za flagą
- **Zakres**: `bot_core/cloud/**`, `scripts/run_cloud_service.py`, runbooki live/paper oraz README.
- **Zmiana**: dodano pakiet cloudowy udostępniający serwer gRPC startowany poleceniem `python scripts/run_cloud_service.py --config config/cloud/server.yaml`. Serwis ładuje runtime Stage6, oferuje marketplace i harmonogramy AI, a gotowość sygnalizuje payloadem `ready`. Runbooki otrzymały checklisty dla operatorów uruchamiających backend za flagą.
- **Działanie dla zespołów**: przygotujcie `config/cloud/server.yaml`, aby ewentualne wdrożenie cloudowe było plug-and-play. W decision logach dokumentujcie zarówno start lokalny (`run_local_bot`), jak i aktywację modułu cloud – szczególnie jeśli wymaga dodatkowych licencji/HWID.

## 2025-11-04 – Doprecyzowanie opisów interfejsów Stage6
- **Zakres**: `bot_core/exchanges/interfaces.py`, `scripts/find_duplicates.py`, runbooki developerskie.
- **Zmiana**: usunięto pozostałe wzmianki o kompatybilności z dawnym modułem `KryptoLowca` z docstringów i opisów narzędzi, aby
  dokumentacja odnosiła się wyłącznie do aktualnej architektury Stage6.
- **Działanie dla zespołów**: podczas przeglądów kodu odwołujcie się już tylko do bieżących modułów (`bot_core`, `core`, `ui`);
  ewentualne pytania migracyjne trzeba kierować do dokumentów w `docs/migrations/`.
- **Aktualizacja**: komunikaty błędów dotyczące magazynu sekretów i metryk AI odnoszą się teraz do „historycznych formatów Stage5”
  zamiast „legacy”, aby utrzymać spójne słownictwo i przygotować repo do twardych testów QA zakazujących dawnej nomenklatury.

## 2025-10-30 – Logowanie Stage6 i migrator bez fallbacków archiwalnych
- **Zakres**: `bot_core/logging/app.py`, migrator Stage6 (`python -m bot_core.runtime.stage6_preset_cli`), dokumentacja runbooków.
- **Zmiana**: usunięto obsługę zmiennych środowiskowych `KRYPT_LOWCA_*` na rzecz wyłącznych prefiksów `BOT_CORE_*`; dawne
  zmienne są ignorowane, więc konfiguracja wymaga jawnego ustawienia nowszych nazw. Pomoc CLI migratora Stage6 ukrywa
  nieaktywne flagi poprzedniej warstwy, dzięki czemu `--help` prezentuje wyłącznie wspierane opcje.
- **Działanie dla zespołów**: zaktualizujcie własne skrypty startowe i pipeline'y CI, aby eksportowały wyłącznie nowe zmienne
  `BOT_CORE_LOG_DIR`, `BOT_CORE_LOG_FILE`, `BOT_CORE_LOGGER_NAME`, `BOT_CORE_LOG_LEVEL`, `BOT_CORE_LOG_FORMAT` i
  `BOT_CORE_LOG_SHIP_VECTOR`. W dokumentacji operacyjnej korzystajcie z odświeżonych przykładów CLI (patrz niżej)
  podczas warsztatów hypercare/migracyjnych.

## 2025-10-25 – Usunięcie archiwalnego pakietu desktopowego
- **Zakres**: `archive/` (czyszczenie), dokumentacja migracyjna oraz README.
- **Zmiana**: skasowano katalog z dawnym botem i zaktualizowano materiały, aby jasno wskazywały brak shimów poprzedniej warstwy.
- **Działanie dla zespołów**: wszystkie odwołania do dawnych namespace'ów muszą korzystać z `bot_core.*`; repozytorium nie
  zawiera już kopii modułów `KryptoLowca` nawet w trybie archiwalnym.

## 2025-10-23 – Aktualizacja komendy Paper Labs
- **Zakres**: `docs/runbooks/PAPER_LABS_CHECKLIST.md`
- **Zmiana**: doprecyzowano wywołanie `python scripts/run_risk_simulation_lab.py` wraz z obowiązkowymi flagami `--config config/core.yaml` i `--output-dir reports/paper_labs`.
- **Działanie dla zespołu Ryzyka**: od kolejnego cyklu Paper Labs używajcie nowej komendy, aby uniknąć uruchomień bez jawnie wskazanych ścieżek konfiguracyjnych i katalogu artefaktów.
## 2025-10-24 – Ujednolicenie komend CLI w runbookach
- **Zakres**: runbooki Stage4–Stage6, backfill, Paper Trading, OEM provisioning, decision log verification oraz checklisty hypercare/portfolio/resilience.
- **Zmiana**: doprecyzowano komendy `python scripts/...` (oraz warianty `PYTHONPATH=. python ...`) w instrukcjach operacyjnych, aby jednoznacznie wskazywały interpreter i wymagane flagi.
- **Działanie dla operatorów**: korzystajcie z ujednoliconych komend przy kolejnych procedurach (paper/demo/live), żeby uniknąć niejasności co do sposobu uruchamiania narzędzi.
- **Notatka**: rozszerzono dokumentację architektoniczną, warsztatową i operacyjną o identyczne prefiksy `python`, dzięki czemu opisy narzędzi w całym repozytorium wskazują pełne polecenia.
