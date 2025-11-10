# Changelog runbooków operacyjnych

## 2025-10-30 – Logowanie Stage6 i migrator bez fallbacków legacy
- **Zakres**: `bot_core/logging/app.py`, migrator Stage6 (`python -m bot_core.runtime.stage6_preset_cli`), dokumentacja runbooków.
- **Zmiana**: usunięto obsługę zmiennych środowiskowych `KRYPT_LOWCA_*` na rzecz wyłącznych prefiksów `BOT_CORE_*`; legacy
  zmienne są ignorowane, więc konfiguracja wymaga jawnego ustawienia nowszych nazw. Pomoc CLI migratora Stage6 ukrywa
  nieaktywne flagi `--legacy-security-*`, dzięki czemu `--help` prezentuje wyłącznie wspierane opcje.
- **Działanie dla zespołów**: zaktualizujcie własne skrypty startowe i pipeline'y CI, aby eksportowały wyłącznie nowe zmienne
  `BOT_CORE_LOG_DIR`, `BOT_CORE_LOG_FILE`, `BOT_CORE_LOGGER_NAME`, `BOT_CORE_LOG_LEVEL`, `BOT_CORE_LOG_FORMAT` i
  `BOT_CORE_LOG_SHIP_VECTOR`. W dokumentacji operacyjnej korzystajcie z odświeżonych przykładów CLI (patrz niżej)
  podczas warsztatów hypercare/migracyjnych.

## 2025-10-25 – Usunięcie archiwalnego pakietu legacy
- **Zakres**: `archive/` (czyszczenie), dokumentacja migracyjna oraz README.
- **Zmiana**: skasowano katalog `archive/legacy_bot` i zaktualizowano materiały, aby jasno wskazywały brak shimów legacy.
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
