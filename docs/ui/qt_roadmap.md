# Roadmap UI Desktop (Qt)

## Cele nadrzędne

1. **Konfiguracja strategii** – wizualne zarządzanie katalogiem pluginów, parametrami `TradingParameters` oraz mapowaniem na reżimy rynkowe.
2. **Monitoring** – widok sytuacyjny łączący dane rynkowe, sygnały strategii oraz historię klasyfikacji `MarketRegimeClassifier`.
3. **Kontrola licencji i fingerprintu** – moduł odpowiedzialny za aktywację, audyt oraz egzekwowanie polityki bezpieczeństwa.

## Strategy Workbench ↔ StrategyRegimeWorkflow

* Mostkowanie z `Strategy Workbench` poprzez wywołanie `python scripts/ui_config_bridge.py --describe-catalog` (lub aliasu pakietowego) w celu pobrania aktualnych opisów strategii, statusów dostępności oraz podpisanych wersji presetów.
* Pobieranie raportów `StrategyRegimeWorkflow` (dostępność presetów, podpisy HMAC, blokady harmonogramu, kandydaci decyzji, podsumowania reżimów, osobna lista fallbacków oraz statystyki aktywacji) przy pomocy `python scripts/ui_config_bridge.py --describe-regime-workflow --regime-workflow-dir var/data/strategy_regime_workflow`. Mostek uzupełnia brakujące znaczniki `issued_at` wartością `1970-01-01T00:00:00Z`, buduje mapę wersji po hashach i potrafi odtworzyć szczegóły presetów na podstawie samego `version_hash` z historii aktywacji. Dodatkowo zwraca agregaty dostępności (`availability_stats`) z rozkładem reżimów, braków danych i blokad licencyjnych, aby panel mógł je prezentować bez dodatkowego przetwarzania.
* Uruchamianie kreatora presetów (`python scripts/ui_config_bridge.py --preset-wizard --wizard-mode build --input preset_payload.json`) z przekazaniem parametrów kontekstowych wybranych w UI i obsługą zwrotów dotyczących zgodności podpisów.
* Prezentowanie operatorowi w UI: raportów dostępności strategii, wyników walidacji podpisów, historii wersji presetów oraz sugerowanych działań naprawczych przekazanych przez mostek.
* Propagowanie do backendu zmian zatwierdzonych w UI (aktualizacje wag reżimów, aktywacje presetów) wraz z metadanymi audytowymi wymaganymi przez `StrategyRegimeWorkflow`.

### Wymagania mostka konfiguracyjnego

* Ścieżka do interpretera Pythona i skryptu mostka są konfigurowane flagami `--strategy-config-python` oraz `--strategy-config-bridge` (lub odpowiadającymi im zmiennymi środowiskowymi), zgodnie z opisem w `ui/README.md`.
* Wszystkie wspierane flagi (`--describe-catalog`, `--describe-regime-workflow`, `--regime-workflow-dir`, `--preset-wizard`, `--wizard-mode`, `--input`, `--config`, `--apply`, `--dump`, `--section`, `--scheduler`) są udokumentowane w `scripts/ui_config_bridge.py` i stanowią źródło prawdy dla roadmapy.

## Kamienie milowe

### M1 – Fundamenty aplikacji (Sprinty 1-2)

* Skeleton aplikacji Qt (Qt6/QML) z modułem startowym i konfiguracją buildów wieloplatformowych.
* Integracja z silnikiem tradingowym poprzez warstwę usługową (`TradingEngine`, `StrategyCatalog`, `StrategyRegimeWorkflow`), wraz z obsługą API mostka udostępniającego raporty dostępności strategii oraz metadane wersji presetów konsumowanych przez powłokę Qt.
* System modułów w UI (pluginy UI) pozwalający na dynamiczne dodawanie widoków.

### M2 – Konfiguracja strategii (Sprinty 3-4)

* Kreator strategii oparty na `TradingParameters` z walidacją w czasie rzeczywistym.
* Biblioteka presetów strategii (trend, day-trading, mean-reversion, arbitraż) z możliwością duplikacji i edycji.
* Widok mapowania strategii na reżimy (wykorzystanie `StrategyRegimeWorkflow`) – edycja wag, progów przełączania, wersjonowanie konfiguracji oraz prezentacja raportów dostępności i historii wersji presetów zwróconych przez API mostka.
* Obsługa podpisów kryptograficznych presetów (walidacja i oznaczanie statusu w UI) w oparciu o dane z backendowego katalogu strategii.

### M3 – Monitoring i telemetria (Sprinty 5-6)

* Panel główny z widżetami: notowania (candlestick + wskaźniki), sygnały pluginów, aktualny reżim i wskaźniki ryzyka.
* Rejestr zdarzeń (logi, alerty ryzyka, decyzje workflow) z możliwością filtrowania i eksportu.
* Analiza post-trade: przebieg pozycji, statystyki portfela, wizualizacja wyników z backtestów.

### M4 – Kontrola licencji i fingerprint (Sprinty 7-8)

* Ekran zarządzania licencjami: aktywacja, przypisanie do urządzenia, wygasanie, historia.
* Moduł fingerprint: generowanie identyfikatora sprzętowego, detekcja zmian, alerty bezpieczeństwa.
* Raportowanie blokad licencyjnych (synchronizacja z backendem, powiadomienia operatora i eksport logów audytowych).
* Integracja z serwerem licencyjnym (REST/gRPC) z obsługą offline cache i harmonogramem odświeżania.

### M5 – Twarde testy i wydanie (Sprinty 9-10)

* Testy E2E (Qt Test, pytest-qt) obejmujące konfigurację, workflow reżimów i autoryzację.
* Profilowanie wydajności (Qt Quick Profiler, cProfile) oraz optymalizacje UI.
* Przygotowanie pakietów instalacyjnych (Windows/MSIX, macOS/.dmg, Linux/AppImage) oraz dokumentacja użytkownika.

## Ryzyka i zależności

* Wysokie skomplikowanie modeli danych (`TradingParameters`, historia reżimów) – konieczny system walidacji i migracji konfiguracji.
* Integracja z backendem licencyjnym – wymagane mocki i środowisko testowe.
* Utrzymanie spójności pomiędzy pluginami strategii a interfejsem użytkownika (synchronizacja opisów, parametrów, wersji).

## Wskaźniki sukcesu

* < 5 minut na pełną konfigurację strategii i aktywację reżimów w UI.
* 99% skuteczność w detekcji zmian fingerprintu i blokadzie nieautoryzowanych urządzeń.
* Stabilność aplikacji w testach długotrwałych (>72h) bez regresji wydajności.
