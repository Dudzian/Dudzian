# Architektura KryptoLowca

## Przegląd systemu

KryptoLowca jest modularnym środowiskiem do tworzenia i uruchamiania strategii
handlowych na giełdach krypto. Cały pipeline operacyjny działa **obowiązkowo w
trybie demo/testnet** do momentu zakończenia audytu bezpieczeństwa i zgodności.
Przed włączeniem trybu LIVE wymagane jest udokumentowane przejście scenariuszy
papier trading oraz zatwierdzenie przez dział compliance.

Centralny przepływ danych wygląda następująco:

1. **Serwis sygnałów** (`KryptoLowca/core/services/signal_service.py`) ładuje
   strategię z rejestru, buduje kontekst (wymuszając tryb demo) i deleguje
   przygotowanie oraz generowanie sygnałów.
2. **Serwis ryzyka** (`KryptoLowca/core/services/risk_service.py`) wykonuje
   kontrolę limitów ekspozycji, ostatnich strat i wymusza blokady ochronne.
3. **Serwis egzekucji** (`KryptoLowca/core/services/execution_service.py`) zamienia
   zatwierdzone sygnały na zlecenia API giełdy poprzez adaptery.

Każda warstwa jest odseparowana kontraktami bazowymi, a komunikacja przebiega w
układzie sygnał → ocena ryzyka → wykonanie. Poniższe sekcje opisują moduły,
które wchodzą w skład tej architektury.

## Strategie bazowe (`KryptoLowca/strategies/base/`)

- `engine.py` zawiera klasę `BaseStrategy`, struktury danych (`StrategyContext`,
  `StrategySignal`) oraz `DataProvider`. Kontekst posiada metodę
  `require_demo_mode`, która blokuje handel LIVE bez formalnego audytu.
  Strategia ma lifecycle: `prepare()` (z wymuszeniem trybu demo),
  `generate_signal()` (zaimplementowana w presetach) oraz hooki na fill/shutdown.
- `registry.py` implementuje prosty rejestr (`StrategyRegistry`) oraz dekorator
  `@strategy` do automatycznej rejestracji presetów. Rejestr jest wykorzystywany
  zarówno przez marketplace jak i silnik auto-tradingu/tests.

## Usługi rdzeniowe (`KryptoLowca/core/services/`)

- `signal_service.py` odpowiada za odszukanie strategii w rejestrze, zbudowanie
  `StrategyContext` z wymuszonym trybem demo oraz wywołanie lifecycle'u
  `prepare()`/`handle_market_data()`.
- `risk_service.py` przeprowadza politykę zarządzania ryzykiem – limity
  notional, kontrolę strat dziennych, odrzucanie sygnałów `HOLD`.
- `execution_service.py` izoluje logikę wysyłania zleceń. Wymaga adaptera giełdy
  implementującego `submit_order` i respektuje ograniczenia (np. brak rozmiaru).

## Konfiguracja (`KryptoLowca/config_manager.py`)

`ConfigManager` spina konfigurację aplikacji i integruje się z marketplace
strategii. Obsługuje szyfrowanie kluczy API, waliduje sekcje (AI, bazy danych,
parametry handlu, giełdy, strategii) oraz pilnuje flag związanych z compliance
(m.in. `require_demo_mode`, `compliance_confirmed`). Każda zmiana konfiguracji
musi uwzględniać zasady KYC/AML i przejść walidację `validate()`.

## Marketplace presetów (`KryptoLowca/strategies/marketplace.py`, `presets/`)

Marketplace udostępnia gotowe presety strategii wraz z metadanymi marketingowymi.
Moduł `marketplace.py` ładuje builtin presety oraz JSON-y z katalogu
`KryptoLowca/strategies/presets/`. Preset (`StrategyPreset`) zawiera parametry do
konfiguracji strategii, rekomendowany balans i tagi. Rejestr strategii oraz
manager konfiguracji korzystają z presetów, aby publikować strategie w GUI i
uruchamiać pipeline demo.

## Bezpieczeństwo, compliance i zgłaszanie incydentów

- **Tryb demo/testnet** jest obowiązkowy aż do przejścia pełnego procesu KYC/AML,
  audytu kodu oraz podpisania akceptacji ryzyka. Wszystkie testy i review muszą
  wykonywać zlecenia na kontach paper-trading.
- KYC/AML: każdy adapter giełdowy musi być skonfigurowany z kontem spełniającym
  wymagania regulacyjne. Zmiany kluczy API wymagają zatwierdzenia przez dział
  compliance i aktualizacji metadanych w `ConfigManager`.
- Zgłaszanie incydentów: incydenty bezpieczeństwa i anomalie handlowe należy
  natychmiast zgłosić do zespołu bezpieczeństwa poprzez kanał `#sec-alerts` oraz
  wypełnić formularz post-incident w ciągu 24h. Logi z usług rdzeniowych i
  dashboardów muszą zostać zabezpieczone na potrzeby analizy.

## Kolejne kroki

- Rozbudowa dokumentacji o szczegółowe diagramy przepływu danych.
- Dodanie testów integracyjnych pipeline'u (sygnał → ryzyko → egzekucja).
- Przygotowanie checklisty audytu LIVE (compliance + ryzyko technologiczne).
