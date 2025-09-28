# Architektura bot_core

## Przegląd systemu

`bot_core` to modularna platforma do budowania i uruchamiania strategii krypto zgodna z etapem "foundation" opisanym w `docs/architecture/phase1_foundation.md`. Cały pipeline operacyjny działa **obowiązkowo w trybie demo/paper** do momentu przejścia pełnego cyklu bezpieczeństwa i zgodności. Dopiero po pozytywnym audycie kodu, potwierdzeniu kontroli ryzyka i podpisaniu akceptacji przez compliance możliwe jest przejście do środowiska LIVE.

Przepływ danych przebiega następująco:

1. Warstwa strategii (lifecycle `prepare` → `handle_market_data` → `generate_signal`) działa na danych dostarczonych przez `bot_core/data`.
2. Wygenerowane sygnały trafiają do `bot_core/risk`, gdzie przechodzą weryfikację profili ryzyka oraz limitów ekspozycji.
3. Zatwierdzone sygnały są przekazywane do `bot_core/execution`, które realizuje zlecenia przy wsparciu adapterów z `bot_core/exchanges`.
4. Całość nadzoruje `bot_core/runtime`, łączące konfigurację, zarządzanie środowiskami i raportowanie alertów z `bot_core/alerts`.

## Proces demo → paper → live

Proces przejścia środowisk jest sekwencyjny i wymusza blokady bezpieczeństwa na każdym etapie:

- **Demo/Testnet:** domyślny tryb uruchomieniowy. `StrategyContext.require_demo_mode` blokuje wykonanie w środowisku innym niż demo, dopóki nie zostaną spełnione wymagania bezpieczeństwa.
- **Paper:** po udokumentowanych wynikach testów demo pipeline przechodzi w tryb paper trading. `bot_core/runtime.bootstrap_environment` weryfikuje konfigurację (`core.yaml`), podpisane poświadczenia, kanały alertów i profile ryzyka przed startem.
- **Live:** aktywacja możliwa wyłącznie po przedstawieniu raportów z testów paper, przeglądu kodu bezpieczeństwa (`bot_core/security`) oraz zatwierdzenia compliance. `bot_core/execution` korzysta wówczas z adapterów `live`, włączane są rozszerzone alerty (kanały krytyczne + throttling) oraz automatyczne eskalacje do zespołu bezpieczeństwa.

Każdy etap wymaga aktualnych podpisów KYC/AML, potwierdzenia limitów w `RiskProfile` i rejestrowania zdarzeń audytowych. Przejście na wyższy poziom bez kompletu dokumentacji jest blokowane przez `runtime` (flagi `require_demo_mode`, `compliance_confirmed`) i walidację konfiguracji.

## Moduły `bot_core`

### `bot_core/exchanges`

Adaptery giełdowe (`BinanceSpotAdapter`, `BinanceFuturesAdapter`, `KrakenSpotAdapter`, `KrakenFuturesAdapter`, `ZondaSpotAdapter`) rozdzielają środowiska `demo`/`paper`/`live`, kontrolują uprawnienia (`read`/`trade`) i egzekwują podpisy HMAC wymagane przez API. Każdy adapter mapuje odpowiedzi na wspólne struktury (`AccountSnapshot`, `OrderStatus`) oraz obsługuje streaming danych, retry i walidację limitów notional.

### `bot_core/data`

Warstwa danych obejmuje `PublicAPIDataSource`, `CachedOHLCVSource` i usługi backfillu. Moduł pracuje dwuwarstwowo:

1. **Parquet jako źródło prawdy** – backfill zapisuje świeczki OHLCV do plików partycjonowanych według `exchange/symbol/granularity/year/month`. Format kolumnowy zapewnia dobrą kompresję, szybkie skany dla backtestów i łatwą integrację z Pandas/Polars.
2. **Manifest w SQLite** – lekka baza (`SQLiteCacheStorage`) przechowuje indeks metadanych (zakresy czasowe, wersje paczek, stany backfillu). Dzięki temu runtime odnajduje właściwe segmenty Parquet bez konieczności wczytywania wszystkich plików.

Źródła normalizują świeczki do UTC, deduplikują zapisy i stosują kontrolowany backoff, aby nie przekraczać limitów publicznych API giełd. Dane są indeksowane per środowisko i giełdę, co umożliwia równoległe testy demo/paper oraz szybkie odtwarzanie historii na potrzeby kontroli ryzyka.

### `bot_core/risk`

Moduł ryzyka (`RiskProfile`, `ThresholdRiskEngine`, `RiskRepository`) wymusza limity dziennych strat, liczbę pozycji, maksymalną ekspozycję per instrument i hard-stop drawdown. Silnik resetuje limity w UTC, blokuje sygnały przekraczające polityki, eskaluje incydenty do alertów oraz aktywuje tryb awaryjny po przekroczeniu progów ochronnych. Profile konfiguruje się w `config/core.yaml`, a walidacja odbywa się w runtime.

### `bot_core/execution`

`ExecutionService` i `ExecutionContext` zamieniają zatwierdzone sygnały na zlecenia, uwzględniając prowizje, poślizg i zasady retry/backoff (`RetryPolicy`). `PaperTradingExecutionService` symuluje fill'e, zapisuje dziennik audytowy i dostarcza metryki do alertów. W trybie live usługa współpracuje z adapterami giełd z segmentu `live` oraz wymaga potwierdzonych uprawnień tradingowych.

### `bot_core/alerts`

`AlertRouter`, `AlertChannel` i `FileAlertAuditLog` obsługują powiadomienia (Telegram, e-mail, SMS, Signal, WhatsApp, Messenger) z kontrolą throttlingu i pełnym audytem zdarzeń (`channel="__suppressed__"` dla zdławionych komunikatów). Warstwa SMS jest modułowa: na starcie korzystamy z lokalnych operatorów (Orange Polska jako referencyjny, następnie inni dostawcy w PL i IS), a globalny agregator (Twilio/Vonage/MessageBird) działa jako fallback ciągłości działania. Alerty są elementem procesów bezpieczeństwa – incydenty krytyczne muszą zostać potwierdzone i przekazane do zespołu bezpieczeństwa w ciągu 24h.

### `bot_core/runtime`

`bootstrap_environment` integruje konfigurację (`load_core_config`), adaptery giełdowe, profile ryzyka, alerty i manager sekretów w `BootstrapContext`. Runtime odpowiada za sekwencjonowanie przejść demo → paper → live, walidację flag compliance, rejestrację kontrolnych checklist oraz inicjalizację repozytoriów danych. Mechanizmy ochronne blokują start środowiska, jeśli brakuje kluczy API, audytów lub aktualnych podpisów regulacyjnych.

### `bot_core/security`

`SecretManager` i `KeyringSecretStorage` przechowują poświadczenia poza repozytorium, rozdzielając klucze `read` i `trade` dla każdego środowiska. Moduł implementuje rotację kluczy co 90 dni (z natychmiastową wymianą po zmianie uprawnień lub incydencie), walidację środowisk (paper/live/testnet) oraz integruje się z politykami IP allowlist. Klucze przechowujemy natywnie (Windows Credential Manager, macOS Keychain, GNOME Keyring/zaszyfrowany magazyn `age` w trybie headless). Wszystkie operacje są logowane i dostępne w dziennikach audytu wykorzystywanych przez compliance.

## Mechanizmy bezpieczeństwa i compliance

- **Wymuszenie trybu demo/paper:** `StrategyContext.require_demo_mode` oraz walidacja konfiguracji w runtime zapobiegają uruchomieniu strategii live bez akceptacji ryzyka.
- **Separacja uprawnień:** adaptery giełdowe stosują osobne klucze API dla `read`/`trade`, a `SecretManager` pilnuje środowisk i rotacji kluczy.
- **Kontrola ryzyka:** `RiskEngine` blokuje sygnały przekraczające limity ekspozycji, liczbę pozycji, dzienny drawdown i wymagania margin.
- **Alerty i audyt:** `AlertRouter` utrzymuje kanały eskalacji, logi audytowe, throttling powiadomień oraz politykę retencji (24 miesiące).
- **Zgłaszanie incydentów:** każde naruszenie bezpieczeństwa lub anomalia handlowa musi być zgłoszona do zespołu bezpieczeństwa (`#sec-alerts`) i opisana w raporcie post-incident w ciągu 24h. Dzienniki danych i alertów są zabezpieczane do analizy.

## Spójność z dokumentacją etapu 1

Dokument został zaktualizowany zgodnie z zakresem `docs/architecture/phase1_foundation.md`. Najważniejsze założenia – modularny podział `bot_core`, separacja środowisk, profile ryzyka, centralny bootstrap oraz audyt alertów – są odzwierciedlone w powyższych sekcjach. Zalecane jest cykliczne review z zespołem architektury po każdej istotnej zmianie modułów.

## Materiały onboardingowe

`docs/ARCHITECTURE.md` dodajemy do listy materiałów startowych dla nowych członków zespołu (obok `docs/architecture/phase1_foundation.md` i checklist bezpieczeństwa). Dokument stanowi punkt wejścia do zrozumienia przepływu demo → paper → live oraz powiązanych mechanizmów ochronnych.

## Kolejne kroki

- Rozbudowa dokumentacji o diagramy przepływu danych i interakcji pomiędzy modułami.
- Dodanie testów integracyjnych pipeline'u (sygnał → ryzyko → egzekucja) z wykorzystaniem środowisk demo i paper.
- Przygotowanie checklisty audytu LIVE (compliance + ryzyko technologiczne) wraz z kryteriami przejścia do produkcji.
