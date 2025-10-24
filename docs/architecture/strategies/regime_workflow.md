# StrategyRegimeWorkflow – zarządzanie presetami strategii

`StrategyRegimeWorkflow` zastępuje legacy `RegimeSwitchWorkflow`, rozszerzając go o
rejestrację wersjonowanych presetów, awaryjne fallbacki, wymuszanie licencji
oraz raporty statystyczne z historii aktywacji. Workflow bazuje na
`StrategyPresetWizard`, dzięki czemu podpisuje presety HMAC zgodnie z katalogiem
strategii i przekazuje komplet metadanych do warstwy decyzyjnej.

## Rejestracja presetów i fallback

* `register_preset(regime=…)` pozwala przypisać dedykowany preset do każdego
  reżimu (`MarketRegime`). Preset jest budowany przez wizarda na podstawie
  wpisów `StrategyCatalog`, a następnie podpisywany kluczem HMAC.
* `register_emergency_preset()` definiuje awaryjny preset uruchamiany, gdy
  właściwy preset nie spełnia wymagań (brak danych, blokada licencji lub okno
  harmonogramu). Fallback przechodzi przez te same kontrole co presety
  standardowe.
* `activate()` wybiera preset odpowiedni dla rozpoznanego reżimu. Jeżeli brak
  dedykowanego presetu lub aktywacja jest zablokowana, workflow włącza preset
  awaryjny i oznacza aktywację flagą `used_fallback=True`.

Wszystkie kontrole (wymagane dane rynkowe, harmonogram, licencje) są realizowane
przed zbudowaniem kandydatów `DecisionCandidate`. W przypadku krytycznego braku
fallbacku workflow zgłasza wyjątek, zabezpieczając pipeline przed podjęciem
niepełnej decyzji.

## Wersjonowanie presetów i metadane HMAC

Każdy zarejestrowany preset otrzymuje strukturę `PresetVersionInfo`, która
zawiera:

- skrót SHA-256 payloadu,
- podpis HMAC (`build_hmac_signature`) z identyfikatorem klucza,
- znacznik czasu `issued_at` (UTC),
- snapshot metadanych strategii (`strategy_keys`, `license_tiers`,
  `risk_classes`, `required_data`, `capabilities`, `tags`).

Version info pozwala audytować zmiany presetów (np. na potrzeby mostów
konfiguracyjnych) i potwierdzać integralność payloadu po stronie
`DecisionOrchestrator`.

## Struktury domenowe

Nowy workflow udostępnia dodatkowe struktury ułatwiające raportowanie i
monitorowanie:

- `RegimePresetActivation` – pełny wynik aktywacji (regime, preset, wersja,
  kandydaci decyzji, flagi fallbacku/blokad, brakujące dane, ostrzeżenia
  licencyjne).
- `PresetAvailability` – raport gotowości presetu względem wymagań danych,
  harmonogramu i licencji.
- `ActivationHistoryStats`, `ActivationTransitionStats`,
  `ActivationCadenceStats`, `ActivationUptimeStats` – statystyki historii
  aktywacji (liczniki reżimów, macierze przejść, kadencja, uptime presetów,
  częstotliwość fallbacków, agregacja powodów blokad i braków danych).

## Katalog strategii i metadane

`StrategyPresetWizard` korzysta z wpisów `StrategyCatalog`, dlatego metadane
presetów są zgodne z definicjami silników strategii. Poniższa tabela przedstawia
wybrane klucze dostępne w katalogu domyślnym:

| Silnik (`StrategyCatalog`) | License tier | Klasy ryzyka | Wymagane dane | Capability | Tagi domyślne |
|----------------------------|--------------|--------------|---------------|------------|---------------|
| `daily_trend_momentum`     | `standard`   | `directional`, `momentum` | `ohlcv`, `technical_indicators` | `trend_d1` | `trend`, `momentum` |
| `scalping`                 | `professional` | `intraday`, `scalping` | `ohlcv`, `order_book` | `scalping` | `intraday`, `scalping` |
| `options_income`           | `enterprise` | `derivatives`, `income` | `options_chain`, `greeks`, `ohlcv` | `options_income` | `options`, `income` |
| `mean_reversion`           | `professional` | `statistical`, `mean_reversion` | `ohlcv`, `spread_history` | `mean_reversion` | `mean_reversion`, `stat_arbitrage` |
| `cross_exchange_arbitrage` | `enterprise` | `arbitrage`, `liquidity` | `order_book`, `latency_monitoring` | `cross_exchange` | `arbitrage`, `liquidity` |

Te metadane są agregowane i deduplikowane w trakcie aktywacji, dzięki czemu
payloady decyzji zawierają znormalizowane listy licencji, klas ryzyka i tagów.

### Domyślny katalog pluginów workflowu

Warstwa tradingowa (`bot_core.trading.strategies`) udostępnia lżejsze plug-iny
sygnałowe, które wykorzystuje `RegimeSwitchWorkflow` oraz `AutoTrader`. Każdy
plugin rejestruje te same metadane, co odpowiadający mu silnik z
`bot_core.strategies.catalog`, dzięki czemu UI otrzymuje spójne informacje o
licencji, wymaganych danych, capability i tagach. Synchronizacja odbywa się
automatycznie przy ładowaniu klas pluginów – wykorzystują one klucz `engine`
do pobrania bieżących metadanych z `DEFAULT_STRATEGY_CATALOG`, co eliminuje
ryzyko rozjazdu przy aktualizacjach katalogu silników.

| Plugin (`StrategyCatalog.default()`) | Silnik w `bot_core.strategies.catalog` | Capability | License tier | Klasy ryzyka | Wymagane dane | Tagi |
|-------------------------------------|----------------------------------------|------------|--------------|--------------|---------------|------|
| `trend_following`                   | `daily_trend_momentum`                 | `trend_d1` | `standard`   | `directional`, `momentum` | `ohlcv`, `technical_indicators` | `trend`, `momentum` |
| `day_trading`                       | `day_trading`                          | `day_trading` | `standard` | `intraday`, `momentum` | `ohlcv`, `technical_indicators` | `intraday`, `momentum` |
| `mean_reversion`                    | `mean_reversion`                       | `mean_reversion` | `professional` | `statistical`, `mean_reversion` | `ohlcv`, `spread_history` | `mean_reversion`, `stat_arbitrage` |
| `arbitrage`                         | `cross_exchange_arbitrage`             | `cross_exchange` | `enterprise` | `arbitrage`, `liquidity` | `order_book`, `latency_monitoring` | `arbitrage`, `liquidity` |
| `grid_trading`                      | `grid_trading`                         | `grid_trading` | `professional` | `market_making` | `order_book`, `ohlcv` | `grid`, `market_making` |
| `volatility_target`                 | `volatility_target`                    | `volatility_target` | `enterprise` | `risk_control`, `volatility` | `ohlcv`, `realized_volatility` | `volatility`, `risk` |
| `scalping`                          | `scalping`                             | `scalping` | `professional` | `intraday`, `scalping` | `ohlcv`, `order_book` | `intraday`, `scalping` |
| `options_income`                    | `options_income`                       | `options_income` | `enterprise` | `derivatives`, `income` | `options_chain`, `greeks`, `ohlcv` | `options`, `income` |
| `statistical_arbitrage`             | `statistical_arbitrage`                | `stat_arbitrage` | `professional` | `statistical`, `mean_reversion` | `ohlcv`, `spread_history` | `stat_arbitrage`, `pairs_trading` |

Każdy plugin udostępnia metodę `metadata()`, a katalog `StrategyCatalog.default()`
zawiera komplet opisów (`describe()`), co pozwala aplikacjom klienckim
weryfikować kompatybilność danych i licencji bez konieczności odpytywania
cięższych silników strategii.

> Wersja domyślna katalogu pluginów waliduje, że każdy silnik z
> `DEFAULT_STRATEGY_CATALOG` posiada przypisaną implementację pluginu.
> Brak pokrycia kończy się wyjątkiem podczas budowy katalogu, dzięki czemu
> testy natychmiast sygnalizują brakujący adapter.
> Dodatkowo rejestr pluginów blokuje duplikaty `engine_key`, więc nie da się
> przypadkowo zarejestrować dwóch wtyczek mapujących na ten sam silnik.
> Jeśli jednak świadomie chcemy zastąpić wbudowany plugin (np. w testach lub
> środowiskach eksperymentalnych), klasa może zadeklarować
> `allow_engine_override=True`, zachowując dziedziczenie metadanych z katalogu
> silników.

### Domyślne wagi strategii w `RegimeSwitchWorkflow`

Workflow rozprowadza ekspozycję na pełny zestaw pluginów – w każdym reżimie
pojawiają się zarówno strategie bazowe, jak i nowe pozycje. Domyślne wagi (po
normalizacji) wyglądają następująco:

* **TREND** – `trend_following` (35%), `volatility_target` (20%),
  `day_trading` (15%), `mean_reversion` (10%), `arbitrage` (10%),
  `grid_trading` (5%), `options_income` (5%).
* **DAILY** – `day_trading` (40%), `scalping` (20%), `trend_following` (10%),
  `volatility_target` (10%), `grid_trading` (10%), `arbitrage` (5%),
  `statistical_arbitrage` (5%).
* **MEAN_REVERSION** – `mean_reversion` (35%), `statistical_arbitrage` (25%),
  `arbitrage` (15%), `grid_trading` (10%), `options_income` (10%),
  `scalping` (5%).

Dzięki temu raporty workflowu zawierają metadane (licencje, capability,
wymagane dane i tagi) dla każdego pluginu dostępnego w katalogu, a UI może
natychmiast zweryfikować gotowość danych i uprawnień.

## Przykładowe przepływy aktywacji

1. **Standardowa aktywacja** – dla reżimu TREND workflow używa presetu
   `daily_trend_momentum`, generuje kandydatów decyzji i zapisuje wpis w
   historii. Metadane wersji potwierdzają podpis HMAC oraz wykorzystany katalog.
2. **Fallback przy brakujących danych** – jeśli reżim MEAN_REVERSION wymaga
   `spread_history`, a dane nie są dostępne, aktywacja zostaje oznaczona jako
   `missing_data`, po czym workflow przełącza się na preset awaryjny (np.
   `scalping`). W historii pojawia się wpis z `used_fallback=True`.
3. **Blokada licencji** – gdy strażnik licencyjny (`get_capability_guard`) nie
   potwierdzi capability presetu (np. `options_income` bez aktywnej licencji),
   aktywacja jest oznaczona `license_blocked`, a workflow wymusza fallback lub
   zgłasza wyjątek przy jego braku. Informacja o blokadzie jest dostępna w
   `RegimePresetActivation.license_issues` oraz agregowana w raportach historii.

## Raporty historii i monitorowanie

Metody `activation_history()`, `activation_history_stats()`,
`activation_transition_stats()`, `activation_cadence_stats()` oraz
`activation_uptime_stats()` umożliwiają:

- budowę dashboardów uptime/użycia strategii,
- analizę przejść między reżimami i fallbackami,
- monitorowanie najczęstszych powodów blokad i braków danych,
- walidację kadencji aktywacji względem harmonogramów i progów klasyfikatora.

Dzięki limitom historii (`activation_history_limit`) workflow zachowuje
pamięć ostatnich aktywacji przy minimalnym narzucie pamięciowym.

## Integracja z warstwą decyzyjną

Workflow buduje kandydatów `DecisionCandidate` w oparciu o zarejestrowany preset
(oraz deduplikowane metadane). Wersja presetu towarzyszy każdemu kandydatowi,
pozwalając `DecisionOrchestrator` i raportom UI odtwarzać kontekst decyzji oraz
licencje wymagane do ich wykonania.

## Testy regresyjne

Scenariusze referencyjne znajdują się w
`tests/strategies/test_regime_workflow.py` i obejmują m.in. przełączanie presetów,
fallback przy brakach danych, walidację podpisów HMAC oraz raporty historii
aktywacji.
