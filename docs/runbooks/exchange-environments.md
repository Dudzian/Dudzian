# Exchange environment switching runbook

This runbook describes how to move a portfolio between paper, testnet and
live exchanges.  The configuration is declarative – the same parameters
are consumed both by the runtime CLI and integration smoke tests.  The
reference YAML lives in `config/environments/exchange_modes.yaml`.

Use the CLI to inspect the available presets:

```bash
python -m bot_core.cli list-environments --environment-config config/environments/exchange_modes.yaml
```

The command marks the default environment declared in the `defaults`
section and prints a summary of the exchange, mode and paper/testnet
flags for each entry.

To inspect the fully merged configuration (including inherited
defaults), run:

```bash
python -m bot_core.cli show-environment --environment live_margin --environment-config config/environments/exchange_modes.yaml
```

The output mirrors the structure of the YAML file, making it easy to
verify watchdog settings, simulator parameters and native adapter
overrides before switching the runtime.

When the YAML file provides `defaults.environment`, the `health-check`
command can consume the configuration without explicitly specifying the
environment name:

```bash
python -m bot_core.cli health-check --environment-config config/environments/exchange_modes.yaml
```

This shortcut is handy for CI smoke tests that should always operate on
the primary paper or staging setup defined in the profile.

Add `--output-format json` when integrating with monitoring pipelines –
the command will emit a machine-readable payload describing the resolved
environment, executed checks and overall status.  Notes about skipped
tests (np. brak poświadczeń dla private API) są również zwracane w
tablicy `notes`, co ułatwia agregację ostrzeżeń.  Użyj wariantu
`--output-format json-pretty`, gdy potrzebujesz czytelnego, wielowierszowego
JSON-u do logów manualnych inspekcji.  Dodaj `--output-path
./artifacts/health.json`, aby zapisać payload na dysku i udostępnić go
kolejnym etapom pipeline'u bez konieczności parsowania stdout.  Sekcja
`paper.simulator` automatycznie zawiera zarówno znane parametry (leverage,
funding), jak i dowolne dodatkowe klucze ustawione flagą
`--paper-simulator-setting` lub w YAML-u.  Dzięki temu monitorujący
pipeline widzi faktyczne wartości przekazane do symulatora – również te,
które pojawią się w nowych wersjach backendu.

Możesz też jawnie wskazać ticker używany przez test publiczny:

```bash
python -m bot_core.cli health-check --public-symbol BTC/EUR --environment-config config/environments/exchange_modes.yaml
```

Parametr `--public-symbol` ma najwyższy priorytet – nadpisuje zarówno
konfigurację środowiska YAML, jak i profil TOML.

Jeżeli potrzebujesz przetestować tylko wybrane komponenty, użyj flagi
`--check`.  Argument można podawać wielokrotnie lub przekazać listę
rozdzieloną przecinkami.  Poniższe wywołanie uruchomi wyłącznie test
prywatnego API i zwróci w JSON-ie listę `requested_checks` opisującą
zadane filtry.  Nazwy testów nie są czułe na wielkość liter, więc
`--check PUBLIC_API` zadziała tak samo jak `--check public_api`.

```bash
python -m bot_core.cli health-check --check private_api --output-format json --environment-config config/environments/exchange_modes.yaml
```

Użyj `--list-checks`, aby wyświetlić dostępne testy health-check zanim
zdefiniujesz filtry CLI lub profile YAML:

```bash
python -m bot_core.cli health-check --list-checks
```

Analogicznie można wymusić weryfikację salda konkretnej waluty oraz
minimalnego depozytu na koncie prywatnym.  Zdefiniuj pola
`health_check.private_asset` oraz `health_check.private_min_balance` w
profilu YAML, aby health-check uznał brak waluty lub zbyt niski balans
za błąd.  Parametry można też nadpisać z CLI:

```bash
python -m bot_core.cli health-check --private-asset USDT --private-min-balance 250 --environment-config config/environments/exchange_modes.yaml
```

Waluty są porównywane w sposób nieczuły na wielkość liter, a minimalny
balans może być liczbą całkowitą lub zmiennoprzecinkową.  Wyniki w
formacie JSON zawierają pola `private_asset` oraz `private_min_balance`,
co ułatwia walidację w pipeline'ach monitorujących.

Symulator paper można dodatkowo dostosować z poziomu CLI.  Wariant
(`spot`, `margin`, `futures`) ustawisz flagą `--paper-variant`, a początkowy
kapitał oraz walutę gotówkową odpowiednio przez `--paper-initial-cash` i
`--paper-cash-asset`.  Te przełączniki nadpisują wartości z profilu YAML,
co pozwala szybko przetestować alternatywne konfiguracje bez modyfikacji
pliku.  Jeżeli chcesz zachować konfigurację kwoty, a jedynie zmienić
walutę, pomiń `--paper-initial-cash` – CLI wykorzysta bieżącą wartość z
menedżera.  Stawkę prowizji symulatora (`fee_rate`) ustawisz flagą
`--paper-fee-rate`, natomiast limity dźwigni, maintenance margin i funding
nadpiszesz za pomocą `--paper-leverage-limit`, `--paper-maintenance-margin`
oraz `--paper-funding-rate`.  Odstęp między naliczeniami funding możesz
kontrolować flagą `--paper-funding-interval` (czas w sekundach).  Wszystkie
te parametry są również odwzorowane w wynikach JSON pod kluczem `paper`,
gdzie sekcja `simulator` prezentuje bieżące wartości konfiguracyjne,
dzięki czemu pipeline’y monitorujące widzą faktycznie użyte ustawienia.
  Jeżeli symulator otrzyma nowe parametry w przyszłych wersjach, możesz je
  natychmiast nadpisać flagą `--paper-simulator-setting klucz=wartość` –
  argument akceptuje wiele powtórzeń i konwertuje wartości na liczby
  zmiennoprzecinkowe, więc `--paper-simulator-setting maintenance_margin_ratio=0.2`
  zadziała bez aktualizacji CLI.  Wartości pozostają widoczne zarówno w
  wynikach JSON, jak i w `manager.get_paper_simulator_settings()`, więc
  kontraktowe testy konfiguracji obejmują również nowe klucze.

Watchdog można stroić bezpośrednio z CLI, gdy środowisko wymaga innych
parametrów niż te zapisane w YAML-u.  Użyj flag `--watchdog-max-attempts`,
`--watchdog-base-delay`, `--watchdog-max-delay` oraz `--watchdog-jitter-min`
i `--watchdog-jitter-max`, aby nadpisać politykę retry.  Parametry
wyłącznika (`failure_threshold`, `recovery_timeout`, `half_open_success_threshold`)
ustawisz odpowiednio przez `--watchdog-failure-threshold`,
`--watchdog-recovery-timeout` i `--watchdog-half-open-success`.  Dzięki temu
możesz chwilowo poluzować lub zaostrzyć strażnika (np. na sandboxie z
większym throttlingiem) bez edycji profili.  Od teraz możesz także
precyzyjnie wskazać, które wyjątki kwalifikują się do ponawiania: dodaj
w YAML-u listę `watchdog.retry_exceptions` (pełne ścieżki modułowe
np. `builtins.TimeoutError`), a z CLI użyj flagi
`--watchdog-retry-exception`, powtarzając ją dla kolejnych klas.  Nazwy
wyjątków pojawiają się w JSON-owym wyniku health-checka, więc pipeline
monitorujący widzi realnie użyte ustawienia watchdog-a.

Natywne adaptery margin/futures również można dopasować do testów bez
modyfikowania YAML-a.  Wskaż tryb adaptera flagą `--native-mode` (do
wyboru `margin` lub `futures`) i nadpisz konkretne parametry wielokrotnym
użyciem `--native-setting` w formacie `klucz=wartość`, np.:

```bash
python -m bot_core.cli health-check \
  --environment live_margin \
  --environment-config config/environments/exchange_modes.yaml \
  --native-mode futures \
  --native-setting margin_mode=isolated \
  --native-setting max_leverage=8
```

Wartości są automatycznie konwertowane na liczby, boole oraz `null`, więc
`--native-setting hedge_mode=true` ustawi przełącznik logiczny.  Wynik
JSON zawiera sekcję `native_adapter` z raportem trybu oraz kompletnych
ustawień przekazanych do `ExchangeManager`, co ułatwia audyt konfiguracji
w pipeline'ach CI.

## Paper simulators

1. **Spot paper** – baseline behaviour for strategies that do not rely on
   leverage.  Enable it by setting:

   ```yaml
   exchange_manager:
     mode: paper
     paper_variant: spot
   ```

2. **Margin paper** – mirrors cross/isolated margin by tracking leverage,
   funding and liquidations.  Configure leverage limits and maintenance
   margin in the `simulator` section:

   ```yaml
   exchange_manager:
     mode: paper
     paper_variant: margin
     simulator:
       leverage_limit: 5.0
       maintenance_margin_ratio: 0.12
       funding_rate: 0.00005
       funding_interval_seconds: 28800
   ```

3. **Futures paper** – extends the margin simulator with futures specific
   funding.  Use the `paper_variant: futures` flag and adjust the
   parameters when matching production risk settings.

Paper simulators emit the same account snapshot fields as native
adapters.  Funding payments and leverage changes are logged to the
telemetry bus so that dashboards can compare paper vs. live runs.

## Testnet (exchange sandbox)

* Set `mode: margin` with `testnet: true`.  The manager instantiates CCXT
  backends by default; for exchanges with native adapters the registry
  declares whether the testnet is available.  If a given exchange does
  not support it, the manager raises a configuration error during
  startup.
* Always provide dedicated API credentials.  Secrets can be passed via
  environment variables referenced in the YAML file.
* Watchdog policies should be more relaxed than in production to account
  for rate limiting on sandboxes.

## Live trading

* Set `mode: margin`/`mode: futures` without `testnet`.  The manager uses
  the native adapter registry to instantiate the correct implementation.
* Configure watchdogs and circuit breakers.  The sample YAML sets a
  three-attempt retry and a one minute recovery window.
* Provide exchange specific settings (e.g. Binance margin type) under the
  `native_adapter.settings` key.
* Telemetry: margin decisions are logged with the
  `order_close_for_reversal` and `margin_event` event types.

## Verification flow

1. Update the YAML profile with the desired environment.
2. Run smoke tests: `pytest tests/smoke -k exchange`.
3. For live/testnet make sure the account snapshot reports positive
   equity and available margin before enabling strategies.

This workflow ensures that multi-strategy runtimes experience identical
behaviour across paper and live deployments, enabling confident rollouts
of leverage-sensitive strategies.
