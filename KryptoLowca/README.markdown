# KryptoŁowca Trading Bot *(legacy namespace)*

KryptoŁowca is a production-grade cryptocurrency trading stack built around a
modular core (`bot_core`) and user-facing launchers that historically lived in
the `KryptoLowca` package. **Cały pakiet `KryptoLowca` jest obecnie traktowany
jako warstwa zgodności wstecznej.** Nowe funkcje i integracje należy budować w
`bot_core.*`, natomiast moduły takie jak `KryptoLowca.ai_models` pozostały tylko
w formie shimów kierujących do `bot_core.ai.legacy_models`.

> **Important:** Legacy source code, które dawniej znajdowało się pod
> `KryptoLowca/bot`, zostało usunięte. Współczesny rozwój musi koncentrować się
> na pakietach opisanych poniżej.

## Architecture Overview

```
bot_core/
├── runtime/
│   ├── metadata.py      # runtime entrypoint + risk profile loaders
│   ├── paper.py         # adapters for paper trading
│   └── ...
├── risk/               # shared risk engines and settings models
└── ...

KryptoLowca/
├── auto_trader/        # headless AutoTrader runtime
│   ├── app.py          # production launcher (integrates risk + runtime metadata)
│   └── paper.py        # paper/sandbox launcher with CLI controls
├── ui/trading/         # modular Tkinter GUI (view/controller/state split)
│   ├── app.py          # TradingGUI facade
│   └── view.py, controller.py, state.py, ...
├── paper_auto_trade_app.py  # compatibility wrapper used by legacy entrypoints
├── run_autotrade_paper.py   # thin launcher delegating to auto_trader.paper
├── run_trading_gui_paper.py # thin launcher delegating to ui.trading.app
└── ...
```

Key characteristics:

- **Shared runtime metadata** – risk profiles, environment presets and
  entrypoint configuration are defined in `config/core.yaml` and accessed
  through `bot_core.runtime.metadata`.
- **Modular Trading GUI** – `bot_core.ui.trading` renders runtime driven
  banners, fraction controls and default notionals while exposing an API for
  AutoTrader integration.
- **Launcher segregation** – AutoTrader, paper trading and GUI launchers each
  wrap the shared runtime, making it possible to run headless services or the
  desktop interface independently.

## Installation

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows PowerShell
   ```

2. **Install dependencies**

   ```bash
   pip install -e .[full]
   ```

   The editable install exposes both `bot_core` and `KryptoLowca` packages.

3. **Provide runtime configuration**

   - Copy `config/core.example.yaml` to `config/core.yaml` (or point the
     `BOT_CORE_CONFIG` environment variable to another location).
   - Define environments, exchanges and risk profiles. Each launcher uses
     `load_risk_manager_settings` to resolve the active profile.

## Running the Components

### Trading GUI

Launch the modular GUI with paper trading defaults:

```bash
python -m KryptoLowca.run_trading_gui_paper
```

The GUI reads runtime metadata on startup, displays the active risk profile in
its banner and offers a "Reload core.yaml" button wired to
`load_risk_manager_settings`.

### AutoTrader

Start the headless AutoTrader with the shared runtime services:

```bash
python -m KryptoLowca.run_autotrade_paper
```

`bot_core.auto_trader.app.AutoTrader` jest kanoniczną implementacją kontrolera
autotradingu. Warstwa `KryptoLowca.auto_trader` utrzymuje jedynie cienki shim
delegujący do modułu `bot_core` (wraz z przekierowaniem alertów), dzięki czemu
legacy importy nadal działają, a logika runtime jest utrzymywana w jednym
miejscu. AutoTrader konsumuje metadane ryzyka, reaguje na przeładowania (GUI,
`SIGHUP`, watcher pliku) i orkiestruje pętlę auto-tradingu oraz telemetrię
Prometheusa.

Emitowane sygnały (`EventType.SIGNAL`) dołączają teraz blok
`metadata.regime_summary` z wynikami wygładzonej oceny reżimu rynku. Analiza
kalibracyjna może odczytać m.in.:

- `metadata.regime_summary.risk_score` – znormalizowany wynik ryzyka (0-1)
  będący podstawą blokad guardrail oraz progów kapitałowych,
- `metadata.regime_summary.risk_level` – etykietę poziomu ryzyka, którą można
  mapować na profile zarządzania kapitałem i polityki hedge.

Dzięki temu analitycy kalibracji mogą bezpośrednio z logów sygnałów odczytać
bieżący stan reżimu oraz porównać go z konfiguracją progów i wag strategii,
bez dodatkowych zapytań do silnika historii reżimów.

## Risk Management Runtime

- Risk profiles are expressed in `config/core.yaml` under `risk_profiles`.
- Each profile is converted into `RiskManagerSettings` by
  `bot_core.runtime.metadata.load_risk_manager_settings`.
- Both the GUI and AutoTrader expose background watchers that reload the active
  profile when `core.yaml` changes, keeping desktop and headless launchers in
  sync.

## Testing

Run the automated test suite:

```bash
pytest
```

Some integration tests expect access to mock CCXT exchanges; set the
appropriate environment variables or skip them using `-m "not ccxt_live"`.

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/x`.
2. Ensure all new modules integrate with the runtime metadata and risk loaders.
3. Add or update tests and documentation where relevant.
4. Submit a pull request with a summary of the change and manual testing notes.

## License

MIT License
