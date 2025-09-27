# KryptoŁowca Trading Bot

KryptoŁowca is a production-grade cryptocurrency trading bot with a graphical user interface (GUI) built using Tkinter. It supports multi-exchange trading, AI-driven predictions, risk management, and comprehensive reporting. Inspired by Cryptohopper, it offers real-time trading, backtesting, and secure API key management.

> **Important:** Legacy source code that previously lived under `KryptoLowca/bot` has been archived in `archive/legacy_bot/`.
> The archived snapshot is kept for reference only and must not be executed in new environments. Always use the modules from
> the top-level `KryptoLowca` package when developing, deploying, or testing the bot.

## Project Structure

```
KryptoŁowca/
├── core/
│   └── trading_engine.py
├── managers/
│   ├── config_manager.py
│   ├── exchange_manager.py
│   ├── database_manager.py
│   ├── ai_manager.py
│   ├── report_manager.py
│   ├── risk_manager_adapter.py
│   └── security_manager.py
├── tests/
│   ├── test_config_manager.py
│   ├── test_exchange_manager.py
│   ├── test_trading_engine.py
│   └── test_security_manager.py
├── trading_gui.py
├── trading_strategies.py
└── README.md
```

## Features

- **Multi-Exchange Support**: Integrates with exchanges like Binance via `ccxt.asyncio`.
- **AI Predictions**: Supports Random Forest, LSTM, and Gradient Boosting models for trading signals.
- **Risk Management**: Dynamic position sizing and stop-loss/take-profit controls.
- **GUI Dashboard**: Real-time metrics (PnL, positions) and interactive charts with Plotly.
- **Secure Key Storage**: API keys encrypted using Fernet (symmetric encryption).
- **Backtesting**: Comprehensive backtesting with performance metrics (Sharpe, Sortino, etc.).
- **Webhooks**: TradingView webhook integration for external signals.
- **Database**: SQLite/PostgreSQL support for trade and log storage.
- **Reporting**: PDF export of trading results.

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Required libraries (see Installation)

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd KryptoŁowca
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install ccxt pyyaml cryptography boto3 pandas numpy plotly tkinterweb aiosqlite sqlalchemy asyncpg torch joblib
   ```

4. **Set up configuration**:
   - Create a `config.yaml` file in the project root:
     ```yaml
     exchange:
       name: binance
       testnet: true
       api_key: your_api_key
       api_secret: your_api_secret
     db:
       db_url: sqlite+aiosqlite:///trading.db
     ai:
       threshold_bps: 5.0
       seq_len: 40
     trade:
       lookback_bars: 100
     risk:
       max_position_risk: 0.02
     ```
   - Replace `your_api_key` and `your_api_secret` with your exchange API credentials.

5. **Secure API keys**:
   - Run `trading_gui.py` and use the "API Keys" section to save encrypted keys with a password.

## Running the Bot

1. **Launch the GUI**:
   ```bash
   python trading_gui.py
   ```

2. **GUI Usage**:
   - Select trading symbols from the "Select Symbols" frame.
   - Configure presets in the "Configuration" frame.
   - Save/load API keys in the "API Keys" frame.
   - Start trading or run backtests using the "Controls" frame.
   - Train AI models in the "AI Training" frame.
   - Monitor real-time metrics in the "Status" frame.

3. **Webhook Setup** (optional):
   - Enable webhooks in the GUI (port 8080 by default).
   - Configure TradingView alerts to send POST requests to `http://localhost:8080/webhook`.

## Testing

Run unit tests to verify functionality:
```bash
python -m pytest tests/
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure all files are in the correct directories and `sys.path` includes `managers` and `core`.
- **Dependency Issues**: Verify all required libraries are installed (`pip list`).
- **Database Errors**: Check `db_url` in `config.yaml` and ensure the database file (`trading.db`) is accessible.
- **API Key Errors**: Use the GUI to save/load encrypted keys correctly.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

MIT License

## Acknowledgments

Inspired by Cryptohopper's trading platform.