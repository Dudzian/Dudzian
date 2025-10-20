"""Legacy wrapper for launching the paper AutoTrader application."""

from __future__ import annotations

from pathlib import Path
import logging
import sys
from typing import Optional


def _ensure_repo_root() -> None:
    """Ensure the repository root is present on ``sys.path`` when executed as a script."""

    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


if __package__ in (None, ""):
    _ensure_repo_root()


from KryptoLowca.auto_trader.paper import (  # noqa: E402
    DEFAULT_PAPER_BALANCE,
    DEFAULT_SYMBOL,
    HeadlessTradingStub,
    PaperAutoTradeApp as _ModernPaperAutoTradeApp,
    PaperAutoTradeOptions,
    main as modern_cli_main,
    parse_cli_args,
)

try:  # pragma: no cover - środowiska bez legacy modułu
    from KryptoLowca.paper_auto_trade_app import PaperAutoTradeApp as LegacyPaperAutoTradeApp
except ImportError:  # pragma: no cover - fallback gdy brak starej implementacji
    LegacyPaperAutoTradeApp = None  # type: ignore[assignment]


PaperAutoTradeApp = _ModernPaperAutoTradeApp

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PAPER_BALANCE",
    "DEFAULT_SYMBOL",
    "HeadlessTradingStub",
    "LegacyPaperAutoTradeApp",
    "PaperAutoTradeApp",
    "PaperAutoTradeOptions",
    "modern_cli_main",
    "parse_cli_args",
    "main",
]


def main(
    use_dummy_feed: bool = True,
    enable_gui: bool = True,
    *,
    core_config_path: str | None = None,
    core_environment: str | None = None,
    symbol: str | None = None,
    paper_balance: float | None = None,
    risk_profile: str | None = None,
) -> None:
    """Backward-compatible launcher delegating to :mod:`KryptoLowca.auto_trader.paper`."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = PaperAutoTradeApp(
        symbol=symbol or DEFAULT_SYMBOL,
        enable_gui=enable_gui,
        use_dummy_feed=use_dummy_feed,
        paper_balance=paper_balance if paper_balance is not None else DEFAULT_PAPER_BALANCE,
        core_config_path=core_config_path,
        core_environment=core_environment,
        risk_profile=risk_profile,
    )
    app.run()


if __name__ == "__main__":  # pragma: no cover - ścieżka CLI
    use_dummy = True
    enable_gui = True
    core_arg: str | None = None
    env_arg: str | None = None
    symbol_arg: str | None = None
    balance_arg: Optional[float] = None
    risk_arg: str | None = None

    for raw in sys.argv[1:]:
        token = raw.strip()
        lowered = token.lower()
        if lowered == "nogui":
            enable_gui = False
            continue
        if lowered.startswith("real"):
            use_dummy = False
            continue
        if lowered.startswith("core="):
            core_arg = raw.split("=", 1)[1] or None
            continue
        if lowered.startswith("env="):
            env_arg = raw.split("=", 1)[1] or None
            continue
        if lowered.startswith("symbol="):
            symbol_arg = raw.split("=", 1)[1] or None
            continue
        if lowered.startswith("balance="):
            value = raw.split("=", 1)[1]
            try:
                balance_arg = float(value)
            except (TypeError, ValueError):
                log.warning("Niepoprawna wartość balance=%s – używam domyślnej", value)
            continue
        if lowered.startswith("risk=") or lowered.startswith("profile="):
            risk_arg = raw.split("=", 1)[1] or None
            continue

    main(
        use_dummy_feed=use_dummy,
        enable_gui=enable_gui,
        core_config_path=core_arg,
        core_environment=env_arg,
        symbol=symbol_arg,
        paper_balance=balance_arg,
        risk_profile=risk_arg,
    )
