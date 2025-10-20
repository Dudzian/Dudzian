"""Warstwa prezentacji Trading GUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import tkinter as tk
from tkinter import ttk

from .state import AppState

if TYPE_CHECKING:  # pragma: no cover - tylko wskazówki typów
    from .controller import TradingSessionController


logger = logging.getLogger(__name__)


class TradingView:
    """Warstwa prezentacji GUI."""

    def __init__(
        self,
        root: tk.Tk,
        state: AppState,
        controller: "TradingSessionController",
        *,
        on_refresh_risk: Optional[Callable[[], None]] = None,
    ):
        self.root = root
        self.state = state
        self.controller = controller
        self._on_refresh_risk = on_refresh_risk
        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.root.title("Trading Bot — AI integrated")
        self.root.geometry("1280x820")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            logger.debug("Motyw clam niedostępny")
        style.configure("LicenseSummary.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("LicenseNotice.TLabel", foreground="#c0392b")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        license_bar = ttk.Frame(main)
        license_bar.pack(fill="x", pady=(0, 8))
        summary_var = self.state.license_summary or tk.StringVar(value="Licencja: —")
        notice_var = self.state.license_notice or tk.StringVar(value="")
        self.state.license_summary = summary_var
        self.state.license_notice = notice_var
        ttk.Label(
            license_bar,
            textvariable=summary_var,
            style="LicenseSummary.TLabel",
        ).pack(side="left")
        self._license_notice_label = ttk.Label(
            license_bar,
            textvariable=notice_var,
            style="LicenseNotice.TLabel",
            wraplength=520,
            justify="right",
        )
        self._license_notice_label.pack(side="right")

        header = ttk.Frame(main)
        header.pack(fill="x")
        ttk.Label(header, text="Network").pack(side="left")
        self.network_combobox = ttk.Combobox(
            header,
            textvariable=self.state.network,
            values=("Testnet", "Live"),
            width=10,
            state="readonly",
        )
        self.network_combobox.pack(side="left", padx=(4, 12))
        ttk.Label(header, text="Mode").pack(side="left")
        self.mode_combobox = ttk.Combobox(
            header,
            textvariable=self.state.mode,
            values=("Spot", "Futures"),
            width=10,
            state="readonly",
        )
        self.mode_combobox.pack(side="left", padx=(4, 12))
        ttk.Label(header, text="Timeframe").pack(side="left")
        ttk.Combobox(
            header,
            textvariable=self.state.timeframe,
            values=("1m", "3m", "5m", "15m", "1h", "4h", "1d"),
            width=10,
            state="readonly",
        ).pack(side="left", padx=(4, 12))
        ttk.Label(header, text="Fraction").pack(side="left")
        self._fraction_spinbox = ttk.Spinbox(
            header,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.state.fraction,
            width=8,
        )
        self._fraction_spinbox.pack(side="left", padx=(4, 12))

        buttons = ttk.Frame(header)
        buttons.pack(side="right")
        self.start_button = ttk.Button(buttons, text="Start", command=self._start_clicked)
        self.start_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Stop", command=self._stop_clicked).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="Odśwież profil ryzyka",
            command=self._refresh_risk_clicked,
        ).pack(side="left")

        risk_info = ttk.Frame(main)
        risk_info.pack(fill="x", pady=(8, 0))
        profile_var = self.state.risk_profile_label or tk.StringVar(value="Profil ryzyka: —")
        limits_var = self.state.risk_limits_label or tk.StringVar(value="Limity ryzyka: —")
        notional_var = self.state.risk_notional_label or tk.StringVar(value="Domyślna kwota: —")
        self.state.risk_profile_label = profile_var
        self.state.risk_limits_label = limits_var
        self.state.risk_notional_label = notional_var
        ttk.Label(risk_info, textvariable=profile_var).pack(side="left")
        ttk.Label(risk_info, textvariable=limits_var).pack(side="left", padx=(12, 0))
        ttk.Label(risk_info, textvariable=notional_var).pack(side="left", padx=(12, 0))

        balances = ttk.Frame(main)
        balances.pack(fill="x", pady=12)
        ttk.Label(balances, text="Account balance:").pack(side="left")
        ttk.Label(balances, textvariable=self.state.account_balance).pack(side="left", padx=(4, 12))
        ttk.Label(balances, text="Paper balance:").pack(side="left")
        ttk.Label(balances, textvariable=self.state.paper_balance).pack(side="left", padx=4)

        content = ttk.Frame(main)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=2)

        self.positions_tree = ttk.Treeview(
            content,
            columns=("symbol", "size", "entry", "pnl"),
            show="headings",
            height=12,
        )
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry", text="Entry")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        scrollbar = ttk.Scrollbar(content, orient="vertical", command=self.positions_tree.yview)
        scrollbar.grid(row=0, column=0, sticky="nse")
        self.positions_tree.configure(yscrollcommand=scrollbar.set)

        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.log_text = tk.Text(right, height=20, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        status_bar = ttk.Frame(main)
        status_bar.pack(fill="x", pady=(12, 0))
        ttk.Label(status_bar, textvariable=self.state.status).pack(side="left")

    # ------------------------------------------------------------------
    def sync_positions(self) -> None:
        self.positions_tree.delete(*self.positions_tree.get_children())
        for symbol, position in self.state.open_positions.items():
            size = position.get("size", "?")
            entry = position.get("entry_price", "?")
            pnl = position.get("unrealized_pnl", "?")
            self.positions_tree.insert("", "end", values=(symbol, size, entry, pnl))

    # ------------------------------------------------------------------
    def append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    # ------------------------------------------------------------------
    def _start_clicked(self) -> None:
        self.controller.start()
        self.append_log("[INFO] Start trading session")

    # ------------------------------------------------------------------
    def _stop_clicked(self) -> None:
        self.controller.stop()
        self.append_log("[INFO] Stop trading session")

    # ------------------------------------------------------------------
    def _refresh_risk_clicked(self) -> None:
        if callable(self._on_refresh_risk):
            try:
                self._on_refresh_risk()
            except Exception:
                logger.exception("Nie udało się odświeżyć profilu ryzyka")

    # ------------------------------------------------------------------
    def configure_fraction_input(
        self,
        *,
        maximum: float,
        increment: float,
    ) -> None:
        spinbox = getattr(self, "_fraction_spinbox", None)
        if spinbox is None:
            return
        try:
            spinbox.config(from_=0.0, to=float(maximum), increment=float(increment))
        except Exception:  # pragma: no cover - defensywne
            logger.debug("Nie udało się zaktualizować konfiguracji spinboxa frakcji", exc_info=True)

    # ------------------------------------------------------------------
    def get_fraction_limits(self) -> tuple[float, float, float]:
        spinbox = getattr(self, "_fraction_spinbox", None)
        if spinbox is None:
            return (0.0, 1.0, 0.01)

        def _extract(option: str, fallback: float) -> float:
            try:
                value = spinbox.cget(option)
            except Exception:
                return fallback
            try:
                return float(value)
            except Exception:
                return fallback

        minimum = _extract("from", 0.0)
        maximum = _extract("to", 1.0)
        increment = _extract("increment", 0.01)
        return (minimum, maximum, increment)

    # ------------------------------------------------------------------
    def configure_network_options(self, *, live_enabled: bool) -> None:
        combo = getattr(self, "network_combobox", None)
        if combo is None:
            return
        values = ["Testnet"]
        if live_enabled:
            values.append("Live")
        try:
            combo.configure(values=tuple(values))
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się zaktualizować listy sieci", exc_info=True)
        if not live_enabled:
            try:
                self.state.network.set("Testnet")
            except Exception:  # pragma: no cover - defensywne
                logger.debug("Nie udało się wymusić trybu Testnet", exc_info=True)

    # ------------------------------------------------------------------
    def configure_mode_options(self, *, futures_enabled: bool) -> None:
        combo = getattr(self, "mode_combobox", None)
        if combo is None:
            return
        values = ["Spot"]
        if futures_enabled:
            values.append("Futures")
        try:
            combo.configure(values=tuple(values))
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się zaktualizować listy trybów", exc_info=True)
        if not futures_enabled:
            try:
                self.state.mode.set("Spot")
            except Exception:  # pragma: no cover - defensywne
                logger.debug("Nie udało się wymusić trybu Spot", exc_info=True)

    # ------------------------------------------------------------------
    def set_start_enabled(self, enabled: bool) -> None:
        button = getattr(self, "start_button", None)
        if button is None:
            return
        try:
            if enabled:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Fallback zmiany stanu przycisku Start", exc_info=True)
            button.configure(state="normal" if enabled else "disabled")


__all__ = ["TradingView"]
