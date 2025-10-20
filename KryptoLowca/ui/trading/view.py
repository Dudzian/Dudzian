"""Warstwa prezentacji Trading GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import tkinter as tk
from tkinter import filedialog, ttk

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

        intel_var = getattr(self.state, "market_intel_label", None)
        summary_value = getattr(self.state, "market_intel_summary", None)
        if not isinstance(summary_value, str) or not summary_value:
            summary_value = "Market intel: —"
        if isinstance(intel_var, tk.Variable):
            setter = getattr(intel_var, "set", None)
            if callable(setter):
                try:
                    setter(summary_value)
                except Exception:  # pragma: no cover - defensywne
                    logger.debug("Nie udało się zsynchronizować etykiety market intel", exc_info=True)
        else:
            intel_var = tk.StringVar(value=summary_value)
            self.state.market_intel_label = intel_var

        history_var = getattr(self.state, "market_intel_history_label", None)
        history_value = getattr(self.state, "market_intel_history_display", "Brak historii market intel")
        if isinstance(history_var, tk.Variable):
            setter = getattr(history_var, "set", None)
            if callable(setter):
                try:
                    setter(history_value)
                except Exception:  # pragma: no cover - defensywne
                    logger.debug("Nie udało się zsynchronizować historii market intel", exc_info=True)
        else:
            history_var = tk.StringVar(value=history_value)
            self.state.market_intel_history_label = history_var

        intel_frame = ttk.Frame(main)
        intel_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(intel_frame, textvariable=intel_var, justify="left").pack(anchor="w")
        ttk.Label(intel_frame, text="Historia market intel:").pack(anchor="w", pady=(4, 0))
        ttk.Label(intel_frame, textvariable=history_var, justify="left").pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Wyczyść historię",
            command=self._clear_history_clicked,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(
            intel_frame,
            text="Kopiuj historię",
            command=self._copy_history_clicked,
        ).pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Zapisz historię",
            command=self._save_history_clicked,
        ).pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Wczytaj historię",
            command=self._load_history_clicked,
        ).pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Otwórz plik historii",
            command=self._open_history_clicked,
        ).pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Ustaw plik historii…",
            command=self._choose_history_destination_clicked,
        ).pack(anchor="w")
        ttk.Button(
            intel_frame,
            text="Przywróć domyślny plik",
            command=self._reset_history_destination_clicked,
        ).pack(anchor="w")
        destination_var = getattr(self.state, "market_intel_history_path_label", None)
        destination_value = getattr(
            self.state,
            "market_intel_history_destination_display",
            "Plik historii: domyślny",
        )
        if isinstance(destination_var, tk.Variable):
            setter = getattr(destination_var, "set", None)
            if callable(setter):
                try:
                    setter(destination_value)
                except Exception:  # pragma: no cover - defensywne
                    logger.debug(
                        "Nie udało się zsynchronizować etykiety pliku historii",
                        exc_info=True,
                    )
        else:
            destination_var = tk.StringVar(value=destination_value)
            self.state.market_intel_history_path_label = destination_var
        ttk.Label(
            intel_frame,
            textvariable=destination_var,
            justify="left",
        ).pack(anchor="w")
        auto_save_var = getattr(self.state, "market_intel_auto_save", None)
        if isinstance(auto_save_var, tk.Variable):
            pass
        else:
            auto_save_var = tk.BooleanVar(value=bool(auto_save_var))
            self.state.market_intel_auto_save = auto_save_var
        ttk.Checkbutton(
            intel_frame,
            text="Auto-zapis historii",
            variable=auto_save_var,
            command=self._auto_save_toggled,
        ).pack(anchor="w", pady=(4, 0))

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
    def _clear_history_clicked(self) -> None:
        try:
            self.controller.clear_market_intel_history()
        except Exception:  # pragma: no cover - defensywnie logujemy
            logger.exception("Nie udało się wyczyścić historii market intel z GUI")

    # ------------------------------------------------------------------
    def _copy_history_clicked(self) -> None:
        try:
            history_text = self.controller.get_market_intel_history_text()
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się pobrać historii market intel")
            return

        try:
            clipboard_clear = getattr(self.root, "clipboard_clear", None)
            clipboard_append = getattr(self.root, "clipboard_append", None)
            if callable(clipboard_clear):
                clipboard_clear()
            if callable(clipboard_append):
                clipboard_append(history_text)
            status = getattr(self.state, "status", None)
            setter = getattr(status, "set", None)
            if callable(setter):
                setter("Skopiowano historię market intel do schowka")
            self.append_log("[INFO] Skopiowano historię market intel do schowka")
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się skopiować historii market intel do schowka")

    # ------------------------------------------------------------------
    def _save_history_clicked(self) -> None:
        try:
            path = self.controller.export_market_intel_history()
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się zapisać historii market intel")
            status = getattr(self.state, "status", None)
            setter = getattr(status, "set", None)
            if callable(setter):
                setter("Nie udało się zapisać historii market intel")
            return

        self.append_log(f"[INFO] Zapisano historię market intel do {path}")

    # ------------------------------------------------------------------
    def _load_history_clicked(self) -> None:
        try:
            entries = self.controller.load_market_intel_history()
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się wczytać historii market intel")
            status = getattr(self.state, "status", None)
            setter = getattr(status, "set", None)
            if callable(setter):
                setter("Nie udało się wczytać historii market intel")
            return

        self.append_log(
            f"[INFO] Wczytano {len(entries)} wpisów historii market intel"
        )

    # ------------------------------------------------------------------
    def _open_history_clicked(self) -> None:
        try:
            path = self.controller.reveal_market_intel_history()
        except FileNotFoundError:
            logger.info("Brak zapisanej historii market intel do otwarcia")
            self.append_log("[WARN] Brak zapisanej historii market intel")
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się otworzyć pliku historii market intel")
            self.append_log("[ERROR] Nie udało się otworzyć pliku historii market intel")
        else:
            self.append_log(f"[INFO] Otwarto plik historii market intel: {path}")

    # ------------------------------------------------------------------
    def _choose_history_destination_clicked(self) -> None:
        try:
            initial_path = self.controller.get_market_intel_history_destination()
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się pobrać bieżącej ścieżki historii market intel")
            initial_path = None

        initialdir: Optional[str] = None
        initialfile: Optional[str] = None
        if isinstance(initial_path, Path):
            try:
                parent = initial_path.parent
                if parent.exists():
                    initialdir = str(parent)
                initialfile = initial_path.name
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się przygotować parametrów dialogu zapisu", exc_info=True)

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".txt",
            filetypes=(("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")),
            initialdir=initialdir,
            initialfile=initialfile,
        )
        if not filename:
            return

        try:
            path = self.controller.set_market_intel_history_destination(filename)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się ustawić nowego pliku historii market intel")
            status = getattr(self.state, "status", None)
            setter = getattr(status, "set", None)
            if callable(setter):
                setter("Nie udało się ustawić pliku historii market intel")
            return

        display_path = path if path is not None else filename
        self.append_log(f"[INFO] Ustawiono plik historii market intel: {display_path}")

    # ------------------------------------------------------------------
    def _reset_history_destination_clicked(self) -> None:
        try:
            self.controller.set_market_intel_history_destination(None)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się przywrócić domyślnego pliku historii market intel")
            status = getattr(self.state, "status", None)
            setter = getattr(status, "set", None)
            if callable(setter):
                setter("Nie udało się przywrócić domyślnego pliku historii market intel")
            return

        self.append_log("[INFO] Przywrócono domyślny plik historii market intel")

    # ------------------------------------------------------------------
    def _auto_save_toggled(self) -> None:
        var = getattr(self.state, "market_intel_auto_save", None)
        if hasattr(var, "get"):
            try:
                enabled = bool(var.get())
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się odczytać market_intel_auto_save", exc_info=True)
                enabled = False
        else:
            enabled = bool(var)

        try:
            self.controller.set_market_intel_auto_save(enabled)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się zmienić trybu auto-zapisu historii market intel")
            return

        message = (
            "Auto-zapis historii market intel włączony"
            if enabled
            else "Auto-zapis historii market intel wyłączony"
        )
        self.append_log(f"[INFO] {message}")

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
