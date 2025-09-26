"""Desktopowy dashboard bota handlowego inspirowany rozwiązaniami komercyjnymi."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - środowisko testowe może nie udostępniać Tk
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - fallback headless
    tk = None  # type: ignore
    ttk = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _can_use_tk() -> bool:
    if tk is None:
        return False
    if os.name == "nt":  # Windows posiada domyślnie środowisko GUI
        return True
    display = os.environ.get("DISPLAY")
    return bool(display)


@dataclass(slots=True)
class DashboardState:
    strategy: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


class DashboardApp:
    """Warstwa prezentacji pulpitu – obsługuje tryb graficzny i headless."""

    def __init__(
        self,
        *,
        config_manager: Any,
        ai_manager: Any,
        exchange_manager: Any,
        risk_manager: Any,
        master: Optional[tk.Misc] = None,
        headless: bool = False,
        log_path: Optional[Path] = None,
    ) -> None:
        self.config_manager = config_manager
        self.ai_manager = ai_manager
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self.headless = headless or not _can_use_tk()
        self.log_path = log_path
        self.state = DashboardState()

        self._tree_strategy: Optional[ttk.Treeview] = None
        self._tree_metrics: Optional[ttk.Treeview] = None
        self._text_logs: Optional[tk.Text] = None
        self._text_alerts: Optional[tk.Text] = None

        if not self.headless:
            self.root = master or tk.Tk()
            self.root.title("KryptoLowca – Dashboard")
            self._build_layout()
        else:
            self.root = None

    # ------------------------------------------------------------------ GUI
    def _build_layout(self) -> None:
        assert not self.headless and tk is not None and ttk is not None

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        frame_strategy = ttk.Frame(notebook)
        frame_metrics = ttk.Frame(notebook)
        frame_logs = ttk.Frame(notebook)
        frame_alerts = ttk.Frame(notebook)

        notebook.add(frame_strategy, text="Strategia")
        notebook.add(frame_metrics, text="Metryki")
        notebook.add(frame_logs, text="Logi")
        notebook.add(frame_alerts, text="Alerty")

        self._tree_strategy = ttk.Treeview(frame_strategy, columns=("value",), show="tree headings")
        self._tree_strategy.heading("#0", text="Parametr")
        self._tree_strategy.heading("value", text="Wartość")
        self._tree_strategy.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._tree_metrics = ttk.Treeview(frame_metrics, columns=("value",), show="tree headings")
        self._tree_metrics.heading("#0", text="Metryka")
        self._tree_metrics.heading("value", text="Wartość")
        self._tree_metrics.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._text_logs = tk.Text(frame_logs, height=20)
        self._text_logs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._text_alerts = tk.Text(frame_alerts, height=20)
        self._text_alerts.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ----------------------------------------------------------------- Update
    def refresh_strategy(self) -> None:
        try:
            strategy = self.config_manager.load_strategy_config()
            if hasattr(strategy, "__dataclass_fields__"):
                data = asdict(strategy)
            elif hasattr(strategy, "__dict__"):
                data = {k: getattr(strategy, k) for k in vars(strategy)}
            else:
                data = dict(strategy)
            self.state.strategy = data
        except Exception as exc:
            logger.error("Nie udało się pobrać konfiguracji strategii: %s", exc)
            return

        if self._tree_strategy is not None:
            self._tree_strategy.delete(*self._tree_strategy.get_children())
            for key, value in sorted(self.state.strategy.items()):
                self._tree_strategy.insert("", tk.END, text=key, values=(value,))

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        self.state.metrics = metrics
        if self._tree_metrics is not None:
            self._tree_metrics.delete(*self._tree_metrics.get_children())
            for key, value in sorted(metrics.items()):
                if isinstance(value, dict):
                    parent = self._tree_metrics.insert("", tk.END, text=key, values=("",))
                    for sub_key, sub_value in sorted(value.items()):
                        self._tree_metrics.insert(parent, tk.END, text=sub_key, values=(sub_value,))
                else:
                    self._tree_metrics.insert("", tk.END, text=key, values=(value,))

    def append_log(self, line: str) -> None:
        timestamped = f"{datetime.now(timezone.utc).isoformat()} | {line.strip()}"
        self.state.logs.append(timestamped)
        if self._text_logs is not None:
            self._text_logs.insert(tk.END, timestamped + "\n")
            self._text_logs.see(tk.END)

    def append_alert(self, alert: Dict[str, Any]) -> None:
        alert_entry = dict(alert)
        alert_entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.state.alerts.append(alert_entry)
        if self._text_alerts is not None:
            line = f"[{alert_entry.get('timestamp')}] {alert_entry.get('severity', 'INFO')}: {alert_entry.get('message')}"
            self._text_alerts.insert(tk.END, line + "\n")
            self._text_alerts.see(tk.END)

    def tail_logs(self, max_lines: int = 200) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()[-max_lines:]
        except Exception as exc:
            logger.error("Nie udało się odczytać logów: %s", exc)
            return
        for line in lines:
            self.append_log(line)

    def mainloop(self) -> None:
        if self.headless or self.root is None:
            logger.info("Dashboard działa w trybie headless – brak pętli GUI")
            return
        self.root.mainloop()


class DashboardController:
    """Łączy warstwę GUI z menedżerami i zapewnia automatyczny refresh."""

    def __init__(
        self,
        *,
        config_manager: Any,
        ai_manager: Any,
        exchange_manager: Any,
        risk_manager: Any,
        refresh_interval: float = 5.0,
        headless: bool = False,
        log_path: Optional[Path] = None,
    ) -> None:
        self.refresh_interval = max(1.0, float(refresh_interval))
        self.app = DashboardApp(
            config_manager=config_manager,
            ai_manager=ai_manager,
            exchange_manager=exchange_manager,
            risk_manager=risk_manager,
            headless=headless,
            log_path=log_path,
        )
        self.config_manager = config_manager
        self.ai_manager = ai_manager
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------- Data sources
    def _collect_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        try:
            if hasattr(self.exchange_manager, "get_api_metrics"):
                metrics["exchange_api"] = self.exchange_manager.get_api_metrics()
        except Exception as exc:
            logger.error("Błąd pobierania metryk giełdy: %s", exc)

        try:
            schedules = getattr(self.ai_manager, "active_schedules", lambda: {})()
            metrics["ai_schedules"] = {
                key: {
                    "interval_s": sched.interval_seconds,
                    "models": list(sched.model_types),
                    "seq_len": sched.seq_len,
                }
                for key, sched in schedules.items()
            }
        except Exception as exc:
            logger.error("Błąd pobierania harmonogramów AI: %s", exc)

        try:
            recent_signals = getattr(self.ai_manager, "_recent_signals", {})
            metrics["ai_recent_signal_count"] = {symbol: len(buffer) for symbol, buffer in recent_signals.items()}
        except Exception:
            metrics.setdefault("ai_recent_signal_count", {})

        try:
            if hasattr(self.risk_manager, "latest_guard_state"):
                metrics["risk_state"] = self.risk_manager.latest_guard_state()
        except Exception:
            pass

        return metrics

    # ----------------------------------------------------------- Refresh loop
    def _refresh_once(self) -> None:
        self.app.refresh_strategy()
        metrics = self._collect_metrics()
        self.app.update_metrics(metrics)
        self.app.tail_logs()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._refresh_once()
            self._stop_event.wait(self.refresh_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.refresh_interval + 1.0)
        self._thread = None

    # -------------------------------------------------------------- Utilities
    def append_alert(self, message: str, *, severity: str = "INFO", **context: Any) -> None:
        payload = {"message": message, "severity": severity, **context}
        self.app.append_alert(payload)

    def append_log(self, line: str) -> None:
        self.app.append_log(line)

    def run(self) -> None:
        self.start()
        try:
            self.app.mainloop()
        finally:
            self.stop()


__all__ = ["DashboardApp", "DashboardController", "DashboardState"]
