"""Testy regresyjne dla shimu `KryptoLowca.auto_trader.app`."""

from __future__ import annotations

from importlib import import_module, reload


def test_shim_reexports_core_symbols(monkeypatch) -> None:
    """Shim powinien delegować klasy do `bot_core.auto_trader.app`."""

    core_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def core_emit(*args: object, **kwargs: object) -> None:
        core_calls.append((args, kwargs))

    monkeypatch.setattr("bot_core.alerts.emit_alert", core_emit)

    shim_module = reload(import_module("KryptoLowca.auto_trader.app"))
    core_module = import_module("bot_core.auto_trader.app")

    assert shim_module.AutoTrader is core_module.AutoTrader
    assert shim_module.RiskDecision is core_module.RiskDecision

    package = import_module("KryptoLowca.auto_trader")

    package_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def package_emit(*args: object, **kwargs: object) -> None:
        package_calls.append((args, kwargs))

    monkeypatch.setattr(package, "emit_alert", package_emit, raising=False)

    shim_module.emit_alert("legacy", severity="WARNING")
    assert package_calls == [(("legacy",), {"severity": "WARNING"})]
    assert core_calls == []

    package_calls.clear()
    monkeypatch.delattr(package, "emit_alert", raising=False)

    shim_module.emit_alert("fallback", severity="INFO")
    shim_module._emit_alert("shadow")

    assert core_calls == [
        (("fallback",), {"severity": "INFO"}),
        (("shadow",), {}),
    ]


def test_shim_ignores_non_callable_emitters(monkeypatch) -> None:
    """Niepoprawne atrybuty ``emit_alert`` nie powinny blokować alertów."""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def core_emit(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("bot_core.alerts.emit_alert", core_emit)

    package = import_module("KryptoLowca.auto_trader")
    monkeypatch.setattr(package, "emit_alert", "not-callable", raising=False)

    shim_module = reload(import_module("KryptoLowca.auto_trader.app"))

    shim_module.emit_alert("from-shim", severity="INFO")

    assert calls == [(("from-shim",), {"severity": "INFO"})]
