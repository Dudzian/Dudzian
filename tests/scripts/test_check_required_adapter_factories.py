from __future__ import annotations

import pytest

import scripts.check_required_adapter_factories as registry_check


def test_check_required_adapter_factories_passes_for_registered_futures() -> None:
    registry_check.validate_required_adapter_factories(
        {
            "deribit_futures": object(),
            "bitmex_futures": object(),
        },
        required=("deribit_futures", "bitmex_futures"),
    )


def test_check_required_adapter_factories_fails_for_missing_entry() -> None:
    with pytest.raises(SystemExit, match="Brak wymaganych fabryk adapterów"):
        registry_check.validate_required_adapter_factories(
            {"deribit_futures": object()},
            required=("deribit_futures", "bitmex_futures"),
        )


def test_check_required_adapter_factories_main_uses_registry_loader(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        registry_check,
        "load_registered_adapter_factories",
        lambda: {
            "deribit_futures": object(),
            "bitmex_futures": object(),
        },
    )

    assert registry_check.main(["--require", "deribit_futures", "--require", "bitmex_futures"]) == 0
    assert "deribit_futures, bitmex_futures" in capsys.readouterr().out
