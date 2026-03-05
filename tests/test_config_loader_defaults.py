from bot_core.ai import config_loader


def test_load_risk_thresholds_handles_missing_package_resource(monkeypatch):
    class _MissingResource:
        def joinpath(self, *_args, **_kwargs):
            return self

        def open(self, *_args, **_kwargs):  # pragma: no cover - exercised in test
            raise FileNotFoundError("resource missing")

    config_loader.reset_threshold_cache()
    monkeypatch.setattr(
        config_loader,
        "_DEFAULT_OVERRIDE_PATH",
        config_loader._ROOT / "nonexistent_risk_thresholds.yaml",
    )
    monkeypatch.delenv(config_loader._ENV_OVERRIDE_VAR, raising=False)
    monkeypatch.setattr(
        config_loader.resources, "files", lambda *_args, **_kwargs: _MissingResource()
    )

    default_thresholds = config_loader._load_default_thresholds()
    monkeypatch.setattr(config_loader, "_DEFAULT_THRESHOLDS", default_thresholds)

    thresholds = config_loader.load_risk_thresholds()

    assert thresholds["auto_trader"]["map_regime_to_signal"]["assessment_confidence"] == 0.5


def test_load_risk_thresholds_handles_missing_defaults_package(monkeypatch):
    config_loader.reset_threshold_cache()
    monkeypatch.setattr(
        config_loader,
        "_DEFAULT_OVERRIDE_PATH",
        config_loader._ROOT / "nonexistent_risk_thresholds.yaml",
    )
    monkeypatch.delenv(config_loader._ENV_OVERRIDE_VAR, raising=False)

    def _missing_package(*_args, **_kwargs):
        raise ModuleNotFoundError("bot_core.ai._defaults missing")

    monkeypatch.setattr(config_loader.resources, "files", _missing_package)

    default_thresholds = config_loader._load_default_thresholds()
    monkeypatch.setattr(config_loader, "_DEFAULT_THRESHOLDS", default_thresholds)

    thresholds = config_loader.load_risk_thresholds()

    assert thresholds["market_regime"]["metrics"]["short_span_min"] == 5
