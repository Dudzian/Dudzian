from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot_core.runtime.pipeline_config_loader import PipelineConfigLoader


def _core_with_schedulers(payload):
    return SimpleNamespace(multi_strategy_schedulers=payload)


def test_resolve_multi_strategy_scheduler_missing_section() -> None:
    loader = PipelineConfigLoader()
    with pytest.raises(ValueError, match="Brak zdefiniowanych schedulerów"):
        loader.resolve_multi_strategy_scheduler(core_config=_core_with_schedulers({}), scheduler_name=None)


def test_resolve_multi_strategy_scheduler_default_selection() -> None:
    loader = PipelineConfigLoader()
    schedulers = {"alpha": object(), "beta": object()}
    resolved = loader.resolve_multi_strategy_scheduler(
        core_config=_core_with_schedulers(schedulers), scheduler_name=None
    )
    assert resolved.scheduler_name == "alpha"
    assert resolved.scheduler_config is schedulers["alpha"]


def test_resolve_multi_strategy_scheduler_explicit_selection() -> None:
    loader = PipelineConfigLoader()
    schedulers = {"alpha": object(), "beta": object()}
    resolved = loader.resolve_multi_strategy_scheduler(
        core_config=_core_with_schedulers(schedulers), scheduler_name="beta"
    )
    assert resolved.scheduler_name == "beta"
    assert resolved.scheduler_config is schedulers["beta"]


def test_resolve_multi_strategy_scheduler_missing_named_scheduler() -> None:
    loader = PipelineConfigLoader()
    with pytest.raises(KeyError, match="Nie znaleziono scheduler-a"):
        loader.resolve_multi_strategy_scheduler(
            core_config=_core_with_schedulers({"alpha": object()}),
            scheduler_name="gamma",
        )


def test_resolve_multi_portfolio_entries_variants() -> None:
    loader = PipelineConfigLoader()
    assert loader.resolve_multi_portfolio_entries({"portfolio_id": "p1", "preset": "x"}) == [
        {"portfolio_id": "p1", "preset": "x"}
    ]
    assert loader.resolve_multi_portfolio_entries(
        [{"portfolio_id": "p1", "preset": "x"}, {"portfolio_id": "p2", "preset": "y"}]
    ) == [
        {"portfolio_id": "p1", "preset": "x"},
        {"portfolio_id": "p2", "preset": "y"},
    ]
    assert loader.resolve_multi_portfolio_entries(
        {"portfolios": {"a": {"portfolio_id": "p1", "preset": "x"}}}
    ) == [{"portfolio_id": "p1", "preset": "x"}]


@pytest.mark.parametrize("invalid", [123, object()])
def test_resolve_multi_portfolio_entries_invalid(invalid: object) -> None:
    loader = PipelineConfigLoader()
    with pytest.raises(TypeError, match="Unsupported portfolio definition structure"):
        loader.resolve_multi_portfolio_entries(invalid)


def test_build_portfolio_binding_validation_and_defaults() -> None:
    loader = PipelineConfigLoader()
    with pytest.raises(ValueError, match="portfolio_id"):
        loader.build_portfolio_binding({"preset": "main"})
    with pytest.raises(ValueError, match="missing primary preset"):
        loader.build_portfolio_binding({"portfolio_id": "paper"})

    binding = loader.build_portfolio_binding(
        {
            "portfolio_id": "paper",
            "preset": "main",
            "fallback": [" alt ", "alt", ""],
            "followers": [
                {"id": "follower-a", "scaling": 2, "risk": 0.5},
                {"portfolio_id": "", "scaling": 3},
            ],
        }
    )
    assert binding.portfolio_id == "paper"
    assert binding.primary_preset == "main"
    assert binding.fallback_presets == ("alt",)
    assert len(binding.followers) == 1
    assert binding.followers[0].portfolio_id == "follower-a"
    assert binding.followers[0].scaling == 2.0
    assert binding.followers[0].risk_multiplier == 0.5
    assert binding.rebalance_cooldown.total_seconds() == 300

    explicit = loader.build_portfolio_binding(
        {
            "portfolio_id": "paper",
            "preset": "main",
            "rebalance_cooldown_seconds": 42,
        }
    )
    assert explicit.rebalance_cooldown.total_seconds() == 42


def test_followers_and_fallback_helpers() -> None:
    loader = PipelineConfigLoader()
    assert loader.build_follower_configs(None) == ()
    assert loader.normalize_fallbacks(None) == ()

    with pytest.raises(TypeError, match="Followers must be"):
        loader.build_follower_configs("bad")
    with pytest.raises(TypeError, match="Fallback presets must"):
        loader.normalize_fallbacks(123)

    followers = loader.build_follower_configs(
        [
            {"portfolio_id": "f1", "enabled": False, "allow_partial": False},
            {"id": "f2", "max_position_value": "100.5"},
            {"id": ""},
        ]
    )
    assert [item.portfolio_id for item in followers] == ["f1", "f2"]
    assert followers[0].enabled is False
    assert followers[0].allow_partial is False
    assert followers[1].max_position_value == 100.5

    assert loader.normalize_fallbacks(["a", " a ", "b", ""]) == ("a", "b")
