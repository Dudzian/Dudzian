from __future__ import annotations

from types import MappingProxyType

import pytest

from bot_core.trading.strategy_aliasing import (
    StrategyAliasResolver,
    normalise_alias_map,
    strategy_key_aliases,
    strategy_name_candidates,
)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Day Trading", {"day trading", "day_trading"}),
        ("SCALPING", {"scalping"}),
        ("grid-trading", {"grid trading", "grid_trading"}),
    ],
)
def test_strategy_key_aliases_normalize_variants(source: str, expected: set[str]) -> None:
    aliases = set(strategy_key_aliases(source))
    assert expected <= aliases


def test_strategy_name_candidates_include_aliases_and_suffixes() -> None:
    candidates = strategy_name_candidates(
        "Intraday_Breakout_Probing",
        {"intraday_breakout": "day_trading"},
        ("_probing",),
    )
    assert "Intraday_Breakout_Probing" in candidates
    assert "intraday_breakout" in candidates
    assert "day_trading" in candidates
    assert "intraday breakout" in candidates
    trimmed = [value for value in candidates if value.endswith("_probing")]
    assert trimmed  # suffix form is preserved


def test_normalise_alias_map_extends_variants_and_ignores_empty() -> None:
    result = normalise_alias_map({" Intraday Breakout ": " day_trading ", "": ""})
    assert result["intraday breakout"] == "day_trading"
    assert result["intraday_breakout"] == "day_trading"
    assert "" not in result


def test_strategy_name_candidates_resolve_alias_map_variants() -> None:
    candidates = strategy_name_candidates(
        "INTRADAY BREAKOUT",
        {"Intraday_Breakout": "day_trading"},
        (),
    )
    assert "day_trading" in candidates


def test_strategy_name_candidates_accept_pre_normalised_map() -> None:
    alias_map = MappingProxyType(normalise_alias_map({"intraday_breakout": "day_trading"}))
    candidates = strategy_name_candidates(
        "Intraday Breakout",
        alias_map,
        (),
        normalised=True,
    )
    assert "day_trading" in candidates
    # ensure alias map remains proxy and untouched
    assert isinstance(alias_map, MappingProxyType)


def test_strategy_alias_resolver_caches_aliases_and_suffixes() -> None:
    resolver = StrategyAliasResolver({"Intraday Breakout": "day_trading"}, ("_probing",))
    assert resolver.alias_map["intraday_breakout"] == "day_trading"
    assert resolver.suffixes == ("_probing",)
    candidates = resolver.candidates("intraday_breakout_probing")
    assert "day_trading" in candidates
    derived = resolver.derive(alias_map={"scalping": "scalping"})
    assert "scalping" in derived.alias_map
    assert "scalping" not in resolver.alias_map
