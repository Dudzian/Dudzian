from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.auto_trader import AutoTrader
from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.strategies.regime_workflow import PresetVersionInfo, RegimePresetActivation


class _DummyEmitter:
    def emit(self, *_args, **_kwargs):
        return None

    def log(self, *_args, **_kwargs):
        return None


def _build_activation(
    regime: MarketRegime,
    preset_name: str,
    *,
    fallback: bool = False,
) -> RegimePresetActivation:
    assessment = MarketRegimeAssessment(
        regime=regime,
        confidence=0.8,
        risk_score=0.3,
        metrics={"volatility": 0.1},
    )
    issued_at = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    version = PresetVersionInfo(
        hash=f"hash-{preset_name}",
        signature={"alg": "HMAC-SHA256", "key_id": "stub"},
        issued_at=issued_at,
        metadata={"name": preset_name},
    )
    preset = {"name": preset_name, "metadata": {"ensemble_weights": {preset_name: 1.0}}}
    return RegimePresetActivation(
        regime=regime,
        assessment=assessment,
        summary=None,
        preset=preset,
        version=version,
        decision_candidates=tuple(),
        activated_at=issued_at,
        preset_regime=regime,
        used_fallback=fallback,
        missing_data=("ohlcv",) if fallback else (),
        blocked_reason="missing_data" if fallback else None,
        recommendation=None,
        license_issues=("license_block",) if fallback else (),
    )


def test_decision_journal_contains_activation_and_guardrail_metadata() -> None:
    journal = InMemoryTradingDecisionJournal()
    trader = AutoTrader(
        emitter=_DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )

    trader._last_guardrail_reasons = ["drawdown"]  # noqa: SLF001 - ustawienie stanu testowego
    trader._exchange_degradation_guardrail_active = True  # noqa: SLF001
    first_activation = _build_activation(MarketRegime.TREND, "trend_alpha")
    trader._apply_strategy_regime_activation(first_activation)  # noqa: SLF001
    trader._log_decision_event("cycle_complete", metadata={})  # noqa: SLF001

    trader._exchange_degradation_guardrail_active = False  # noqa: SLF001
    trader._last_guardrail_reasons = []  # noqa: SLF001
    second_activation = _build_activation(MarketRegime.MEAN_REVERSION, "mean_beta", fallback=True)
    trader._apply_strategy_regime_activation(second_activation)  # noqa: SLF001
    trader._log_decision_event("cycle_complete", metadata={})  # noqa: SLF001

    records = list(journal.export())
    assert len(records) == 2

    first_meta = json.loads(records[0]["activation"])
    assert first_meta["preset_name"] == "trend_alpha"
    assert first_meta["used_fallback"] is False

    last_meta = json.loads(records[-1]["activation"])
    assert last_meta["preset_name"] == "mean_beta"
    assert last_meta["used_fallback"] is True
    assert "license_block" in last_meta.get("license_issues", [])

    guardrail_meta = json.loads(records[-1]["guardrail_transition"])
    assert guardrail_meta["previous_active"] is True
    assert guardrail_meta["active"] is False
