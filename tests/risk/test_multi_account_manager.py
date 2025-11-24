from bot_core.risk.portfolio import AccountExposure, MultiAccountRiskManager


def test_collect_snapshot_handles_missing_portfolio() -> None:
    manager = MultiAccountRiskManager()
    exposures = [AccountExposure(account_id="acct-1", portfolio=None, equity=0.0)]

    snapshot = manager.collect_snapshot(exposures, {})

    assert snapshot.accounts[0].account_id == "acct-1"
    assert snapshot.accounts[0].equity == 0.0
    assert snapshot.symbol_limits == {}
    assert snapshot.aggregate_metrics.risk_level.name == "VERY_LOW"


def test_collect_snapshot_skips_invalid_positions() -> None:
    manager = MultiAccountRiskManager()
    exposures = [
        AccountExposure(
            account_id="acct-2",
            portfolio={"BTCUSDT": None},
            equity=100.0,
        )
    ]

    snapshot = manager.collect_snapshot(exposures, {})

    assert snapshot.accounts[0].account_id == "acct-2"
    assert snapshot.symbol_limits == {}
    # brak poprawnego portfolio nie powinien powodować wyjątków ani NaN
    assert snapshot.accounts[0].metrics.overall_risk_score >= 0.0
