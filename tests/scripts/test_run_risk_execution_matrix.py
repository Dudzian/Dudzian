from types import SimpleNamespace

from scripts.ci import run_risk_execution_matrix


def test_commands_contract_shape() -> None:
    commands = run_risk_execution_matrix.COMMANDS

    assert len(commands) == 4
    for command in commands:
        assert command[0] == run_risk_execution_matrix.sys.executable
        assert "pytest" in command
        assert "tests/test_trading_controller.py" in command

    assert "tests/test_runtime_pipeline.py" in commands[0]
    assert "tests/exchanges" in commands[1]
    assert "tests/integration/exchanges" in commands[1]


def test_commands_cover_expected_boundary_selectors() -> None:
    selectors = " ".join(
        command[command.index("-k") + 1]
        for command in run_risk_execution_matrix.COMMANDS
        if "-k" in command
    )

    assert "risk" in selectors
    assert "performance_guard" in selectors
    assert "ai_failover" in selectors
    assert "execution" in selectors
    assert "place_order" in selectors
    assert "create_order" in selectors
    assert "retry" in selectors
    assert "non_filled" in selectors
    assert "partial" in selectors
    assert "duplicate" in selectors
    assert "final_label" in selectors
    assert "validator" in selectors
    assert "direction_mismatch" in selectors
    assert "without_exchange_credentials" in selectors


def test_main_fail_fast_returns_first_non_zero(monkeypatch) -> None:
    calls: list[list[str]] = []
    returncodes = iter([0, 9, 0])

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is False
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(run_risk_execution_matrix.subprocess, "run", fake_run)

    result = run_risk_execution_matrix.main()

    assert result == 9
    assert calls == [
        run_risk_execution_matrix.COMMANDS[0],
        run_risk_execution_matrix.COMMANDS[1],
    ]
