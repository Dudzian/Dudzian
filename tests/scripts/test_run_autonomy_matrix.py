from types import SimpleNamespace

from scripts.ci import run_autonomy_matrix


def test_commands_contract_shape() -> None:
    commands = run_autonomy_matrix.COMMANDS

    assert len(commands) == 5
    for command in commands:
        assert command[0] == run_autonomy_matrix.sys.executable
        assert "pytest" in command
        assert "tests/test_trading_controller.py" in command


def test_commands_cover_expected_boundary_selectors() -> None:
    selectors = " ".join(
        command[command.index("-k") + 1]
        for command in run_autonomy_matrix.COMMANDS
        if "-k" in command
    )

    assert "direction_mismatch" in selectors
    assert "validator" in selectors
    assert "duplicate" in selectors
    assert "final" in selectors
    assert "close_ranked" in selectors
    assert "partial" in selectors
    assert "final_label" in selectors
    assert "missing" in selectors
    assert "unknown" in selectors
    assert "rejected" in selectors
    assert "handoff" in selectors


def test_main_fail_fast_returns_first_non_zero(monkeypatch) -> None:
    calls: list[list[str]] = []
    returncodes = iter([0, 7, 0])

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is False
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(run_autonomy_matrix.subprocess, "run", fake_run)

    result = run_autonomy_matrix.main()

    assert result == 7
    assert calls == [
        run_autonomy_matrix.COMMANDS[0],
        run_autonomy_matrix.COMMANDS[1],
    ]
