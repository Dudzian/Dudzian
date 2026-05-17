from types import SimpleNamespace

from scripts.ci import run_recovery_matrix


def test_commands_contract_shape() -> None:
    commands = run_recovery_matrix.COMMANDS

    assert len(commands) == 4
    for command in commands:
        assert command[0] == run_recovery_matrix.sys.executable
        assert "pytest" in command
        assert "tests/test_trading_controller.py" in command


def test_commands_cover_expected_boundary_selectors() -> None:
    selectors = " ".join(
        command[command.index("-k") + 1]
        for command in run_recovery_matrix.COMMANDS
        if "-k" in command
    )

    for required_token in (
        "restart",
        "after_restart",
        "restore",
        "replay",
        "duplicate",
        "final_label",
        "outcome_label",
        "open_outcome",
        "tracker",
        "no_tracker",
        "pending_close",
        "close_ranked",
        "risk",
        "execution",
    ):
        assert required_token in selectors


def test_main_fail_fast_returns_first_non_zero(monkeypatch) -> None:
    calls: list[list[str]] = []
    returncodes = iter([0, 7, 0])

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is False
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(run_recovery_matrix.subprocess, "run", fake_run)

    result = run_recovery_matrix.main()

    assert result == 7
    assert calls == [
        run_recovery_matrix.COMMANDS[0],
        run_recovery_matrix.COMMANDS[1],
    ]
