from types import SimpleNamespace

from scripts.ci import run_observability_matrix


def test_commands_contract_shape() -> None:
    commands = run_observability_matrix.COMMANDS

    assert len(commands) == 5
    for command in commands:
        assert command[0] == run_observability_matrix.sys.executable
        assert "pytest" in command


def test_commands_cover_expected_observability_selectors() -> None:
    selectors = " ".join(" ".join(command) for command in run_observability_matrix.COMMANDS)

    for required_token in (
        "journal",
        "feed",
        "alert",
        "severity",
        "signal_skipped",
        "order_executed",
        "test_runtime_service_defaults.py",
        "fallback",
        "decision_envelope",
        "test_alerts.py",
        "test_decision_payload_normalizer.py",
    ):
        assert required_token in selectors


def test_main_fail_fast_returns_first_non_zero(monkeypatch) -> None:
    calls: list[list[str]] = []
    returncodes = iter([0, 9, 0])

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is False
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(run_observability_matrix.subprocess, "run", fake_run)

    result = run_observability_matrix.main()

    assert result == 9
    assert calls == [
        run_observability_matrix.COMMANDS[0],
        run_observability_matrix.COMMANDS[1],
    ]
