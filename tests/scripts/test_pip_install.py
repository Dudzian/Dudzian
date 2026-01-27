from scripts.ci import pip_install


def test_pip_install_splits_separator(monkeypatch) -> None:
    captured = {}

    def fake_call(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return 0

    monkeypatch.setattr(pip_install.subprocess, "call", fake_call)

    result = pip_install.main(["--", "-e", ".[dev]"])

    assert result == 0
    assert captured["cmd"][0:4] == [pip_install.sys.executable, "-m", "pip", "install"]
    assert "--" not in captured["cmd"]
    assert "-e" in captured["cmd"]
    assert ".[dev]" in captured["cmd"]


def test_pip_install_requires_separator_for_pip_args(monkeypatch) -> None:
    def fake_call(*_args, **_kwargs):
        raise AssertionError("pip should not be invoked without -- separator")

    monkeypatch.setattr(pip_install.subprocess, "call", fake_call)

    result = pip_install.main(["-e", ".[dev]"])

    assert result == 2


def test_pip_install_requires_args_after_separator(monkeypatch) -> None:
    def fake_call(*_args, **_kwargs):
        raise AssertionError("pip should not be invoked without args after --")

    monkeypatch.setattr(pip_install.subprocess, "call", fake_call)

    result = pip_install.main(["--"])

    assert result == 2
