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
