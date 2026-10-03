from pathlib import Path

from scripts import cryptohunter_activation_request as activation


def test_production_cleanup_is_explicitly_rejected_without_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        activation,
        "load_production_trust",
        lambda path: object(),
    )
    result = activation.main(["cleanup", "--environment", "PRODUCTION", "--output", str(tmp_path)])
    assert result == 2
    assert "cleanup is TEST_ONLY" in capsys.readouterr().err
    assert not (tmp_path / "cleanup-evidence.json").exists()


def test_test_only_cleanup_reports_exact_environment(tmp_path: Path) -> None:
    assert (
        activation.main(["cleanup", "--environment", "TEST_ONLY", "--output", str(tmp_path)]) == 0
    )
    assert '"environment":"TEST_ONLY"' in (tmp_path / "cleanup-evidence.json").read_text(
        encoding="utf-8"
    )
