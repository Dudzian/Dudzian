from __future__ import annotations

from pathlib import Path

import pytest

from ui.backend.decision_log_repository import DecisionLogRepository


def test_decision_log_repository_loads_jsonl_and_skips_invalid_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"event":"first","timestamp":"2025-01-01T00:00:00+00:00"}',
                "",
                '{"event":"broken"',
                '["not-a-mapping"]',
                '{"event":"second","timestamp":"2025-01-01T00:01:00+00:00"}',
            ]
        ),
        encoding="utf-8",
    )
    repository = DecisionLogRepository()

    with caplog.at_level("WARNING"):
        entries = list(repository.load_jsonl_entries(log_path, limit=0))

    assert [entry["event"] for entry in entries] == ["first", "second"]
    assert "Pominięto uszkodzony wpis decision logu" in caplog.text


def test_decision_log_repository_applies_tail_limit(tmp_path: Path) -> None:
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"event":"first"}',
                '{"event":"second"}',
                '{"event":"third"}',
            ]
        ),
        encoding="utf-8",
    )
    repository = DecisionLogRepository()

    entries = list(repository.load_jsonl_entries(log_path, limit=2))

    assert [entry["event"] for entry in entries] == ["second", "third"]


def test_decision_log_repository_skips_valid_json_non_mapping(tmp_path: Path) -> None:
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"event":"first"}',
                '["valid-but-non-mapping"]',
                '{"event":"second"}',
            ]
        ),
        encoding="utf-8",
    )
    repository = DecisionLogRepository()

    entries = repository.load_jsonl_entries(log_path, limit=0)

    assert [entry["event"] for entry in entries] == ["first", "second"]


def test_decision_log_repository_reraises_file_not_found(tmp_path: Path) -> None:
    repository = DecisionLogRepository()

    with pytest.raises(FileNotFoundError):
        list(repository.load_jsonl_entries(tmp_path / "missing.jsonl", limit=0))


def test_decision_log_repository_wraps_io_errors(tmp_path: Path) -> None:
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text('{"event":"first"}\n', encoding="utf-8")
    repository = DecisionLogRepository()

    original_open = Path.open

    def _broken_open(self: Path, *args: object, **kwargs: object):
        if self == log_path:
            raise OSError("disk failure")
        return original_open(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Path, "open", _broken_open)
        with pytest.raises(
            RuntimeError,
            match=r"Nie udało się odczytać decision logu '.*decision\.jsonl': disk failure",
        ):
            list(repository.load_jsonl_entries(log_path, limit=0))
