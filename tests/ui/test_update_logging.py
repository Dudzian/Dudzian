from __future__ import annotations

from pathlib import Path

from ui.backend.logging import get_update_logger


def test_update_logger_writes_to_file(tmp_path: Path) -> None:
    logger = get_update_logger(tmp_path)
    logger.info("Aktualizacja offline rozpoczęta")

    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "offline_update.log"
    assert log_file.exists(), "Nie utworzono pliku logu aktualizacji"
    contents = log_file.read_text(encoding="utf-8")
    assert "Aktualizacja offline rozpoczęta" in contents

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
