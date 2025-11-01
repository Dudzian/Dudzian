from __future__ import annotations

from pathlib import Path

from bot_core.marketplace import validate_exchange_presets


REPO_ROOT = Path(__file__).resolve().parents[2]
_EXCHANGES_DIR = REPO_ROOT / "config" / "exchanges"
_PRESETS_DIR = REPO_ROOT / "config" / "marketplace" / "presets" / "exchanges"


def test_committed_exchange_presets_are_signed_and_current() -> None:
    results = validate_exchange_presets(
        exchanges_dir=_EXCHANGES_DIR,
        output_dir=_PRESETS_DIR,
        version_strategy="spec-hash",
    )

    assert results, "Oczekiwano co najmniej jednego presetu giełdowego w repozytorium."

    problems = {
        result.preset_id: result.issues
        for result in results
        if result.issues or not result.exists or not result.verified or not result.up_to_date
    }

    assert not problems, f"Wykryto problemy z presetami giełdowymi: {problems}"
