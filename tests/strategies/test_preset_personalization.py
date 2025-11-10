import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pytest

from bot_core.marketplace import PresetRepository
from bot_core.security.signing import build_hmac_signature
from bot_core.strategies.personalization.preferences import (
    PresetPreferencePersonalizer,
    RiskTarget,
    UserPreferenceConfig,
)


def _write_signed_preset(path: Path, *, preset_id: str, signing_key: bytes) -> None:
    payload = {
        "name": preset_id,
        "metadata": {
            "id": preset_id,
            "version": "1.0.0",
            "license": {
                "module_id": f"module::{preset_id}",
                "fingerprint": "device",
                "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        },
        "strategies": [
            {
                "name": f"{preset_id}-strategy",
                "engine": "mean_reversion",
                "parameters": {
                    "budget": 1000,
                    "risk_multiplier": 1.0,
                    "leverage": 2.0,
                    "max_positions": 3,
                },
            }
        ],
    }
    signature = build_hmac_signature(payload, key=signing_key, key_id="catalog")
    document = {"preset": payload, "signature": signature}
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture()
def personalization_preset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "personalization"
    base.mkdir()
    monkeypatch.chdir(base)
    presets = base / "presets"
    presets.mkdir()
    return presets


@pytest.fixture()
def signing_key() -> bytes:
    return b"catalog-secret"


def _sample_preset() -> Mapping[str, object]:
    return {
        "name": "alpha",
        "strategies": [
            {
                "name": "mean-reversion",
                "parameters": {
                    "budget": 1000,
                    "risk_multiplier": 1.0,
                    "leverage": 2.0,
                    "max_positions": 3,
                },
            }
        ],
    }


def test_user_preference_config_from_mapping() -> None:
    payload = {"riskTarget": "aggressive", "budget": "2500", "maxPositions": 5}
    config = UserPreferenceConfig.from_mapping(payload)
    assert config.risk_target is RiskTarget.AGGRESSIVE
    assert config.budget == pytest.approx(2500.0)
    assert config.max_positions == 5


def test_personalizer_builds_overrides() -> None:
    personalizer = PresetPreferencePersonalizer(
        risk_scalars={
            RiskTarget.CONSERVATIVE: 0.5,
            RiskTarget.BALANCED: 1.0,
            RiskTarget.AGGRESSIVE: 1.8,
        }
    )
    config = UserPreferenceConfig.from_mapping(
        {"risk_target": "aggressive", "budget": 5000, "max_positions": 7}
    )
    overrides = personalizer.build_overrides(_sample_preset(), config)
    assert "mean-reversion" in overrides
    params = overrides["mean-reversion"]
    assert params["budget"] == pytest.approx(5000)
    assert params["risk_multiplier"] == pytest.approx(1.8)
    assert params["leverage"] == pytest.approx(3.6)
    assert params["max_positions"] == 7


def test_personalizer_accepts_preset_document(personalization_preset_dir: Path, signing_key: bytes) -> None:
    preset_path = personalization_preset_dir / "alpha.json"
    _write_signed_preset(preset_path, preset_id="alpha", signing_key=signing_key)
    repository = PresetRepository(personalization_preset_dir)
    document = repository.load_all()[0]
    config = UserPreferenceConfig.from_mapping(
        {"risk_target": "conservative", "budget": 800}
    )
    overrides = PresetPreferencePersonalizer().build_overrides(document, config)
    assert overrides
    strategy_overrides = overrides.get("alpha-strategy") or next(iter(overrides.values()))
    assert strategy_overrides["budget"] == pytest.approx(800)
