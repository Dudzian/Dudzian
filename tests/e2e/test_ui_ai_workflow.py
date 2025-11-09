from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import pytest
pytestqt = pytest.importorskip("pytestqt")
from PySide6.QtCore import QObject
from pytestqt.qtbot import QtBot

from bot_core.ai.manager import AIManager
from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.training import ModelTrainer
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, TradingDecisionEvent
from core.security.license_verifier import FingerprintResult, LicenseVerificationOutcome
from ui.backend.licensing_controller import LicensingController
from ui.backend.onboarding_service import OnboardingService


class StubLicenseVerifier:
    """Prosty weryfikator licencji zwracający sukces bez I/O."""

    def __init__(self, *, fingerprint: str = "FP-DEMO") -> None:
        self.fingerprint = fingerprint
        self.recorded_payloads: list[str] = []

    def read_fingerprint(self) -> FingerprintResult:
        return FingerprintResult(self.fingerprint)

    def verify_license_text(
        self, text: str, *, fingerprint: str | None = None
    ) -> LicenseVerificationOutcome:
        self.recorded_payloads.append(text)
        data = json.loads(text)
        license_id = data.get("id", "demo-license")
        return LicenseVerificationOutcome(True, "ok", license_id=license_id, fingerprint=fingerprint)

    def verify_license_file(
        self, path: str | Path, *, fingerprint: str | None = None
    ) -> LicenseVerificationOutcome:
        return self.verify_license_text(Path(path).read_text(encoding="utf-8"), fingerprint=fingerprint)


class StubSecretStore(QObject):
    """Minimalna implementacja magazynu sekretów zapamiętująca zapisane klucze."""

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[Mapping[str, object]] = []

    def save_exchange_credentials(self, credentials) -> None:  # type: ignore[override]
        payload = {
            "exchange": credentials.exchange,
            "api_key": credentials.api_key,
            "api_secret": credentials.api_secret,
            "api_passphrase": credentials.api_passphrase,
        }
        self.saved.append(payload)

    def security_details_token(self) -> str:
        return "secure"


@pytest.mark.qt
def test_end_to_end_license_strategy_and_paper_flow(qtbot: QtBot, tmp_path: Path) -> None:
    verifier = StubLicenseVerifier()
    licensing = LicensingController(verifier=verifier)
    qtbot.addCleanup(licensing.deleteLater)

    with qtbot.waitSignal(licensing.fingerprintChanged):
        licensing.refreshFingerprint()
    assert licensing.fingerprint == verifier.fingerprint

    payload = json.dumps({"id": "QA-LICENSE"})
    assert licensing.applyLicenseText(payload) is True
    assert licensing.licenseAccepted is True

    descriptors = (
        ("alpha", "Alpha Strategy"),
        ("beta", "Beta Strategy"),
    )

    def _strategies():
        from bot_core.strategies.public import StrategyDescriptor

        return (
            StrategyDescriptor(
                name=name,
                engine=name,
                title=title,
                license_tier="standard",
                risk_classes=("swing",),
                required_data=("ohlcv",),
                tags=("demo",),
                metadata={},
            )
            for name, title in descriptors
        )

    onboarding = OnboardingService(
        strategy_loader=lambda: tuple(_strategies()),
        secret_store=StubSecretStore(),
    )
    qtbot.addCleanup(onboarding.deleteLater)

    assert onboarding.refreshStrategies() is True
    assert onboarding.selectStrategy("alpha") is True
    assert onboarding.importApiKey("binance", "key", "secret") is True
    assert onboarding.configurationReady is True

    licensing.finalizeOnboarding(
        True,
        strategy_title=onboarding.selectedStrategyTitle,
        exchange_id="binance",
        onboarding_status_id="wizard.completed",
    )

    journal = InMemoryTradingDecisionJournal()
    ai_manager = AIManager(
        model_dir=tmp_path / "models",
        decision_journal=journal,
    )

    vectors: list[FeatureVector] = []
    for idx in range(32):
        vectors.append(
            FeatureVector(
                timestamp=1_700_000_000 + idx * 60,
                symbol="BTCUSDT",
                features={"momentum": float(idx) / 10.0, "volume_ratio": 1.0},
                target_bps=float(idx % 3) * 5.0,
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": ["BTCUSDT"]})

    ai_manager.schedule_walk_forward_retraining(
        "demo-job",
        interval=timedelta(minutes=15),
        dataset_provider=lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=6, validation_split=0.25),
        quality_thresholds={"min_directional_accuracy": 0.0, "max_mae": 50.0},
        repository_base=tmp_path / "repository",
        model_type="alpha",
        symbol="BTCUSDT",
        attach_to_decision=True,
        decision_name="paper-alpha",
        decision_repository_root=tmp_path / "decision_repo",
        set_default_decision=True,
    )

    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)
    assert results, "trening powinien zostać wykonany"

    score = ai_manager.score_decision_features(
        {"momentum": 0.15, "volume_ratio": 1.0},
        model_name="paper-alpha",
    )
    assert isinstance(score.expected_return_bps, float)

    decision_event = TradingDecisionEvent(
        event_type="paper_simulation",
        timestamp=now,
        environment="paper",
        portfolio="demo-portfolio",
        risk_profile="balanced",
        schedule="paper.alpha.schedule",
        strategy="paper-alpha",
        metadata={
            "ai": {
                "expected_return_bps": float(score.expected_return_bps),
                "success_probability": float(score.success_probability),
            }
        },
    )
    journal.record(decision_event)

    exported = tuple(journal.export())
    assert exported and exported[-1]["strategy"] == "paper-alpha"
    assert exported[-1]["environment"] == "paper"
