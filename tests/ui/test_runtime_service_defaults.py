import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6", reason="UI/QML tests require PySide6")
from PySide6.QtCore import QCoreApplication, QMetaObject, Qt, Q_ARG

from ui.backend.runtime_service import RuntimeService
from tests.ui._qt_invoke_safe import (
    assert_has_overload,
    invoke_safe_qvariantmap,
    invoke_safe_variant,
)

_WIN32_SKIP_VARIANTMAP_ONLY = "win32: only QVariantMap overload present; invokeMethod marshalling risk"


def test_runtime_service_uses_demo_loader_when_no_journal() -> None:
    service = RuntimeService()

    result = service.loadRecentDecisions(5)

    assert result, "Oczekiwano wpisów demonstracyjnych przy pustej konfiguracji"
    assert service.errorMessage == ""
    snapshot = service.feedTransportSnapshot
    assert snapshot["status"] == "initializing"
    assert snapshot["mode"] in {"demo", "file"}
    assert service.aiRegimeBreakdown == []


def test_runtime_service_refreshes_runtime_metadata(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="Runtime metadata test requires pyarrow-backed AI repository")
    try:
        from bot_core.ai import FilesystemModelRepository
        from bot_core.ai.models import ModelArtifact
    except ModuleNotFoundError as exc:
        pytest.skip(f"Runtime metadata test requires optional AI dependencies: {exc}")


    registry_dir = tmp_path / "models"
    registry_dir.mkdir()
    runtime_template = Path("config/runtime.yaml").read_text(encoding="utf-8")
    runtime_config_path = tmp_path / "runtime.yaml"
    runtime_config_path.write_text(
        runtime_template.replace("model_registry_path: models", f"model_registry_path: {registry_dir}"),
        encoding="utf-8",
    )

    repository = FilesystemModelRepository(registry_dir)
    now = datetime.now(timezone.utc)
    artifact = ModelArtifact(
        feature_names=("regime", "strategy"),
        model_state={
            "policies": {
                "trend": {
                    "regime": "trend",
                    "total_plays": 5,
                    "strategies": [
                        {
                            "name": "trend_following",
                            "plays": 5,
                            "total_reward": 3.5,
                            "total_squared_reward": 2.1,
                            "last_reward": 0.8,
                            "updated_at": now.isoformat(),
                        }
                    ],
                }
            }
        },
        trained_at=now,
        metrics={"summary": {"total_plays": 5}},
        metadata={"updated_at": now.isoformat()},
        target_scale=1.0,
        training_rows=5,
        validation_rows=0,
        test_rows=0,
        feature_scalers={},
    )
    repository.save(
        artifact,
        "adaptive_strategy_policy.json",
        version=now.strftime("%Y%m%dT%H%M%S"),
        aliases=("latest",),
        activate=True,
    )

    service = RuntimeService(
        decision_loader=lambda limit: [],
        runtime_config_path=runtime_config_path,
    )

    service.refreshRuntimeMetadata()

    assert service.retrainNextRun, "Oczekiwano obliczenia harmonogramu retrainingu"
    summary = service.adaptiveStrategySummary
    assert "trend" in summary
    assert "trend_following" in summary
    breakdown = service.aiRegimeBreakdown
    assert breakdown
    assert breakdown[0]["bestStrategy"] == "trend_following"


def test_runtime_service_exposes_cloud_status(monkeypatch) -> None:
    dummy_options = SimpleNamespace(
        client=SimpleNamespace(
            address="127.0.0.1:50052",
            fallback_entrypoint="cloud-demo",
            allow_local_fallback=True,
            auto_connect=True,
            metadata={},
            metadata_env={},
            metadata_files={},
        ),
        metadata=[],
        tls_credentials=None,
        authority_override=None,
        config_path=Path("config/cloud/client.yaml"),
    )

    def _fake_load(self, invalidate=False):
        self._cloud_client_options = dummy_options
        self._update_cloud_status(status="configured", target=dummy_options.client.address)
        return dummy_options

    def _fake_handshake(self, force=False):
        self._cloud_session_token = "token"
        self._update_cloud_status(
            status="configured",
            target=dummy_options.client.address,
            handshake={"status": "ok", "licenseId": "LIC", "fingerprint": "HW"},
        )
        return True

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_load_cloud_client_options", _fake_load)
    monkeypatch.setattr(RuntimeService, "_refresh_cloud_handshake", _fake_handshake)

    service = RuntimeService(decision_loader=lambda limit: [], cloud_runtime_enabled=True)
    status = service.cloudRuntimeStatus
    assert status["status"] in {"configured", "ready", "initializing"}
    assert status["target"] == "127.0.0.1:50052"
    assert status["handshake"]["status"] == "ok"


def test_runtime_service_trigger_operator_action_reports_handled_for_request_aliases() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])
    entry = {
        "event": "risk_blocked",
        "timestamp": "2025-01-02T09:15:00+00:00",
        "id": "decision-123",
        "portfolio": "paper-main",
        "strategy": "momentum_v2",
    }

    assert service.triggerOperatorAction("requestFreeze", entry) is True
    assert service.lastOperatorAction["action"] == "freeze"

    assert service.triggerOperatorAction("requestUnfreeze", entry) is True
    assert service.lastOperatorAction["action"] == "unfreeze"

    assert service.triggerOperatorAction("requestUnblock", entry) is True
    assert service.lastOperatorAction["action"] == "unblock"


def test_runtime_service_operator_action_can_be_invoked_via_qt_metaobject() -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(decision_loader=lambda limit: [])

    if sys.platform == "win32":
        # Win32/PySide6: invoke only the safer primitive QVariant path.
        try:
            assert_has_overload(service, "triggerOperatorAction(QString,QVariant)")
        except AssertionError:
            assert_has_overload(service, "triggerOperatorAction(QString,QVariantMap)")
            pytest.skip(_WIN32_SKIP_VARIANTMAP_ONLY)

        ok = QMetaObject.invokeMethod(
            service,
            "triggerOperatorAction",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QString", "requestFreeze"),
            Q_ARG("QVariant", 0),
        )
    else:
        entry_payload = {
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
            "id": "decision-meta",
        }
        ok = QMetaObject.invokeMethod(
            service,
            "triggerOperatorAction",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QString", "requestFreeze"),
            Q_ARG("QVariantMap", invoke_safe_qvariantmap(entry_payload)),
        )

    assert app is not None
    assert ok is True
    assert service.lastOperatorAction["action"] == "freeze"
    if sys.platform == "win32":
        assert service.lastOperatorAction["entry"] == {}




def test_runtime_service_operator_action_can_be_invoked_via_qvariant_signature() -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(decision_loader=lambda limit: [])
    if sys.platform == "win32":
        try:
            assert_has_overload(service, "triggerOperatorAction(QString,QVariant)")
        except AssertionError:
            assert_has_overload(service, "triggerOperatorAction(QString,QVariantMap)")
            pytest.skip(_WIN32_SKIP_VARIANTMAP_ONLY)

        ok = QMetaObject.invokeMethod(
            service,
            "triggerOperatorAction",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QString", "requestFreeze"),
            Q_ARG("QVariant", 0),
        )
    else:
        entry_payload = {
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
            "id": "decision-variant",
        }
        ok = QMetaObject.invokeMethod(
            service,
            "triggerOperatorAction",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QString", "requestFreeze"),
            Q_ARG("QVariant", invoke_safe_variant(entry_payload)),
        )

    assert app is not None
    assert ok is True
    assert service.lastOperatorAction["action"] == "freeze"
    if sys.platform == "win32":
        assert service.lastOperatorAction["entry"] == {}
    else:
        assert service.lastOperatorAction["entry"]["id"] == "decision-variant"


def test_runtime_service_request_freeze_can_be_invoked_via_qvariant_signature() -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(decision_loader=lambda limit: [])
    if sys.platform == "win32":
        try:
            assert_has_overload(service, "requestFreeze(QVariant)")
        except AssertionError:
            try:
                assert_has_overload(service, "requestFreeze()")
            except AssertionError:
                assert_has_overload(service, "requestFreeze(QVariantMap)")
                pytest.skip(_WIN32_SKIP_VARIANTMAP_ONLY)
            ok = QMetaObject.invokeMethod(
                service,
                "requestFreeze",
                Qt.ConnectionType.DirectConnection,
            )
        else:
            ok = QMetaObject.invokeMethod(
                service,
                "requestFreeze",
                Qt.ConnectionType.DirectConnection,
                Q_ARG("QVariant", 0),
            )
    else:
        entry_payload = {
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
            "id": "decision-freeze-variant",
        }
        ok = QMetaObject.invokeMethod(
            service,
            "requestFreeze",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QVariant", invoke_safe_variant(entry_payload)),
        )

    assert app is not None
    assert ok is True
    assert service.lastOperatorAction["action"] == "freeze"
    if sys.platform == "win32":
        assert service.lastOperatorAction["entry"] == {}
    else:
        assert service.lastOperatorAction["entry"]["id"] == "decision-freeze-variant"

def test_runtime_service_trigger_operator_action_normalizes_qjsvalue_like_entry() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])

    class _FakeQjsValue:
        def toVariant(self):
            return {
                "record": {
                    "event": "risk_blocked",
                    "timestamp": "2025-01-02T09:15:00+00:00",
                    "id": "decision-qjs-trigger",
                }
            }

    assert service.triggerOperatorAction("requestFreeze", _FakeQjsValue()) is True
    assert service.lastOperatorAction["action"] == "freeze"
    assert service.lastOperatorAction["entry"]["id"] == "decision-qjs-trigger"


def test_runtime_service_operator_action_normalizes_qjsvalue_like_entry() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])

    class _FakeQjsValue:
        def toVariant(self):
            return {
                "record": {
                    "event": "risk_blocked",
                    "timestamp": "2025-01-02T09:15:00+00:00",
                    "id": "decision-123",
                }
            }

    assert service.requestFreeze(_FakeQjsValue()) is True
    assert service.lastOperatorAction["action"] == "freeze"
    assert service.lastOperatorAction["entry"]["id"] == "decision-123"


def test_runtime_service_operator_action_normalizes_topyobject_fallback() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])

    class _FakeQjsValue:
        def toPyObject(self):
            return {
                "record": {
                    "event": "risk_blocked",
                    "timestamp": "2025-01-02T09:15:00+00:00",
                    "id": "decision-456",
                }
            }

    assert service.requestFreeze(_FakeQjsValue()) is True
    assert service.lastOperatorAction["action"] == "freeze"
    assert service.lastOperatorAction["entry"]["id"] == "decision-456"


def test_runtime_service_operator_action_is_recorded_for_unmappable_payload() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])

    class _UnmappableEntry:
        pass

    assert service.requestFreeze(_UnmappableEntry()) is True
    assert service.lastOperatorAction["action"] == "freeze"
    assert service.lastOperatorAction["entry"] == {}
