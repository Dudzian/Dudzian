from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "snippet",
    [
        "void deletePreviewReady(const QString& relativePath, const QVariantMap& result);",
        "void purgePreviewReady(const QVariantMap& result);",
        "void archivePreviewReady(const QString& destination, bool overwrite, const QString& format, const QVariantMap& result);",
        "using BridgeCallback = std::function<void(const BridgeResult&)>;",
        "void runBridge(const QStringList& arguments, BridgeCallback&& callback);",
        "Q_INVOKABLE bool openReportLocation(const QString& relativePath);",
    ],
)
def test_controller_header_contains_async_api(snippet: str) -> None:
    header = Path("ui/src/reporting/ReportCenterController.hpp").read_text(encoding="utf-8")
    assert snippet in header


def test_qml_has_busy_overlay_and_signal_connections() -> None:
    qml = Path("ui/qml/components/ReportBrowser.qml").read_text(encoding="utf-8")
    assert "Rectangle {\n        anchors.fill: parent\n        visible: browser.isBusy" in qml
    assert "function onDeletePreviewReady" in qml
    assert "function onPurgePreviewReady" in qml
    assert "function onArchivePreviewReady" in qml
    assert "function onArchiveFinished" in qml
    assert "enabled: !browser.isBusy" in qml
