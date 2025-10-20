import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

Pane {
    id: footer
    implicitHeight: 44
    padding: 12

    background: Rectangle {
        color: Qt.darker(footer.palette.window, 1.3)
    }

    RowLayout {
        anchors.fill: parent
        spacing: 16

        Label {
            text: qsTr("Instrument: %1").arg(appController.instrumentLabel)
        }

        Label {
            text: qsTr("Status: %1").arg(appController.connectionStatus)
        }

        Label {
            text: qsTr("Samples: %1")
                  .arg(ohlcvModel && ohlcvModel.count !== undefined ? ohlcvModel.count : 0)
        }

        Label {
            text: qsTr("Okna: %1")
                  .arg(((Window.window && Window.window.extraWindowCount !== undefined)
                        ? Window.window.extraWindowCount : 0) + 1)
        }

        Label {
            text: licenseController.licenseActive
                    ? qsTr("Licencja: %1 • ważna do %2 • FP %3")
                          .arg(licenseController.licenseProfile)
                          .arg(licenseController.licenseExpiresAt.length > 0
                                   ? Qt.formatDateTime(new Date(licenseController.licenseExpiresAt), "yyyy-MM-dd")
                                   : qsTr("-")
                          )
                          .arg(licenseController.licenseFingerprint)
                    : qsTr("Licencja: nieaktywna")
            color: licenseController.licenseActive
                    ? palette.windowText
                    : Qt.rgba(0.94, 0.36, 0.32, 1)
        }

        Label {
            readonly property bool active: alertsModel && alertsModel.hasActiveAlerts
            readonly property bool hasUnacked: alertsModel && alertsModel.hasUnacknowledgedAlerts
            text: active
                    ? hasUnacked
                          ? qsTr("Alerty: %1K / %2W • %3 niepotw.")
                                .arg(alertsModel.criticalCount)
                                .arg(alertsModel.warningCount)
                                .arg(alertsModel.unacknowledgedCount)
                          : qsTr("Alerty: %1K / %2W")
                                .arg(alertsModel.criticalCount)
                                .arg(alertsModel.warningCount)
                    : qsTr("Alerty: brak")
            color: alertsModel && alertsModel.criticalCount > 0
                    ? Qt.rgba(0.94, 0.36, 0.32, 1)
                    : alertsModel && alertsModel.warningCount > 0
                          ? Qt.rgba(0.96, 0.7, 0.25, 1)
                          : hasUnacked
                                ? Qt.rgba(0.96, 0.68, 0.26, 1)
                                : palette.windowText
            font.bold: (alertsModel && alertsModel.criticalCount > 0) || hasUnacked
        }

        Label {
            text: appController.reduceMotionActive
                    ? qsTr("Animacje: ograniczone")
                    : qsTr("Animacje: pełne")
            color: appController.reduceMotionActive
                    ? Qt.rgba(0.96, 0.74, 0.23, 1)
                    : palette.windowText
        }

        Label {
            objectName: "telemetryStatusLabel"
            readonly property bool backlog: appController.telemetryPendingRetryCount > 0
            text: backlog
                    ? qsTr("Telemetria: offline (%1 próbek w kolejce)")
                          .arg(appController.telemetryPendingRetryCount)
                    : qsTr("Telemetria: online")
            color: backlog ? Qt.rgba(0.94, 0.36, 0.32, 1) : palette.windowText
            font.bold: backlog
            hoverEnabled: true
            ToolTip.visible: hovered && backlog
            ToolTip.delay: 400
            ToolTip.text: qsTr("Aplikacja ponawia wysyłkę próbek telemetrii do serwisu MetricsService")
        }

        Item { Layout.fillWidth: true }

        Label {
            id: clockLabel
            text: Qt.formatDateTime(new Date(), "HH:mm:ss")

            Timer {
                interval: 1000
                running: true
                repeat: true
                onTriggered: clockLabel.text = Qt.formatDateTime(new Date(), "HH:mm:ss")
            }
        }
    }
}
