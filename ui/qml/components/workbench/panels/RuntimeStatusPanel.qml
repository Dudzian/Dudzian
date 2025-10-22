import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "runtimePanel"
    property var runtimeStatus: ({})

    Layout.fillWidth: true
    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 8

        Label {
            text: qsTr("Status runtime")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Połączenie"); font.bold: true }
            Label {
                text: root.runtimeStatus.connection || qsTr("brak")
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Offline"); font.bold: true }
            Label { text: Utils.formatBoolean(root.runtimeStatus.offlineMode) }

            Label { text: qsTr("Automatyzacja"); font.bold: true }
            Label { text: Utils.formatBoolean(root.runtimeStatus.automationRunning) }

            Label { text: qsTr("Redukcja animacji"); font.bold: true }
            Label { text: Utils.formatBoolean(root.runtimeStatus.reduceMotion) }

            Label { text: qsTr("Daemon offline"); font.bold: true }
            Label { text: root.runtimeStatus.offlineDaemonStatus || qsTr("brak") }
        }

        ColumnLayout {
            spacing: 4
            Layout.fillWidth: true

            Label { text: qsTr("Parametry wydajności"); font.bold: true }
            Label {
                text: root.runtimeStatus.performanceGuard
                    ? qsTr("FPS docelowe: %1, limit jank: %2 ms").arg(root.runtimeStatus.performanceGuard.fpsTarget || 0)
                        .arg(Utils.formatNumber(root.runtimeStatus.performanceGuard.jankThresholdMs, 1))
                    : qsTr("Brak danych performance guard")
                wrapMode: Text.WordWrap
            }
        }

        ColumnLayout {
            spacing: 4
            Layout.fillWidth: true

            Label { text: qsTr("Odświeżanie ryzyka"); font.bold: true }
            Label {
                text: root.runtimeStatus.riskRefresh
                    ? qsTr("Co %1 s, kolejne za %2 s")
                        .arg(root.runtimeStatus.riskRefresh.intervalSeconds || 0)
                        .arg(root.runtimeStatus.riskRefresh.nextRefreshInSeconds || 0)
                    : qsTr("Brak zaplanowanych aktualizacji")
            }
        }
    }
}
