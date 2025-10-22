import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "strategyControlPanel"

    property var controlState: ({})

    signal startSchedulerRequested()
    signal stopSchedulerRequested()
    signal riskRefreshRequested()

    Layout.fillWidth: true
    Layout.columnSpan: 2

    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            text: qsTr("Sterowanie strategią")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Harmonogram aktywny"); font.bold: true }
            Label { text: Utils.formatBoolean(controlState.schedulerRunning) }

            Label { text: qsTr("Automatyzacja offline"); font.bold: true }
            Label { text: Utils.formatBoolean(controlState.automationRunning) }

            Label { text: qsTr("Tryb offline"); font.bold: true }
            Label { text: Utils.formatBoolean(controlState.offlineMode) }

            Label { text: qsTr("Ostatnia akcja"); font.bold: true }
            Label {
                text: Utils.formatText(controlState.lastActionMessage, qsTr("Brak"))
                wrapMode: Text.WordWrap
                color: controlState.lastActionSuccess === false ? "#C0392B" : palette.text
            }

            Label { text: qsTr("Czas ostatniej akcji"); font.bold: true }
            Label { text: Utils.formatTimestamp(controlState.lastActionAt) }

            Label { text: qsTr("Ostatnie odświeżenie ryzyka"); font.bold: true }
            Label { text: Utils.formatTimestamp(controlState.lastRiskRefreshAt) }

            Label { text: qsTr("Następne odświeżenie ryzyka"); font.bold: true }
            Label { text: Utils.formatTimestamp(controlState.nextRiskRefreshDueAt) }

            Label { text: qsTr("Wymuszone odświeżenia"); font.bold: true }
            Label { text: Utils.formatNumber(controlState.manualRefreshCount, 0) }
        }

        RowLayout {
            spacing: 12
            Layout.fillWidth: true

            Button {
                text: qsTr("Uruchom harmonogram")
                enabled: !controlState.schedulerRunning
                onClicked: root.startSchedulerRequested()
            }

            Button {
                text: qsTr("Zatrzymaj harmonogram")
                enabled: controlState.schedulerRunning
                onClicked: root.stopSchedulerRequested()
            }

            Button {
                text: qsTr("Odśwież ryzyko teraz")
                Layout.alignment: Qt.AlignLeft
                onClicked: root.riskRefreshRequested()
            }
        }
    }
}
