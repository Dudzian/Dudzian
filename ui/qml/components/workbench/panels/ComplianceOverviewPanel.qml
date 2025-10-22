import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "complianceOverviewPanel"
    property var complianceSummary: ({})

    readonly property real scoreValue: {
        const summary = root.complianceSummary || {}
        const value = Number(summary.complianceScore)
        if (!isFinite(value))
            return 0
        if (value < 0)
            return 0
        if (value > 1)
            return 1
        return value
    }

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
            text: qsTr("Zgodność operacyjna")
            font.pointSize: 15
            font.bold: true
        }

        RowLayout {
            spacing: 12
            Layout.fillWidth: true

            ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: root.scoreValue
            }

            Label {
                text: Utils.formatPercent(root.scoreValue, 0)
                font.bold: true
                Layout.alignment: Qt.AlignVCenter
            }
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Licencja aktywna"); font.bold: true }
            Label { text: Utils.formatBoolean(root.complianceSummary.licenseActive) }

            Label { text: qsTr("Alerty sygnałowe"); font.bold: true }
            Label { text: (root.complianceSummary.openAlerts !== undefined ? root.complianceSummary.openAlerts : 0).toString() }

            Label { text: qsTr("Naruszenia ryzyka"); font.bold: true }
            Label { text: (root.complianceSummary.outstandingBreaches !== undefined ? root.complianceSummary.outstandingBreaches : 0).toString() }

            Label { text: qsTr("Ostatnie naruszenie"); font.bold: true }
            Label { text: Utils.formatTimestamp(root.complianceSummary.lastBreachAt) }

            Label { text: qsTr("Ostatnia ocena ryzyka"); font.bold: true }
            Label { text: Utils.formatTimestamp(root.complianceSummary.lastRiskAssessmentAt) }

            Label { text: qsTr("Następna ocena"); font.bold: true }
            Label { text: Utils.formatTimestamp(root.complianceSummary.nextRiskReviewAt) }

            Label { text: qsTr("Automatyzacja wstrzymana"); font.bold: true }
            Label { text: Utils.formatBoolean(root.complianceSummary.automationPaused) }
        }

        ColumnLayout {
            spacing: 4
            Layout.fillWidth: true

            Repeater {
                model: Array.isArray(root.complianceSummary.notes) ? root.complianceSummary.notes : []

                delegate: Label {
                    text: "• " + Utils.formatText(modelData, qsTr("Brak informacji"))
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}
