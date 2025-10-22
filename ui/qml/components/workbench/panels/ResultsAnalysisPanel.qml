import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "resultsPanel"
    property var portfolioSummary: ({})
    property var riskSnapshot: ({})

    Layout.fillWidth: true
    Layout.fillHeight: true
    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            text: qsTr("Analiza wyników")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 8
            Layout.fillWidth: true

            Label { text: qsTr("Profil"); font.bold: true }
            Label { text: root.portfolioSummary.profileLabel || qsTr("Live") }

            Label { text: qsTr("Wartość portfela"); font.bold: true }
            Label { text: Utils.formatNumber(root.portfolioSummary.latestValue, 2) }

            Label { text: qsTr("Liczba próbek"); font.bold: true }
            Label { text: (root.portfolioSummary.entryCount || 0).toString() }

            Label { text: qsTr("Min / Max"); font.bold: true }
            Label {
                text: Utils.formatNumber(root.portfolioSummary.minValue, 2) +
                      " / " + Utils.formatNumber(root.portfolioSummary.maxValue, 2)
            }

            Label { text: qsTr("Maksymalne DD"); font.bold: true }
            Label { text: Utils.formatPercent(root.portfolioSummary.maxDrawdown) }

            Label { text: qsTr("Średnie DD"); font.bold: true }
            Label { text: Utils.formatPercent(root.portfolioSummary.averageDrawdown) }

            Label { text: qsTr("Dźwignia"); font.bold: true }
            Label {
                text: Utils.formatNumber(root.portfolioSummary.averageLeverage, 2) +
                      qsTr(" (max %1)").arg(Utils.formatNumber(root.portfolioSummary.maxLeverage, 2))
            }

            Label { text: qsTr("Maks. ekspozycja"); font.bold: true }
            Label { text: Utils.formatPercent(root.portfolioSummary.maxExposureUtilization) }

            Label { text: qsTr("Naruszenia"); font.bold: true }
            Label {
                text: (root.portfolioSummary.totalBreaches || 0) +
                      (root.portfolioSummary.anyBreach ? qsTr(" (wystąpiły)") : qsTr(" (brak)"))
            }
        }

        ColumnLayout {
            spacing: 8
            Layout.fillWidth: true

            Label {
                text: qsTr("Ekspozycje ryzyka (%1)").arg(root.riskSnapshot.profileLabel || qsTr("n/d"))
                font.bold: true
            }

            Label {
                text: root.riskSnapshot.generatedAt
                    ? qsTr("Stan na %1").arg(root.riskSnapshot.generatedAt)
                    : ""
                color: palette.mid
            }

            ListView {
                id: exposureList
                objectName: "exposureList"
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(contentHeight, 180)
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { }
                model: root.riskSnapshot.exposures || []

                delegate: RowLayout {
                    width: ListView.view ? ListView.view.width : parent.width
                    spacing: 12

                    Label {
                        text: modelData.code || "?"
                        Layout.fillWidth: true
                        font.bold: true
                    }

                    Label {
                        text: Utils.formatPercent(modelData.current)
                        Layout.alignment: Qt.AlignRight
                    }

                    Label {
                        text: Utils.formatPercent(modelData.max)
                        Layout.alignment: Qt.AlignRight
                    }

                    Label {
                        text: Utils.formatPercent(modelData.threshold)
                        Layout.alignment: Qt.AlignRight
                        color: modelData.breach ? palette.negative : palette.mid
                    }
                }
            }

            Label {
                visible: !(root.riskSnapshot.exposures && root.riskSnapshot.exposures.length)
                text: qsTr("Brak danych ekspozycji.")
                color: palette.mid
            }
        }
    }
}
