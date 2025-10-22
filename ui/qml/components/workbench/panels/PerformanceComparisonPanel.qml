import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "performanceComparisonPanel"
    property var performanceComparison: ({})

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
            text: qsTr("Porównanie wyników")
            font.pointSize: 15
            font.bold: true
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: root.performanceComparison.benchmarkName || qsTr("Benchmark")
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Label {
                text: qsTr("Od %1").arg(Utils.formatTimestamp(root.performanceComparison.since))
                color: palette.mid
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 3
            columnSpacing: 16
            rowSpacing: 6

            Label { text: qsTr("Strategia") }
            Label {
                text: Utils.formatPercent(root.performanceComparison.strategyReturn || 0, 2)
                font.bold: true
            }
            Item { }

            Label { text: qsTr("Benchmark") }
            Label {
                text: Utils.formatPercent(root.performanceComparison.benchmarkReturn || 0, 2)
                color: palette.mid
            }
            Item { }

            Label { text: qsTr("Nadwyżka") }
            Label {
                text: Utils.formatSignedPercent((root.performanceComparison.strategyReturn || 0) - (root.performanceComparison.benchmarkReturn || 0), 2)
                color: ((root.performanceComparison.strategyReturn || 0) - (root.performanceComparison.benchmarkReturn || 0)) >= 0 ? palette.highlight : palette.link
            }
            Item { }

            Label { text: qsTr("Alfa") }
            Label { text: Utils.formatPercent(root.performanceComparison.alpha || 0, 2) }
            Label { text: qsTr("Beta: %1").arg(Utils.formatNumber(root.performanceComparison.beta, 2)) }

            Label { text: qsTr("Sharpe") }
            Label { text: Utils.formatNumber(root.performanceComparison.sharpe, 2) }
            Label { text: qsTr("Sortino: %1").arg(Utils.formatNumber(root.performanceComparison.sortino, 2)) }

            Label { text: qsTr("Zmienność") }
            Label { text: Utils.formatPercent(root.performanceComparison.volatility || 0, 2) }
            Label { text: qsTr("Tracking error: %1").arg(Utils.formatPercent(root.performanceComparison.trackingError || 0, 2)) }

            Label { text: qsTr("Maks. obsunięcie") }
            Label { text: Utils.formatPercent(root.performanceComparison.maxDrawdown || 0, 2) }
            Label { text: qsTr("Aktualizacja: %1").arg(Utils.formatTimestamp(root.performanceComparison.updatedAt)) }
        }

        Label {
            visible: root.performanceComparison.notes && root.performanceComparison.notes.length > 0
            text: root.performanceComparison.notes
            color: palette.mid
            wrapMode: Text.WordWrap
        }
    }
}
