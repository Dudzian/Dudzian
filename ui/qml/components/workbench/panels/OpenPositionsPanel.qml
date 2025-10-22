import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "openPositionsPanel"
    property var openPositions: []

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
            text: qsTr("Otwarte pozycje")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: positionsList
            objectName: "openPositionsList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.openPositions || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Label {
                        text: (modelData.symbol || qsTr("Instrument")) +
                              (modelData.exchange ? qsTr(" @ %1").arg(modelData.exchange) : "")
                        font.bold: true
                    }

                    Label {
                        text: modelData.side || (modelData.signedQuantity < 0
                            ? qsTr("Short")
                            : (modelData.signedQuantity > 0 ? qsTr("Long") : ""))
                        color: {
                            const sideText = (modelData.side || "").toString().toLowerCase()
                            return sideText.indexOf("short") !== -1 ? palette.negative : palette.highlight
                        }
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: modelData.account ? qsTr("Konto: %1").arg(modelData.account) : ""
                        color: palette.mid
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6

                    Label { text: qsTr("Wolumen") }
                    Label {
                        text: Utils.formatNumber(modelData.quantity, 4)
                        font.bold: true
                    }

                    Label { text: qsTr("Cena wejścia") }
                    Label { text: Utils.formatNumber(modelData.entryPrice, 2) }

                    Label { text: qsTr("Cena rynkowa") }
                    Label { text: Utils.formatNumber(modelData.markPrice, 2) }

                    Label { text: qsTr("Dźwignia") }
                    Label { text: Utils.formatNumber(modelData.leverage, 2) }

                    Label { text: qsTr("Marża") }
                    Label { text: Utils.formatNumber(modelData.margin, 2) }

                    Label { text: qsTr("P/L niezreal.") }
                    Label {
                        text: Utils.formatNumber(modelData.unrealizedPnl, 2)
                        color: modelData.unrealizedPnl >= 0 ? palette.highlight : palette.negative
                        font.bold: true
                    }

                    Label { text: qsTr("Ostatnia aktualizacja") }
                    Label {
                        text: Utils.formatTimestamp(modelData.lastUpdate)
                        color: palette.mid
                    }
                }

                Rectangle {
                    visible: index < (ListView.view.count - 1)
                    height: 1
                    color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.25)
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                }
            }
        }

        Label {
            visible: (root.openPositions || []).length === 0
            text: qsTr("Brak aktywnych pozycji do wyświetlenia.")
            color: palette.mid
        }
    }
}
