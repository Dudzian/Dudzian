import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "tradeHistoryPanel"
    property var tradeHistory: []

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
            text: qsTr("Historia transakcji")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: tradesList
            objectName: "tradeHistoryList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.tradeHistory || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 4

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Label {
                        text: Utils.formatTimestamp(modelData.timestamp)
                        color: palette.mid
                    }

                    Label {
                        text: (modelData.symbol || qsTr("Instrument")) +
                              (modelData.exchange ? qsTr(" @ %1").arg(modelData.exchange) : "")
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: modelData.status || ""
                        color: {
                            const statusText = (modelData.status || "").toString().toLowerCase()
                            return statusText.indexOf("fill") !== -1 ? palette.highlight : palette.mid
                        }
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6

                    Label { text: qsTr("Kierunek") }
                    Label {
                        text: modelData.side || ""
                        color: {
                            const sideText = (modelData.side || "").toString().toLowerCase()
                            return sideText.indexOf("sell") !== -1 || sideText.indexOf(qsTr("Sprzedaż").toLowerCase()) !== -1
                                ? palette.negative
                                : palette.highlight
                        }
                        font.bold: true
                    }

                    Label { text: qsTr("Wolumen") }
                    Label { text: Utils.formatNumber(modelData.quantity, 4) }

                    Label { text: qsTr("Cena wykonania") }
                    Label { text: Utils.formatNumber(modelData.price, 2) }

                    Label { text: qsTr("P/L zreal.") }
                    Label {
                        text: Utils.formatNumber(modelData.pnl, 2)
                        color: modelData.pnl >= 0 ? palette.highlight : palette.negative
                    }

                    Label { text: qsTr("Prowizja") }
                    Label { text: Utils.formatNumber(modelData.fee, 2) }

                    Label { text: qsTr("Zlecenie") }
                    Label {
                        text: modelData.orderId || "–"
                        color: palette.mid
                    }

                    Label { text: qsTr("Notatki") }
                    Label {
                        text: Utils.formatText(modelData.notes, "–")
                        wrapMode: Text.WordWrap
                        maximumLineCount: 2
                        color: palette.buttonText
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
            visible: (root.tradeHistory || []).length === 0
            text: qsTr("Brak transakcji do wyświetlenia.")
            color: palette.mid
        }
    }
}
