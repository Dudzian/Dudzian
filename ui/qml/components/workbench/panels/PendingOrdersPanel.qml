import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "pendingOrdersPanel"
    property var pendingOrders: []

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
            text: qsTr("Oczekujące zlecenia")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: pendingOrdersList
            objectName: "pendingOrdersList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.pendingOrders || []

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
                        text: modelData.side || ""
                        color: {
                            const sideText = (modelData.side || "").toString().toLowerCase()
                            if (sideText.indexOf("short") !== -1)
                                return palette.negative
                            if (sideText.indexOf("sell") !== -1)
                                return palette.negative
                            return palette.highlight
                        }
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: modelData.type ? qsTr("Typ: %1").arg(modelData.type) : ""
                        color: palette.mid
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6

                    Label { text: qsTr("ID") }
                    Label {
                        text: modelData.clientOrderId || modelData.id || "–"
                        font.family: "Monospace"
                    }

                    Label { text: qsTr("Wolumen") }
                    Label { text: Utils.formatNumber(modelData.quantity, 4) }

                    Label { text: qsTr("Wypełniono") }
                    Label { text: Utils.formatNumber(modelData.filledQuantity, 4) }

                    Label { text: qsTr("Pozostało") }
                    Label { text: Utils.formatNumber(modelData.remainingQuantity, 4) }

                    Label { text: qsTr("Cena") }
                    Label { text: Utils.formatNumber(modelData.price, 2) }

                    Label { text: qsTr("Śr. cena") }
                    Label { text: Utils.formatNumber(modelData.averagePrice, 2) }

                    Label { text: qsTr("Status") }
                    Label {
                        text: Utils.formatText(modelData.status)
                        font.bold: true
                    }

                    Label { text: qsTr("TIF") }
                    Label { text: Utils.formatText(modelData.timeInForce) }

                    Label { text: qsTr("Reduce only") }
                    Label { text: Utils.formatBoolean(modelData.reduceOnly) }

                    Label { text: qsTr("Post only") }
                    Label { text: Utils.formatBoolean(modelData.postOnly) }

                    Label { text: qsTr("Utworzono") }
                    Label {
                        text: Utils.formatTimestamp(modelData.createdAt)
                        color: palette.mid
                    }

                    Label { text: qsTr("Aktualizacja") }
                    Label {
                        text: Utils.formatTimestamp(modelData.updatedAt || modelData.expiresAt)
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
            visible: (root.pendingOrders || []).length === 0
            text: qsTr("Brak aktywnych zleceń oczekujących.")
            color: palette.mid
        }
    }
}
