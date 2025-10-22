import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "signalAlertsPanel"
    property var signalAlerts: []

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
            text: qsTr("Sygnały i alerty")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: signalAlertsList
            objectName: "signalAlertsList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.signalAlerts || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Label {
                        text: modelData.message || qsTr("Sygnał strategii")
                        font.bold: true
                        wrapMode: Text.WrapAnywhere
                        Layout.fillWidth: true
                    }

                    Label {
                        text: modelData.direction || ""
                        color: {
                            const direction = (modelData.direction || "").toString().toLowerCase()
                            if (direction.indexOf("short") !== -1)
                                return palette.negative
                            if (direction.indexOf("sell") !== -1)
                                return palette.negative
                            return palette.highlight
                        }
                        font.bold: true
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6

                    Label { text: qsTr("Symbol") }
                    Label {
                        text: modelData.symbol || "–"
                        font.bold: true
                    }

                    Label { text: qsTr("Kategoria") }
                    Label { text: Utils.formatText(modelData.category) }

                    Label { text: qsTr("Konf.") }
                    Label { text: Utils.formatNumber(modelData.confidence, 2) }

                    Label { text: qsTr("Wpływ") }
                    Label { text: Utils.formatNumber(modelData.impact, 2) }

                    Label { text: qsTr("Priorytet") }
                    Label { text: Utils.formatText(modelData.priority) }

                    Label { text: qsTr("Utworzono") }
                    Label {
                        text: Utils.formatTimestamp(modelData.generatedAt)
                        color: palette.mid
                    }

                    Label { text: qsTr("Wygasa") }
                    Label {
                        text: Utils.formatTimestamp(modelData.expiresAt)
                        color: palette.mid
                    }

                    Label { text: qsTr("Tagi") }
                    Label {
                        text: Array.isArray(modelData.tags) && modelData.tags.length
                            ? modelData.tags.join(", ")
                            : "–"
                        wrapMode: Text.Wrap
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
            visible: (root.signalAlerts || []).length === 0
            text: qsTr("Brak aktywnych sygnałów.")
            color: palette.mid
        }
    }
}
