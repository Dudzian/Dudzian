import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "automationRulesPanel"
    property var automationRules: []

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
            text: qsTr("Reguły automatyzacji")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: rulesList
            objectName: "automationRulesList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.automationRules || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label {
                        text: modelData.name || qsTr("Reguła automatyczna")
                        font.bold: true
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }

                    Rectangle {
                        property alias hovered: hoverHandler.hovered
                        width: 12
                        height: 12
                        radius: 6
                        color: modelData.enabled ? palette.highlight : palette.mid
                        border.color: palette.windowText
                        border.width: modelData.critical ? 2 : 1
                        ToolTip.visible: hovered
                        ToolTip.text: modelData.enabled ? qsTr("Aktywna") : qsTr("Nieaktywna")
                        HoverHandler { id: hoverHandler }
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6

                    Label { text: qsTr("Typ") }
                    Label { text: Utils.formatText(modelData.type) }

                    Label { text: qsTr("Warunek") }
                    Label {
                        text: Utils.formatText(modelData.trigger)
                        wrapMode: Text.Wrap
                    }

                    Label { text: qsTr("Akcja") }
                    Label {
                        text: Utils.formatText(modelData.action)
                        wrapMode: Text.Wrap
                    }

                    Label { text: qsTr("Skuteczność") }
                    Label { text: Utils.formatPercent(modelData.successRate, 1) }

                    Label { text: qsTr("Błędy") }
                    Label { text: Utils.formatNumber(modelData.errorCount, 0) }

                    Label { text: qsTr("Ostatnie uruchomienie") }
                    Label {
                        text: Utils.formatTimestamp(modelData.lastTriggeredAt)
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

                Label {
                    visible: Utils.formatText(modelData.description).length > 1
                    text: Utils.formatText(modelData.description)
                    wrapMode: Text.Wrap
                    color: palette.mid
                }

                Rectangle {
                    visible: index < (ListView.view.count - 1)
                    height: 1
                    color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.25)
                    Layout.fillWidth: true
                    Layout.topMargin: 6
                }
            }
        }

        Label {
            visible: !rulesList.count
            text: qsTr("Brak zdefiniowanych reguł automatyzacji.")
            color: palette.mid
            horizontalAlignment: Text.AlignHCenter
            Layout.alignment: Qt.AlignHCenter
        }
    }
}
