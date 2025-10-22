import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "capitalAllocationPanel"
    property var capitalAllocation: []

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
            text: qsTr("Alokacja kapitału")
            font.pointSize: 15
            font.bold: true
        }

        Label {
            visible: (root.capitalAllocation || []).length === 0
            text: qsTr("Brak danych o alokacji kapitału")
            color: palette.mid
        }

        ListView {
            id: allocationList
            objectName: "capitalAllocationList"
            visible: (root.capitalAllocation || []).length > 0
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(220, contentHeight)
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}
            model: root.capitalAllocation || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label {
                        text: modelData.segment || qsTr("Segment")
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: Utils.formatPercent(modelData.weight || 0, 1)
                        font.bold: true
                    }
                }

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 1
                    value: Math.min(1, Math.max(0, modelData.weight || 0))
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 4

                    Label { text: qsTr("Cel") }
                    Label {
                        text: Utils.formatPercent(modelData.targetWeight || 0, 1)
                        color: palette.mid
                    }

                    Label { text: qsTr("Odchylenie") }
                    Label {
                        text: Utils.formatSignedPercent(modelData.deltaWeight || ((modelData.weight || 0) - (modelData.targetWeight || 0)), 1)
                        color: (modelData.deltaWeight || ((modelData.weight || 0) - (modelData.targetWeight || 0))) >= 0 ? palette.highlight : palette.link
                    }

                    Label { text: qsTr("Notional") }
                    Label { text: Utils.formatNumber(modelData.notional || modelData.currentValue, 0) }

                    Label { text: qsTr("Hedge") }
                    Label { text: Utils.formatBoolean(modelData.hedged) }

                    Label { text: qsTr("Dźwignia") }
                    Label { text: Utils.formatNumber(modelData.leverage, 2) }
                }

                Label {
                    visible: modelData.notes && modelData.notes.length > 0
                    text: modelData.notes
                    color: palette.mid
                    wrapMode: Text.WordWrap
                }

                Rectangle {
                    visible: index < (ListView.view.count - 1)
                    Layout.topMargin: 8
                    Layout.fillWidth: true
                    height: 1
                    color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.15)
                }
            }
        }
    }
}
