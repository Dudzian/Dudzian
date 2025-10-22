import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "executionDiagnosticsPanel"
    property var executionDiagnostics: ({})

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
            text: qsTr("Diagnostyka wykonania zleceń")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            id: metricsGrid
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 16
            rowSpacing: 6

            Label { text: qsTr("Dostawca") }
            Label {
                text: Utils.formatText(root.executionDiagnostics.provider)
                font.bold: true
            }

            Label { text: qsTr("Średnie opóźnienie") }
            Label { text: Utils.formatNumber(root.executionDiagnostics.avgLatencyMs, 1) + qsTr(" ms") }

            Label { text: qsTr("P95 opóźnienia") }
            Label { text: Utils.formatNumber(root.executionDiagnostics.p95LatencyMs, 1) + qsTr(" ms") }

            Label { text: qsTr("Maks. opóźnienie") }
            Label { text: Utils.formatNumber(root.executionDiagnostics.maxLatencyMs, 1) + qsTr(" ms") }

            Label { text: qsTr("Wskaźnik realizacji") }
            Label { text: Utils.formatPercent(root.executionDiagnostics.fillRate, 2) }

            Label { text: qsTr("Wskaźnik odrzuceń") }
            Label { text: Utils.formatPercent(root.executionDiagnostics.rejectRate, 2) }

            Label { text: qsTr("Poślizg (bps)") }
            Label { text: Utils.formatNumber(root.executionDiagnostics.slippageBps, 2) }

            Label { text: qsTr("Aktualizacja") }
            Label {
                text: Utils.formatTimestamp(root.executionDiagnostics.lastUpdated)
                color: palette.mid
            }
        }

        Label {
            visible: Utils.formatText(root.executionDiagnostics.notes).length > 1
            text: Utils.formatText(root.executionDiagnostics.notes)
            wrapMode: Text.Wrap
            color: palette.mid
        }

        GroupBox {
            title: qsTr("Niedawne incydenty")
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: incidentsList
                objectName: "executionIncidentsList"
                anchors.fill: parent
                clip: true
                spacing: 8
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { }
                model: root.executionDiagnostics.recentIncidents || []

                delegate: ColumnLayout {
                    width: ListView.view ? ListView.view.width : parent.width
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: modelData.type || qsTr("incydent")
                            font.bold: true
                            color: modelData.resolved ? palette.highlight : palette.negative
                        }

                        Label {
                            text: Utils.formatTimestamp(modelData.timestamp)
                            color: palette.mid
                            Layout.alignment: Qt.AlignRight
                            Layout.fillWidth: true
                        }
                    }

                    Label {
                        text: Utils.formatText(modelData.message)
                        wrapMode: Text.Wrap
                    }

                    Rectangle {
                        visible: index < (ListView.view.count - 1)
                        height: 1
                        color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.25)
                        Layout.fillWidth: true
                        Layout.topMargin: 4
                    }
                }
            }
        }

        Label {
            visible: !(root.executionDiagnostics.recentIncidents && root.executionDiagnostics.recentIncidents.length)
            text: qsTr("Brak incydentów wykonania do wyświetlenia.")
            color: palette.mid
            horizontalAlignment: Text.AlignHCenter
            Layout.alignment: Qt.AlignHCenter
        }
    }
}
