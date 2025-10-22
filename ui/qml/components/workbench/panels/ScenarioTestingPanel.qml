import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "scenarioTestingPanel"
    property var scenarioTests: []

    readonly property int scenarioCount: Array.isArray(root.scenarioTests) ? root.scenarioTests.length : 0

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
            text: qsTr("Testy scenariuszy i stress testy")
            font.pointSize: 15
            font.bold: true
        }

        Label {
            visible: root.scenarioCount === 0
            text: qsTr("Brak zdefiniowanych scenariuszy do weryfikacji.")
            color: Qt.lighter(palette.text, 1.6)
            wrapMode: Text.WordWrap
        }

        Repeater {
            model: root.scenarioCount === 0 ? [] : root.scenarioTests

            delegate: Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: Qt.darker(palette.base, 1.02)
                    radius: 6
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: Utils.formatText(modelData.name, qsTr("Scenariusz"))
                            font.bold: true
                            Layout.fillWidth: true
                        }

                        Label {
                            visible: !!modelData.severity
                            text: Utils.formatText(modelData.severity, "")
                            color: "#546E7A"
                            font.capitalization: Font.Capitalize
                        }

                        Label {
                            text: Utils.formatText(modelData.status, qsTr("Brak statusu"))
                            color: modelData.success === false ? "#C62828" : "#2E7D32"
                            font.bold: true
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Label {
                            text: qsTr("Uruchomień: %1").arg(modelData.runCount !== undefined ? modelData.runCount : 0)
                        }

                        Label {
                            text: qsTr("Ostatnie uruchomienie: %1").arg(Utils.formatTimestamp(modelData.lastRunAt))
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                        }
                    }

                    GridLayout {
                        columns: 2
                        columnSpacing: 12
                        rowSpacing: 6
                        Layout.fillWidth: true

                        Label { text: qsTr("Wpływ PnL"); font.bold: true }
                        Label { text: Utils.formatSignedPercent(modelData.pnlImpact, 2) }

                        Label { text: qsTr("Maks. obsunięcie"); font.bold: true }
                        Label { text: Utils.formatPercent(modelData.maxDrawdown, 2) }

                        Label { text: qsTr("Zużycie depozytu"); font.bold: true }
                        Label { text: Utils.formatPercent(modelData.marginUsage, 2) }

                        Label { text: qsTr("Wpływ płynności"); font.bold: true }
                        Label { text: Utils.formatPercent(modelData.liquidityImpact, 2) }

                        Label { text: qsTr("Czas trwania"); font.bold: true }
                        Label { text: Utils.formatDuration(modelData.durationSeconds) }

                        Label { text: qsTr("Zakończono"); font.bold: true }
                        Label { text: Utils.formatTimestamp(modelData.completedAt) }
                    }

                    Label {
                        visible: !!modelData.description
                        text: modelData.description
                        wrapMode: Text.WordWrap
                    }

                    ColumnLayout {
                        visible: Array.isArray(modelData.notes) && modelData.notes.length > 0
                        spacing: 2

                        Repeater {
                            model: Array.isArray(modelData.notes) ? modelData.notes : []

                            delegate: Label {
                                text: "• " + Utils.formatText(modelData, qsTr("Brak notatek"))
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }
            }
        }
    }
}
