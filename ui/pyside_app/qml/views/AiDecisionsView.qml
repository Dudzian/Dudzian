import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "../components" as Components

Rectangle {
    id: root
    objectName: "aiDecisionsView"
    color: Qt.rgba(0, 0, 0, 0)
    radius: 18
    border.color: designSystem ? designSystem.color("border") : "#3C3F44"
    border.width: 1
    implicitHeight: 420

    property var runtimeService
    property var designSystem
    property var aiSnapshot: ({})
    property var lastDecision: aiSnapshot.lastDecision || ({})
    property var telemetry: aiSnapshot.telemetry || ({})
    property alias historyCount: historyList.count
    property string currentMode: lastDecision.mode || ""
    property real confidenceValue: lastDecision.confidence !== undefined ? Number(lastDecision.confidence) : 0.0

    signal snapshotUpdated()

    function refreshSnapshot() {
        aiSnapshot = runtimeService ? runtimeService.aiGovernorSnapshot || ({}) : ({})
        lastDecision = aiSnapshot.lastDecision || ({})
        telemetry = aiSnapshot.telemetry || ({})
        snapshotUpdated()
    }

    Component.onCompleted: refreshSnapshot()

    Connections {
        target: runtimeService
        function onAiGovernorSnapshotChanged() {
            root.refreshSnapshot()
        }
    }

    Rectangle {
        id: frostLayer
        anchors.fill: parent
        radius: root.radius
        gradient: Gradient {
            GradientStop { position: 0; color: designSystem ? designSystem.color("gradientHeroStart") : "#1f1f35" }
            GradientStop { position: 1; color: designSystem ? designSystem.color("gradientHeroEnd") : "#0d0d14" }
        }
        opacity: 0.6
        z: -2
    }

    MultiEffect {
        anchors.fill: frostLayer
        source: frostLayer
        blurEnabled: true
        blurRadius: 32
        saturation: 0.95
        brightness: 0.08
        z: -1
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Text {
                text: designSystem && designSystem.iconGlyph ? designSystem.iconGlyph("fingerprint") : "\uf0c2"
                font.pixelSize: 22
                font.bold: true
                font.family: designSystem && designSystem.fontAwesomeFamily ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                color: designSystem ? designSystem.color("accent") : "#5bc8ff"
            }
            ColumnLayout {
                Layout.fillWidth: true
                Text {
                    text: qsTr("Decyzje AI")
                    font.bold: true
                    font.pointSize: 16
                    color: designSystem ? designSystem.color("textPrimary") : "#fdfdfd"
                }
                Text {
                    text: lastDecision.reason ? lastDecision.reason : qsTr("Oczekiwanie na sygnał governora")
                    font.pointSize: 11
                    color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                    wrapMode: Text.WordWrap
                }
            }
            Components.IconButton {
                designSystem: designSystem
                subtle: true
                iconName: "refresh"
                text: qsTr("Odśwież")
                onClicked: {
                    if (runtimeService)
                        runtimeService.reloadAiGovernorSnapshot()
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 12
            color: Qt.rgba(0, 0, 0, 0.25)
            border.color: designSystem ? designSystem.color("border") : "#2d2f37"
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Text {
                        text: currentMode.length > 0 ? currentMode.toUpperCase() : qsTr("BRAK TRYBU")
                        font.bold: true
                        font.pointSize: 14
                        color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: qsTr("Confidence %1%2").arg(Math.round(confidenceValue * 100)).arg("%")
                        font.pointSize: 11
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                    }
                }

                ProgressBar {
                    Layout.fillWidth: true
                    value: Math.min(Math.max(confidenceValue, 0.0), 1.0)
                    from: 0
                    to: 1
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Text {
                        text: telemetry && telemetry.riskMetrics && telemetry.riskMetrics.risk_score !== undefined
                              ? qsTr("Ryzyko: %1").arg(Number(telemetry.riskMetrics.risk_score).toFixed(2))
                              : qsTr("Ryzyko: n/d")
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                    Text {
                        text: telemetry && telemetry.cycleMetrics && telemetry.cycleMetrics.cycle_latency_p95_ms !== undefined
                              ? qsTr("p95 cyklu: %1 ms").arg(Math.round(Number(telemetry.cycleMetrics.cycle_latency_p95_ms)))
                              : qsTr("p95 cyklu: n/d")
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                    Item { Layout.fillWidth: true }
                }
            }
        }

        Text {
            text: qsTr("Historia rekomendacji")
            font.bold: true
            font.pointSize: 13
            color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
        }

        ListView {
            id: historyList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 6
            model: aiSnapshot.history || []
            delegate: Rectangle {
                width: ListView.view.width
                height: implicitHeight
                radius: 10
                color: Qt.rgba(0, 0, 0, 0.2)
                border.color: designSystem ? designSystem.color("border") : "#2d2f37"
                border.width: 1
                implicitHeight: historyColumn.implicitHeight + 12

                ColumnLayout {
                    id: historyColumn
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 4
                    RowLayout {
                        spacing: 6
                        Text {
                            text: designSystem && designSystem.iconGlyph ? designSystem.iconGlyph("mode_wizard") : "\uf0c2"
                            font.family: designSystem && designSystem.fontAwesomeFamily ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                            font.pixelSize: 16
                            color: designSystem ? designSystem.color("accent") : "#5bc8ff"
                        }
                        Text {
                            text: modelData.mode ? modelData.mode.toUpperCase() : qsTr("n/d")
                            font.bold: true
                            color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: modelData.confidence !== undefined ? qsTr("%1% confidence").arg(Math.round(Number(modelData.confidence) * 100)) : ""
                            color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                            font.pointSize: 10
                        }
                    }
                    Text {
                        text: modelData.reason || qsTr("Brak uzasadnienia")
                        wrapMode: Text.WordWrap
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                }
            }
        }

        Text {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            visible: historyList.count === 0
            text: qsTr("Brak historii AI Governora")
            color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
        }
    }
}
