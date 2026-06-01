import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiControlCenterRoot"
    property var previewState
    contentWidth: availableWidth
    clip: true
    property var autonomyModes: ["Advisory", "Supervised dry-run", "Autonomous paper", "Research mode"]
    property var policyModes: ["conservative", "balanced", "aggressive"]
    property var modules: [
        ({ name: "Market scanner", status: "ready" }),
        ({ name: "Signal scorer", status: "ready" }),
        ({ name: "Strategy governor", status: "guarded" }),
        ({ name: "Risk governor", status: "guarded" }),
        ({ name: "Execution guard", status: "blocked live" }),
        ({ name: "Recovery monitor", status: "preview-only" }),
        ({ name: "Telemetry monitor", status: "preview-only" }),
        ({ name: "Kill-switch", status: "blocked live" })
    ]

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        Label { objectName: "aiControlCenterTitle"; text: qsTr("AI Center / Centrum autonomii"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Operacyjny AI/Governor Center: autonomy mode selector, readiness, coverage, confidence threshold i local governor recommendation. Zero backendu, zero live, zero exchange I/O."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 4 : 2
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model identity"); description: qsTr("Decision Governor Preview Core"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model family/type"); description: qsTr("heuristic/governor/ML-ready preview"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model version/build"); description: qsTr("preview-build 7.2-local"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Current autonomy mode • Autonomy level"); description: qsTr("%1 / 5 • mode %2").arg(previewState.autonomyLevel).arg(previewState.autonomyMode); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Training/readiness percent • Model readiness"); description: qsTr("Model readiness %1%").arg(previewState.modelReadiness); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Training coverage %"); description: qsTr("Training coverage %1%").arg(previewState.trainingCoverage); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Data coverage percent • Data coverage %"); description: qsTr("Data coverage %1%").arg(previewState.dataCoverage); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Safety state"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled • Runtime loop not started"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Autonomy mode selector")
            description: qsTr("Advisory, Supervised dry-run, Autonomous paper, Research mode — zmienia tylko lokalny UI state.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.autonomyModes
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.autonomyMode === modelData
                        width: Math.max(160, modeLabel.implicitWidth + 28)
                        height: 42
                        radius: 12
                        color: active ? Qt.rgba(0.33, 0.78, 1.0, 0.20) : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        Label { id: modeLabel; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: active }
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: previewState.setAutonomyMode(modelData) }
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Confidence threshold slider/styled control"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                Label { text: previewState.confidenceThreshold + "%"; color: designSystem.color("textPrimary"); font.bold: true }
            }
            Slider {
                id: confidenceSlider
                Layout.fillWidth: true
                from: 40
                to: 95
                value: previewState.confidenceThreshold
                onMoved: previewState.confidenceThreshold = Math.round(value)
                background: Rectangle { x: confidenceSlider.leftPadding; y: confidenceSlider.topPadding + confidenceSlider.availableHeight / 2 - height / 2; width: confidenceSlider.availableWidth; height: 8; radius: 4; color: designSystem.color("surfaceMuted"); Rectangle { width: confidenceSlider.visualPosition * parent.width; height: parent.height; radius: 4; color: designSystem.color("accent") } }
                handle: Rectangle { x: confidenceSlider.leftPadding + confidenceSlider.visualPosition * (confidenceSlider.availableWidth - width); y: confidenceSlider.topPadding + confidenceSlider.availableHeight / 2 - height / 2; width: 22; height: 22; radius: 11; color: designSystem.color("textPrimary"); border.color: designSystem.color("accent"); border.width: 2 }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Decision policy preview")
            description: qsTr("conservative / balanced / aggressive")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.policyModes
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.decisionPolicyPreview === modelData
                        width: 150
                        height: 40
                        radius: 12
                        color: active ? Qt.rgba(0.33, 0.78, 1.0, 0.20) : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        Label { anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: active }
                        MouseArea { anchors.fill: parent; onClicked: previewState.setDecisionPolicy(modelData) }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("AI modules")
            description: qsTr("ready / guarded / preview-only / blocked live")
            GridLayout {
                Layout.fillWidth: true
                columns: width > 900 ? 4 : 2
                rowSpacing: 8
                columnSpacing: 8
                Repeater {
                    model: root.modules
                    delegate: Rectangle {
                        required property var modelData
                        Layout.fillWidth: true
                        implicitHeight: 56
                        radius: 12
                        color: designSystem.color("surfaceMuted")
                        border.color: modelData.status === "blocked live" ? designSystem.color("critical") : designSystem.color("border")
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            Label { text: modelData.name; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                            Label { text: modelData.status; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                        }
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); iconName: "mode_wizard"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.generateGovernorRecommendation() }
                Label { text: previewState.lastGovernorDecision; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }
    }
}
