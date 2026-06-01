import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiDecisionsView"
    property var runtimeService
    property var previewState
    property var aiSnapshot: runtimeService && runtimeService.aiGovernorSnapshot ? runtimeService.aiGovernorSnapshot : ({})
    property var lastDecision: aiSnapshot.lastDecision || ({})
    property var decisionTimeline: aiSnapshot.history || []
    property var recommendedModes: lastDecision.recommendedModes || []
    property int timelineCount: decisionTimeline.length
    property int recommendationCount: recommendedModes.length
    property string currentMode: lastDecision.mode || "preview"
    contentWidth: availableWidth
    clip: true

    function countAction(fragment) {
        var count = 0
        for (var i = 0; i < previewState.decisionPreviewRows.length; ++i) {
            if (previewState.decisionPreviewRows[i].action.indexOf(fragment) >= 0)
                count += 1
        }
        return count
    }

    function refreshSnapshot() {
        aiSnapshot = runtimeService && runtimeService.aiGovernorSnapshot ? runtimeService.aiGovernorSnapshot : ({})
        lastDecision = aiSnapshot.lastDecision || ({})
        decisionTimeline = aiSnapshot.history || []
        recommendedModes = lastDecision.recommendedModes || []
        timelineCount = decisionTimeline.length
        recommendationCount = recommendedModes.length
        currentMode = lastDecision.mode || "preview"
    }

    Component.onCompleted: refreshSnapshot()
    onRuntimeServiceChanged: refreshSnapshot()

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        Label { objectName: "aiDecisionsTitle"; text: qsTr("Decyzje"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Dynamiczny local decision stream reaguje na paper simulation i Generate governor recommendation. Lista nie jest pusta i nie używa backendu."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("blocked live count"); description: String(root.countAction("BLOCKED LIVE")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("paper simulated orders count"); description: String(root.countAction("PAPER")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("no order count"); description: String(root.countAction("NO ORDER")); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("AI / Governor decision preview")
            description: qsTr("Każdy rekord: Timestamp, symbol, action, confidence, reason, Risk reason, Strategy source, Safety block state, paper/session state.")
            RowLayout {
                Layout.fillWidth: true
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate next decision"); iconName: "mode_wizard"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.generateNextDecision() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); subtle: true; onClicked: previewState.generateGovernorRecommendation() }
                Label { text: qsTr("Paper session state: %1").arg(previewState.paperSessionState); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
            }
            Repeater {
                model: previewState.decisionPreviewRows
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: decisionColumn.implicitHeight + 22
                    radius: 14
                    color: designSystem.color("surfaceMuted")
                    border.color: modelData.action === "BLOCKED LIVE" ? designSystem.color("critical") : designSystem.color("border")
                    border.width: 1
                    ColumnLayout {
                        id: decisionColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Label { text: modelData.symbol; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                            Label { text: qsTr("action: %1").arg(modelData.action); color: modelData.action === "BLOCKED LIVE" ? designSystem.color("warning") : designSystem.color("accent"); font.bold: true }
                            Label { text: qsTr("confidence %1").arg(modelData.confidence); color: designSystem.color("textSecondary") }
                            Label { text: qsTr("Timestamp %1").arg(modelData.timestamp); color: designSystem.color("textSecondary") }
                        }
                        Label { text: qsTr("reason: %1").arg(modelData.reason); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Risk reason: %1").arg(modelData.riskReason); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Strategy source: %1 • Safety block state: %2 • paper/session state: %3").arg(modelData.strategy).arg(modelData.safety).arg(modelData.paperState); color: designSystem.color("textPrimary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
