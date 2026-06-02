import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiDecisionsPreviewPanel"
    property var runtimeService
    property var previewState
    property var governorSnapshot: ({})
    property int timelineCount: root.hasHistoryRowsSource() ? root.historyRows().length : root.filteredDecisionRows().length
    property int recommendationCount: root.recommendedModes().length
    property string currentMode: root.lastDecisionValue("mode", "")
    contentWidth: availableWidth
    clip: true

    function decisionRows() {
        if (!root.previewState || !root.previewState.decisionPreviewRows) return []
        return root.previewState.decisionPreviewRows
    }
    function hasHistoryRowsSource() {
        return root.governorSnapshot && root.governorSnapshot.history !== undefined && root.governorSnapshot.history !== null
    }
    function historyRows() {
        if (!root.hasHistoryRowsSource()) return []
        return root.governorSnapshot.history
    }
    function lastDecisionValue(key, fallback) {
        if (!root.governorSnapshot || !root.governorSnapshot.lastDecision) return fallback
        var value = root.governorSnapshot.lastDecision[key]
        return value === undefined || value === null ? fallback : value
    }
    function recommendedModes() {
        return root.lastDecisionValue("recommendedModes", [])
    }
    function refreshAiGovernorSnapshot() {
        if (!root.runtimeService || !root.runtimeService.aiGovernorSnapshot) {
            root.governorSnapshot = {}
            return
        }
        root.governorSnapshot = root.runtimeService.aiGovernorSnapshot
    }
    function filteredDecisionRows() {
        var out = []
        var rows = root.decisionRows()
        var decisionFilter = root.previewState && root.previewState.decisionFilter ? root.previewState.decisionFilter : "all"
        var decisionPairFilter = root.previewState && root.previewState.decisionPairFilter ? root.previewState.decisionPairFilter : "All pairs"
        for (var i = 0; i < rows.length; ++i) {
            var row = rows[i]
            if (decisionFilter === "paper" && row.action.indexOf("PAPER") < 0) continue
            if (decisionFilter === "blocked" && row.action !== "BLOCKED LIVE") continue
            if (decisionFilter === "no-order" && row.action !== "NO ORDER" && row.action !== "HOLD" && row.action !== "WAIT") continue
            if (decisionPairFilter !== "All pairs" && row.symbol !== decisionPairFilter) continue
            out.push(row)
        }
        return out
    }
    function countAction(kind) {
        var count = 0
        var rows = root.decisionRows()
        for (var i = 0; i < rows.length; ++i) {
            var action = rows[i].action
            if (kind === "blocked" && action === "BLOCKED LIVE") count += 1
            if (kind === "paper" && action.indexOf("PAPER") >= 0) count += 1
            if (kind === "no-order" && (action === "NO ORDER" || action === "HOLD" || action === "WAIT")) count += 1
        }
        return count
    }

    onRuntimeServiceChanged: root.refreshAiGovernorSnapshot()
    Component.onCompleted: root.refreshAiGovernorSnapshot()

    Connections {
        target: root.runtimeService
        ignoreUnknownSignals: true
        function onAiGovernorSnapshotChanged() { root.refreshAiGovernorSnapshot() }
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "aiDecisionsTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "aiDecisionsTitle"; text: qsTr("Decyzje"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Decision stream reads shared local-only paper bridge/state rows from Generate next decision, Generate governor recommendation and Simulate terminal order. All rows are Paper preview records; no live orders are emitted."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("blocked live count"); description: String(root.countAction("blocked")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("paper simulated orders count"); description: String(root.countAction("paper")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("no order count"); description: String(root.countAction("no-order")); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Decision filters")
            description: qsTr("Filters: all, paper, blocked, no-order, selected pair.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("all"); subtle: previewState.decisionFilter !== "all"; onClicked: previewState.decisionFilter = "all" }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("paper"); subtle: previewState.decisionFilter !== "paper"; onClicked: previewState.decisionFilter = "paper" }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("blocked"); subtle: previewState.decisionFilter !== "blocked"; onClicked: previewState.decisionFilter = "blocked" }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("no-order"); subtle: previewState.decisionFilter !== "no-order"; onClicked: previewState.decisionFilter = "no-order" }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("selected pair"); subtle: previewState.decisionPairFilter === "All pairs"; onClicked: previewState.decisionPairFilter = previewState.selectedPairs.length > 0 ? previewState.selectedPairs[0] : "All pairs" }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("All pairs"); subtle: previewState.decisionPairFilter !== "All pairs"; onClicked: previewState.decisionPairFilter = "All pairs" }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("AI / Governor decision preview")
            description: qsTr("Each record includes Timestamp, symbol, action, confidence, reason, Risk reason, Strategy source and Safety block state.")
            RowLayout {
                Layout.fillWidth: true
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate next decision"); iconName: "mode_wizard"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.generateNextDecision() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); subtle: true; onClicked: previewState.generateGovernorRecommendation() }
                Label { text: qsTr("Paper session state: %1").arg(previewState.paperSessionStatus); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
            }
            ListView {
                Layout.fillWidth: true
                Layout.preferredHeight: 520
                clip: true
                spacing: 8
                model: root.filteredDecisionRows()
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view ? ListView.view.width : 900
                    height: decisionColumn.implicitHeight + 22
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
                            Rectangle { implicitWidth: Math.max(110, actionLabel.implicitWidth + 22); implicitHeight: 26; radius: 13; color: Qt.rgba(0.33, 0.78, 1, 0.14); border.color: modelData.action === "BLOCKED LIVE" ? designSystem.color("critical") : designSystem.color("accent"); Label { id: actionLabel; anchors.centerIn: parent; text: modelData.action; color: designSystem.color("textPrimary"); font.bold: true; font.pixelSize: 11 } }
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
