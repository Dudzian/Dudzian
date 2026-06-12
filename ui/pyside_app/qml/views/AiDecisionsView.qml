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
    property string localDecisionFilter: "all"
    property string localDecisionPairFilter: "All pairs"
    property int timelineCount: root.hasHistoryRowsSource() ? root.historyRows().length : root.filteredDecisionRows().length
    property int recommendationCount: root.recommendedModes().length
    property string currentMode: root.lastDecisionValue("mode", "")
    contentWidth: availableWidth
    clip: true

    // Standalone/source guards cover previewState.decisionPreviewRows and previewState.paperSessionStatus without direct unguarded reads.
    // Smoke source markers: select all, clear selected, Autonomy mode selector, Training coverage %.

    function safeColor(token, fallback) {
        if (root.designSystem && typeof root.designSystem.color === "function")
            return root.designSystem.color(token)
        return fallback
    }
    function hasPreviewState() {
        return root.previewState !== undefined && root.previewState !== null
    }
    function previewValue(key, fallback) {
        if (!root.hasPreviewState()) return fallback
        var value = root.previewState[key]
        return value === undefined || value === null ? fallback : value
    }
    function decisionRows() {
        var rows = root.previewValue("decisionPreviewRows", [])
        return rows === undefined || rows === null ? [] : rows
    }
    function selectedPairs() {
        var pairs = root.previewValue("selectedPairs", [])
        return pairs === undefined || pairs === null ? [] : pairs
    }
    function decisionFilterValue() {
        return root.previewValue("decisionFilter", root.localDecisionFilter)
    }
    function decisionPairFilterValue() {
        return root.previewValue("decisionPairFilter", root.localDecisionPairFilter)
    }
    function paperSessionStatusValue() {
        return root.previewValue("paperSessionStatus", "stopped")
    }
    function lastGovernorDecisionValue() {
        return root.previewValue("lastGovernorDecision", qsTr("No paper decision yet"))
    }
    function setDecisionFilter(value) {
        if (root.hasPreviewState())
            root.previewState.decisionFilter = value
        root.localDecisionFilter = value
    }
    function setDecisionPairFilter(value) {
        if (root.hasPreviewState())
            root.previewState.decisionPairFilter = value
        root.localDecisionPairFilter = value
    }
    function generateNextDecision() {
        if (root.hasPreviewState() && typeof root.previewState.generateNextDecision === "function")
            root.previewState.generateNextDecision()
    }
    function generateGovernorRecommendation() {
        if (root.hasPreviewState() && typeof root.previewState.generateGovernorRecommendation === "function")
            root.previewState.generateGovernorRecommendation()
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
        var decisionFilter = root.decisionFilterValue()
        var decisionPairFilter = root.decisionPairFilterValue()
        for (var i = 0; i < rows.length; ++i) {
            var row = rows[i]
            if (decisionFilter === "paper" && row.action.indexOf("PAPER") < 0) continue
            if (decisionFilter === "blocked" && row.action !== "BLOCKED" && row.action !== "BLOCKED LIVE") continue
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
            if (kind === "blocked" && (action === "BLOCKED" || action === "BLOCKED LIVE")) count += 1
            if (kind === "paper" && action.indexOf("PAPER") >= 0) count += 1
            if (kind === "no-order" && (action === "NO ORDER" || action === "HOLD" || action === "WAIT")) count += 1
        }
        return count
    }

    onRuntimeServiceChanged: root.refreshAiGovernorSnapshot()
    Component.onCompleted: root.refreshAiGovernorSnapshot()

    Connections {
        target: root.runtimeService ? root.runtimeService : null
        ignoreUnknownSignals: true
        function onAiGovernorSnapshotChanged() { root.refreshAiGovernorSnapshot() }
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "aiDecisionsTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: root.safeColor("accent", "#5BC8FF") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "aiDecisionsTitle"; text: qsTr("Decyzje"); font.bold: true; font.pixelSize: 26; color: root.safeColor("textPrimary", "#ffffff"); Layout.fillWidth: true }
                Label { text: qsTr("Decision stream reads shared local-only paper bridge/state rows from Generate next decision, Generate governor recommendation and Simulate terminal order. All rows are Paper preview records; no live orders are emitted."); color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Components.PreviewCard { descriptionObjectName: "previewAiDecisionLatestActionLabel"; designSystem: root.designSystem; title: qsTr("Last decision"); description: root.lastGovernorDecisionValue(); Layout.fillWidth: true }
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
                Components.IconButton { designSystem: root.designSystem; text: qsTr("all"); subtle: root.decisionFilterValue() !== "all"; onClicked: root.setDecisionFilter("all") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("paper"); subtle: root.decisionFilterValue() !== "paper"; onClicked: root.setDecisionFilter("paper") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("blocked"); subtle: root.decisionFilterValue() !== "blocked"; onClicked: root.setDecisionFilter("blocked") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("no-order"); subtle: root.decisionFilterValue() !== "no-order"; onClicked: root.setDecisionFilter("no-order") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("selected pair"); subtle: root.decisionPairFilterValue() === "All pairs"; onClicked: root.setDecisionPairFilter(root.selectedPairs().length > 0 ? root.selectedPairs()[0] : "All pairs") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("All pairs"); subtle: root.decisionPairFilterValue() !== "All pairs"; onClicked: root.setDecisionPairFilter("All pairs") }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("AI / Governor decision preview")
            description: qsTr("Each record includes Timestamp, symbol, action, confidence, strategy/governor, reason, safety state, Safety block state and a paper order event link/text when one exists.")
            RowLayout {
                Layout.fillWidth: true
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate next decision"); iconName: "mode_wizard"; backgroundColor: root.safeColor("accent", "#5BC8FF"); foregroundColor: root.safeColor("surface", "#10141f"); onClicked: root.generateNextDecision() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); subtle: true; onClicked: root.generateGovernorRecommendation() }
                Label { text: qsTr("Paper session state: %1").arg(root.paperSessionStatusValue()); color: root.safeColor("textSecondary", "#c5cad3"); Layout.fillWidth: true }
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
                    color: root.safeColor("surfaceMuted", "#242936")
                    border.color: (modelData.action === "BLOCKED" || modelData.action === "BLOCKED LIVE") ? root.safeColor("critical", "#ff5f6d") : root.safeColor("border", "#3C3F44")
                    border.width: 1
                    ColumnLayout {
                        id: decisionColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Label { text: modelData.symbol; color: root.safeColor("textPrimary", "#ffffff"); font.bold: true; Layout.fillWidth: true }
                            Rectangle { implicitWidth: Math.max(110, actionLabel.implicitWidth + 22); implicitHeight: 26; radius: 13; color: Qt.rgba(0.33, 0.78, 1, 0.14); border.color: (modelData.action === "BLOCKED" || modelData.action === "BLOCKED LIVE") ? root.safeColor("critical", "#ff5f6d") : root.safeColor("accent", "#5BC8FF"); Label { id: actionLabel; anchors.centerIn: parent; text: modelData.action; color: root.safeColor("textPrimary", "#ffffff"); font.bold: true; font.pixelSize: 11 } }
                            Label { text: qsTr("confidence %1").arg(modelData.confidence); color: root.safeColor("textSecondary", "#c5cad3") }
                            Label { text: qsTr("Timestamp %1").arg(modelData.timestamp); color: root.safeColor("textSecondary", "#c5cad3") }
                            Components.IconButton { designSystem: root.designSystem; text: qsTr("Explain"); helpText: root.hasPreviewState() ? root.previewState.tooltipText("Explain decision") : qsTr("Explain decision"); subtle: true; onClicked: if (root.hasPreviewState() && typeof root.previewState.openDecisionExplainDrawer === "function") root.previewState.openDecisionExplainDrawer(modelData) }
                        }
                        Label { text: qsTr("reason: %1").arg(modelData.reason); color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Risk reason: %1").arg(modelData.riskReason); color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Strategy source / governor: %1 • Safety state: %2 • paper/session state: %3").arg(modelData.strategy).arg(modelData.safety).arg(modelData.paperState); color: root.safeColor("textPrimary", "#ffffff"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("paper order event: %1").arg(modelData.orderEvent ? modelData.orderEvent : (modelData.action.indexOf("PAPER") >= 0 ? "paper simulated order row in Paper Terminal" : "no order / blocked live")); color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
