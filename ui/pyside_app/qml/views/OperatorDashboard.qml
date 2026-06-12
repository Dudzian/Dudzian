import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "operatorDashboardRoot"
    property var previewState
    property bool defaultDashboard: true
    contentWidth: availableWidth
    clip: true
    implicitWidth: 1040
    implicitHeight: 680

    function statusColor(status) {
        if (status === "blocked") return designSystem.color("critical")
        if (status === "simulated") return designSystem.color("accent")
        return designSystem.color("warning")
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            Rectangle { objectName: "operatorDashboardTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6
                Label { objectName: "operatorDashboardTitle"; text: qsTr("Dashboard"); font.bold: true; font.pixelSize: 28; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Kokpit operatora dla bezpiecznego Paper Preview. Live trading disabled, Exchange route disabled, Order submission disabled, order submission disabled, API keys not required."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Bot status: Demo/Paper Preview"); description: qsTr("Paper session status: %1 • Runtime loop not started • Sandbox/testnet planned").arg(previewState.paperSessionStatus); Layout.preferredWidth: 340 }
        }

        GridLayout {
            objectName: "operatorDashboardSafetySummary"
            Layout.fillWidth: true
            columns: width > 1100 ? 4 : 2
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("AI/Governor status • AI / Governor mode • Autonomy level"); description: qsTr("Active AI model / governor engine: %1 • autonomy mode %2 • autonomy level %3/5").arg(previewState.activeGovernorEngine).arg(previewState.autonomyMode).arg(previewState.autonomyLevel); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model readiness %"); description: qsTr("Model readiness %1% • Training/coverage %2% • Data coverage %3%").arg(previewState.modelReadiness).arg(previewState.trainingCoverage).arg(previewState.dataCoverage); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected exchanges"); description: qsTr("%1 selected: %2").arg(previewState.selectedExchanges.length).arg(previewState.selectedExchanges.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected coins/pairs"); description: qsTr("%1 selected from %2 preview pairs: %3").arg(previewState.selectedPairs.length).arg(previewState.previewMarketPairs.length).arg(previewState.selectedPairs.slice(0, 8).join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardBestScannerOpportunity"; descriptionObjectName: "previewDashboardBestOpportunityLabel"; designSystem: root.designSystem; title: qsTr("Best scanner opportunity"); description: qsTr("%1 • candidates %2 • safe preview scanner").arg(previewState.scannerBestOpportunity).arg(previewState.scannerCandidateCount); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies"); description: qsTr("%1 active strategies: %2").arg(previewState.activeStrategies.length).arg(previewState.activeStrategies.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardFeed"; descriptionObjectName: "previewDashboardGovernorDecisionLabel"; designSystem: root.designSystem; title: qsTr("Last AI/governor decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true; Components.IconButton { designSystem: root.designSystem; text: qsTr("Wyjaśnij ostatnią decyzję"); helpText: previewState.tooltipText("Explain decision"); onClicked: previewState.openDecisionExplainDrawer(previewState.decisionPreviewRows.length > 0 ? previewState.decisionPreviewRows[0] : null) } }
            Components.PreviewCard { objectName: "operatorDashboardRiskControls"; designSystem: root.designSystem; title: qsTr("Risk state"); description: qsTr("%1 • Risk profile %2 • max position %3").arg(previewState.riskState).arg(previewState.riskProfile).arg(previewState.maxPosition); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper session PnL / equity"); description: qsTr("Paper session equity: %1 • Paper session PnL: %2 • Session ticks: %3 • Portfolio report / selected range: %4 %5").arg(previewState.formatMoney(previewState.paperEquity, "USDT")).arg(previewState.formatUsd(previewState.paperPnl)).arg(previewState.paperSessionTicks).arg(previewState.portfolioSelectedRange).arg(previewState.formatUsd(previewState.portfolioAllTimePnlUsd)); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last paper order"); description: previewState.paperOrderRows.length > 0 ? qsTr("%1 • %2 • %3 • %4").arg(previewState.paperOrderRows[0].timestamp).arg(previewState.paperOrderRows[0].pair).arg(previewState.paperOrderRows[0].action).arg(previewState.paperOrderRows[0].status) : qsTr("No local-only paper bridge/state orders yet"); Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardAlertSummary"; designSystem: root.designSystem; title: qsTr("Alert Center summary"); description: qsTr("unread alerts: %1 • critical count: %2 • last alert: %3").arg(previewState.alertUnreadCount).arg(previewState.alertCriticalCount).arg(previewState.alertRows.length > 0 ? previewState.alertRows[0].title : "—"); Layout.fillWidth: true; Components.IconButton { designSystem: root.designSystem; text: qsTr("Otwórz Alerty"); helpText: previewState.tooltipText("Alert Center"); onClicked: previewState.showPanel("alertsPanel") } }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last governor/paper decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Simulation status"); description: previewState.simulationStatusLabel + " • running=" + previewState.simulationRunning + " paused=" + previewState.simulationPaused; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Simulation speed / tick count"); description: qsTr("speed x%1 • interval %2 ms • ticks %3 • last tick %4").arg(previewState.simulationSpeed).arg(previewState.simulationTickIntervalMs).arg(previewState.simulationTickCount).arg(previewState.simulationLastTickAt); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last simulated scan"); description: qsTr("pair %1 • action %2 • order %3").arg(previewState.simulationLastPair).arg(previewState.simulationLastAction).arg(previewState.simulationLastOrder); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Safety boundary"); description: previewState.simulationSafetyBoundary; Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Szybkie akcje / Quick actions")
            description: qsTr("Przyciski zmieniają lokalny preview state: session status, ticks, orders, blocked, no-order, simulated, Paper PnL/equity, last governor decision i order blotter. Nie uruchamiają runtime loop ani real orders.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Start Paper Preview"); helpText: previewState.tooltipText("Start Paper Preview"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.startPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Pause"); helpText: previewState.tooltipText("Pause"); subtle: true; onClicked: previewState.pausePaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Stop"); helpText: previewState.tooltipText("Stop"); subtle: true; onClicked: previewState.stopPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Reset"); helpText: previewState.tooltipText("Reset"); subtle: true; onClicked: previewState.resetPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate Next Tick"); helpText: previewState.tooltipText("Generate Next Tick"); iconName: "refresh"; onClicked: previewState.generatePaperTick() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Run 10 ticks"); helpText: previewState.tooltipText("Run 10 paper ticks"); onClicked: previewState.runTenMockTicks() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Start Scanner"); helpText: previewState.tooltipText("Start scanner"); onClicked: previewState.startMarketScannerPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("AI Recommended Risk"); helpText: previewState.tooltipText("AI recommended risk"); onClicked: previewState.applyAiRecommendedRiskProfile() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Open Alerts"); helpText: previewState.tooltipText("Open Alerts"); onClicked: previewState.showPanel("alertsPanel") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Open Settings"); helpText: previewState.tooltipText("Open Settings"); onClicked: previewState.showPanel("settingsPanel") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Open Help"); helpText: previewState.tooltipText("Open Help"); onClicked: previewState.showPanel("helpGlossaryPanel") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate Diagnostic Bundle"); helpText: previewState.tooltipText("Generate diagnostic bundle"); onClicked: previewState.generateDiagnosticBundle() }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: previewState.simulationScenarios
                    delegate: Components.IconButton {
                        required property string modelData
                        designSystem: root.designSystem
                        text: modelData
                        helpText: previewState.tooltipText("Market scenario")
                        subtle: previewState.simulationScenario !== modelData
                        onClicked: previewState.setSimulationScenario(modelData)
                    }
                }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Simulation speed x1"); helpText: previewState.tooltipText("Simulation speed"); subtle: previewState.simulationSpeed !== 1; onClicked: previewState.setSimulationSpeed(1) }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Simulation speed x3"); helpText: previewState.tooltipText("Simulation speed"); subtle: previewState.simulationSpeed !== 3; onClicked: previewState.setSimulationSpeed(3) }
            }
            GridLayout {
                Layout.fillWidth: true
                columns: width > 900 ? 6 : 2
                rowSpacing: 8
                columnSpacing: 8
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper session status"); description: previewState.paperSessionStatus; Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Session ticks"); description: String(previewState.paperSessionTicks); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Orders"); description: String(previewState.paperOrdersCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("blocked"); description: String(previewState.paperBlockedCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("no-order"); description: String(previewState.paperNoOrderCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("simulated"); description: String(previewState.paperSimulatedCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Market scenario"); description: previewState.simulationScenario + " / " + previewState.simulationMarketMode; Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Live-like paper simulation"); description: qsTr("Paper loop local-only: no exchange API, no real orders, no secret reads, production runtime loop not started."); Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Paper order blotter")
            description: qsTr("Trading table with Time, Pair, Action, Status, Confidence, Reason. status chips: simulated, blocked, no order. action chips: PAPER BUY, PAPER SELL, HOLD, WAIT, NO ORDER, BLOCKED.")
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 34
                radius: 10
                color: designSystem.color("surfaceMuted")
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    Label { text: qsTr("Time"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 }
                    Label { text: qsTr("Pair"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 110 }
                    Label { text: qsTr("Action"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 130 }
                    Label { text: qsTr("Status"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 110 }
                    Label { text: qsTr("Confidence"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 100 }
                    Label { text: qsTr("Reason"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                }
            }
            ListView {
                objectName: "operatorDashboardOrderList"
                Layout.fillWidth: true
                Layout.preferredHeight: 280
                clip: true
                spacing: 8
                model: previewState.paperOrderRows
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view ? ListView.view.width : 900
                    height: blotterRow.implicitHeight + 18
                    radius: 12
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    RowLayout {
                        id: blotterRow
                        anchors.fill: parent
                        anchors.margins: 9
                        Label { text: modelData.timestamp; color: designSystem.color("textPrimary"); Layout.preferredWidth: 90 }
                        Label { text: modelData.pair; color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 110 }
                        Rectangle { Layout.preferredWidth: 130; implicitHeight: 26; radius: 13; color: Qt.rgba(0.33, 0.78, 1, 0.16); border.color: designSystem.color("accent"); Label { anchors.centerIn: parent; text: modelData.action; color: designSystem.color("textPrimary"); font.bold: true; font.pixelSize: 11 } }
                        Rectangle { Layout.preferredWidth: 110; implicitHeight: 26; radius: 13; color: Qt.rgba(1, 1, 1, 0.05); border.color: root.statusColor(modelData.status); Label { anchors.centerIn: parent; text: modelData.status; color: root.statusColor(modelData.status); font.bold: true; font.pixelSize: 11 } }
                        Label { text: modelData.confidence; color: designSystem.color("textSecondary"); Layout.preferredWidth: 100 }
                        Label { text: modelData.reason; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
