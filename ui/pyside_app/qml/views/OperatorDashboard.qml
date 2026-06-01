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

    ColumnLayout {
        width: root.availableWidth
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6
                Label { objectName: "operatorDashboardTitle"; text: qsTr("Dashboard"); font.bold: true; font.pixelSize: 28; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Final-product cockpit dla safe paper/dry-run preview. Lokalny UI state, zero runtime loop, zero Exchange I/O, zero real order submission."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Bot status: Demo/Paper Preview"); description: qsTr("Paper session status: %1 • Runtime loop not started • API keys not required").arg(previewState.paperSessionState); Layout.preferredWidth: 320 }
        }

        GridLayout {
            objectName: "operatorDashboardSafetySummary"
            Layout.fillWidth: true
            columns: width > 1100 ? 4 : 2
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("AI/Governor status • AI / Governor mode • Autonomy level"); description: qsTr("Active AI model / governor engine: Decision Governor Preview Core • autonomy mode %1 • autonomy level %2/5").arg(previewState.autonomyMode).arg(previewState.autonomyLevel); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model readiness %"); description: qsTr("Model readiness %1% • Training/coverage %2% • Data coverage %3%").arg(previewState.modelReadiness).arg(previewState.trainingCoverage).arg(previewState.dataCoverage); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected exchanges"); description: qsTr("%1 selected: %2").arg(previewState.selectedExchanges.length).arg(previewState.selectedExchanges.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected coins/pairs"); description: qsTr("%1 selected: %2").arg(previewState.selectedPairs.length).arg(previewState.selectedPairs.slice(0, 8).join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies"); description: qsTr("%1 active strategies: %2").arg(previewState.activeStrategies.length).arg(previewState.activeStrategies.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardFeed"; designSystem: root.designSystem; title: qsTr("Last AI/governor decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardRiskControls"; designSystem: root.designSystem; title: qsTr("Risk state"); description: qsTr("%1 • Risk profile %2 • riskLocked=%3").arg(previewState.riskState).arg(previewState.riskProfile).arg(previewState.riskLocked); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Mock PnL / equity preview"); description: qsTr("Mock equity: %1 USDT • Mock PnL: %2 USDT • ticks: %3").arg(previewState.mockEquity.toFixed(2)).arg(previewState.mockPnl.toFixed(2)).arg(previewState.paperTicks); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Paper / dry-run session cockpit")
            description: qsTr("Start Paper Preview, Pause, Stop, Reset, Next Tick i Run 10 mock ticks zmieniają wyłącznie lokalny UI state. No real order, no real exchange I/O.")
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Start Paper Preview"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.startPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Pause"); subtle: true; onClicked: previewState.pausePaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Stop"); subtle: true; onClicked: previewState.stopPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Reset"); subtle: true; onClicked: previewState.resetPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate Next Tick"); iconName: "refresh"; onClicked: previewState.generatePaperTick() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Run 10 mock ticks"); onClicked: previewState.runTenMockTicks() }
            }
            GridLayout {
                Layout.fillWidth: true
                columns: 3
                rowSpacing: 8
                columnSpacing: 10
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Mock open paper positions"); description: previewState.openPaperPositions.map(function(p) { return p.pair + " " + p.side + " " + p.pnl + " (" + p.label + ")" }).join(" • "); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Mock closed paper trades"); description: previewState.closedPaperTrades.map(function(p) { return p.pair + " " + p.side + " " + p.pnl + " (" + p.label + ")" }).join(" • "); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper safety label • Safety kill-switch"); description: qsTr("Preview only • simulated preview only • no real order • order submission disabled • NO ORDER — preview only"); Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Mock paper orders list")
            description: qsTr("timestamp • pair • side • size • price • status: simulated / blocked / no order • reason")
            Repeater {
                model: previewState.paperOrdersPreview
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: orderRow.implicitHeight + 18
                    radius: 12
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    ColumnLayout {
                        id: orderRow
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.margins: 10
                        spacing: 4
                        Label { text: qsTr("%1 • %2 • %3 • size %4 • price %5 • status %6").arg(modelData.timestamp).arg(modelData.pair).arg(modelData.side).arg(modelData.size).arg(modelData.price).arg(modelData.status); color: designSystem.color("textPrimary"); font.bold: true; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: modelData.reason; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Safety locks")
            description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • Runtime loop not started")
        }
    }
}
