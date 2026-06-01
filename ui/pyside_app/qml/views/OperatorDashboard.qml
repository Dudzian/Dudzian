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
                Label { text: qsTr("Operator cockpit for safe dry-run and Paper preview. Live trading disabled, Exchange route disabled, Order submission disabled, order submission disabled, API keys not required in preview."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Bot status: Demo/Paper Preview"); description: qsTr("Paper session status: %1 • Runtime loop not started • Sandbox/testnet planned").arg(previewState.paperSessionState); Layout.preferredWidth: 340 }
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
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected coins/pairs"); description: qsTr("%1 selected from %2 preview pairs: %3").arg(previewState.selectedPairs.length).arg(previewState.previewMarketPairs.length).arg(previewState.selectedPairs.slice(0, 8).join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies"); description: qsTr("%1 active strategies: %2").arg(previewState.activeStrategies.length).arg(previewState.activeStrategies.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardFeed"; designSystem: root.designSystem; title: qsTr("Last AI/governor decision"); description: previewState.lastGovernorDecision; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "operatorDashboardRiskControls"; designSystem: root.designSystem; title: qsTr("Risk state"); description: qsTr("%1 • Risk profile %2 • max position %3").arg(previewState.riskState).arg(previewState.riskProfile).arg(previewState.maxPosition); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper PnL / equity preview"); description: qsTr("Paper equity: %1 USDT • Paper PnL: %2 USDT • Session ticks: %3").arg(previewState.previewEquity.toFixed(2)).arg(previewState.previewPnl.toFixed(2)).arg(previewState.paperTicks); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Paper / dry-run session cockpit")
            description: qsTr("Controls update local UI state only. Safe dry-run stays inside the preview shell; no exchange route and no real orders.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Start Paper Preview"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.startPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Pause"); subtle: true; onClicked: previewState.pausePaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Stop"); subtle: true; onClicked: previewState.stopPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Reset"); subtle: true; onClicked: previewState.resetPaperPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate Next Tick"); iconName: "refresh"; onClicked: previewState.generatePaperTick() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Run 10 paper ticks"); onClicked: previewState.runTenMockTicks() }
            }
            GridLayout {
                Layout.fillWidth: true
                columns: width > 900 ? 5 : 2
                rowSpacing: 8
                columnSpacing: 8
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Session ticks"); description: String(previewState.paperTicks); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Orders"); description: String(previewState.paperOrdersCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("blocked"); description: String(previewState.blockedOrdersCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("no-order"); description: String(previewState.noOrderCount); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("simulated"); description: String(previewState.simulatedOrdersCount); Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Paper order blotter")
            description: qsTr("Trading table with Time, Pair, Action, Status, Confidence, Reason. status chips: simulated, blocked, no order. action chips: PAPER BUY, PAPER SELL, HOLD, WAIT, NO ORDER, BLOCKED LIVE.")
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
                model: previewState.paperOrdersPreview
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
