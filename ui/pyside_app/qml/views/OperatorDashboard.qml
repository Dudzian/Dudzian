import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "operatorDashboardRoot"
    property bool defaultDashboard: true
    contentWidth: availableWidth
    clip: true
    implicitWidth: 1040
    implicitHeight: 680

    ColumnLayout {
        width: root.availableWidth
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6
                Label { objectName: "operatorDashboardTitle"; text: qsTr("Dashboard"); font.bold: true; font.pixelSize: 28; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Final-product safe preview shell for Dudzian Bot. Static local preview / no live data, but final application flow and layout."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Bot status: Demo/Paper Preview"); description: qsTr("Preview only • Runtime loop not started • API keys not required"); Layout.preferredWidth: 280 }
        }

        GridLayout {
            objectName: "operatorDashboardSafetySummary"
            Layout.fillWidth: true
            columns: 3
            rowSpacing: 12
            columnSpacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("AI/Governor status • AI / Governor mode"); description: qsTr("Active AI model / governor engine: Decision Governor Preview Core • advisory / supervised dry-run"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model readiness %"); description: qsTr("Model readiness 72% • Training/coverage 68% • static local preview metric"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Autonomy level"); description: qsTr("Autonomy level: supervised dry-run. Execution guard blocks all live actions."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected exchanges"); description: qsTr("Demo Exchange active • Binance, Bybit, OKX, KuCoin, Coinbase available as UI-only dry-run"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected coins/pairs"); description: qsTr("BTC/USDT, ETH/USDT, SOL/USDT selected • BNB/USDT, XRP/USDT, ADA/USDT, DOGE/USDT available"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies"); description: qsTr("Momentum Guard, Range Guard enabled preview • Volatility Breakout Preview guarded"); Layout.fillWidth: true }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Components.PreviewCard {
                objectName: "operatorDashboardFeed"
                designSystem: root.designSystem
                title: qsTr("Last AI/governor decision")
                description: qsTr("BTC/USDT HOLD • confidence 0.81 • reason: momentum neutralny, risk state guarded • NO ORDER — preview only")
                Layout.fillWidth: true
            }
            Components.PreviewCard {
                objectName: "operatorDashboardRiskControls"
                designSystem: root.designSystem
                title: qsTr("Risk state")
                description: qsTr("Risk state: guarded preview • max drawdown 4.0% • risk kill-switch armed • execution guard blocking live orders")
                Layout.fillWidth: true
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Safety locks")
            description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • Runtime loop not started")
        }
    }
}
