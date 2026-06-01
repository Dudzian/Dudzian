import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "riskControlsPreviewPanel"
    property var runtimeService
    property string policyModePreview: "advisory / shadow"
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 16
        Label { objectName: "riskControlsPreviewTitle"; text: qsTr("Ryzyko"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Final risk control module in safe product preview. All controls are styled dark UI and preview-only state."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        GridLayout {
            Layout.fillWidth: true
            columns: 3
            rowSpacing: 12
            columnSpacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk state"); description: qsTr("Guarded preview • max drawdown 4.0% • supervised dry-run"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk kill-switch"); description: qsTr("Risk kill-switch armed. Runtime loop not started."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Opportunity governor mode"); description: qsTr("advisory / shadow • execution guard blocks live orders"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Risk limits preview")
            description: qsTr("Max position, max open positions, stop loss, take profit, max slippage and max drawdown are local preview values.")
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 10
                columnSpacing: 14
                Label { text: qsTr("Max position"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "$2,500 preview"; Layout.fillWidth: true }
                Label { text: qsTr("Max open positions"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 25; value: 3; Layout.fillWidth: true }
                Label { text: qsTr("Stop loss"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 200; value: 11; textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) + "%" }; Layout.fillWidth: true }
                Label { text: qsTr("Take profit"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 200; value: 18; textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) + "%" }; Layout.fillWidth: true }
                Label { text: qsTr("Max slippage"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 100; value: 35; textFromValue: function(value, locale) { return Number(value / 100).toLocaleString(locale, 'f', 2) + "%" }; Layout.fillWidth: true }
                Label { text: qsTr("Max drawdown"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 200; value: 40; textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) + "%" }; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Safety controls")
            description: qsTr("Preview-only state. Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • Runtime loop not started.")
            RowLayout { Layout.fillWidth: true; Label { text: qsTr("Risk kill-switch"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true } Components.StyledSwitch { designSystem: root.designSystem; checked: true } }
            RowLayout { Layout.fillWidth: true; Label { text: qsTr("Opportunity governor mode"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true } Label { text: root.policyModePreview; color: root.designSystem.color("textSecondary"); font.bold: true } }
            RowLayout { Layout.fillWidth: true; Label { text: qsTr("Execution guard state"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true } Label { text: qsTr("BLOCKING LIVE ORDERS"); color: root.designSystem.color("warning"); font.bold: true } }
        }
    }
}
