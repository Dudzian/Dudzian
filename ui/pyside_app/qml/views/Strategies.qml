import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "strategiesPreviewPanel"
    property var runtimeService
    property var previewState
    property var strategyCards: [
        ({ name: "Momentum Guard", floor: "0.72", cooldown: "90s", allocation: "8%", timeframe: "15m", riskProfile: "Balanced", status: "enabled" }),
        ({ name: "Range Guard", floor: "0.68", cooldown: "120s", allocation: "6%", timeframe: "30m", riskProfile: "Conservative", status: "enabled" }),
        ({ name: "Volatility Breakout Preview", floor: "0.80", cooldown: "180s", allocation: "4%", timeframe: "5m", riskProfile: "Aggressive", status: "guarded" }),
        ({ name: "Mean Reversion Preview", floor: "0.70", cooldown: "240s", allocation: "5%", timeframe: "1h", riskProfile: "Conservative", status: "disabled" }),
        ({ name: "Trend Follow Preview", floor: "0.76", cooldown: "300s", allocation: "7%", timeframe: "4h", riskProfile: "Balanced", status: "guarded" }),
        ({ name: "Liquidity Sweep Preview", floor: "0.82", cooldown: "360s", allocation: "3%", timeframe: "3m", riskProfile: "Aggressive", status: "disabled" })
    ]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "strategiesTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "strategiesPreviewTitle"; text: qsTr("Strategie"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Docelowe moduły strategii dla Paper Preview. Każda karta ma enabled toggle, confidence floor, cooldown, timeframe, max allocation, allowed pairs count, risk profile i local Save Preview action. Zmiany są tylko lokalnym UI state, bez runtime config write i bez live execution."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies summary"); description: qsTr("%1 active strategies: %2").arg(previewState.activeStrategies.length).arg(previewState.activeStrategies.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Preview-only save state"); description: previewState.lastStrategySaveStatus + qsTr(" • Save Preview changes local label and activeStrategies only — no runtime write."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution policy"); description: qsTr("Live trading disabled • Exchange route disabled • Order submission disabled"); Layout.fillWidth: true }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1100 ? 2 : 1
            rowSpacing: 10
            columnSpacing: 10
            Repeater {
                model: root.strategyCards
                delegate: Components.PreviewCard {
                    required property var modelData
                    property bool enabledPreview: previewState.hasValue(previewState.activeStrategies, modelData.name)
                    designSystem: root.designSystem
                    title: modelData.name
                    description: qsTr("Module status: %1 • enable toggle, confidence floor, cooldown, timeframe, max allocation, allowed pairs count and risk profile are editable preview fields.").arg(enabledPreview ? (modelData.status === "guarded" ? "guarded" : "enabled") : "disabled")
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("enable toggle"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Components.StyledSwitch { designSystem: root.designSystem; checked: enabledPreview; onToggled: { enabledPreview = checked; previewState.setStrategyActive(modelData.name, checked) } }
                    }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        rowSpacing: 8
                        columnSpacing: 12
                        Label { text: qsTr("confidence floor"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.floor; placeholderText: qsTr("0.70"); Layout.fillWidth: true }
                        Label { text: qsTr("cooldown"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.cooldown; placeholderText: qsTr("120s"); Layout.fillWidth: true }
                        Label { text: qsTr("timeframe"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.timeframe; placeholderText: qsTr("15m"); Layout.fillWidth: true }
                        Label { text: qsTr("max allocation"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.allocation; placeholderText: qsTr("5%"); Layout.fillWidth: true }
                        Label { text: qsTr("allowed pairs count"); color: root.designSystem.color("textSecondary") }
                        Components.StyledSpinBox { designSystem: root.designSystem; from: 0; to: 128; value: previewState.selectedPairs.length; Layout.fillWidth: true }
                        Label { text: qsTr("risk profile"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.riskProfile; placeholderText: qsTr("Balanced"); Layout.fillWidth: true }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton { designSystem: root.designSystem; text: qsTr("Save Preview"); iconName: "copy"; onClicked: { previewState.setStrategyActive(modelData.name, enabledPreview); previewState.saveStrategyPreview(modelData.name); saveStatus.text = qsTr("local Save Preview action updated local UI state — no runtime config write") } }
                        Label { id: saveStatus; text: enabledPreview ? qsTr("status: enabled/guarded") : qsTr("status: disabled"); color: root.designSystem.color("textSecondary"); Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
