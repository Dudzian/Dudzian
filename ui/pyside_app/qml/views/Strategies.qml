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
        ({ name: "Momentum Guard", floor: "0.72", cooldown: "90s", maxPosition: "$2,500", timeframe: "15m", riskProfile: "Balanced" }),
        ({ name: "Range Guard", floor: "0.68", cooldown: "120s", maxPosition: "$1,800", timeframe: "30m", riskProfile: "Conservative" }),
        ({ name: "Volatility Breakout Preview", floor: "0.80", cooldown: "180s", maxPosition: "$1,200", timeframe: "5m", riskProfile: "Aggressive" }),
        ({ name: "Mean Reversion Preview", floor: "0.70", cooldown: "240s", maxPosition: "$1,000", timeframe: "1h", riskProfile: "Conservative" }),
        ({ name: "Trend Follow Preview", floor: "0.76", cooldown: "300s", maxPosition: "$1,600", timeframe: "4h", riskProfile: "Balanced" }),
        ({ name: "Liquidity Sweep Preview", floor: "0.82", cooldown: "360s", maxPosition: "$900", timeframe: "3m", riskProfile: "Aggressive" })
    ]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        Label { objectName: "strategiesPreviewTitle"; text: qsTr("Strategie"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Klikalne strategy cards w safe demo/offline preview. Save Preview aktualizuje wyłącznie local UI state; nie zapisuje runtime config i nie uruchamia live execution."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies summary"); description: qsTr("%1 active strategies: %2").arg(previewState.activeStrategies.length).arg(previewState.activeStrategies.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Preview-only save state"); description: qsTr("Save Preview changes local label and activeStrategies only — no runtime write."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution policy"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled"); Layout.fillWidth: true }
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
                    description: qsTr("Strategy card • Toggle enabled/disabled local state • Confidence floor, cooldown, max position preview, timeframe, allowed pairs, risk profile.")
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Enabled preview toggle"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Components.StyledSwitch { designSystem: root.designSystem; checked: enabledPreview; onToggled: { enabledPreview = checked; previewState.setStrategyActive(modelData.name, checked) } }
                    }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        rowSpacing: 8
                        columnSpacing: 12
                        Label { text: qsTr("confidence floor"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.floor; Layout.fillWidth: true }
                        Label { text: qsTr("cooldown"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.cooldown; Layout.fillWidth: true }
                        Label { text: qsTr("max position preview"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.maxPosition; Layout.fillWidth: true }
                        Label { text: qsTr("timeframe"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.timeframe; Layout.fillWidth: true }
                        Label { text: qsTr("allowed pairs"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: previewState.selectedPairs.slice(0, 6).join(", "); Layout.fillWidth: true }
                        Label { text: qsTr("risk profile"); color: root.designSystem.color("textSecondary") }
                        Components.StyledTextField { designSystem: root.designSystem; text: modelData.riskProfile; Layout.fillWidth: true }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton { designSystem: root.designSystem; text: qsTr("Save Preview"); iconName: "copy"; onClicked: { previewState.setStrategyActive(modelData.name, enabledPreview); saveStatus.text = qsTr("Save Preview updated local UI state — no runtime write") } }
                        Label { id: saveStatus; text: enabledPreview ? qsTr("Enabled in local preview") : qsTr("Disabled in local preview"); color: root.designSystem.color("textSecondary"); Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
