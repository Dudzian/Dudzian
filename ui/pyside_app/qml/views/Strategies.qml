import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "strategiesPreviewPanel"
    property var runtimeService
    property var demoStrategies: [
        ({ name: qsTr("Momentum Guard"), enabled: true, floor: "0.72", cooldown: "90s", maxPosition: "$2,500", source: "trend/momentum preview", safety: qsTr("advisory only • no live execution") }),
        ({ name: qsTr("Range Guard"), enabled: true, floor: "0.68", cooldown: "120s", maxPosition: "$1,800", source: "range/risk preview", safety: qsTr("risk governor locked") }),
        ({ name: qsTr("Volatility Breakout Preview"), enabled: false, floor: "0.80", cooldown: "180s", maxPosition: "$1,200", source: "breakout preview", safety: qsTr("execution guard blocks live") })
    ]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 16
        Label { objectName: "strategiesPreviewTitle"; text: qsTr("Strategie"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Final strategy configuration module in safe demo/offline preview. No runtime write / no live execution / Order submission disabled."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Active strategies summary"); description: qsTr("Momentum Guard + Range Guard enabled preview; Volatility Breakout Preview disabled until safety review."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Preview-only save state"); description: qsTr("Save Preview updates UI label only; no runtime write and no live config mutation."); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution policy"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled"); Layout.fillWidth: true }
        }

        Repeater {
            model: root.demoStrategies
            delegate: Components.PreviewCard {
                required property var modelData
                designSystem: root.designSystem
                title: modelData.name
                description: qsTr("Strategy card with styled controls. Source: %1. Safety: %2").arg(modelData.source).arg(modelData.safety)
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: qsTr("Enabled preview toggle"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                    Components.StyledSwitch { designSystem: root.designSystem; checked: modelData.enabled }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 12
                    Label { text: qsTr("Confidence floor"); color: root.designSystem.color("textSecondary") }
                    Components.StyledTextField { designSystem: root.designSystem; text: modelData.floor; Layout.fillWidth: true }
                    Label { text: qsTr("Cooldown"); color: root.designSystem.color("textSecondary") }
                    Components.StyledTextField { designSystem: root.designSystem; text: modelData.cooldown; Layout.fillWidth: true }
                    Label { text: qsTr("Max position preview"); color: root.designSystem.color("textSecondary") }
                    Components.StyledTextField { designSystem: root.designSystem; text: modelData.maxPosition; Layout.fillWidth: true }
                    Label { text: qsTr("Safety label"); color: root.designSystem.color("textSecondary") }
                    Label { text: modelData.safety; color: root.designSystem.color("textPrimary"); font.bold: true; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Components.IconButton { designSystem: root.designSystem; text: qsTr("Save Preview"); iconName: "copy"; onClicked: saveStatus.text = qsTr("Preview-only save state updated — no runtime write") }
                    Label { id: saveStatus; text: qsTr("Local UI state ready"); color: root.designSystem.color("textSecondary"); Layout.fillWidth: true }
                }
            }
        }
    }
}
