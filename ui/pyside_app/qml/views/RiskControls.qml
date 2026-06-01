import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "riskControlsPreviewPanel"
    property var runtimeService
    property var previewState
    property var profiles: ["Conservative", "Balanced", "Aggressive"]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "riskControlsTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "riskControlsTitle"; text: qsTr("Ryzyko"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Kokpit ryzyka dla bezpiecznego Paper Preview. Segmented Conservative/Balanced/Aggressive aktualizuje lokalne wartości limitów; live trading, exchange route i order submission pozostają wyłączone."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Risk profile segmented control")
            description: qsTr("Choose Conservative, Balanced or Aggressive. Active segment updates riskProfile, maxPosition, maxOpenPositions, stopLoss, takeProfit, maxSlippage, maxDrawdown, dailyLossLimit, per-symbol exposure and riskState.")
            RowLayout {
                objectName: "riskProfileSegmentedControl"
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.profiles
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.riskProfile === modelData
                        Layout.fillWidth: true
                        implicitHeight: 44
                        radius: 14
                        color: active ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        Label { anchors.centerIn: parent; text: modelData; color: active ? designSystem.color("surface") : designSystem.color("textPrimary"); font.bold: true }
                        MouseArea { anchors.fill: parent; onClicked: previewState.setRiskProfile(modelData) }
                    }
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 4 : 2
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk state"); description: previewState.riskState; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max position"); description: previewState.maxPosition; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max open positions"); description: String(previewState.maxOpenPositions); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Stop loss"); description: previewState.stopLoss; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Take profit"); description: previewState.takeProfit; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max slippage"); description: previewState.maxSlippage; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max drawdown"); description: previewState.maxDrawdown; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Daily loss limit"); description: previewState.dailyLossLimit; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Per-symbol exposure"); description: previewState.perSymbolExposure; Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Safety boundary")
            description: qsTr("Safety kill-switch armed • live disabled • exchange route disabled • order submission disabled • paper bridge not connected/planned • API keys not required.")
        }
    }
}
