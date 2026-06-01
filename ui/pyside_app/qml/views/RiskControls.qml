import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "riskControlsPreviewPanel"
    property var runtimeService
    property var previewState
    property string policyModePreview: "advisory / shadow"
    property var riskProfiles: ["Conservative", "Balanced", "Aggressive"]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        Label { objectName: "riskControlsPreviewTitle"; text: qsTr("Ryzyko"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Finalny moduł kontroli ryzyka w safe product preview. Wszystkie kontrolki są dark UI i preview-only; Live pozostaje blocked."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 3 : 1
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk state"); description: previewState.riskState; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk kill-switch"); description: qsTr("Risk kill-switch armed=%1 • Live trading disabled • Order submission disabled").arg(previewState.riskLocked); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution guard state"); description: qsTr("Execution guard state: blocked live • paper preview only • exchange/order disabled"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Risk limits")
            description: qsTr("Max position, Max open positions, Stop loss, Take profit, Max slippage, Max drawdown, Daily loss limit, Per-symbol exposure")
            GridLayout {
                Layout.fillWidth: true
                columns: width > 900 ? 4 : 2
                rowSpacing: 8
                columnSpacing: 12
                Label { text: qsTr("Max position"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 1; to: 100000; value: 2500; Layout.fillWidth: true }
                Label { text: qsTr("Max open positions"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox { designSystem: root.designSystem; from: 1; to: 20; value: 4; Layout.fillWidth: true }
                Label { text: qsTr("Stop loss"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "1.8%"; Layout.fillWidth: true }
                Label { text: qsTr("Take profit"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "3.2%"; Layout.fillWidth: true }
                Label { text: qsTr("Max slippage"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "0.12%"; Layout.fillWidth: true }
                Label { text: qsTr("Max drawdown"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "4.0%"; Layout.fillWidth: true }
                Label { text: qsTr("Daily loss limit"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { objectName: "riskDailyLossLimit"; designSystem: root.designSystem; text: "2.5%"; Layout.fillWidth: true }
                Label { text: qsTr("Per-symbol exposure"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { objectName: "riskPerSymbolExposure"; designSystem: root.designSystem; text: "1.0% equity"; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Governor and guard controls")
            description: qsTr("Opportunity governor mode, Execution guard state, Risk kill-switch i preview risk profile.")
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Risk kill-switch"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                Components.StyledSwitch { designSystem: root.designSystem; checked: previewState.riskLocked; onToggled: { previewState.riskLocked = checked; previewState.riskState = checked ? "guarded preview • kill-switch armed • live blocked" : "preview unlocked for paper only • live blocked" } }
            }
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 8
                columnSpacing: 12
                Label { text: qsTr("Opportunity governor mode"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField { designSystem: root.designSystem; text: "guarded opportunity scan"; Layout.fillWidth: true }
                Label { text: qsTr("Execution guard state"); color: root.designSystem.color("textSecondary") }
                Label { text: qsTr("blocked live • paper route simulated only"); color: root.designSystem.color("textPrimary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.riskProfiles
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.riskProfile === modelData
                        width: 160
                        height: 42
                        radius: 12
                        color: active ? Qt.rgba(0.33, 0.78, 1.0, 0.20) : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        Label { anchors.centerIn: parent; text: qsTr("Apply preview risk profile: %1").arg(modelData); color: designSystem.color("textPrimary"); font.pixelSize: 11; font.bold: active }
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: previewState.setRiskProfile(modelData) }
                    }
                }
            }
        }

        Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Live status"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled • Runtime loop not started") }
    }
}
