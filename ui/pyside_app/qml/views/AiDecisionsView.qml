import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiDecisionsView"
    property var runtimeService
    property var aiSnapshot: runtimeService && runtimeService.aiGovernorSnapshot ? runtimeService.aiGovernorSnapshot : ({})
    property var lastDecision: aiSnapshot.lastDecision || ({})
    property var decisionTimeline: aiSnapshot.history || []
    property var recommendedModes: lastDecision.recommendedModes || []
    property int timelineCount: decisionTimeline.length
    property int recommendationCount: recommendedModes.length
    property string currentMode: lastDecision.mode || "preview"
    property var decisions: [
        ({ timestamp: "12:04:18Z", symbol: "BTC/USDT", action: "HOLD", confidence: "0.81", reason: qsTr("Momentum neutralny; trend nie pokonał confidence floor."), riskReason: qsTr("Max drawdown guard within preview limit; position cap not used."), strategy: "Momentum Guard", safety: qsTr("NO ORDER — preview only") }),
        ({ timestamp: "12:03:42Z", symbol: "ETH/USDT", action: "WAIT", confidence: "0.74", reason: qsTr("Training coverage preview wskazuje brak przewagi po kosztach."), riskReason: qsTr("Risk governor waits for lower slippage and fresh telemetry."), strategy: "Range Guard", safety: qsTr("Exchange I/O disabled") }),
        ({ timestamp: "12:02:57Z", symbol: "SOL/USDT", action: "BLOCKED LIVE", confidence: "0.69", reason: qsTr("Volatility breakout requires live guard, which is intentionally disabled."), riskReason: qsTr("Execution guard blocks order route; risk kill-switch armed."), strategy: "Volatility Breakout Preview", safety: qsTr("Live trading disabled, Order submission disabled") }),
        ({ timestamp: "12:01:33Z", symbol: "BNB/USDT", action: "NO ORDER", confidence: "0.61", reason: qsTr("Advisory preview rejected low confidence setup."), riskReason: qsTr("Below model readiness confidence floor."), strategy: "Strategy governor", safety: qsTr("Preview only / no order") })
    ]
    contentWidth: availableWidth
    clip: true

    function refreshSnapshot() {
        aiSnapshot = runtimeService && runtimeService.aiGovernorSnapshot ? runtimeService.aiGovernorSnapshot : ({})
        lastDecision = aiSnapshot.lastDecision || ({})
        decisionTimeline = aiSnapshot.history || []
        recommendedModes = lastDecision.recommendedModes || []
        timelineCount = decisionTimeline.length
        recommendationCount = recommendedModes.length
        currentMode = lastDecision.mode || "preview"
    }

    Component.onCompleted: refreshSnapshot()
    onRuntimeServiceChanged: refreshSnapshot()
    ColumnLayout {
        width: root.availableWidth
        spacing: 16
        Label { objectName: "aiDecisionsTitle"; text: qsTr("Decyzje"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Real-looking AI / Governor decision stream using static local preview rows. No empty history as the primary experience."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last decision"); description: qsTr("BTC/USDT HOLD • confidence 0.81 • NO ORDER — preview only"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Safety block state"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Decision source"); description: qsTr("Decision Governor Preview Core • Momentum Guard / Range Guard / Risk governor"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("AI / Governor decision preview")
            description: qsTr("Symbol, Action, Confidence, Reason, Risk reason, Strategy source, Timestamp, Safety block state, Preview only / no order.")
            Repeater {
                model: root.decisions
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: decisionColumn.implicitHeight + 24
                    radius: 14
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    border.width: 1
                    ColumnLayout {
                        id: decisionColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Label { text: modelData.symbol; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                            Label { text: modelData.action; color: modelData.action === "BLOCKED LIVE" ? designSystem.color("warning") : designSystem.color("accent"); font.bold: true }
                            Label { text: qsTr("Confidence %1").arg(modelData.confidence); color: designSystem.color("textSecondary") }
                            Label { text: qsTr("Timestamp %1").arg(modelData.timestamp); color: designSystem.color("textSecondary") }
                        }
                        Label { text: qsTr("Reason: %1").arg(modelData.reason); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Risk reason: %1").arg(modelData.riskReason); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Label { text: qsTr("Strategy source: %1 • Safety block state: %2").arg(modelData.strategy).arg(modelData.safety); color: designSystem.color("textPrimary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
