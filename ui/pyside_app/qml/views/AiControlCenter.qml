import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiControlCenterRoot"
    property var runtimeService
    property var previewState
    property var policies: ["Conservative policy", "Balanced policy", "Opportunity policy"]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "aiControlCenterTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "aiControlCenterTitle"; text: qsTr("AI Center"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Centrum autonomii for supervised Paper preview. Model, governor and policy controls are visible while live execution remains locked."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1100 ? 3 : 1
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Decision Governor Preview Core"); description: qsTr("Model family/type: policy ensemble • Model version/build: preview-7.4 • Active AI model / governor engine") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Readiness badge"); description: qsTr("READY FOR PAPER PREVIEW • Live trading disabled • Exchange route disabled") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Current autonomy mode • Opportunity governor mode"); description: qsTr("%1 • Autonomy level %2/5").arg(previewState.autonomyMode).arg(previewState.autonomyLevel) }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Model readiness")
            description: qsTr("Visual readiness indicators: Model readiness %, Training/readiness percent, Training/coverage, Data coverage percent.")
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 10
                columnSpacing: 12
                Label { text: qsTr("Model readiness %"); color: designSystem.color("textPrimary") }
                ProgressBar { objectName: "aiReadinessProgressBar"; from: 0; to: 100; value: previewState.modelReadiness; Layout.fillWidth: true }
                Label { text: qsTr("Training/readiness percent • Training/coverage"); color: designSystem.color("textPrimary") }
                ProgressBar { objectName: "aiTrainingCoverageProgressBar"; from: 0; to: 100; value: previewState.trainingCoverage; Layout.fillWidth: true }
                Label { text: qsTr("Data coverage percent"); color: designSystem.color("textPrimary") }
                ProgressBar { objectName: "aiDataCoverageProgressBar"; from: 0; to: 100; value: previewState.dataCoverage; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Autonomy and policy controls")
            description: qsTr("Autonomy level 1–5, policy selector and confidence threshold change local preview state only.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Advisory"); subtle: previewState.autonomyMode !== "Advisory"; onClicked: previewState.setAutonomyMode("Advisory") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Supervised dry-run"); subtle: previewState.autonomyMode !== "Supervised dry-run"; onClicked: previewState.setAutonomyMode("Supervised dry-run") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Autonomous paper"); subtle: previewState.autonomyMode !== "Autonomous paper"; onClicked: previewState.setAutonomyMode("Autonomous paper") }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater { model: root.policies; delegate: Components.IconButton { required property string modelData; designSystem: root.designSystem; text: modelData; subtle: previewState.decisionPolicyPreview !== modelData; onClicked: previewState.setDecisionPolicy(modelData) } }
            }
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("confidence threshold"); color: designSystem.color("textPrimary") }
                Components.StyledSpinBox { objectName: "aiConfidenceThresholdControl"; designSystem: root.designSystem; from: 50; to: 95; value: previewState.confidenceThreshold; onValueModified: previewState.setConfidenceThreshold(value); Layout.preferredWidth: 120 }
                Label { text: qsTr("%1% • policy: %2").arg(previewState.confidenceThreshold).arg(previewState.decisionPolicyPreview); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Governor recommendation stream")
            description: qsTr("Generate governor recommendation changes lastGovernorDecision, appends a decision stream row, rotates pair/action and updates timestamp.")
            RowLayout {
                Layout.fillWidth: true
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); iconName: "mode_wizard"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.generateGovernorRecommendation() }
                Label { text: previewState.lastGovernorDecision; color: designSystem.color("textPrimary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 3 : 1
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Market scanner"); description: qsTr("AI scans eligible preview pairs only; exchange route disabled.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Strategy governor"); description: qsTr("Routes signals to enabled preview strategies.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk governor"); description: qsTr("Applies risk profile and kill-switch before any paper action.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution guard"); description: qsTr("Live route locked; order submission disabled.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Recovery monitor"); description: qsTr("Runtime loop not started; recovery bridge planned.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Telemetry monitor"); description: qsTr("Safe preview feed with heartbeat freshness.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Kill-switch"); description: qsTr("Armed in preview and visible to the operator.") }
        }
    }
}
