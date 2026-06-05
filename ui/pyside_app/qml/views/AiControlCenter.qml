import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiControlCenterRoot"
    property var runtimeService
    property var previewState
    property var policies: ["Polityka konserwatywna", "Polityka zbalansowana", "Polityka oportunistyczna"]
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
                Label { text: qsTr("Centrum autonomii dla nadzorowanego Paper Preview. Model, Governor i polityki są klikalne lokalnie, a live trading, exchange route i order submission pozostają wyłączone."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1100 ? 3 : 1
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Decision Governor Preview Core"); description: qsTr("Active AI model / governor engine: %1 • Model family/type: policy ensemble • Model version/build: %2").arg(previewState.activeGovernorEngine).arg(previewState.modelVersionBuild) }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Readiness badge"); description: qsTr("READY FOR PAPER PREVIEW • Model readiness %1% • Live trading disabled • Exchange route disabled • Order submission disabled • API keys not required").arg(previewState.modelReadiness) }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Current autonomy mode • Opportunity governor mode"); description: qsTr("%1 • Autonomy level %2/5 • confidence threshold %3% • decision policy: %4").arg(previewState.autonomyMode).arg(previewState.autonomyLevel).arg(previewState.confidenceThreshold).arg(previewState.decisionPolicyPreview) }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Model readiness")
            description: qsTr("StyledProgressBar indicators: Model readiness %, Training/readiness percent, Training/coverage, Data coverage percent. Ciemne tło, cyan fill, rounded corners i czytelny procent bez natywnego wyglądu systemowego.")
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 10
                columnSpacing: 12
                Label { text: qsTr("Model readiness %"); color: designSystem.color("textPrimary") }
                Components.StyledProgressBar { objectName: "aiReadinessProgressBar"; designSystem: root.designSystem; value: previewState.modelReadiness; label: previewState.modelReadiness + "%"; Layout.fillWidth: true }
                Label { text: qsTr("Training/readiness percent • Training/coverage"); color: designSystem.color("textPrimary") }
                Components.StyledProgressBar { objectName: "aiTrainingCoverageProgressBar"; designSystem: root.designSystem; value: previewState.trainingCoverage; label: previewState.trainingCoverage + "%"; Layout.fillWidth: true }
                Label { text: qsTr("Data coverage percent"); color: designSystem.color("textPrimary") }
                Components.StyledProgressBar { objectName: "aiDataCoverageProgressBar"; designSystem: root.designSystem; value: previewState.dataCoverage; label: previewState.dataCoverage + "%"; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Autonomy and policy controls")
            description: qsTr("Autonomy level 1–5, policy selector and confidence threshold change local preview state only.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Doradczy"); subtle: previewState.autonomyMode !== "Advisory"; onClicked: previewState.setAutonomyMode("Advisory") }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Nadzorowany dry-run"); subtle: previewState.autonomyMode !== "Supervised dry-run"; onClicked: previewState.setAutonomyMode("Supervised dry-run") }
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
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Generate governor recommendation"); helpText: previewState.tooltipText("Generate governor recommendation"); iconName: "mode_wizard"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.generateGovernorRecommendation() }
                Label { text: previewState.lastGovernorDecision; color: designSystem.color("textPrimary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 3 : 1
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Market scanner"); description: qsTr("AI scans eligible preview pairs only; exchange route disabled. AI candidates: %1").arg(previewState.scannerCandidateCount) }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Strategy governor"); description: qsTr("Routes signals to enabled preview strategies.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Risk governor"); description: qsTr("Applies risk profile and kill-switch before any paper action.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Execution guard"); description: qsTr("Live route locked; order submission disabled.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Recovery monitor"); description: qsTr("Runtime loop not started; recovery bridge planned.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Telemetry monitor"); description: qsTr("Safe preview feed with heartbeat freshness.") }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Kill-switch"); description: qsTr("Armed in preview and visible to the operator.") }
        }
    }
}
