import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "aiControlCenterRoot"
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 16
        Label { objectName: "aiControlCenterTitle"; text: qsTr("AI Center / Centrum autonomii"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Centrum autonomii governora: model identity, readiness, data coverage, confidence, last decision and autonomy stack. Safe preview: no live, no order, no exchange I/O."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        GridLayout {
            Layout.fillWidth: true
            columns: 3
            rowSpacing: 12
            columnSpacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model identity"); description: qsTr("Decision Governor Preview Core"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model family/type"); description: qsTr("heuristic/governor/ML-ready preview"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Model version/build"); description: qsTr("preview-build 7.1-local • static local preview"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Training/readiness percent"); description: qsTr("Model readiness 72% • training readiness preview 72%"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Data coverage percent"); description: qsTr("Training/coverage 68% • Data coverage 68% • no live data"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Current autonomy mode"); description: qsTr("advisory / supervised dry-run • autonomy level 2 of 5"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Confidence preview"); description: qsTr("0.81 confidence preview for BTC/USDT HOLD"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Last decision and reason"); description: qsTr("BTC/USDT HOLD — momentum neutralny, risk governor guarded, NO ORDER — preview only"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Safety state"); description: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled • Runtime loop not started"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Autonomy Stack")
            description: qsTr("Każdy moduł działa jako static local preview; execution guard blokuje zlecenia.")
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 10
                columnSpacing: 12
                Repeater {
                    model: [
                        "Market scanner — mock/local preview feed",
                        "Strategy governor — advisory only",
                        "Risk governor — guarded limits",
                        "Execution guard — order route blocked",
                        "Recovery monitor — dry-run heartbeat",
                        "Telemetry monitor — local freshness checks",
                        "Safety kill-switch / Kill-switch — armed / preview"
                    ]
                    delegate: Rectangle {
                        required property string modelData
                        Layout.fillWidth: true
                        height: 44
                        radius: 12
                        color: designSystem.color("surfaceMuted")
                        border.color: designSystem.color("border")
                        Label { anchors.fill: parent; anchors.margins: 10; text: modelData; color: designSystem.color("textPrimary"); verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight }
                    }
                }
            }
        }
    }
}
