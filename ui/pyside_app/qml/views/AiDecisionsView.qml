import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "../components" as Components

Rectangle {
    id: root
    objectName: "aiDecisionsView"
    color: Qt.rgba(0, 0, 0, 0)
    radius: 18
    border.color: designSystem ? designSystem.color("border") : "#3C3F44"
    border.width: 1
    implicitHeight: 460

    property var runtimeService
    property var designSystem
    property var aiSnapshot: ({})
    property var lastDecision: aiSnapshot.lastDecision || ({})
    property var telemetry: aiSnapshot.telemetry || ({})
    property var decisionTimeline: aiSnapshot.history || []
    property var recommendedModes: lastDecision.recommendedModes || []
    property alias timelineCount: timelineView.count
    property alias recommendationCount: modeRepeater.count
    property string currentMode: lastDecision.mode || ""
    property real confidenceValue: lastDecision.confidence !== undefined ? Number(lastDecision.confidence) : 0.0
    property string selectedModeFilter: ""
    property int filteredTimelineCount: 0

    signal snapshotUpdated()

    function refreshSnapshot() {
        aiSnapshot = runtimeService ? runtimeService.aiGovernorSnapshot || ({}) : ({})
        lastDecision = aiSnapshot.lastDecision || ({})
        telemetry = aiSnapshot.telemetry || ({})
        decisionTimeline = aiSnapshot.history || []
        recommendedModes = lastDecision.recommendedModes || []
        snapshotUpdated()
        updateFilteredCount()
    }

    function confidencePercent(value) {
        return Math.round(Math.max(0.0, Math.min(1.0, value)) * 100)
    }

    function modeToken(mode) {
        if (!mode)
            return ""
        return String(mode).toLowerCase()
    }

    function modeGlyph(mode) {
        var token = modeToken(mode)
        switch (token) {
        case "scalping":
            return "\uf56b" // bolt
        case "hedge":
            return "\uf3ed" // shield-alt
        case "grid":
            return "\uf00a" // th
        default:
            return designSystem && designSystem.iconGlyph ? designSystem.iconGlyph("mode_wizard") : "\uf0c2"
        }
    }

    function modeColor(mode) {
        var token = modeToken(mode)
        if (!designSystem)
            return "#5bc8ff"
        switch (token) {
        case "scalping":
            return designSystem.color("accent")
        case "hedge":
            return designSystem.color("warning")
        case "grid":
            return designSystem.color("success")
        default:
            return designSystem.color("textPrimary")
        }
    }

    function matchesFilter(mode) {
        if (!selectedModeFilter || selectedModeFilter.length === 0)
            return true
        return modeToken(mode) === selectedModeFilter
    }

    function updateFilteredCount() {
        var items = decisionTimeline || []
        var count = 0
        for (var i = 0; i < items.length; ++i) {
            if (matchesFilter(items[i].mode))
                count += 1
        }
        filteredTimelineCount = count
    }

    function telemetryValue(section, key) {
        var container = telemetry && telemetry[section] ? telemetry[section] : ({})
        if (container[key] === undefined)
            return "—"
        var value = Number(container[key])
        if (isNaN(value))
            return container[key]
        if (section === "cycleMetrics")
            return Math.round(value).toLocaleString(Qt.locale(), "f", 0) + " ms"
        return value.toFixed(2)
    }

    Component.onCompleted: refreshSnapshot()

    Connections {
        target: runtimeService
        function onAiGovernorSnapshotChanged() {
            root.refreshSnapshot()
        }
    }

    onDecisionTimelineChanged: updateFilteredCount()
    onSelectedModeFilterChanged: updateFilteredCount()

    Rectangle {
        id: frostLayer
        anchors.fill: parent
        radius: root.radius
        gradient: Gradient {
            GradientStop { position: 0; color: designSystem ? designSystem.color("gradientHeroStart") : "#1f1f35" }
            GradientStop { position: 1; color: designSystem ? designSystem.color("gradientHeroEnd") : "#0d0d14" }
        }
        opacity: 0.65
        z: -2
    }

    MultiEffect {
        anchors.fill: frostLayer
        source: frostLayer
        blurEnabled: true
        blur: 1.0
        blurMax: 28
        saturation: 0.9
        brightness: 0.08
        z: -1
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Text {
                text: modeGlyph(currentMode)
                font.pixelSize: 26
                font.bold: true
                font.family: designSystem && designSystem.fontAwesomeFamily ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                color: modeColor(currentMode)
            }
            ColumnLayout {
                Layout.fillWidth: true
                Text {
                    text: qsTr("Decyzje AI")
                    font.bold: true
                    font.pointSize: 17
                    color: designSystem ? designSystem.color("textPrimary") : "#fdfdfd"
                }
                Text {
                    text: lastDecision.reason ? lastDecision.reason : qsTr("Oczekiwanie na rekomendację governora")
                    font.pointSize: 11
                    color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                    wrapMode: Text.WordWrap
                }
            }
            Components.IconButton {
                designSystem: designSystem
                subtle: true
                iconName: "refresh"
                text: qsTr("Odśwież")
                onClicked: runtimeService && runtimeService.reloadAiGovernorSnapshot()
            }
        }

        Rectangle {
            id: detailCard
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            radius: 14
            color: Qt.rgba(0, 0, 0, 0.35)
            border.color: designSystem ? designSystem.color("border") : "#2d2f37"
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Text {
                        text: currentMode.length > 0 ? currentMode.toUpperCase() : qsTr("BRAK TRYBU")
                        font.bold: true
                        font.pointSize: 15
                        color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: qsTr("Confidence %1%2").arg(confidencePercent(confidenceValue)).arg("%")
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                }

                ProgressBar {
                    Layout.fillWidth: true
                    value: Math.max(0.0, Math.min(1.0, confidenceValue))
                    from: 0
                    to: 1
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Text {
                        text: qsTr("Ryzyko: %1").arg(lastDecision.riskScore !== undefined ? Number(lastDecision.riskScore).toFixed(2) : "—")
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                    Text {
                        text: qsTr("Koszt: %1 bps").arg(lastDecision.transactionCostBps !== undefined ? Number(lastDecision.transactionCostBps).toFixed(1) : "—")
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                    Text {
                        text: qsTr("p95 cyklu: %1").arg(telemetryValue("cycleMetrics", "cycle_latency_p95_ms"))
                        color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                        font.pointSize: 11
                    }
                    Item { Layout.fillWidth: true }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: recommendationCount > 0
            Repeater {
                id: modeRepeater
                model: recommendedModes
                delegate: Rectangle {
                    required property var modelData
                    readonly property string modeName: modelData && modelData.mode ? String(modelData.mode) : String(modelData)
                    visible: modeName.length > 0
                    implicitWidth: modeRow.implicitWidth + 20
                    implicitHeight: modeRow.implicitHeight + 20
                    radius: 14
                    color: Qt.rgba(0, 0, 0, selectedModeFilter === modeToken(modeName) ? 0.55 : 0.3)
                    border.width: selectedModeFilter === modeToken(modeName) ? 2 : 1
                    border.color: modeColor(modeName)

                    RowLayout {
                        id: modeRow
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 6
                        Text {
                            text: modeGlyph(modeName)
                            font.pixelSize: 14
                            font.family: designSystem && designSystem.fontAwesomeFamily ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                            color: modeColor(modeName)
                        }
                        Text {
                            text: modeName.toUpperCase()
                            font.bold: true
                            color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
                            font.pointSize: 11
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            var token = modeToken(modeName)
                            if (selectedModeFilter === token)
                                selectedModeFilter = ""
                            else
                                selectedModeFilter = token
                        }
                    }
                }
            }
        }

        ListView {
            id: timelineView
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
            clip: true
            model: decisionTimeline
            delegate: Item {
                width: ListView.view.width
                height: entryCard.implicitHeight + 10
                visible: matchesFilter(modelData.mode)

                Rectangle {
                    id: entryCard
                    anchors.fill: parent
                    radius: 12
                    color: Qt.rgba(0, 0, 0, 0.28)
                    border.color: designSystem ? designSystem.color("border") : "#2d2f37"
                    border.width: 1
                    anchors.margins: 2

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        RowLayout {
                            spacing: 8
                            Text {
                                text: modeGlyph(modelData.mode)
                                font.pixelSize: 16
                                font.family: designSystem && designSystem.fontAwesomeFamily ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                                color: modeColor(modelData.mode)
                            }
                            Text {
                                text: modelData.mode ? String(modelData.mode).toUpperCase() : qsTr("n/d")
                                font.bold: true
                                color: designSystem ? designSystem.color("textPrimary") : "#ffffff"
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: modelData.timestamp || ""
                                font.pointSize: 10
                                color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                            }
                        }
                        Text {
                            text: modelData.reason || qsTr("Brak uzasadnienia")
                            wrapMode: Text.WordWrap
                            color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                            font.pointSize: 11
                        }
                        RowLayout {
                            spacing: 10
                            Text {
                                text: qsTr("Confidence %1%2").arg(confidencePercent(modelData.confidence || 0.0)).arg("%")
                                font.pointSize: 10
                                color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                            }
                            Text {
                                text: modelData.regime ? qsTr("Regime %1").arg(modelData.regime) : ""
                                font.pointSize: 10
                                color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
                            }
                        }
                    }
                }
            }
        }

        Text {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            visible: filteredTimelineCount === 0
            text: qsTr("Brak historii AI Governora")
            color: designSystem ? designSystem.color("textSecondary") : "#c5cad3"
        }
    }
}
