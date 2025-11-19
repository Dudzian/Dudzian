import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts
import "../design-system" as Design
import "../design-system/charts" as Charts

Item {
    id: root
    implicitWidth: 1080
    implicitHeight: 720

    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var snapshot: runtimeService && runtimeService.aiGovernorSnapshot ? runtimeService.aiGovernorSnapshot : ({})
    property var feedTransportSnapshot: runtimeService && runtimeService.feedTransportSnapshot ? runtimeService.feedTransportSnapshot : ({})
    property var feedSlaReport: runtimeService && runtimeService.feedSlaReport ? runtimeService.feedSlaReport : ({})
    property var feedHealth: runtimeService && runtimeService.feedHealth ? runtimeService.feedHealth : ({})
    property var history: snapshot && snapshot.history ? snapshot.history : []
    property int selectedIndex: (history.length > 0 ? 0 : -1)
    property var selectedDecision: (selectedIndex >= 0 && selectedIndex < history.length) ? history[selectedIndex] : ({})

    function refresh() {
        if (!runtimeService)
            return
        snapshot = runtimeService.reloadAiGovernorSnapshot()
        feedTransportSnapshot = runtimeService.feedTransportSnapshot
        feedSlaReport = runtimeService.feedSlaReport
        feedHealth = runtimeService.feedHealth
        history = snapshot && snapshot.history ? snapshot.history : []
        if (history.length === 0)
            selectedIndex = -1
        else if (selectedIndex >= history.length)
            selectedIndex = 0
    }

    function severityColor(state) {
        switch (state) {
        case "critical": return Design.Palette.warning
        case "warning": return Design.Palette.accent
        case "degraded": return Design.Palette.accent
        default: return Design.Palette.success
        }
    }

    function confidenceSeries() {
        var values = []
        for (var i = history.length - 1; i >= 0; --i) {
            var entry = history[i]
            var c = entry.confidence !== undefined ? Number(entry.confidence) * 100.0 : 0.0
            values.push({ "value": c })
        }
        return values
    }

    function latencySeries() {
        var values = []
        for (var i = history.length - 1; i >= 0; --i) {
            var entry = history[i]
            var latency = 0.0
            if (entry.telemetry && entry.telemetry.cycleLatency && entry.telemetry.cycleLatency.p95Ms !== undefined)
                latency = Number(entry.telemetry.cycleLatency.p95Ms)
            else if (entry.decision && entry.decision.latencyMs !== undefined)
                latency = Number(entry.decision.latencyMs)
            values.push({ "value": latency })
        }
        return values
    }

    function outcomeSeries() {
        var values = []
        for (var i = history.length - 1; i >= 0; --i) {
            var entry = history[i]
            var decision = entry.decision || {}
            var state = decision.state || ""
            var mapped = 0
            if (state === "executed" || state === "approved")
                mapped = 1
            else if (state === "rejected" || state === "declined")
                mapped = -1
            values.push({ "value": mapped })
        }
        return values
    }

    Connections {
        target: runtimeService
        ignoreUnknownSignals: true
        function onAiGovernorSnapshotChanged() {
            root.snapshot = runtimeService.aiGovernorSnapshot
            root.history = root.snapshot.history || []
            if (root.history.length === 0)
                root.selectedIndex = -1
            else if (root.selectedIndex >= root.history.length)
                root.selectedIndex = 0
        }
        function onFeedTransportSnapshotChanged() { root.feedTransportSnapshot = runtimeService.feedTransportSnapshot }
        function onFeedSlaReportChanged() { root.feedSlaReport = runtimeService.feedSlaReport }
        function onFeedHealthChanged() { root.feedHealth = runtimeService.feedHealth }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        padding: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Label {
                text: qsTr("Historia decyzji AI")
                font.bold: true
                font.pointSize: 17
                color: Design.Palette.textPrimary
            }
            Item { Layout.fillWidth: true }
            Button {
                text: qsTr("Odśwież")
                icon.name: "refresh"
                enabled: runtimeService !== null
                onClicked: refresh()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            visible: feedSlaReport && feedSlaReport.sla_state && feedSlaReport.sla_state !== "ok"
            color: severityColor(feedSlaReport ? feedSlaReport.sla_state : "ok")
            radius: 8
            implicitHeight: visible ? 64 : 0
            opacity: 0.9

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 2
                Label {
                    text: qsTr("Alert SLA feedu decyzji (%1)").arg(feedSlaReport.sla_state || "")
                    font.bold: true
                    color: Design.Palette.textOnAccent
                }
                Label {
                    text: qsTr("p95: %1 ms • reconnecty: %2 • downtime: %3 s")
                          .arg(Number(feedSlaReport.p95_ms || 0).toFixed(0))
                          .arg(feedSlaReport.reconnects || 0)
                          .arg(Number(feedSlaReport.downtime_seconds || 0).toFixed(0))
                    color: Design.Palette.textOnAccent
                    wrapMode: Text.WordWrap
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            Frame {
                Layout.fillHeight: true
                Layout.preferredWidth: 360
                Layout.alignment: Qt.AlignTop
                background: Rectangle { radius: 10; color: Design.Palette.surface }

                ListView {
                    id: decisionList
                    anchors.fill: parent
                    model: history
                    clip: true
                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 90
                        color: ListView.isCurrentItem ? Design.Palette.surfaceStrong : Design.Palette.surface
                        radius: 6
                        border.color: ListView.isCurrentItem ? Design.Palette.accent : Design.Palette.border
                        border.width: ListView.isCurrentItem ? 1.2 : 1
                        anchors.margins: 4

                        MouseArea {
                            anchors.fill: parent
                            onClicked: root.selectedIndex = index
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 4
                            Label {
                                text: (modelData.timestamp || "") + " • " + (modelData.mode || qsTr("tryb"))
                                font.bold: true
                                color: Design.Palette.textPrimary
                            }
                            Label {
                                text: modelData.reason || qsTr("Brak uzasadnienia")
                                color: Design.Palette.textSecondary
                                wrapMode: Text.Wrap
                            }
                            Label {
                                text: qsTr("Confidence: %1% • Reżim: %2")
                                      .arg(Number(modelData.confidence || 0).toLocaleString(Qt.locale(), "f", 0))
                                      .arg(modelData.regime || qsTr("n/d"))
                                color: Design.Palette.textSecondary
                            }
                        }
                    }
                }
            }

            Frame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                background: Rectangle { radius: 10; color: Design.Palette.surfaceStrong }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Label {
                            text: selectedDecision.mode || qsTr("Brak decyzji")
                            font.bold: true
                            font.pointSize: 15
                            color: Design.Palette.textPrimary
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: selectedDecision.timestamp || ""
                            color: Design.Palette.textSecondary
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        Label {
                            text: selectedDecision.reason || qsTr("Oczekiwanie na sygnał")
                            wrapMode: Text.WordWrap
                            color: Design.Palette.textSecondary
                            Layout.fillWidth: true
                        }
                        ColumnLayout {
                            spacing: 2
                            Label {
                                text: qsTr("Confidence: %1%")
                                      .arg(Number(selectedDecision.confidence || 0).toLocaleString(Qt.locale(), "f", 0))
                                color: Design.Palette.textPrimary
                            }
                            Label {
                                text: qsTr("Outcome: %1")
                                      .arg(selectedDecision.decision && selectedDecision.decision.state
                                           ? selectedDecision.decision.state
                                           : qsTr("n/d"))
                                color: Design.Palette.textSecondary
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Label { text: qsTr("Transport") ; color: Design.Palette.textSecondary }
                            Label {
                                text: feedTransportSnapshot && feedTransportSnapshot.status
                                      ? qsTr("%1 • %2")
                                            .arg(feedTransportSnapshot.status)
                                            .arg(feedTransportSnapshot.mode || "grpc")
                                      : qsTr("brak danych")
                                color: Design.Palette.textPrimary
                                font.bold: true
                            }
                            Label {
                                visible: feedTransportSnapshot && feedTransportSnapshot.lastError
                                text: qsTr("Ostatni błąd: %1").arg(feedTransportSnapshot.lastError || "")
                                color: Design.Palette.warning
                                wrapMode: Text.Wrap
                            }
                        }
                        ColumnLayout {
                            spacing: 6
                            Label { text: qsTr("Latencja p95") ; color: Design.Palette.textSecondary }
                            Label {
                                text: selectedDecision.telemetry && selectedDecision.telemetry.cycleLatency && selectedDecision.telemetry.cycleLatency.p95Ms !== undefined
                                      ? qsTr("%1 ms").arg(Number(selectedDecision.telemetry.cycleLatency.p95Ms).toFixed(0))
                                      : qsTr("n/d")
                                color: Design.Palette.textPrimary
                            }
                            Label {
                                text: qsTr("Rekomendowane tryby: %1")
                                      .arg(selectedDecision.recommendedModes && selectedDecision.recommendedModes.length > 0
                                           ? selectedDecision.recommendedModes.join(", ")
                                           : qsTr("brak"))
                                color: Design.Palette.textSecondary
                                wrapMode: Text.Wrap
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 10

                        Charts.PnlChart {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: 220
                            title: qsTr("Confidence (%)")
                            data: confidenceSeries()
                            lineColor: Design.Palette.accent
                        }

                        ChartView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: 220
                            antialiasing: true
                            legend.visible: false
                            theme: ChartView.ChartThemeDark
                            animationOptions: ChartView.NoAnimation
                            backgroundColor: Design.Palette.surface

                            ValueAxis { id: xAxis; min: 0; max: Math.max(1, history.length - 1); labelsVisible: false }
                            ValueAxis {
                                id: latencyAxis
                                min: 0
                                max: {
                                    var values = latencySeries()
                                    if (!values || values.length === 0)
                                        return 1
                                    var maxVal = 0
                                    for (var i = 0; i < values.length; ++i) {
                                        var val = Number(values[i].value || 0)
                                        if (val > maxVal)
                                            maxVal = val
                                    }
                                    return Math.max(1, maxVal) * 1.1
                                }
                            }
                            ValueAxis { id: outcomeAxis; min: -1.2; max: 1.2; labelsVisible: true; labelsColor: Design.Palette.textSecondary }

                            LineSeries {
                                axisX: xAxis
                                axisY: latencyAxis
                                color: Design.Palette.accent
                                width: 2
                                Component.onCompleted: rebuild()
                                function rebuild() {
                                    clear()
                                    var values = latencySeries()
                                    for (var i = 0; i < values.length; ++i)
                                        append(i, Number(values[i].value || 0))
                                }
                                Connections { target: root; function onHistoryChanged() { rebuild() } }
                            }

                            LineSeries {
                                axisX: xAxis
                                axisY: outcomeAxis
                                color: Design.Palette.success
                                width: 2
                                Component.onCompleted: rebuild()
                                function rebuild() {
                                    clear()
                                    var values = outcomeSeries()
                                    for (var i = 0; i < values.length; ++i)
                                        append(i, Number(values[i].value || 0))
                                }
                                Connections { target: root; function onHistoryChanged() { rebuild() } }
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Źródłowe sygnały")
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 6

                            Repeater {
                                model: selectedDecision && selectedDecision.signals ? selectedDecision.signals : []
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Label {
                                        text: modelData && modelData.name ? modelData.name : qsTr("sygnał")
                                        color: Design.Palette.textPrimary
                                        font.bold: true
                                    }
                                    Label {
                                        text: modelData && modelData.source ? modelData.source : ""
                                        color: Design.Palette.textSecondary
                                    }
                                    Item { Layout.fillWidth: true }
                                    Label {
                                        text: qsTr("w=%1").arg(Number(modelData.weight || 0).toLocaleString(Qt.locale(), "f", 2))
                                        color: Design.Palette.textSecondary
                                    }
                                    Label {
                                        text: modelData.value !== undefined ? Number(modelData.value).toLocaleString(Qt.locale(), "f", 3) : ""
                                        color: Design.Palette.textPrimary
                                    }
                                }
                            }

                            Label {
                                visible: !(selectedDecision && selectedDecision.signals && selectedDecision.signals.length > 0)
                                text: qsTr("Brak zarejestrowanych sygnałów źródłowych")
                                color: Design.Palette.textSecondary
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }
            }
        }
    }
}
