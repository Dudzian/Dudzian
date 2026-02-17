import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "../styles" as Styles
import "../design-system" as Design
import "." as Dashboard

Item {
    id: root
    implicitWidth: 1024
    implicitHeight: 640
    width: implicitWidth
    height: implicitHeight

    property var telemetryProviderObj: (typeof telemetryProvider !== "undefined" ? telemetryProvider : null)
    readonly property string telemetryLastUpdated: root.telemetryProviderObj && root.telemetryProviderObj.lastUpdated
                                          ? root.telemetryProviderObj.lastUpdated
                                          : ""
    readonly property string telemetryErrorMessage: root.telemetryProviderObj && root.telemetryProviderObj.errorMessage
                                           ? root.telemetryProviderObj.errorMessage
                                           : ""
    readonly property var telemetryIoQueues: root.telemetryProviderObj && root.telemetryProviderObj.ioQueues
                                    ? root.telemetryProviderObj.ioQueues
                                    : []
    readonly property var guardrailSummary: root.telemetryProviderObj && root.telemetryProviderObj.guardrailSummary
                                   ? root.telemetryProviderObj.guardrailSummary
                                   : ({})
    readonly property var retrainingRuns: root.telemetryProviderObj && root.telemetryProviderObj.retraining
                                 ? root.telemetryProviderObj.retraining
                                 : []
    property var dashboardSettingsController: null
    property var complianceController: null
    property var runtimeServiceObj: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var reportController: null
    property int refreshIntervalMs: dashboardSettingsController ? dashboardSettingsController.refreshIntervalMs : 4000
    readonly property var defaultCardOrder: ["feed_sla", "io_queue", "guardrails", "retraining", "compliance", "risk_journal", "ai_decisions"]
    readonly property var effectiveCardOrder: (dashboardSettingsController
                                            && dashboardSettingsController.visibleCardOrder
                                            && dashboardSettingsController.visibleCardOrder.length > 0)
                                            ? dashboardSettingsController.visibleCardOrder
                                            : defaultCardOrder
    readonly property var effectiveGridCardOrder: cardOrderWithoutStandaloneCards(effectiveCardOrder)
    property var aiDecisions: []
    property string aiDecisionError: ""
    property string retrainSchedulerNextRun: runtimeServiceObj && runtimeServiceObj.retrainNextRun
                                            ? runtimeServiceObj.retrainNextRun
                                            : ""
    property string adaptiveStrategySummary: runtimeServiceObj && runtimeServiceObj.adaptiveStrategySummary
                                              ? runtimeServiceObj.adaptiveStrategySummary
                                              : ""
    property string regimeActivationSummary: runtimeServiceObj && runtimeServiceObj.regimeActivationSummary
                                             ? runtimeServiceObj.regimeActivationSummary
                                             : ""
    property var riskMetrics: runtimeServiceObj && runtimeServiceObj.riskMetrics ? runtimeServiceObj.riskMetrics : ({})
    property var riskTimeline: runtimeServiceObj && runtimeServiceObj.riskTimeline ? runtimeServiceObj.riskTimeline : []
    property var lastOperatorAction: runtimeServiceObj && runtimeServiceObj.lastOperatorAction ? runtimeServiceObj.lastOperatorAction : ({})
    property var longPollMetrics: runtimeServiceObj && runtimeServiceObj.longPollMetrics ? runtimeServiceObj.longPollMetrics : []
    onLongPollMetricsChanged: root.syncLongPollMetricsModel()
    ListModel {
        id: longPollMetricsListModel
    }
    property alias longPollMetricsModel: longPollMetricsListModel
    property var cycleMetrics: runtimeServiceObj && runtimeServiceObj.cycleMetrics ? runtimeServiceObj.cycleMetrics : ({})
    property var feedTransportSnapshot: runtimeServiceObj && runtimeServiceObj.feedTransportSnapshot
                                        ? runtimeServiceObj.feedTransportSnapshot
                                        : ({})
    property var feedHealth: runtimeServiceObj && runtimeServiceObj.feedHealth ? runtimeServiceObj.feedHealth : ({})
    property var feedSlaReport: runtimeServiceObj && runtimeServiceObj.feedSlaReport ? runtimeServiceObj.feedSlaReport : ({})
    property var feedAlertHistory: runtimeServiceObj && runtimeServiceObj.feedAlertHistory ? runtimeServiceObj.feedAlertHistory : []
    property var feedAlertChannels: runtimeServiceObj && runtimeServiceObj.feedAlertChannels ? runtimeServiceObj.feedAlertChannels : []
    property var aiRegimeBreakdown: runtimeServiceObj && runtimeServiceObj.aiRegimeBreakdown
                                    ? runtimeServiceObj.aiRegimeBreakdown
                                    : []
    property string feedLatencyAlertState: "ok"
    property real feedLatencyAlertValue: 0.0
    property string feedLatencyAlertTicket: ""
    readonly property url feedSlaRunbookUrl: Qt.resolvedUrl("../../docs/runbooks/operations/feed_sla.md")

    function componentForCard(cardId) {
        switch (cardId) {
        case "feed_sla":
            return feedSlaCardComponent
        case "io_queue":
            return ioCardComponent
        case "guardrails":
        case "guardrail":
        case "guardrails_card":
        case "guardrail_card":
            return guardrailCardComponent
        case "retraining":
            return retrainingCardComponent
        case "compliance":
            return complianceCardComponent
        case "risk_journal":
            return riskJournalCardComponent
        case "ai_decisions":
            return aiDecisionCardComponent
        default:
            console.warn("RuntimeOverview: unknown card id", cardId)
            return null
        }
    }

    function cardOrderWithoutStandaloneCards(cardOrder) {
        const filtered = []
        const source = cardOrder || []
        for (let i = 0; i < source.length; ++i) {
            if (source[i] !== "ai_decisions" && source[i] !== "risk_journal")
                filtered.push(source[i])
        }
        return filtered
    }

    function hasAiDecisionsCard(cardOrder) {
        const source = cardOrder || []
        for (let i = 0; i < source.length; ++i) {
            if (source[i] === "ai_decisions")
                return true
        }
        return false
    }

    function hasRiskJournalCard(cardOrder) {
        const source = cardOrder || []
        for (let i = 0; i < source.length; ++i) {
            if (source[i] === "risk_journal")
                return true
        }
        return false
    }

    function refreshTelemetry() {
        if (!root.telemetryProviderObj)
            return
        root.telemetryProviderObj.refreshTelemetry()
    }

    function refreshDecisions() {
        if (!root.runtimeServiceObj)
            return
        const result = root.runtimeServiceObj.loadRecentDecisions(10)
        if (Array.isArray(result))
            root.aiDecisions = result
        root.riskMetrics = root.runtimeServiceObj ? root.runtimeServiceObj.riskMetrics : ({})
        root.riskTimeline = root.runtimeServiceObj ? root.runtimeServiceObj.riskTimeline : []
        root.lastOperatorAction = root.runtimeServiceObj ? root.runtimeServiceObj.lastOperatorAction : ({})
        root.longPollMetrics = root.runtimeServiceObj ? root.runtimeServiceObj.longPollMetrics : []
        root.syncLongPollMetricsModel()
        root.cycleMetrics = root.runtimeServiceObj ? root.runtimeServiceObj.cycleMetrics : ({})
        root.feedTransportSnapshot = root.runtimeServiceObj ? root.runtimeServiceObj.feedTransportSnapshot : ({})
        root.feedHealth = root.runtimeServiceObj ? root.runtimeServiceObj.feedHealth : ({})
        root.feedSlaReport = root.runtimeServiceObj ? root.runtimeServiceObj.feedSlaReport : ({})
        root.feedAlertHistory = root.runtimeServiceObj && root.runtimeServiceObj.feedAlertHistory
                ? root.runtimeServiceObj.feedAlertHistory
                : []
        root.feedAlertChannels = root.runtimeServiceObj && root.runtimeServiceObj.feedAlertChannels
                ? root.runtimeServiceObj.feedAlertChannels
                : []
        root.syncLatencyAlert()
        root.aiRegimeBreakdown = root.runtimeServiceObj ? root.runtimeServiceObj.aiRegimeBreakdown : []
        root.adaptiveStrategySummary = root.runtimeServiceObj ? root.runtimeServiceObj.adaptiveStrategySummary : ""
        root.regimeActivationSummary = root.runtimeServiceObj ? root.runtimeServiceObj.regimeActivationSummary : ""
    }

    function syncLongPollMetricsModel() {
        longPollMetricsListModel.clear()
        const source = root.longPollMetrics || []
        const size = source.length || 0
        for (let i = 0; i < size; ++i) {
            const entry = source[i]
            if (!entry || typeof entry !== "object") {
                longPollMetricsListModel.append({})
                continue
            }
            if (Array.isArray(entry)) {
                longPollMetricsListModel.append({ raw: entry })
                continue
            }
            longPollMetricsListModel.append(Object.assign({}, entry))
        }
    }

    function refreshAll() {
        root.refreshTelemetry()
        root.refreshDecisions()
    }

    onRuntimeServiceObjChanged: {
        root.longPollMetrics = root.runtimeServiceObj && root.runtimeServiceObj.longPollMetrics
                ? root.runtimeServiceObj.longPollMetrics
                : []
        root.syncLongPollMetricsModel()
    }

    Component.onCompleted: root.syncLongPollMetricsModel()

    function slaSeverityColor(state) {
        if (state === "critical")
            return Qt.rgba(0.9, 0.2, 0.25, 1)
        if (state === "warning")
            return Qt.rgba(0.95, 0.65, 0.2, 1)
        return Styles.AppTheme.textPrimary
    }

    function statusLabel(slaState, transportStatus) {
        if (transportStatus)
            return qsTr("%1 • %2")
                    .arg(transportStatus)
                    .arg(root.feedTransportSnapshot.mode || "demo")
        if (slaState && slaState !== "ok")
            return qsTr("SLA: %1").arg(slaState)
        return qsTr("Status transportu nieznany")
    }

    function syncLatencyAlert() {
        const report = root.feedSlaReport || ({})
        const severity = report.latency_state || "ok"
        const previous = root.feedLatencyAlertState
        const p95 = report.p95_ms !== undefined ? Number(report.p95_ms) : 0.0
        root.feedLatencyAlertState = severity
        root.feedLatencyAlertValue = p95
        if (severity === previous)
            return
        if (severity !== "warning" && severity !== "critical")
            return
        const threshold = severity === "critical"
                ? Number(report.latency_critical_ms || 0)
                : Number(report.latency_warning_ms || 0)
        root.feedLatencyAlertTicket = "sla-feed-" + Date.now()
        if (root.reportController && root.reportController.logOperationalAlert) {
            const title = severity === "critical"
                    ? qsTr("ALERT: SLA feedu przekroczone")
                    : qsTr("Ostrzeżenie SLA feedu")
            const message = qsTr("Latencja p95 = %1 ms (limit %2 ms)")
                    .arg(p95.toFixed(0))
                    .arg(threshold.toFixed(0))
            root.reportController.logOperationalAlert("sla.feed", {
                severity: severity,
                title: title,
                message: message,
                latency_ms: p95,
                warning_ms: Number(report.latency_warning_ms || 0),
                critical_ms: Number(report.latency_critical_ms || 0),
                ticket: root.feedLatencyAlertTicket
            })
        }
    }

    Timer {
        id: refreshTimer
        interval: Math.max(1500, root.refreshIntervalMs)
        repeat: true
        // W testach headless (QQmlEngine.load(Item)) komponent zwykle nie jest osadzony
        // w Window, więc wyłączamy auto-refresh aby uniknąć wyścigu z asercjami.
        running: !!root.telemetryProviderObj && !!root.Window.window
        triggeredOnStart: true
        onTriggered: root.refreshAll()
    }

    Connections {
        target: root.telemetryProviderObj
        ignoreUnknownSignals: true
        function onErrorMessageChanged() {
            errorBanner.visible = root.telemetryErrorMessage.length > 0
        }
    }

    Connections {
        target: root.runtimeServiceObj
        ignoreUnknownSignals: true

        function onDecisionsChanged() {
            if (root.runtimeServiceObj)
                root.aiDecisions = root.runtimeServiceObj.decisions
        }

        function onErrorMessageChanged() {
            if (!root.runtimeServiceObj)
                return
            root.aiDecisionError = root.runtimeServiceObj.errorMessage
        }

        function onRiskMetricsChanged() {
            if (!root.runtimeServiceObj)
                return
            root.riskMetrics = root.runtimeServiceObj.riskMetrics
        }

        function onRiskTimelineChanged() {
            if (!root.runtimeServiceObj)
                return
            root.riskTimeline = root.runtimeServiceObj.riskTimeline
        }

        function onOperatorActionChanged() {
            if (!root.runtimeServiceObj)
                return
            root.lastOperatorAction = root.runtimeServiceObj.lastOperatorAction
        }

        function onLongPollMetricsChanged() {
            if (!root.runtimeServiceObj)
                return
            root.longPollMetrics = root.runtimeServiceObj.longPollMetrics
            root.syncLongPollMetricsModel()
        }

        function onCycleMetricsChanged() {
            if (!root.runtimeServiceObj)
                return
            root.cycleMetrics = root.runtimeServiceObj.cycleMetrics
        }

        function onFeedTransportSnapshotChanged() {
            if (!root.runtimeServiceObj)
                return
            root.feedTransportSnapshot = root.runtimeServiceObj.feedTransportSnapshot
        }

        function onFeedHealthChanged() {
            if (!root.runtimeServiceObj)
                return
            root.feedHealth = root.runtimeServiceObj.feedHealth
        }

        function onFeedSlaReportChanged() {
            if (!root.runtimeServiceObj)
                return
            root.feedSlaReport = root.runtimeServiceObj.feedSlaReport
            root.syncLatencyAlert()
        }

        function onFeedAlertHistoryChanged() {
            if (!root.runtimeServiceObj)
                return
            root.feedAlertHistory = root.runtimeServiceObj.feedAlertHistory
        }

        function onFeedAlertChannelsChanged() {
            if (!root.runtimeServiceObj)
                return
            root.feedAlertChannels = root.runtimeServiceObj.feedAlertChannels
        }

        function onAiRegimeBreakdownChanged() {
            if (!root.runtimeServiceObj)
                return
            root.aiRegimeBreakdown = root.runtimeServiceObj.aiRegimeBreakdown
        }

        function onAdaptiveStrategySummaryChanged() {
            if (!root.runtimeServiceObj)
                return
            root.adaptiveStrategySummary = root.runtimeServiceObj.adaptiveStrategySummary
        }

        function onRegimeActivationSummaryChanged() {
            if (!root.runtimeServiceObj)
                return
            root.regimeActivationSummary = root.runtimeServiceObj.regimeActivationSummary
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Text {
                id: lastUpdatedLabel
                objectName: "runtimeOverviewLastUpdated"
                text: qsTr("Ostatnia aktualizacja: %1")
                      .arg(root.telemetryLastUpdated.length > 0
                           ? root.telemetryLastUpdated
                           : qsTr("n/d"))
                color: Styles.AppTheme.textSecondary
                font.pointSize: 12
            }

            Item { Layout.fillWidth: true }

            Button {
                id: manualRefreshButton
                objectName: "manualRefreshButton"
                text: qsTr("Odśwież")
                enabled: !!root.telemetryProviderObj
                onClicked: root.refreshAll()
            }
        }

        Rectangle {
            id: errorBanner
            objectName: "runtimeOverviewErrorBanner"
            Layout.fillWidth: true
            visible: false
            color: Qt.rgba(0.75, 0.25, 0.28, 0.9)
            radius: 6
            implicitHeight: visible ? 36 : 0

            Text {
                anchors.centerIn: parent
                text: root.telemetryErrorMessage
                color: "white"
                font.pointSize: 11
            }
        }

        Design.StrategyAiPanel {
            id: strategyAiPanel
            objectName: "runtimeOverviewStrategyAiPanel"
            Layout.fillWidth: true
            runtimeService: root.runtimeServiceObj
            longPollMetricsModel: root.longPollMetricsModel
            feedTransportSnapshot: root.feedTransportSnapshot
            aiRegimes: root.aiRegimeBreakdown
            adaptiveSummary: root.adaptiveStrategySummary
            activationSummary: root.regimeActivationSummary
        }

        Design.ResponsiveGrid {
            id: cardsGrid
            Layout.fillWidth: true
            Layout.fillHeight: true
            minColumnWidth: 360
            maxColumns: 3

            Loader {
                id: aiDecisionCardLoader
                objectName: "runtimeOverviewAiCard"
                Layout.fillWidth: true
                Layout.fillHeight: true
                asynchronous: false
                sourceComponent: aiDecisionCardComponent
                active: root.hasAiDecisionsCard(root.effectiveCardOrder)
                onStatusChanged: {
                    if (status === Loader.Error)
                        console.error("Card load error:", "ai_decisions", errorString())
                }
            }

            Loader {
                id: riskJournalCardLoader
                objectName: "runtimeOverviewRiskJournalCard"
                Layout.fillWidth: true
                Layout.fillHeight: true
                asynchronous: false
                sourceComponent: riskJournalCardComponent
                active: root.hasRiskJournalCard(root.effectiveCardOrder)
                onStatusChanged: {
                    if (status === Loader.Error)
                        console.error("Card load error:", "risk_journal", errorString())
                }
            }

            Repeater {
                id: cardRepeater
                model: root.effectiveGridCardOrder
                delegate: Loader {
                    readonly property string cardId: modelData
                    objectName: "runtimeOverviewCardLoader_" + cardId
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    asynchronous: false
                    sourceComponent: root.componentForCard(cardId)
                    active: sourceComponent !== null
                    onStatusChanged: {
                        if (status === Loader.Error)
                            console.error("Card load error:", cardId, errorString())
                    }
                }
            }
        }

        Label {
            visible: root.effectiveCardOrder.length === 0
            text: qsTr("Wszystkie karty zostały ukryte w ustawieniach dashboardu")
            color: Styles.AppTheme.textSecondary
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
        }
    }

    Component {
        id: feedSlaCardComponent
        Rectangle {
            id: feedSlaCard
            objectName: "runtimeOverviewFeedSlaCard"
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1

            property var report: root.feedSlaReport || ({})
            property var alertHistory: root.feedAlertHistory || []
            property var alertChannels: root.feedAlertChannels || []
            property string alertChannelsText: ""
            property string sla_state: report && report.sla_state ? report.sla_state : "ok"
            property string severity: feedSlaCard.sla_state
            readonly property bool latencyAlertActive: !!(report && report.latency_state && report.latency_state !== "ok")

            onAlertChannelsChanged: {
                var channels = alertChannels || []
                alertChannelsText = channels.map(function(channel) {
                    return channel.name + " (" + (channel.status || channel.state || "n/a") + ")"
                }).join(", ")
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10

                Text {
                    text: qsTr("SLA decision feed")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                Item {
                    id: slaStateLabel
                    objectName: "runtimeOverviewSlaStateLabel"
                    Layout.fillWidth: true
                    property string text: statusLabel(feedSlaCard.sla_state, root.feedTransportSnapshot.status)
                    property color color: slaSeverityColor(feedSlaCard.severity)
                    implicitHeight: slaStateText.implicitHeight

                    Text {
                        id: slaStateText
                        objectName: parent.objectName
                        anchors.fill: parent
                        text: parent.text
                        color: parent.color
                        font.pointSize: 13
                        font.bold: true
                        elide: Text.ElideRight
                        wrapMode: Text.NoWrap
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 16

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        Item {
                            objectName: "runtimeOverviewSlaLatency"
                            property string text: report && report.p95_ms !== undefined
                                                  ? qsTr("Latencja p95: %1 ms (limit %2 ms)")
                                                        .arg(Number(report.p95_ms).toFixed(0))
                                                        .arg(report.latency_warning_ms || 0)
                                                  : qsTr("Brak pomiarów latencji")
                            property color color: slaSeverityColor(report ? report.latency_state : "ok")
                            Layout.fillWidth: true
                            implicitHeight: slaLatencyText.implicitHeight

                            Text {
                                id: slaLatencyText
                                objectName: parent.objectName
                                anchors.fill: parent
                                text: parent.text
                                color: parent.color
                                wrapMode: Text.WordWrap
                            }
                        }

                        Label {
                            objectName: "runtimeOverviewSlaP50"
                            text: report && report.p50_ms !== undefined
                                  ? qsTr("Latencja p50: %1 ms").arg(Number(report.p50_ms).toFixed(0))
                                  : qsTr("Latencja p50: n/d")
                            color: Styles.AppTheme.textSecondary
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        Item {
                            objectName: "runtimeOverviewSlaReconnects"
                            property string text: report && report.reconnects !== undefined
                                                  ? qsTr("Reconnecty: %1 / próg %2")
                                                        .arg(report.reconnects)
                                                        .arg(report.reconnects_warning || 0)
                                                  : qsTr("Reconnecty: n/d")
                            property color color: slaSeverityColor(report ? report.reconnects_state : "ok")
                            Layout.fillWidth: true
                            implicitHeight: slaReconnectsText.implicitHeight

                            Text {
                                id: slaReconnectsText
                                objectName: parent.objectName
                                anchors.fill: parent
                                text: parent.text
                                color: parent.color
                                wrapMode: Text.WordWrap
                            }
                        }

                        Item {
                            objectName: "runtimeOverviewSlaDowntime"
                            property string text: report && report.downtime_seconds !== undefined
                                                  ? qsTr("Downtime: %1 s / próg %2 s")
                                                        .arg(Number(report.downtime_seconds || 0).toFixed(1))
                                                        .arg(report.downtime_warning_seconds || 0)
                                                  : qsTr("Downtime: n/d")
                            property color color: slaSeverityColor(report ? report.downtime_state : "ok")
                            Layout.fillWidth: true
                            implicitHeight: slaDowntimeText.implicitHeight

                            Text {
                                id: slaDowntimeText
                                objectName: parent.objectName
                                anchors.fill: parent
                                text: parent.text
                                color: parent.color
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }

                Item {
                    objectName: "runtimeOverviewSlaLastError"
                    property string text: qsTr("Ostatni błąd: %1").arg(root.feedHealth.lastError)
                    property color color: Styles.AppTheme.warning
                    visible: !!(root.feedHealth && root.feedHealth.lastError && root.feedHealth.lastError.length > 0)
                    Layout.fillWidth: true
                    implicitHeight: slaLastErrorText.implicitHeight

                    Text {
                        id: slaLastErrorText
                        objectName: parent.objectName
                        anchors.fill: parent
                        text: parent.text
                        color: parent.color
                        wrapMode: Text.WordWrap
                    }
                }

                Item {
                    objectName: "runtimeOverviewSlaRetry"
                    property string text: qsTr("Następny reconnect za %1 s")
                                          .arg(report && report.nextRetrySeconds !== undefined && report.nextRetrySeconds !== null
                                               ? Number(report.nextRetrySeconds).toFixed(1)
                                               : "n/d")
                    property color color: Styles.AppTheme.textSecondary
                    visible: report && report.nextRetrySeconds !== undefined && report.nextRetrySeconds !== null
                    Layout.fillWidth: true
                    implicitHeight: slaRetryText.implicitHeight

                    Text {
                        id: slaRetryText
                        objectName: parent.objectName
                        anchors.fill: parent
                        text: parent.text
                        color: parent.color
                        wrapMode: Text.WordWrap
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Label {
                        text: qsTr("Ostatnie alerty")
                        font.bold: true
                        color: Styles.AppTheme.textPrimary
                    }

                    ListView {
                        id: slaAlertList
                        objectName: "runtimeOverviewSlaAlertList"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.min(160, count * 48)
                        interactive: false
                        clip: true
                        model: alertHistory
                        delegate: RowLayout {
                            spacing: 8
                            Layout.fillWidth: true

                            Rectangle {
                                width: 10
                                height: 10
                                radius: 5
                                color: slaSeverityColor(modelData.severity || "ok")
                                Layout.alignment: Qt.AlignVCenter
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2

                                Label {
                                    text: (modelData.label || modelData.metric || qsTr("SLA")) +
                                          ": " + (modelData.formattedValue || "n/d")
                                    color: Styles.AppTheme.textPrimary
                                    font.bold: true
                                }

                                Label {
                                    text: qsTr("%1 • %2")
                                          .arg(modelData.severity || "ok")
                                          .arg(modelData.timestamp ? modelData.timestamp.toString() : "")
                                    color: Styles.AppTheme.textSecondary
                                }
                            }
                        }
                    }

                    Item {
                        objectName: "runtimeOverviewSlaEscalationStatus"
                        Layout.fillWidth: true
                        property string text: alertChannels && alertChannels.length > 0
                                              ? qsTr("Kanały eskalacji: %1")
                                                    .arg(feedSlaCard.alertChannelsText)
                                              : qsTr("Kanały eskalacji nieaktywne")
                        property color color: Styles.AppTheme.textSecondary
                        implicitHeight: slaEscalationText.implicitHeight

                        Text {
                            id: slaEscalationText
                            objectName: parent.objectName
                            anchors.fill: parent
                            text: parent.text
                            color: parent.color
                            elide: Text.ElideRight
                            wrapMode: Text.NoWrap
                        }
                    }
                }
            }

            Rectangle {
                anchors.fill: parent
                radius: feedSlaCard.radius
                color: feedSlaCard.severity === "critical"
                       ? Qt.rgba(0.42, 0.09, 0.13, 0.82)
                       : Qt.rgba(0.4, 0.3, 0.08, 0.78)
                border.color: Styles.AppTheme.warning
                border.width: feedSlaCard.latencyAlertActive ? 1 : 0
                visible: feedSlaCard.latencyAlertActive
                z: 5

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 10

                    Label {
                        text: feedSlaCard.severity === "critical"
                              ? qsTr("HyperCare: natychmiastowa eskalacja SLA")
                              : qsTr("HyperCare: monitoruj decision feed")
                        color: Qt.rgba(1, 1, 1, 0.95)
                        font.bold: true
                        font.pixelSize: 17
                    }

                    Label {
                        wrapMode: Text.WordWrap
                        color: Qt.rgba(1, 1, 1, 0.85)
                        text: feedSlaCard.report && feedSlaCard.report.p95_ms !== undefined
                              ? qsTr("p95 = %1 ms • progi: %2 / %3 ms")
                                    .arg(Number(feedSlaCard.report.p95_ms || 0).toFixed(0))
                                    .arg(Number(feedSlaCard.report.latency_warning_ms || 0).toFixed(0))
                                    .arg(Number(feedSlaCard.report.latency_critical_ms || 0).toFixed(0))
                              : qsTr("Brak danych o latencji feedu")
                    }

                    RowLayout {
                        Layout.alignment: Qt.AlignLeft
                        spacing: 12

                        Button {
                            text: qsTr("Otwórz runbook SLA")
                            icon.name: "link"
                            onClicked: Qt.openUrlExternally(root.feedSlaRunbookUrl)
                        }

                        Button {
                            visible: root.reportController && root.reportController.logOperationalAlert
                            text: qsTr("Zaloguj eskalację")
                            onClicked: {
                                if (!root.reportController || !root.reportController.logOperationalAlert)
                                    return
                                root.reportController.logOperationalAlert("sla.feed", {
                                                          severity: feedSlaCard.severity,
                                                          title: qsTr("Manualna eskalacja SLA"),
                                                          message: qsTr("Potwierdzono incydent HyperCare (%1)")
                                                                   .arg(root.feedLatencyAlertTicket),
                                                         latency_ms: Number(feedSlaCard.report.p95_ms || 0)
                                                     })
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: ioCardComponent
        Rectangle {
            objectName: "runtimeOverviewIoCard"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.rowSpan: cardsGrid.columns === 1 ? 1 : 2
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Text {
                    text: qsTr("Kolejki I/O")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ColumnLayout {
                        width: parent.width
                        spacing: 8
                        objectName: "runtimeOverviewIoRepeater"

                        Repeater {
                            model: root.telemetryIoQueues
                            delegate: Frame {
                                Layout.fillWidth: true
                                background: Rectangle {
                                    radius: 6
                                    color: Qt.rgba(0.12, 0.14, 0.18, 0.85)
                                }

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 4

                                    Text {
                                        text: qsTr("%1 • %2").arg(model.environment).arg(model.queue)
                                        font.bold: true
                                        color: Styles.AppTheme.textPrimary
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 12

                                        Text {
                                            text: qsTr("Timeouty: %1").arg(Number(model.timeoutTotal).toFixed(0))
                                            color: Styles.AppTheme.textSecondary
                                        }
                                        Text {
                                            text: model.timeoutAvgSeconds !== null && model.timeoutAvgSeconds !== undefined
                                                  ? qsTr("Śr. czas: %1 s").arg(Number(model.timeoutAvgSeconds).toFixed(3))
                                                  : qsTr("Śr. czas: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 12

                                        Text {
                                            text: qsTr("Oczekiwania: %1").arg(Number(model.rateLimitWaitTotal).toFixed(0))
                                            color: Styles.AppTheme.textSecondary
                                        }
                                        Text {
                                            text: model.rateLimitWaitAvgSeconds !== null && model.rateLimitWaitAvgSeconds !== undefined
                                                  ? qsTr("Śr. oczekiwanie: %1 s").arg(Number(model.rateLimitWaitAvgSeconds).toFixed(3))
                                                  : qsTr("Śr. oczekiwanie: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }
                                        Text {
                                            text: qsTr("Poziom: %1").arg(model.severity)
                                            color: model.severity === "error" ? Qt.rgba(0.9, 0.25, 0.3, 1)
                                                 : model.severity === "warning" ? Qt.rgba(0.95, 0.65, 0.2, 1)
                                                 : model.severity === "info" ? Qt.rgba(0.35, 0.7, 0.9, 1)
                                                 : Styles.AppTheme.textSecondary
                                        }
                                    }
                                }
                            }
                        }

                        Label {
                            visible: root.telemetryIoQueues.length === 0
                            text: qsTr("Brak danych z kolejki I/O")
                            color: Styles.AppTheme.textSecondary
                        }
                    }
                }
            }
        }
    }

    Component {
        id: guardrailCardComponent
        Rectangle {
            id: guardrailCard
            objectName: "runtimeOverviewGuardrailCard"
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1
            property var summary: root.guardrailSummary

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Text {
                    text: qsTr("Guardrail'e")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                ColumnLayout {
                    spacing: 4

                    Text {
                        text: qsTr("Łączna liczba kolejek: %1")
                              .arg(Number(guardrailCard.summary.totalQueues || 0).toFixed(0))
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Błędy: %1 • Ostrzeżenia: %2")
                              .arg(Number(guardrailCard.summary.errorQueues || 0).toFixed(0))
                              .arg(Number(guardrailCard.summary.warningQueues || 0).toFixed(0))
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Informacje: %1 • Stabilne: %2")
                              .arg(Number(guardrailCard.summary.infoQueues || 0).toFixed(0))
                              .arg(Number(guardrailCard.summary.normalQueues || 0).toFixed(0))
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Timeouty: %1 • Oczekiwania: %2")
                              .arg(Number(guardrailCard.summary.totalTimeouts || 0).toFixed(0))
                              .arg(Number(guardrailCard.summary.totalRateLimitWaits || 0).toFixed(0))
                        color: Styles.AppTheme.textSecondary
                    }
                }
            }
        }
    }

    Component {
        id: retrainingCardComponent
        Rectangle {
            objectName: "runtimeOverviewRetrainingCard"
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Text {
                    text: qsTr("Retraining")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                ColumnLayout {
                    width: parent.width
                    spacing: 6
                    objectName: "runtimeOverviewRetrainingRepeater"

                    Repeater {
                        model: root.retrainingRuns
                        delegate: Frame {
                            Layout.fillWidth: true
                            background: Rectangle {
                                radius: 6
                                color: Qt.rgba(0.12, 0.14, 0.18, 0.85)
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 4

                                Text {
                                    text: qsTr("Status: %1").arg(model.status)
                                    font.bold: true
                                    color: Styles.AppTheme.textPrimary
                                }

                                Text {
                                    text: qsTr("Liczba cykli: %1").arg(model.runs)
                                    color: Styles.AppTheme.textSecondary
                                }

                                Text {
                                    text: model.averageDurationSeconds !== null && model.averageDurationSeconds !== undefined
                                          ? qsTr("Śr. czas: %1 s").arg(Number(model.averageDurationSeconds).toFixed(2))
                                          : qsTr("Śr. czas: n/d")
                                    color: Styles.AppTheme.textSecondary
                                }

                                Text {
                                    text: model.averageDriftScore !== null && model.averageDriftScore !== undefined
                                          ? qsTr("Śr. dryf: %1").arg(Number(model.averageDriftScore).toFixed(3))
                                          : qsTr("Śr. dryf: n/d")
                                    color: Styles.AppTheme.textSecondary
                                }
                            }
                        }
                    }

                    Label {
                        visible: root.retrainingRuns.length === 0
                        text: qsTr("Brak danych retrainingu")
                        color: Styles.AppTheme.textSecondary
                    }
                }
            }
        }
    }

    Component {
        id: complianceCardComponent
        CompliancePanel {
            objectName: "runtimeOverviewCompliancePanel"
            telemetryProviderOverride: root.telemetryProviderObj
            complianceControllerOverride: root.complianceController
        }
    }

    Component {
        id: riskJournalCardComponent
        Rectangle {
            objectName: "runtimeOverviewRiskJournalCardSurface"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.rowSpan: cardsGrid.columns === 1 ? 1 : 2
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1

            Dashboard.RiskJournalPanel {
                objectName: "riskJournalPanel"
                anchors.fill: parent
                anchors.margins: 16
                runtimeService: root.runtimeServiceObj
                metrics: root.runtimeServiceObj ? root.runtimeServiceObj.riskMetrics : ({})
                timeline: root.runtimeServiceObj ? root.runtimeServiceObj.riskTimeline : []
                lastOperatorAction: root.runtimeServiceObj ? root.runtimeServiceObj.lastOperatorAction : ({})
                onFreezeRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeServiceObj ? root.runtimeServiceObj.lastOperatorAction : ({})
                }
                onUnfreezeRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeServiceObj ? root.runtimeServiceObj.lastOperatorAction : ({})
                }
                onUnblockRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeServiceObj ? root.runtimeServiceObj.lastOperatorAction : ({})
                }
            }
        }
    }

    Component {
        id: aiDecisionCardComponent
        Item {
            objectName: "runtimeOverviewAiCardContent"
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                id: aiDecisionCardSurface
                anchors.fill: parent
                color: Styles.AppTheme.surfaceStrong
                radius: 8
                border.color: Styles.AppTheme.surfaceSubtle
                border.width: 1

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                Text {
                    text: qsTr("Decyzje AI")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                Rectangle {
                    objectName: "runtimeOverviewAiErrorBanner"
                    Layout.fillWidth: true
                    visible: root.aiDecisionError.length > 0
                    color: Qt.rgba(0.75, 0.25, 0.28, 0.9)
                    radius: 6
                    implicitHeight: visible ? 32 : 0

                    Text {
                        anchors.centerIn: parent
                        text: root.aiDecisionError
                        color: "white"
                        font.pointSize: 11
                    }
                }

                GroupBox {
                    title: qsTr("Monitoring modeli AI")
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 6

                        Text {
                            text: root.aiDecisions.length > 0
                                  ? qsTr("Ostatnia decyzja: %1 (%2)")
                                        .arg(root.aiDecisions[0].decision && root.aiDecisions[0].decision.model
                                                ? root.aiDecisions[0].decision.model
                                                : qsTr("n/d"))
                                        .arg(root.aiDecisions[0].marketRegime && root.aiDecisions[0].marketRegime.regime
                                                ? root.aiDecisions[0].marketRegime.regime
                                                : qsTr("n/d"))
                                  : qsTr("Brak zarejestrowanych decyzji modeli")
                            color: Styles.AppTheme.textSecondary
                            wrapMode: Text.WordWrap
                        }

                        Text {
                            text: root.retrainSchedulerNextRun
                                  ? qsTr("Następny retraining: %1").arg(root.retrainSchedulerNextRun)
                                  : qsTr("Harmonogram retrainingu: nieaktywny")
                            color: Styles.AppTheme.textSecondary
                            wrapMode: Text.WordWrap
                        }

                        Text {
                            text: root.adaptiveStrategySummary
                                  ? root.adaptiveStrategySummary
                                  : qsTr("Brak aktywnych presetów adaptacyjnych")
                            color: Styles.AppTheme.textSecondary
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                GroupBox {
                    objectName: "runtimeOverviewCycleMetricsGroup"
                    title: qsTr("Metryki cyklu decyzyjnego")
                    Layout.fillWidth: true

                    ColumnLayout {
                        id: cycleMetricsSummary
                        spacing: 6
                        readonly property real cycleCount: Number(root.cycleMetrics.cycles_total || 0)
                        readonly property real strategySwitches: Number(root.cycleMetrics.strategy_switch_total || 0)
                        readonly property real guardrailBlocks: Number(root.cycleMetrics.guardrail_blocks_total || 0)
                        readonly property bool guardrailAlert: guardrailBlocks > 0
                        readonly property bool strategyAlert: strategySwitches > 5

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("Łączna liczba cykli")
                                color: Styles.AppTheme.textSecondary
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                objectName: "runtimeOverviewCycleCount"
                                text: cycleMetricsSummary.cycleCount.toFixed(0)
                                font.bold: true
                                color: Styles.AppTheme.textPrimary
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("Przełączenia strategii")
                                color: Styles.AppTheme.textSecondary
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                objectName: "runtimeOverviewStrategySwitches"
                                readonly property bool alert: cycleMetricsSummary.strategyAlert
                                text: cycleMetricsSummary.strategySwitches.toFixed(0)
                                font.bold: alert
                                color: alert ? Qt.rgba(0.95, 0.65, 0.2, 1) : Styles.AppTheme.textPrimary
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("Blokady guardrail")
                                color: Styles.AppTheme.textSecondary
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                objectName: "runtimeOverviewGuardrailBlocks"
                                readonly property bool alert: cycleMetricsSummary.guardrailAlert
                                text: cycleMetricsSummary.guardrailBlocks.toFixed(0)
                                font.bold: alert
                                color: alert ? Qt.rgba(0.9, 0.25, 0.3, 1) : Styles.AppTheme.textPrimary
                            }
                        }

                        Text {
                            objectName: "runtimeOverviewGuardrailAlert"
                            visible: cycleMetricsSummary.guardrailAlert
                            text: qsTr("Blokady guardrail wymagają uwagi – sprawdź kartę Guardrail'e")
                            color: Qt.rgba(0.9, 0.25, 0.3, 1)
                            font.bold: true
                            wrapMode: Text.WordWrap
                        }

                        Text {
                            objectName: "runtimeOverviewCycleLatency"
                            visible: root.cycleMetrics && (root.cycleMetrics.cycle_latency_p50_ms !== undefined || root.cycleMetrics.cycle_latency_p95_ms !== undefined)
                            text: {
                                const metrics = root.cycleMetrics || {}
                                const p50 = metrics.cycle_latency_p50_ms
                                const p95 = metrics.cycle_latency_p95_ms
                                const hasP50 = typeof p50 === "number" && !isNaN(p50)
                                const hasP95 = typeof p95 === "number" && !isNaN(p95)
                                if (!hasP50 && !hasP95)
                                    return qsTr("Brak próbek latencji cyklu decyzyjnego")
                                const p50Label = hasP50 ? Number(p50).toLocaleString(Qt.locale(), "f", 0) : qsTr("n/d")
                                const p95Label = hasP95 ? Number(p95).toLocaleString(Qt.locale(), "f", 0) : qsTr("n/d")
                                return qsTr("Opóźnienie cyklu p50: %1 ms • p95: %2 ms").arg(p50Label).arg(p95Label)
                            }
                            color: Styles.AppTheme.textSecondary
                            wrapMode: Text.WordWrap
                        }
                    }
                }


                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ColumnLayout {
                        width: parent.width
                        spacing: 8
                        objectName: "runtimeOverviewAiRepeater"

                        Repeater {
                            model: root.aiDecisions
                            delegate: Frame {
                                Layout.fillWidth: true
                                background: Rectangle {
                                    radius: 6
                                    color: Qt.rgba(0.12, 0.14, 0.18, 0.85)
                                }

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 4

                                    Text {
                                        text: qsTr("%1 • %2").arg(model.timestamp || "").arg(model.strategy || model.event)
                                        font.bold: true
                                        color: Styles.AppTheme.textPrimary
                                    }

                                    Text {
                                        text: model.environment && model.portfolio
                                              ? qsTr("Środowisko: %1 • Portfel: %2").arg(model.environment).arg(model.portfolio)
                                              : qsTr("Środowisko: %1").arg(model.environment || "n/d")
                                        color: Styles.AppTheme.textSecondary
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 12

                                        Text {
                                            text: model.decision && model.decision.state
                                                  ? qsTr("Decyzja: %1").arg(model.decision.state)
                                                  : qsTr("Decyzja: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }

                                        Text {
                                            text: model.decision && model.decision.signal
                                                  ? qsTr("Sygnał: %1").arg(model.decision.signal)
                                                  : qsTr("Sygnał: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }

                                        Text {
                                            text: model.decision && model.decision.shouldTrade !== undefined
                                                  ? (model.decision.shouldTrade ? qsTr("Handel: TAK") : qsTr("Handel: NIE"))
                                                  : qsTr("Handel: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 12

                                        Text {
                                            text: model.decision && model.decision.model
                                                  ? qsTr("Model: %1").arg(model.decision.model)
                                                  : qsTr("Model: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }

                                        Text {
                                            text: model.marketRegime && model.marketRegime.regime
                                                  ? qsTr("Reżim: %1").arg(model.marketRegime.regime)
                                                  : qsTr("Reżim: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }

                                        Text {
                                            text: model.marketRegime && model.marketRegime.riskLevel
                                                  ? qsTr("Poziom ryzyka: %1").arg(model.marketRegime.riskLevel)
                                                  : qsTr("Poziom ryzyka: n/d")
                                            color: Styles.AppTheme.textSecondary
                                        }
                                    }

                                    Text {
                                        visible: model.metadata && model.metadata.strategy_recommendation
                                        text: qsTr("Rekomendacja: %1").arg(model.metadata.strategy_recommendation)
                                        color: Styles.AppTheme.textSecondary
                                    }
                                }
                            }
                        }

                        Label {
                            objectName: "runtimeOverviewAiEmptyLabel"
                            visible: root.aiDecisions.length === 0 && root.aiDecisionError.length === 0
                            text: qsTr("Brak zarejestrowanych decyzji AI")
                            color: Styles.AppTheme.textSecondary
                        }
                    }
                }
            }
        }
    }


}
}
