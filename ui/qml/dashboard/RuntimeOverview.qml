import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles
import "." as Dashboard

Item {
    id: root
    implicitWidth: 1024
    implicitHeight: 640

    property var telemetryProvider: (typeof telemetryProvider !== "undefined" ? telemetryProvider : null)
    property var dashboardSettingsController: (typeof dashboardSettingsController !== "undefined" ? dashboardSettingsController : null)
    property var complianceController: (typeof complianceController !== "undefined" ? complianceController : null)
    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property int refreshIntervalMs: dashboardSettingsController ? dashboardSettingsController.refreshIntervalMs : 4000
    readonly property var defaultCardOrder: ["io_queue", "guardrails", "retraining", "compliance", "risk_journal", "ai_decisions"]
    property var aiDecisions: []
    property string aiDecisionError: ""
    property string retrainSchedulerNextRun: runtimeService && runtimeService.retrainNextRun
                                            ? runtimeService.retrainNextRun
                                            : ""
    property string adaptiveStrategySummary: runtimeService && runtimeService.adaptiveStrategySummary
                                              ? runtimeService.adaptiveStrategySummary
                                              : ""
    property var riskMetrics: runtimeService && runtimeService.riskMetrics ? runtimeService.riskMetrics : ({})
    property var riskTimeline: runtimeService && runtimeService.riskTimeline ? runtimeService.riskTimeline : []
    property var lastOperatorAction: runtimeService && runtimeService.lastOperatorAction ? runtimeService.lastOperatorAction : ({})

    function componentForCard(cardId) {
        switch (cardId) {
        case "io_queue":
            return ioCardComponent
        case "guardrails":
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
            return null
        }
    }

    function refreshTelemetry() {
        if (!root.telemetryProvider)
            return
        root.telemetryProvider.refreshTelemetry()
    }

    function refreshDecisions() {
        if (!root.runtimeService)
            return
        const result = root.runtimeService.loadRecentDecisions(10)
        if (Array.isArray(result))
            root.aiDecisions = result
        root.riskMetrics = root.runtimeService ? root.runtimeService.riskMetrics : ({})
        root.riskTimeline = root.runtimeService ? root.runtimeService.riskTimeline : []
        root.lastOperatorAction = root.runtimeService ? root.runtimeService.lastOperatorAction : ({})
    }

    function refreshAll() {
        root.refreshTelemetry()
        root.refreshDecisions()
    }

    Timer {
        id: refreshTimer
        interval: Math.max(1500, root.refreshIntervalMs)
        repeat: true
        running: !!root.telemetryProvider
        triggeredOnStart: true
        onTriggered: root.refreshAll()
    }

    Connections {
        target: root.telemetryProvider
        ignoreUnknownSignals: true
        function onErrorMessageChanged() {
            errorBanner.visible = root.telemetryProvider.errorMessage.length > 0
        }
    }

    Connections {
        target: root.runtimeService
        ignoreUnknownSignals: true

        function onDecisionsChanged() {
            if (root.runtimeService)
                root.aiDecisions = root.runtimeService.decisions
        }

        function onErrorMessageChanged() {
            if (!root.runtimeService)
                return
            root.aiDecisionError = root.runtimeService.errorMessage
        }

        function onRiskMetricsChanged() {
            if (!root.runtimeService)
                return
            root.riskMetrics = root.runtimeService.riskMetrics
        }

        function onRiskTimelineChanged() {
            if (!root.runtimeService)
                return
            root.riskTimeline = root.runtimeService.riskTimeline
        }

        function onOperatorActionChanged() {
            if (!root.runtimeService)
                return
            root.lastOperatorAction = root.runtimeService.lastOperatorAction
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
                text: root.telemetryProvider && root.telemetryProvider.lastUpdated.length > 0
                      ? qsTr("Ostatnia aktualizacja: %1").arg(root.telemetryProvider.lastUpdated)
                      : qsTr("Ostatnia aktualizacja: n/d")
                color: Styles.AppTheme.textSecondary
                font.pointSize: 12
            }

            Item { Layout.fillWidth: true }

            Button {
                id: manualRefreshButton
                text: qsTr("Odśwież")
                enabled: !!root.telemetryProvider
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
                text: root.telemetryProvider ? root.telemetryProvider.errorMessage : ""
                color: "white"
                font.pointSize: 11
            }
        }

        GridLayout {
            id: cardsGrid
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: width > 980 ? 3 : 1
            rowSpacing: 16
            columnSpacing: 16

            Repeater {
                id: cardRepeater
                model: root.dashboardSettingsController ? root.dashboardSettingsController.visibleCardOrder : root.defaultCardOrder
                delegate: Loader {
                    readonly property string cardId: modelData
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    sourceComponent: root.componentForCard(cardId)
                    active: sourceComponent !== null
                }
            }
        }

        Label {
            visible: root.dashboardSettingsController && root.dashboardSettingsController.visibleCardOrder.length === 0
            text: qsTr("Wszystkie karty zostały ukryte w ustawieniach dashboardu")
            color: Styles.AppTheme.textSecondary
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
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
                            model: root.telemetryProvider ? root.telemetryProvider.ioQueues : []
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
                            visible: !root.telemetryProvider || root.telemetryProvider.ioQueues.length === 0
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
            objectName: "runtimeOverviewGuardrailCard"
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
                    text: qsTr("Guardrail'e")
                    font.bold: true
                    font.pointSize: 15
                    color: Styles.AppTheme.textPrimary
                }

                ColumnLayout {
                    spacing: 4

                    Text {
                        text: qsTr("Łączna liczba kolejek: %1").arg(root.telemetryProvider ? root.telemetryProvider.guardrailSummary.totalQueues : 0)
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Błędy: %1 • Ostrzeżenia: %2").arg(root.telemetryProvider ? root.telemetryProvider.guardrailSummary.errorQueues : 0)
                                                                                 .arg(root.telemetryProvider ? root.telemetryProvider.guardrailSummary.warningQueues : 0)
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Informacje: %1 • Stabilne: %2")
                              .arg(root.telemetryProvider ? root.telemetryProvider.guardrailSummary.infoQueues : 0)
                              .arg(root.telemetryProvider ? root.telemetryProvider.guardrailSummary.normalQueues : 0)
                        color: Styles.AppTheme.textSecondary
                    }
                    Text {
                        text: qsTr("Timeouty: %1 • Oczekiwania: %2")
                              .arg(root.telemetryProvider ? Number(root.telemetryProvider.guardrailSummary.totalTimeouts).toFixed(0) : "0")
                              .arg(root.telemetryProvider ? Number(root.telemetryProvider.guardrailSummary.totalRateLimitWaits).toFixed(0) : "0")
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
                        model: root.telemetryProvider ? root.telemetryProvider.retraining : []
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
                        visible: !root.telemetryProvider || root.telemetryProvider.retraining.length === 0
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
            telemetryProvider: root.telemetryProvider
            complianceController: root.complianceController
        }
    }

    Component {
        id: riskJournalCardComponent
        Rectangle {
            objectName: "runtimeOverviewRiskJournalCard"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.rowSpan: cardsGrid.columns === 1 ? 1 : 2
            color: Styles.AppTheme.surfaceStrong
            radius: 8
            border.color: Styles.AppTheme.surfaceSubtle
            border.width: 1

            Dashboard.RiskJournalPanel {
                anchors.fill: parent
                anchors.margins: 16
                runtimeService: root.runtimeService
                metrics: root.riskMetrics
                timeline: root.riskTimeline
                lastOperatorAction: root.lastOperatorAction
                onFreezeRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeService ? root.runtimeService.lastOperatorAction : ({})
                }
                onUnfreezeRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeService ? root.runtimeService.lastOperatorAction : ({})
                }
                onUnblockRequested: function(entry) {
                    root.lastOperatorAction = root.runtimeService ? root.runtimeService.lastOperatorAction : ({})
                }
            }
        }
    }

    Component {
        id: aiDecisionCardComponent
        Rectangle {
            objectName: "runtimeOverviewAiCard"
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
