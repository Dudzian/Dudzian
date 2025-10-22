import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "./StrategyWorkbenchViewModel.qml" as ViewModel
import "./panels"

Item {
    id: root
    property alias viewModel: viewModel

    // Możliwość podmiany kontrolerów zewnętrznych
    property var appController: null
    property var strategyController: null
    property var riskModel: null
    property var riskHistoryModel: null
    property var licenseController: null

    implicitWidth: 960
    implicitHeight: 540

    ViewModel.StrategyWorkbenchViewModel {
        id: viewModel
        objectName: "strategyWorkbenchViewModel"
        appController: root.appController ? root.appController : (typeof appController !== "undefined" ? appController : null)
        strategyController: root.strategyController ? root.strategyController : (typeof strategyController !== "undefined" ? strategyController : null)
        riskModel: root.riskModel ? root.riskModel : (typeof riskModel !== "undefined" ? riskModel : null)
        riskHistoryModel: root.riskHistoryModel ? root.riskHistoryModel : (typeof riskHistoryModel !== "undefined" ? riskHistoryModel : null)
        licenseController: root.licenseController ? root.licenseController : (typeof licenseController !== "undefined" ? licenseController : null)
    }

    ScrollView {
        anchors.fill: parent
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            width: parent.width
            spacing: 16
            padding: 16

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: Qt.darker(palette.base, 1.05)
                    radius: 8
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                    Label {
                        text: viewModel.demoModeActive
                            ? qsTr("Tryb demo: %1").arg(viewModel.demoModeTitle)
                            : qsTr("Tryb live")
                        font.bold: true
                        Layout.alignment: Qt.AlignVCenter
                    }

                    Label {
                        text: viewModel.demoModeActive
                            ? viewModel.demoModeDescription
                            : qsTr("Dane pobierane z warstwy runtime i licencji")
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        maximumLineCount: 3
                        elide: Text.ElideRight
                    }

                    ComboBox {
                        id: demoPresetSelector
                        objectName: "demoPresetSelector"
                        Layout.preferredWidth: 220
                        model: viewModel.demoPresets
                        textRole: "title"
                        onActivated: function(index) {
                            if (index < 0 || index >= viewModel.demoPresets.length)
                                return
                            viewModel.activateDemoMode(viewModel.demoPresets[index].id)
                        }
                    }

                    Button {
                        id: demoDisableButton
                        objectName: "demoDisableButton"
                        text: qsTr("Wyłącz demo")
                        enabled: viewModel.demoModeActive
                        onClicked: viewModel.disableDemoMode()
                    }
                }
            }

            GridLayout {
                columns: 2
                columnSpacing: 16
                rowSpacing: 16
                Layout.fillWidth: true

                StrategyDashboardPanel {
                    schedulerEntries: viewModel.schedulerEntries
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                InstrumentOverviewPanel {
                    instrumentDetails: viewModel.instrumentDetails
                    runtimeStatus: viewModel.runtimeStatus
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                StrategyControlPanel {
                    controlState: viewModel.controlState
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                    onStartSchedulerRequested: viewModel.startScheduler()
                    onStopSchedulerRequested: viewModel.stopScheduler()
                    onRiskRefreshRequested: viewModel.triggerRiskRefresh()
                }

                ExchangeManagementPanel {
                    exchangeConnections: viewModel.exchangeConnections
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.columnSpan: 2
                }

                AiConfigurationPanel {
                    configuration: viewModel.aiConfiguration
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                ResultsAnalysisPanel {
                    portfolioSummary: viewModel.portfolioSummary
                    riskSnapshot: viewModel.riskSnapshot
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                OpenPositionsPanel {
                    openPositions: viewModel.openPositions
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                PendingOrdersPanel {
                    pendingOrders: viewModel.pendingOrders
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                TradeHistoryPanel {
                    tradeHistory: viewModel.tradeHistory
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                SignalAlertsPanel {
                    signalAlerts: viewModel.signalAlerts
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                RiskTimelinePanel {
                    riskTimeline: viewModel.riskTimeline
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                ActivityLogPanel {
                    activityLog: viewModel.activityLog
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                }

                RuntimeStatusPanel {
                    runtimeStatus: viewModel.runtimeStatus
                    Layout.fillWidth: true
                }

                LicenseStatusPanel {
                    licenseStatus: viewModel.licenseStatus
                    Layout.fillWidth: true
                }
            }
        }
    }
}
