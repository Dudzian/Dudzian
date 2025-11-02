import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "." as Views

Item {
    id: root
    implicitWidth: 1080
    implicitHeight: 720

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var riskModel: appController ? appController.riskModel : (typeof riskModel !== "undefined" ? riskModel : null)
    property var riskHistoryModel: appController ? appController.riskHistoryModel : (typeof riskHistoryModel !== "undefined" ? riskHistoryModel : null)
    property var alertsModel: (typeof alertsModel !== "undefined" ? alertsModel : null)
    property var alertsFilterModel: (typeof alertsFilterModel !== "undefined" ? alertsFilterModel : null)
    property var decisionModel: (typeof decisionLogModel !== "undefined" ? decisionLogModel : null)
    property var decisionFilterModel: (typeof decisionFilterModel !== "undefined" ? decisionFilterModel : null)
    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)

    property var lastDecisions: []
    property string runtimeError: ""

    function refreshRuntime() {
        if (!runtimeService)
            return
        const result = runtimeService.loadRecentDecisions(15)
        if (Array.isArray(result))
            lastDecisions = result
        runtimeError = runtimeService.errorMessage || ""
    }

    Timer {
        id: runtimeRefreshTimer
        interval: 4000
        repeat: true
        running: !!runtimeService
        triggeredOnStart: true
        onTriggered: root.refreshRuntime()
    }

    Connections {
        target: runtimeService
        ignoreUnknownSignals: true

        function onDecisionsChanged() {
            root.lastDecisions = runtimeService.decisions
        }

        function onErrorMessageChanged() {
            root.runtimeError = runtimeService.errorMessage
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 16
        padding: 18

        Label {
            text: qsTr("Centrum operacyjne")
            font.pixelSize: 26
            font.bold: true
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Item {
                SplitView.fillWidth: true
                SplitView.preferredWidth: parent.width * 0.55

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    Views.PortfolioDashboard {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        appController: root.appController
                        riskModel: root.riskModel
                        riskHistoryModel: root.riskHistoryModel
                        alertsModel: root.alertsModel
                    }

                    Views.RiskControls {
                        Layout.fillWidth: true
                        Layout.preferredHeight: parent.height * 0.45
                        appController: root.appController
                        limitsModel: root.appController ? root.appController.riskLimitsModel : null
                        costModel: root.appController ? root.appController.riskCostModel : null
                    }
                }
            }

            Frame {
                SplitView.fillWidth: true
                SplitView.preferredWidth: parent.width * 0.45
                padding: 14
                background: Rectangle {
                    radius: 10
                    color: Qt.darker(palette.base, 1.05)
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    Components.DecisionAlertLogPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredHeight: parent.height * 0.55
                        alertsModel: root.alertsModel
                        alertsFilterModel: root.alertsFilterModel
                        decisionModel: root.decisionModel
                        decisionFilterModel: root.decisionFilterModel
                        onExportCompleted: (url) => console.info("Decision log exported", url)
                    }

                    Frame {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.08) }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8
                            padding: 10

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 6

                                Label {
                                    text: qsTr("Ostatnie decyzje AI")
                                    font.pixelSize: 18
                                    font.bold: true
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTr("Odśwież")
                                    enabled: !!root.runtimeService
                                    onClicked: root.refreshRuntime()
                                }
                            }

                            Label {
                                visible: root.runtimeError.length > 0
                                text: root.runtimeError
                                color: Qt.rgba(0.78, 0.25, 0.28, 1)
                                font.pointSize: 11
                            }

                            ListView {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                model: root.lastDecisions
                                clip: true
                                delegate: Frame {
                                    width: ListView.view.width
                                    padding: 8
                                    background: Rectangle { radius: 6; color: Qt.darker(palette.base, 1.02) }

                                    ColumnLayout {
                                        anchors.fill: parent
                                        spacing: 4

                                        Label {
                                            text: modelData.strategy ? qsTr("Strategia: %1").arg(modelData.strategy) : qsTr("Strategia nieznana")
                                            font.bold: true
                                        }

                                        Label {
                                            text: modelData.timestamp
                                            color: palette.mid
                                            font.pointSize: 10
                                        }

                                        Label {
                                            text: modelData.decision && modelData.decision.state ? qsTr("Stan: %1").arg(modelData.decision.state) : qsTr("Stan niedostępny")
                                        }

                                        Label {
                                            text: modelData.marketRegime && modelData.marketRegime.regime ? qsTr("Reżim: %1 (%2)").arg(modelData.marketRegime.regime).arg(modelData.marketRegime.riskLevel || qsTr("n/d")) : qsTr("Brak danych o reżimie")
                                            color: palette.mid
                                        }
                                    }
                                }

                                placeholderText: root.runtimeError.length === 0 ? qsTr("Brak zarejestrowanych decyzji") : ""
                            }
                        }
                    }
                }
            }
        }
    }
}

