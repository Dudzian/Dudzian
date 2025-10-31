import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 1024
    implicitHeight: 640

    property var telemetryProvider: (typeof telemetryProvider !== "undefined" ? telemetryProvider : null)
    property var dashboardSettingsController: (typeof dashboardSettingsController !== "undefined" ? dashboardSettingsController : null)
    property int refreshIntervalMs: dashboardSettingsController ? dashboardSettingsController.refreshIntervalMs : 4000
    readonly property var defaultCardOrder: ["io_queue", "guardrails", "retraining"]

    function componentForCard(cardId) {
        switch (cardId) {
        case "io_queue":
            return ioCardComponent
        case "guardrails":
            return guardrailCardComponent
        case "retraining":
            return retrainingCardComponent
        default:
            return null
        }
    }

    function refreshTelemetry() {
        if (!root.telemetryProvider)
            return
        root.telemetryProvider.refreshTelemetry()
    }

    Timer {
        id: refreshTimer
        interval: Math.max(1500, root.refreshIntervalMs)
        repeat: true
        running: !!root.telemetryProvider
        triggeredOnStart: true
        onTriggered: root.refreshTelemetry()
    }

    Connections {
        target: root.telemetryProvider
        ignoreUnknownSignals: true
        function onErrorMessageChanged() {
            errorBanner.visible = root.telemetryProvider.errorMessage.length > 0
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
                onClicked: root.refreshTelemetry()
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
}
