import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "." as Design
import "components" as Components

Components.Card {
    id: root
    objectName: "strategyAiPanel"
    title: qsTr("AI Governor i strategie adaptacyjne")
    subtitle: feedTransportSnapshot && feedTransportSnapshot.status
              ? qsTr("Transport %1 (%2)").arg(feedTransportSnapshot.status).arg(feedTransportSnapshot.mode || "local")
              : qsTr("Transport: brak danych")

    property var runtimeService: null
    property var feedTransportSnapshot: ({})
    property var aiRegimes: []
    property string adaptiveSummary: ""
    property string activationSummary: ""

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 16

        RowLayout {
            id: statusRow
            objectName: "strategyAiStatusRow"
            spacing: 12
            Layout.fillWidth: true

            Design.Icon {
                id: transportIcon
                iconName: feedTransportSnapshot && feedTransportSnapshot.mode === "grpc" ? "cloud" : "bolt"
                iconSize: 28
                iconColor: feedTransportSnapshot && feedTransportSnapshot.status === "connected"
                           ? Design.Palette.success
                           : Design.Palette.warning
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    id: statusLabel
                    objectName: "strategyAiTransportLabel"
                    text: feedTransportSnapshot && feedTransportSnapshot.label
                          ? qsTr("%1 • %2").arg(feedTransportSnapshot.label).arg(feedTransportSnapshot.status || qsTr("unknown"))
                          : qsTr("Źródło lokalne • %1").arg(feedTransportSnapshot && feedTransportSnapshot.status
                                                           ? feedTransportSnapshot.status
                                                           : qsTr("unknown"))
                    color: Design.Palette.textPrimary
                    font.pixelSize: Design.Typography.title
                    wrapMode: Text.WordWrap
                }

                Label {
                    text: feedTransportSnapshot && feedTransportSnapshot.latencyP95
                          ? qsTr("Latency p95: %1 ms • reconnects: %2")
                                .arg(Number(feedTransportSnapshot.latencyP95).toFixed(0))
                                .arg(feedTransportSnapshot.reconnects || 0)
                          : qsTr("Brak pomiarów latencji decyzji")
                    color: Design.Palette.textSecondary
                    font.pixelSize: Design.Typography.body
                }

                Label {
                    visible: !!(feedTransportSnapshot && feedTransportSnapshot.lastError && feedTransportSnapshot.lastError.length > 0)
                    text: qsTr("Ostatni błąd: %1").arg(feedTransportSnapshot.lastError)
                    color: Design.Palette.warning
                    font.pixelSize: Design.Typography.caption
                    wrapMode: Text.WordWrap
                }
            }

            Button {
                id: refreshButton
                text: qsTr("Odśwież AI")
                icon.name: "refresh"
                onClicked: {
                    if (root.runtimeService)
                        root.runtimeService.refreshRuntimeMetadata()
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 10

            Repeater {
                id: regimeRepeater
                model: aiRegimes || []
                delegate: Rectangle {
                    width: parent ? parent.width : 0
                    height: 64
                    radius: 14
                    color: Qt.rgba(1, 1, 1, 0.04)
                    border.color: Qt.rgba(1, 1, 1, 0.06)
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 10

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Label {
                                text: modelData && modelData.regime ? modelData.regime : qsTr("(brak nazwy)")
                                font.pixelSize: Design.Typography.title
                                color: Design.Palette.textPrimary
                                font.bold: true
                            }

                            Label {
                                text: modelData && modelData.bestStrategy
                                      ? qsTr("Najlepsza strategia: %1").arg(modelData.bestStrategy)
                                      : qsTr("Brak zwycięzcy")
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.body
                            }
                        }

                        ColumnLayout {
                            Layout.alignment: Qt.AlignVCenter
                            spacing: 2

                            Label {
                                text: modelData && modelData.meanReward !== undefined
                                      ? qsTr("μ=%1").arg(Number(modelData.meanReward).toFixed(2))
                                      : "—"
                                color: Design.Palette.textPrimary
                                font.pixelSize: Design.Typography.body
                            }

                            Label {
                                text: modelData && modelData.plays !== undefined
                                      ? qsTr("n=%1").arg(modelData.plays)
                                      : "—"
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.caption
                            }
                        }
                    }
                }
            }

            Label {
                visible: !aiRegimes || aiRegimes.length === 0
                text: qsTr("Brak danych polityk adaptacyjnych – uruchom retraining AI.")
                color: Design.Palette.textSecondary
                font.pixelSize: Design.Typography.body
                wrapMode: Text.WordWrap
            }
        }

        Label {
            objectName: "strategyAiActivationSummary"
            visible: activationSummary.length > 0
            text: activationSummary
            color: Design.Palette.textSecondary
            font.pixelSize: Design.Typography.body
            wrapMode: Text.WordWrap
        }

        Label {
            objectName: "strategyAiAdaptiveSummary"
            visible: adaptiveSummary.length > 0
            text: adaptiveSummary
            color: Design.Palette.textSecondary
            font.pixelSize: Design.Typography.caption
            wrapMode: Text.WordWrap
        }
    }
}
