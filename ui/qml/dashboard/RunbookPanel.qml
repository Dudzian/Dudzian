import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 720
    implicitHeight: 560

    property var runbookController: (typeof runbookController !== "undefined" ? runbookController : null)
    property int refreshIntervalMs: 5000
    property var pendingAction: ({ runbookId: "", actionId: "", label: "", confirmMessage: "" })
    property var actionStatus: ({})

    function refreshAlerts() {
        if (!root.runbookController)
            return
        root.runbookController.refreshAlerts()
    }

    Timer {
        id: refreshTimer
        interval: Math.max(2000, root.refreshIntervalMs)
        repeat: true
        running: !!root.runbookController
        triggeredOnStart: true
        onTriggered: root.refreshAlerts()
    }

    Connections {
        target: root.runbookController
        ignoreUnknownSignals: true
        function onErrorMessageChanged() {
            errorBanner.visible = root.runbookController.errorMessage.length > 0
        }
        function onActionStatusChanged() {
            try {
                root.actionStatus = JSON.parse(root.runbookController.actionStatus)
                actionStatusBanner.visible = root.actionStatus && root.actionStatus.status
            } catch (err) {
                root.actionStatus = ({})
                actionStatusBanner.visible = false
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Text {
                id: lastUpdatedLabel
                objectName: "runbookPanelLastUpdated"
                text: root.runbookController && root.runbookController.lastUpdated.length > 0
                      ? qsTrId("runbookPanel.lastUpdated").arg(root.runbookController.lastUpdated)
                      : qsTrId("runbookPanel.lastUpdatedFallback")
                color: Styles.AppTheme.textSecondary
                font.pointSize: 12
            }

            Item { Layout.fillWidth: true }

            Button {
                id: refreshButton
                text: qsTrId("runbookPanel.refresh")
                enabled: !!root.runbookController
                onClicked: root.refreshAlerts()
            }
        }

        Rectangle {
            id: errorBanner
            objectName: "runbookPanelErrorBanner"
            Layout.fillWidth: true
            visible: false
            color: Qt.rgba(0.75, 0.25, 0.28, 0.9)
            radius: 6
            implicitHeight: visible ? 36 : 0

            Text {
                anchors.centerIn: parent
                text: root.runbookController ? root.runbookController.errorMessage : ""
                color: "white"
                font.pointSize: 11
            }
        }

        Rectangle {
            id: actionStatusBanner
            objectName: "runbookPanelActionBanner"
            Layout.fillWidth: true
            visible: false
            color: root.actionStatus && root.actionStatus.status === "success" ? Qt.rgba(0.2, 0.55, 0.3, 0.9)
                                                                               : Qt.rgba(0.75, 0.25, 0.28, 0.9)
            radius: 6
            implicitHeight: visible ? 48 : 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 8

                Text {
                    text: root.actionStatus && root.actionStatus.status === "success"
                          ? qsTrId("runbookPanel.actionSuccess").arg(root.actionStatus.action_id)
                          : qsTrId("runbookPanel.actionFailure")
                    color: "white"
                    font.pointSize: 11
                }

                Text {
                    text: root.actionStatus && root.actionStatus.message ? root.actionStatus.message
                                                                          : root.actionStatus && root.actionStatus.stdout
                                                                                ? root.actionStatus.stdout
                                                                                : ""
                    color: "white"
                    font.pointSize: 10
                    wrapMode: Text.Wrap
                }
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                width: parent.width
                spacing: 10
                objectName: "runbookPanelContainer"

                Repeater {
                    id: alertsRepeater
                    objectName: "runbookPanelRepeater"
                    model: root.runbookController ? root.runbookController.alerts : []

                    Frame {
                        Layout.fillWidth: true
                        readonly property var alert: (typeof modelData !== "undefined" && modelData !== null) ? modelData : model
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0.16, 0.18, 0.22, 0.9)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 6

                            readonly property string envText: "" + (alert.environment || alert["environment"] || "")
                            readonly property string queueText: "" + (alert.queue || alert["queue"] || "")
                            readonly property string msgText: "" + (alert.message || alert["message"] || "")

                            Text {
                                text: envText.length > 0
                                      ? qsTr("%1 • %2").arg(envText).arg(queueText)
                                      : queueText.length > 0 ? queueText : qsTrId("runbookPanel.defaultTitle")
                                font.bold: true
                                color: Styles.AppTheme.textPrimary
                            }

                            Text {
                                text: msgText
                                color: Styles.AppTheme.textSecondary
                                wrapMode: Text.Wrap
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                readonly property string severityValue: (alert.severity || alert["severity"] || "")

                                Rectangle {
                                    width: 12
                                    height: 12
                                    radius: 6
                                    color: severityValue === "error" ? Qt.rgba(0.9, 0.25, 0.3, 1)
                                           : severityValue === "warning" ? Qt.rgba(0.95, 0.65, 0.2, 1)
                                           : Qt.rgba(0.35, 0.7, 0.9, 1)
                                }

                                Text {
                                    text: qsTrId("runbookPanel.severity").arg(severityValue)
                                    color: Styles.AppTheme.textSecondary
                                }

                                Item { Layout.fillWidth: true }

                                readonly property var tsRaw: (alert.timestamp !== undefined ? alert.timestamp : alert["timestamp"])
                                readonly property string tsText: (tsRaw === undefined || tsRaw === null) ? "" : ("" + tsRaw)

                                Text {
                                    visible: tsText.length > 0
                                    text: tsText
                                    color: Styles.AppTheme.textSecondary
                                    font.pointSize: 11
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Text {
                                    text: (alert.runbookTitle || alert["runbookTitle"] || "").length > 0
                                          ? qsTrId("runbookPanel.runbookAssigned").arg(alert.runbookTitle || alert["runbookTitle"])
                                          : qsTrId("runbookPanel.runbookMissing")
                                    color: Styles.AppTheme.textSecondary
                                    wrapMode: Text.Wrap
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTrId("runbookPanel.openRunbook")
                                    visible: (alert.runbookPath || alert["runbookPath"] || "").length > 0
                                    onClicked: {
                                        if (root.runbookController)
                                            root.runbookController.openRunbook(alert.runbookPath || alert["runbookPath"])
                                    }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                visible: (alert.manualSteps || alert["manualSteps"] || []).length > 0

                                Text {
                                    text: qsTrId("runbookPanel.manualSteps")
                                    font.bold: true
                                    color: Styles.AppTheme.textPrimary
                                }

                                Repeater {
                                    model: alert.manualSteps || alert["manualSteps"] || []

                                    Text {
                                        text: "• " + modelData
                                        color: Styles.AppTheme.textSecondary
                                        wrapMode: Text.Wrap
                                    }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                visible: (alert.automaticActions || alert["automaticActions"] || []).length > 0

                                Text {
                                    text: qsTrId("runbookPanel.automaticActions")
                                    font.bold: true
                                    color: Styles.AppTheme.textPrimary
                                }

                                Repeater {
                                    model: alert.automaticActions || alert["automaticActions"] || []

                                    Button {
                                        readonly property string actionId: "" + ((modelData && (modelData["id"] !== undefined ? modelData["id"] : modelData.id)) || ("idx_" + index))
                                        readonly property var actionLabel: (modelData && (modelData["label"] !== undefined ? modelData["label"] : modelData.label)) || ""

                                        objectName: "runbookActionButton_" + actionId
                                        text: actionLabel
                                        onClicked: root.requestRunbookAction(alert.runbookId || alert["runbookId"], modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                Label {
                    visible: !root.runbookController || root.runbookController.alerts.length === 0
                    text: qsTrId("runbookPanel.emptyState")
                    color: Styles.AppTheme.textSecondary
                }
            }
        }
    }

    Dialog {
        id: confirmActionDialog
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        title: qsTrId("runbookPanel.confirmActionTitle")

        contentItem: ColumnLayout {
            spacing: 8
            width: Math.max(320, implicitWidth)

            Text {
                text: root.pendingAction.confirmMessage && root.pendingAction.confirmMessage.length > 0
                      ? root.pendingAction.confirmMessage
                      : qsTrId("runbookPanel.confirmActionText").arg(root.pendingAction.label)
                wrapMode: Text.WordWrap
            }
        }

        onAccepted: root.executePendingAction()
    }

    function requestRunbookAction(runbookId, actionPayload) {
        if (!root.runbookController)
            return
        root.pendingAction = ({
            runbookId: runbookId,
            actionId: actionPayload.id,
            label: actionPayload.label,
            confirmMessage: actionPayload.confirmMessage
        })

        if (actionPayload.confirmMessage && actionPayload.confirmMessage.length > 0) {
            confirmActionDialog.open()
        } else {
            root.executePendingAction()
        }
    }

    function executePendingAction() {
        if (!root.runbookController || !root.pendingAction.runbookId || !root.pendingAction.actionId)
            return
        root.runbookController.runAction(root.pendingAction.runbookId, root.pendingAction.actionId)
    }
}
