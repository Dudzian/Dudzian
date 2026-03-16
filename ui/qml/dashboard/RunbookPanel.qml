import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 720
    implicitHeight: 560

    property var runbookControllerObj: null
    property int refreshIntervalMs: 5000
    property var pendingAction: ({ runbookId: "", actionId: "", label: "", confirmMessage: "" })
    property var actionStatus: ({})

    Component.onCompleted: {
        // QQmlContext property: "runbookController" (set via rootContext().setContextProperty).
        try {
            if (!runbookControllerObj)
                runbookControllerObj = runbookController
            if (runbookControllerObj)
                refreshAlerts()
        } catch (e) {
            // Keep null when the panel is loaded without the context property.
        }
    }

    function refreshAlerts() {
        if (!root.runbookControllerObj)
            return
        root.runbookControllerObj.refreshAlerts()
    }

    function listSize(value) {
        if (value === undefined || value === null)
            return 0
        if (typeof value.length === "number")
            return value.length
        if (typeof value.length === "function")
            try {
                return value.length()
            } catch (err) {}
        if (typeof value.count === "number")
            return value.count
        if (typeof value.count === "function")
            try {
                return value.count()
            } catch (err) {}
        try {
            if (typeof value.size === "function")
                return value.size()
        } catch (err) {}
        return 0
    }

    Timer {
        id: refreshTimer
        interval: Math.max(2000, root.refreshIntervalMs)
        repeat: true
        running: !!root.runbookControllerObj
        triggeredOnStart: true
        onTriggered: root.refreshAlerts()
    }

    Connections {
        target: root.runbookControllerObj
        ignoreUnknownSignals: true
        function onErrorMessageChanged() {
            errorBanner.visible = root.runbookControllerObj.errorMessage.length > 0
        }
        function onActionStatusChanged() {
            try {
                root.actionStatus = JSON.parse(root.runbookControllerObj.actionStatus)
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
                text: root.runbookControllerObj && root.runbookControllerObj.lastUpdated.length > 0
                      ? qsTrId("runbookPanel.lastUpdated").arg(root.runbookControllerObj.lastUpdated)
                      : qsTrId("runbookPanel.lastUpdatedFallback")
                color: Styles.AppTheme.textSecondary
                font.pointSize: 12
            }

            Item { Layout.fillWidth: true }

            Button {
                id: refreshButton
                text: qsTrId("runbookPanel.refresh")
                enabled: !!root.runbookControllerObj
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
                text: root.runbookControllerObj ? root.runbookControllerObj.errorMessage : ""
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
                    model: root.runbookControllerObj ? root.runbookControllerObj.alerts : []

                    Frame {
                        id: alertFrame
                        function safeName(value) {
                            var text = (value === undefined || value === null) ? "" : ("" + value)
                            return text.replace(/[^A-Za-z0-9_]/g, "_")
                        }
                        objectName: {
                            var runbookId = alertFrame.getField(alertFrame.alert, "runbookId", "")
                            if (runbookId !== undefined && runbookId !== null && ("" + runbookId).length > 0)
                                return "runbookAlertFrame_" + safeName(runbookId) + "_" + index
                            return index === 0 ? "runbookAlertFrame_first" : "runbookAlertFrame_" + index
                        }
                        Layout.fillWidth: true
                        readonly property int alertIndex: index
                        readonly property var alert: {
                            var m = alertsRepeater.model
                            var i = alertFrame.alertIndex
                            try {
                                if (m && typeof m.get === "function")
                                    return m.get(i)
                                if (m && root.listSize(m) > i) {
                                    var v = m[i]
                                    if (v !== undefined)
                                        return v
                                }
                            } catch (err) {}
                            return (typeof modelData !== "undefined" && modelData !== null) ? modelData : null
                        }
                        function getField(obj, key, fallback) {
                            if (!obj)
                                return fallback
                            var v = obj[key]
                            if (v === undefined) {
                                try {
                                    v = obj[key]
                                } catch (err) {
                                    v = undefined
                                }
                            }
                            return (v === undefined || v === null) ? fallback : v
                        }
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0.16, 0.18, 0.22, 0.9)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 6

                            readonly property string envText: "" + alertFrame.getField(alert, "environment", "")
                            readonly property string queueText: "" + alertFrame.getField(alert, "queue", "")
                            readonly property string msgText: "" + alertFrame.getField(alert, "message", "")

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

                                readonly property string severityValue: alertFrame.getField(alert, "severity", "")

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

                                readonly property var tsRaw: alertFrame.getField(alert, "timestamp", undefined)
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
                                    text: (alertFrame.getField(alert, "runbookTitle", "")).length > 0
                                          ? qsTrId("runbookPanel.runbookAssigned").arg(alertFrame.getField(alert, "runbookTitle", ""))
                                          : qsTrId("runbookPanel.runbookMissing")
                                    color: Styles.AppTheme.textSecondary
                                    wrapMode: Text.Wrap
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTrId("runbookPanel.openRunbook")
                                    visible: (alertFrame.getField(alert, "runbookPath", "")).length > 0
                                    onClicked: {
                                        if (root.runbookControllerObj)
                                            root.runbookControllerObj.openRunbook(alertFrame.getField(alert, "runbookPath", ""))
                                    }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                visible: root.listSize(alertFrame.getField(alert, "manualSteps", null)) > 0

                                Text {
                                    text: qsTrId("runbookPanel.manualSteps")
                                    font.bold: true
                                    color: Styles.AppTheme.textPrimary
                                }

                                Repeater {
                                    model: alertFrame.getField(alert, "manualSteps", []) || []

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
                                visible: root.listSize(alertFrame.getField(alert, "automaticActions", null)) > 0

                                Text {
                                    text: qsTrId("runbookPanel.automaticActions")
                                    font.bold: true
                                    color: Styles.AppTheme.textPrimary
                                }

                                Repeater {
                                    objectName: "runbookActionsRepeater_" + alertFrame.alertIndex
                                    model: alertFrame.getField(alert, "automaticActions", []) || []

                                    Button {
                                        readonly property var rawId: alertFrame.getField(modelData, "id", undefined)
                                        readonly property string actionId: (rawId === undefined || rawId === null || ("" + rawId).length === 0)
                                                                          ? ("idx_" + index)
                                                                          : ("" + rawId)
                                        readonly property string actionLabel: "" + alertFrame.getField(modelData, "label", "")

                                        objectName: "runbookActionButton_" + actionId
                                        text: actionLabel
                                        onClicked: root.requestRunbookAction(alertFrame.getField(alert, "runbookId", ""), modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                Label {
                    visible: !root.runbookControllerObj
                             || root.listSize(root.runbookControllerObj.alerts) === 0
                    text: qsTrId("runbookPanel.emptyState")
                    color: Styles.AppTheme.textSecondary
                }
            }
        }
    }

    Dialog {
        id: confirmActionDialog
        parent: root
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
        if (!root.runbookControllerObj)
            return
        var actionId = ""
        var label = ""
        var confirmMessage = ""

        if (actionPayload) {
            actionId = actionPayload["id"] !== undefined ? actionPayload["id"] : actionPayload.id
            label = actionPayload["label"] !== undefined ? actionPayload["label"] : actionPayload.label
            confirmMessage = actionPayload["confirmMessage"] !== undefined ? actionPayload["confirmMessage"] : actionPayload.confirmMessage
        }

        actionId = (actionId === undefined || actionId === null) ? "" : ("" + actionId)
        label = (label === undefined || label === null) ? "" : ("" + label)
        confirmMessage = (confirmMessage === undefined || confirmMessage === null) ? "" : ("" + confirmMessage)

        root.pendingAction = ({
            runbookId: runbookId,
            actionId: actionId,
            label: label,
            confirmMessage: confirmMessage
        })

        if (confirmMessage.length > 0) {
            confirmActionDialog.open()
        } else {
            root.executePendingAction()
        }
    }

    function executePendingAction() {
        if (!root.runbookControllerObj || !root.pendingAction.runbookId || !root.pendingAction.actionId)
            return
        root.runbookControllerObj.runAction(root.pendingAction.runbookId, root.pendingAction.actionId)
    }
}
