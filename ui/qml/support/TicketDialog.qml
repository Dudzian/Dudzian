import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Dialog {
    id: root
    objectName: "ticketDialog"
    modal: true
    title: qsTrId("ticketDialog.title")
    standardButtons: Dialog.NoButton
    implicitWidth: 560
    implicitHeight: 420
    closePolicy: Popup.NoAutoClose

    property var diagnosticsController: (typeof diagnosticsController !== "undefined" ? diagnosticsController : null)

    signal diagnosticsRequested()

    function reset() {
        if (diagnosticsController) {
            diagnosticsController.description = ""
        }
        descriptionField.text = ""
    }

    function triggerExport() {
        if (!diagnosticsController)
            return
        diagnosticsController.description = descriptionField.text
        diagnosticsController.outputDirectory = outputField.text
        diagnosticsController.generateDiagnostics()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16

        Label {
            text: qsTrId("ticketDialog.descriptionLabel")
            font.bold: true
        }

        TextArea {
            id: descriptionField
            objectName: "ticketDialogDescription"
            Layout.fillWidth: true
            Layout.fillHeight: true
            placeholderText: qsTrId("ticketDialog.descriptionPlaceholder")
            wrapMode: TextEdit.Wrap
            enabled: !diagnosticsController || !diagnosticsController.busy
        }

        Label {
            text: qsTrId("ticketDialog.outputLabel")
            font.bold: true
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: outputField
                objectName: "ticketDialogOutputField"
                Layout.fillWidth: true
                text: diagnosticsController ? diagnosticsController.outputDirectory : ""
                enabled: !diagnosticsController || !diagnosticsController.busy
            }

            Button {
                id: refreshButton
                objectName: "ticketDialogRefreshButton"
                text: qsTrId("ticketDialog.action.useDefault")
                enabled: !!diagnosticsController && !diagnosticsController.busy
                onClicked: outputField.text = diagnosticsController.outputDirectory
            }
        }

        Rectangle {
            id: statusBanner
            objectName: "ticketDialogStatusBanner"
            Layout.fillWidth: true
            implicitHeight: diagnosticsController && diagnosticsController.statusMessageId !== "ticketDialog.status.idle" ? 48 : 0
            radius: 6
            visible: diagnosticsController && diagnosticsController.statusMessageId !== "ticketDialog.status.idle"
            color: diagnosticsController && diagnosticsController.statusMessageId === "ticketDialog.status.error" ? Qt.rgba(0.75, 0.15, 0.15, 0.85)
                   : diagnosticsController && diagnosticsController.statusMessageId === "ticketDialog.status.success" ? Qt.rgba(0.16, 0.55, 0.28, 0.85)
                   : Qt.rgba(0.14, 0.5, 0.8, 0.4)

            Text {
                anchors.centerIn: parent
                color: "white"
                font.pointSize: 11
                text: diagnosticsController
                      ? qsTrId(diagnosticsController.statusMessageId)
                            + (diagnosticsController.statusDetails && diagnosticsController.statusDetails.length > 0
                                   ? "\n" + diagnosticsController.statusDetails
                                   : "")
                      : ""
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }
        }
    }

    footer: DialogButtonBox {
        spacing: 12

        Button {
            text: qsTrId("ticketDialog.action.generate")
            DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole
            enabled: !!diagnosticsController && !diagnosticsController.busy
            onClicked: root.triggerExport()
        }

        Button {
            text: diagnosticsController && diagnosticsController.busy ? qsTrId("ticketDialog.action.closeDisabled") : qsTrId("ticketDialog.action.close")
            DialogButtonBox.buttonRole: DialogButtonBox.RejectRole
            enabled: !diagnosticsController || !diagnosticsController.busy
            onClicked: root.close()
        }
    }

    Connections {
        target: diagnosticsController
        ignoreUnknownSignals: true

        function onOutputDirectoryChanged() {
            if (diagnosticsController)
                outputField.text = diagnosticsController.outputDirectory
        }

        function onExportCompleted(path) {
            statusBanner.color = Qt.rgba(0.16, 0.55, 0.28, 0.85)
            descriptionField.text = ""
        }

        function onExportFailed() {
            statusBanner.color = Qt.rgba(0.75, 0.15, 0.15, 0.85)
        }
    }
}
