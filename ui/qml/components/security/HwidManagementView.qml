import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

Item {
    id: root
    property var activationControllerRef: typeof activationController !== "undefined" ? activationController : null
    property var licenseControllerRef: typeof licenseController !== "undefined" ? licenseController : null

    implicitWidth: parent ? parent.width : 640

    function fingerprintJson() {
        if (!activationControllerRef || !activationControllerRef.fingerprint)
            return "{}"
        return JSON.stringify(activationControllerRef.fingerprint, null, 2)
    }

    function resolvedFingerprint() {
        if (!activationControllerRef || !activationControllerRef.fingerprint)
            return ""
        var document = activationControllerRef.fingerprint
        if (document.fingerprint)
            return document.fingerprint
        if (document.payload && document.payload.fingerprint)
            return document.payload.fingerprint
        return ""
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Fingerprint urządzenia")
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Odśwież")
                enabled: !!activationControllerRef
                onClicked: activationControllerRef.refresh()
            }

            Button {
                text: qsTr("Eksportuj…")
                enabled: !!activationControllerRef
                onClicked: exportDialog.open()
            }
        }

        TextArea {
            id: fingerprintText
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            text: fingerprintJson()
            wrapMode: TextEdit.Wrap
            readOnly: true
            font.family: "monospace"
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Fingerprint oczekiwany") }
            Label {
                text: licenseControllerRef ? (licenseControllerRef.expectedFingerprint || qsTr("brak")) : qsTr("brak")
                font.family: "monospace"
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Fingerprint lokalny") }
            Label {
                text: resolvedFingerprint() || qsTr("brak")
                font.family: "monospace"
                wrapMode: Text.WordWrap
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: overrideField
                Layout.fillWidth: true
                placeholderText: qsTr("Wpisz fingerprint do nadpisania")
            }

            Button {
                text: qsTr("Zapisz jako oczekiwany")
                enabled: licenseControllerRef && overrideField.text.length > 0
                onClicked: {
                    if (!licenseControllerRef)
                        return
                    licenseControllerRef.saveExpectedFingerprint(overrideField.text.trim())
                    overrideField.text = ""
                }
            }

            Button {
                text: qsTr("Nadpisz konfigurację")
                enabled: licenseControllerRef && overrideField.text.length > 0
                onClicked: {
                    if (!licenseControllerRef)
                        return
                    licenseControllerRef.overrideExpectedFingerprint(overrideField.text.trim())
                    overrideField.text = ""
                }
            }
        }
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Zapisz dokument fingerprint")
        fileMode: FileDialog.SaveFile
        defaultSuffix: "json"
        nameFilters: [qsTr("Dokumenty JSON (*.json)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (!activationControllerRef)
                return
            if (selectedFile)
                activationControllerRef.exportFingerprint(selectedFile)
        }
    }
}
