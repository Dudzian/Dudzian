import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

Dialog {
    id: root
    property var updateController: (typeof offlineUpdateController !== "undefined" ? offlineUpdateController : null)
    property string selectedFile: ""
    property string signingKey: ""

    title: qsTr("Aktualizacja offline (.kbot)")
    modal: true
    standardButtons: Dialog.Close
    width: 640

    function resetState() {
        selectedFile = ""
        signingKey = ""
        if (updateController && updateController.setSigningKey)
            updateController.setSigningKey("")
    }

    onClosed: resetState()

    FileDialog {
        id: packageDialog
        title: qsTr("Wybierz pakiet .kbot")
        nameFilters: [qsTr("Pakiety Kryptołowca (*.kbot)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (packageDialog.selectedFile && packageDialog.selectedFile.length > 0)
                selectedFile = packageDialog.selectedFile
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: qsTr("Wybierz podpisany pakiet aktualizacji w formacie .kbot i zaimportuj go do katalogu aktualizacji desktopowych.")
        }

        GroupBox {
            title: qsTr("Pakiet aktualizacji")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    TextField {
                        Layout.fillWidth: true
                        placeholderText: qsTr("Ścieżka do pakietu .kbot")
                        text: root.selectedFile
                        readOnly: true
                    }
                    Button {
                        text: qsTr("Wybierz plik")
                        onClicked: packageDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    TextField {
                        id: keyField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Klucz podpisu HMAC (opcjonalnie)")
                        text: root.signingKey
                        echoMode: TextInput.Password
                        onEditingFinished: {
                            root.signingKey = text
                            if (updateController)
                                updateController.setSigningKey(text)
                        }
                    }
                    Button {
                        text: qsTr("Wyczyść")
                        onClicked: {
                            keyField.text = ""
                            root.signingKey = ""
                            if (updateController)
                                updateController.setSigningKey("")
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: qsTr("Fingerprint docelowy")
                    }
                    TextField {
                        Layout.fillWidth: true
                        readOnly: updateController && updateController.fingerprint.length > 0
                        text: updateController ? updateController.fingerprint : ""
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Status")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 4

                Label {
                    Layout.fillWidth: true
                    text: updateController ? qsTrId(updateController.statusMessageId) : qsTr("Brak statusu")
                    color: updateController && updateController.statusMessageId.indexOf("error") >= 0
                           ? palette.negative
                           : palette.text
                }

                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: updateController ? updateController.statusDetails : ""
                    color: palette.mid
                }

                Label {
                    Layout.fillWidth: true
                    text: updateController && updateController.lastPackageId.length > 0
                          ? qsTr("Ostatnia paczka: %1").arg(updateController.lastPackageId)
                          : ""
                    color: palette.mid
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Item { Layout.fillWidth: true }
            Button {
                text: updateController && updateController.busy ? qsTr("Importowanie...") : qsTr("Importuj")
                enabled: updateController && !updateController.busy && root.selectedFile.length > 0
                onClicked: {
                    if (!updateController)
                        return
                    const ok = updateController.importPackage(root.selectedFile)
                    if (ok)
                        root.selectedFile = ""
                }
            }
        }
    }
}
