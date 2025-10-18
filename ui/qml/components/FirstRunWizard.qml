import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

FocusScope {
    id: wizard
    anchors.fill: parent
    visible: !licenseController.licenseActive
    enabled: visible
    property int step: 0
    property string fingerprintValue: activationController && activationController.fingerprint
                                      && activationController.fingerprint.payload
                                      ? activationController.fingerprint.payload.fingerprint || ""
                                      : ""

    onVisibleChanged: {
        if (visible) {
            step = 0
            if (activationController)
                activationController.refresh()
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.76)
    }

    Pane {
        id: card
        anchors.centerIn: parent
        width: Math.min(parent.width - 80, 720)
        padding: 24
        background: Rectangle {
            color: Qt.darker(card.palette.window, 1.05)
            radius: 14
            border.color: Qt.rgba(0.2, 0.6, 0.9, 0.4)
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 18

            Label {
                text: qsTr("Kreator pierwszego uruchomienia")
                font.pixelSize: 24
                font.bold: true
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: step === 0
                      ? qsTr("Zweryfikuj fingerprint urządzenia, zapisz go w konfiguracji i przejdź do aktywacji licencji.")
                      : step === 1
                        ? qsTr("Zaimportuj licencję OEM zgodną z fingerprintem urządzenia.")
                        : qsTr("Licencja aktywowana – możesz korzystać z aplikacji.")
            }

            StackLayout {
                id: stack
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: step

                // Step 0 - Fingerprint preview
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12

                        Label {
                            text: qsTr("Fingerprint")
                            font.pixelSize: 18
                            font.bold: true
                        }

                        TextArea {
                            id: fingerprintArea
                            Layout.fillWidth: true
                            Layout.preferredHeight: 120
                            readOnly: true
                            wrapMode: TextEdit.Wrap
                            text: activationController && activationController.fingerprint
                                  ? JSON.stringify(activationController.fingerprint, null, 2)
                                  : qsTr("Brak danych fingerprintu")
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Button {
                                text: qsTr("Kopiuj do schowka")
                                onClicked: Qt.application.clipboard.setText(fingerprintArea.text)
                            }

                            Button {
                                text: qsTr("Zapisz do pliku")
                                onClicked: exportDialog.open()
                            }

                            Button {
                                text: qsTr("Zapisz w konfiguracji")
                                enabled: fingerprintValue.length > 0
                                onClicked: licenseController.saveExpectedFingerprint(fingerprintValue)
                            }

                            Item { Layout.fillWidth: true }
                        }

                        ListView {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.min(contentHeight, 160)
                            model: activationController && activationController.fingerprint
                                   && activationController.fingerprint.payload
                                   ? activationController.fingerprint.payload.component_list || []
                                   : []
                            clip: true

                            delegate: Frame {
                                required property var modelData
                                Layout.fillWidth: true
                                padding: 8

                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: 2
                                    Label { text: modelData.name || qsTr("Komponent") }
                                    Label {
                                        visible: !!modelData.normalized
                                        text: qsTr("Normalized: %1").arg(modelData.normalized)
                                        color: palette.mid
                                    }
                                    Label {
                                        visible: !!modelData.digest
                                        text: qsTr("Digest: %1").arg(modelData.digest)
                                        font.family: "monospace"
                                    }
                                }
                            }
                        }
                    }
                }

                // Step 1 - License activation
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12

                        Label {
                            text: qsTr("Wczytaj licencję OEM")
                            font.pixelSize: 18
                            font.bold: true
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Button {
                                text: qsTr("Wybierz plik")
                                onClicked: licenseFileDialog.open()
                            }

                            Button {
                                text: qsTr("Zastosuj payload")
                                enabled: licensePayload.text.length > 0
                                onClicked: licenseController.applyLicenseText(licensePayload.text)
                            }

                            Button {
                                text: qsTr("Wyczyść")
                                onClicked: licensePayload.text = ""
                            }

                            Item { Layout.fillWidth: true }
                        }

                        TextArea {
                            id: licensePayload
                            Layout.fillWidth: true
                            Layout.preferredHeight: 140
                            wrapMode: TextEdit.Wrap
                            placeholderText: qsTr("Wklej payload licencji JSON lub base64")
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            visible: licenseController.statusMessage.length > 0
                            text: licenseController.statusMessage
                            color: licenseController.statusIsError
                                   ? Qt.rgba(0.92, 0.36, 0.32, 1)
                                   : Qt.rgba(0.36, 0.72, 0.46, 1)
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            visible: licenseController.licenseActive
                            text: qsTr("Aktywowano profil %1 – ważna do %2")
                                  .arg(licenseController.licenseProfile)
                                  .arg(licenseController.licenseExpiresAt)
                        }
                    }
                }

                // Step 2 - Success
                Item {
                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 12

                        Label {
                            text: qsTr("Licencja aktywna")
                            font.pixelSize: 22
                            font.bold: true
                        }

                        Label {
                            text: qsTr("Fingerprint: %1").arg(licenseController.licenseFingerprint)
                            color: palette.mid
                        }

                        Label {
                            text: qsTr("Profil: %1 (ważna do %2)")
                                  .arg(licenseController.licenseProfile)
                                  .arg(licenseController.licenseExpiresAt)
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Button {
                    text: qsTr("Wstecz")
                    enabled: step > 0
                    onClicked: step = Math.max(0, step - 1)
                }

                Item { Layout.fillWidth: true }

                Button {
                    text: step === 1 ? qsTr("Zakończ") : qsTr("Dalej")
                    enabled: step === 0 || licenseController.licenseActive
                    visible: step < 2
                    onClicked: {
                        if (step === 0) {
                            step = 1
                        } else if (step === 1 && licenseController.licenseActive) {
                            step = 2
                        }
                    }
                }

                Button {
                    text: qsTr("Zamknij")
                    visible: step === 2
                    onClicked: wizard.visible = false
                }
            }
        }
    }

    FileDialog {
        id: licenseFileDialog
        title: qsTr("Wybierz plik licencji")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("Dokumenty JSON (*.json *.jsonl)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (selectedFile)
                licenseController.loadLicenseUrl(selectedFile)
        }
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Eksport fingerprintu")
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("Dokument JSON (*.json)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (selectedFile)
                activationController.exportFingerprint(selectedFile)
        }
    }

    Connections {
        target: activationController
        function onFingerprintChanged() {
            fingerprintValue = activationController && activationController.fingerprint
                               && activationController.fingerprint.payload
                               ? activationController.fingerprint.payload.fingerprint || ""
                               : ""
        }
    }

    Connections {
        target: licenseController
        function onLicenseActiveChanged() {
            if (licenseController.licenseActive) {
                step = 2
                wizard.visible = false
            }
        }
    }
}
