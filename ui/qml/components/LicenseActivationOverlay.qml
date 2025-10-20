import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

FocusScope {
    id: overlay
    anchors.fill: parent

    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.72)
    }

    Pane {
        id: card
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        width: Math.min(parent.width - 96, 640)
        padding: 24
        background: Rectangle {
            color: Qt.darker(card.palette.window, 1.1)
            radius: 12
            border.color: Qt.rgba(0.3, 0.6, 0.9, 0.5)
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 16

            Label {
                text: qsTr("Aktywacja licencji OEM")
                font.pixelSize: 22
                font.bold: true
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: licenseController.expectedFingerprintAvailable
                      ? qsTr("Zabezpieczony pakiet oczekuje licencji podpisanej dla fingerprintu <b>%1</b>.")
                          .arg(licenseController.expectedFingerprint)
                      : qsTr("Zabezpieczony pakiet oczekuje licencji podpisanej dla tej stacji. Przygotuj plik JSON/JSONL lub kod QR wygenerowany przez dział OEM.")
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                color: Qt.rgba(0.75, 0.75, 0.78, 1)
                text: qsTr("1. Wybierz plik z nośnika USB lub zeskanowany payload.\n2. Zweryfikuj, że fingerprint i data ważności są zgodne z dokumentacją.\n3. Po aktywacji UI odblokuje dostęp do danych i modułów live.")
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: Qt.darker(card.palette.window, 1.25)
                    radius: 8
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Button {
                            text: qsTr("Wczytaj licencję z pliku (USB)")
                            onClicked: licenseFileDialog.open()
                        }

                        Button {
                            text: qsTr("Zastosuj payload tekstowy")
                            enabled: manualInput.text.length > 0
                            onClicked: licenseController.applyLicenseText(manualInput.text)
                        }

                        Button {
                            text: qsTr("Automatyczna aktywacja")
                            enabled: licenseController && !licenseController.provisioningInProgress
                            onClicked: licenseController.autoProvision(activationController ? activationController.fingerprint : ({}))
                        }

                        Button {
                            text: qsTr("Wyczyść pole")
                            onClicked: manualInput.text = ""
                        }

                        Item { Layout.fillWidth: true }
                    }

                    TextArea {
                        id: manualInput
                        Layout.fillWidth: true
                        Layout.preferredHeight: 140
                        wrapMode: TextEdit.Wrap
                        placeholderText: qsTr("Wklej payload JSON lub base64 z kodu QR")
                        selectByMouse: true
                        color: palette.text
                        background: Rectangle {
                            radius: 6
                            color: Qt.darker(card.palette.base, 1.1)
                            border.color: Qt.rgba(0.3, 0.6, 0.9, 0.4)
                        }
                    }
                }
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: licenseController.statusMessage.length > 0
                text: licenseController.statusMessage
                color: licenseController.statusIsError
                        ? Qt.rgba(0.94, 0.36, 0.32, 1)
                        : Qt.rgba(0.36, 0.74, 0.52, 1)
            }

            BusyIndicator {
                running: licenseController && licenseController.provisioningInProgress
                visible: running
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: licenseController.licenseActive
                text: qsTr("Aktywowano edycję %1. Utrzymanie do %2.")
                          .arg(licenseController.licenseEdition)
                          .arg(licenseController.licenseMaintenanceUntil || qsTr("bez terminu"))
            }
        }
    }

    FileDialog {
        id: licenseFileDialog
        title: qsTr("Wybierz dokument licencyjny OEM")
        nameFilters: [qsTr("Dokumenty JSON (*.json *.jsonl)"), qsTr("Wszystkie pliki (*)")]
        fileMode: FileDialog.OpenFile
        onAccepted: {
            if (selectedFile)
                licenseController.loadLicenseUrl(selectedFile)
        }
    }

    Connections {
        target: licenseController
        function onLicenseActiveChanged() {
            if (licenseController.licenseActive)
                manualInput.text = ""
        }
    }
}
