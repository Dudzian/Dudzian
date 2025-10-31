import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../styles" as Styles

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
            if (licenseController)
                licenseController.autoProvision(activationController ? activationController.fingerprint : ({}))
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Styles.AppTheme.overlayBackground()
    }

    Pane {
        id: card
        objectName: "firstRunWizardCard"
        anchors.centerIn: parent
        width: Math.min(parent.width - 80, 720)
        padding: Styles.AppTheme.spacingXl
        background: Rectangle {
            color: Styles.AppTheme.cardBackground(0.95)
            radius: Styles.AppTheme.radiusLarge
            border.color: Styles.AppTheme.accentMuted
        }

        ColumnLayout {
            anchors.fill: parent
            objectName: "firstRunWizardColumn"
            spacing: Styles.AppTheme.spacingLg

            Label {
                text: qsTr("Kreator pierwszego uruchomienia")
                font.pixelSize: Styles.AppTheme.fontSizeHeadline
                font.family: Styles.AppTheme.fontFamily
                font.bold: true
                color: Styles.AppTheme.textPrimary
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: step === 0
                      ? qsTr("Zweryfikuj fingerprint urządzenia, zapisz go w konfiguracji i przejdź do aktywacji licencji.")
                      : step === 1
                        ? qsTr("Zaimportuj licencję OEM zgodną z fingerprintem urządzenia.")
                        : qsTr("Licencja aktywowana – możesz korzystać z aplikacji.")
                font.family: Styles.AppTheme.fontFamily
                font.pixelSize: Styles.AppTheme.fontSizeBody
                color: Styles.AppTheme.textSecondary
            }

            StackLayout {
                id: stack
                objectName: "firstRunWizardSteps"
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: step

                // Step 0 - Fingerprint preview
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Fingerprint")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
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
                            font.family: Styles.AppTheme.monoFontFamily
                            color: Styles.AppTheme.textPrimary
                            background: Rectangle {
                                radius: Styles.AppTheme.radiusSmall
                                color: Styles.AppTheme.surfaceSubtle
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm

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
                                padding: Styles.AppTheme.spacingSm
                                background: Rectangle {
                                    radius: Styles.AppTheme.radiusSmall
                                    color: Styles.AppTheme.cardBackground(0.85)
                                }

                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: Styles.AppTheme.spacingXs
                                    Label {
                                        text: modelData.name || qsTr("Komponent")
                                        font.family: Styles.AppTheme.fontFamily
                                        font.pixelSize: Styles.AppTheme.fontSizeBody
                                        color: Styles.AppTheme.textPrimary
                                    }
                                    Label {
                                        visible: !!modelData.normalized
                                        text: qsTr("Normalized: %1").arg(modelData.normalized)
                                        font.family: Styles.AppTheme.fontFamily
                                        font.pixelSize: Styles.AppTheme.fontSizeCaption
                                        color: Styles.AppTheme.textTertiary
                                    }
                                    Label {
                                        visible: !!modelData.digest
                                        text: qsTr("Digest: %1").arg(modelData.digest)
                                        font.family: Styles.AppTheme.monoFontFamily
                                        font.pixelSize: Styles.AppTheme.fontSizeCaption
                                        color: Styles.AppTheme.textSecondary
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
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Wczytaj licencję OEM")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm

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
                                text: qsTr("Automatyczna aktywacja")
                                enabled: licenseController && !licenseController.provisioningInProgress
                                onClicked: licenseController.autoProvision(activationController ? activationController.fingerprint : ({}))
                            }

                            Button {
                                text: qsTr("Wyczyść")
                                onClicked: licensePayload.text = ""
                            }

                            Item { Layout.fillWidth: true }
                        }

                        BusyIndicator {
                            running: licenseController && licenseController.provisioningInProgress
                            visible: running
                        }

                        TextArea {
                            id: licensePayload
                            Layout.fillWidth: true
                            Layout.preferredHeight: 140
                            wrapMode: TextEdit.Wrap
                            placeholderText: qsTr("Wklej payload licencji JSON lub base64")
                            font.family: Styles.AppTheme.monoFontFamily
                            color: Styles.AppTheme.textPrimary
                            background: Rectangle {
                                radius: Styles.AppTheme.radiusSmall
                                color: Styles.AppTheme.surfaceSubtle
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            visible: licenseController.statusMessage.length > 0
                            text: licenseController.statusMessage
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: licenseController.statusIsError
                                   ? Styles.AppTheme.negative
                                   : Styles.AppTheme.positive
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            visible: licenseController.licenseActive
                            text: qsTr("Aktywowano edycję %1 – utrzymanie do %2")
                                  .arg(licenseController.licenseEdition)
                                  .arg(licenseController.licenseMaintenanceUntil || qsTr("bez terminu"))
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }
                    }
                }

                // Step 2 - Success
                Item {
                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Licencja aktywna")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            text: qsTr("Fingerprint: %1").arg(licenseController.licenseFingerprint)
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Label {
                            text: qsTr("Edycja: %1 (utrzymanie do %2)")
                                  .arg(licenseController.licenseEdition)
                                  .arg(licenseController.licenseMaintenanceUntil || qsTr("bez terminu"))
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: Styles.AppTheme.spacingSm

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
