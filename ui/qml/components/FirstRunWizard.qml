import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../styles" as Styles

FocusScope {
    id: wizard
    implicitWidth: 720
    implicitHeight: 520

    property var activationControllerRef: (typeof activationController !== "undefined" ? activationController : null)
    property var licenseControllerRef: (typeof licenseController !== "undefined" ? licenseController : null)
    visible: licenseControllerRef ? !Boolean(licenseControllerRef.licenseActive) : true
    enabled: visible
    property int step: 0
    property bool activationStarted: false
    property string fingerprintValue: activationControllerRef && activationControllerRef.fingerprint
                                      && activationControllerRef.fingerprint.payload
                                      ? activationControllerRef.fingerprint.payload.fingerprint || ""
                                      : ""

    function fileDialogPath(value) {
        if (!value)
            return ""
        if (value.toLocalFile) {
            var local = value.toLocalFile()
            if (local && local.length > 0)
                return local
        }
        if (value.toString)
            return value.toString()
        return "" + value
    }

    function canAutoProvision() {
        return !!(licenseControllerRef && typeof licenseControllerRef.autoProvision === "function")
    }

    function triggerAutoProvision() {
        if (!canAutoProvision())
            return
        licenseControllerRef.autoProvision(activationControllerRef ? activationControllerRef.fingerprint : ({}))
    }

    function beginActivationFlow() {
        step = 0
        if (activationControllerRef)
            activationControllerRef.refresh()
        triggerAutoProvision()
    }

    onVisibleChanged: {
        if (!visible) {
            activationStarted = false
            return
        }
        if (!activationStarted) {
            activationStarted = true
            beginActivationFlow()
        }
    }

    Component.onCompleted: {
        if (visible && !activationStarted) {
            activationStarted = true
            beginActivationFlow()
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
                            text: activationControllerRef && activationControllerRef.fingerprint
                                  ? JSON.stringify(activationControllerRef.fingerprint, null, 2)
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
                                enabled: licenseControllerRef && fingerprintValue.length > 0
                                onClicked: licenseControllerRef.saveExpectedFingerprint(fingerprintValue)
                            }

                            Item { Layout.fillWidth: true }
                        }

                        ListView {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.min(contentHeight, 160)
                            model: activationControllerRef && activationControllerRef.fingerprint
                                   && activationControllerRef.fingerprint.payload
                                   ? activationControllerRef.fingerprint.payload.component_list || []
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
                                enabled: licenseControllerRef && licensePayload.text.length > 0
                                onClicked: licenseControllerRef.applyLicenseText(licensePayload.text)
                            }

                            Button {
                                text: qsTr("Automatyczna aktywacja")
                                enabled: canAutoProvision() && !Boolean(licenseControllerRef && licenseControllerRef.provisioningInProgress)
                                onClicked: triggerAutoProvision()
                            }

                            Button {
                                text: qsTr("Wyczyść")
                                onClicked: licensePayload.text = ""
                            }

                            Item { Layout.fillWidth: true }
                        }

                        BusyIndicator {
                            running: Boolean(licenseControllerRef && licenseControllerRef.provisioningInProgress)
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
                            readonly property string statusText: (licenseControllerRef && licenseControllerRef.statusMessage)
                                                                 ? licenseControllerRef.statusMessage
                                                                 : ""
                            visible: statusText.length > 0
                            text: statusText
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: licenseControllerRef && licenseControllerRef.statusIsError
                                   ? Styles.AppTheme.negative
                                   : Styles.AppTheme.positive
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            visible: licenseControllerRef ? Boolean(licenseControllerRef.licenseActive) : false
                            text: qsTr("Aktywowano edycję %1 – utrzymanie do %2")
                                  .arg(licenseControllerRef ? licenseControllerRef.licenseEdition : "")
                                  .arg(licenseControllerRef && licenseControllerRef.licenseMaintenanceUntil ? licenseControllerRef.licenseMaintenanceUntil : qsTr("bez terminu"))
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
                            text: qsTr("Fingerprint: %1").arg(licenseControllerRef ? licenseControllerRef.licenseFingerprint : "")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Label {
                            text: qsTr("Edycja: %1 (utrzymanie do %2)")
                                  .arg(licenseControllerRef ? licenseControllerRef.licenseEdition : "")
                                  .arg(licenseControllerRef && licenseControllerRef.licenseMaintenanceUntil ? licenseControllerRef.licenseMaintenanceUntil : qsTr("bez terminu"))
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
                    enabled: step === 0 || Boolean(licenseControllerRef && licenseControllerRef.licenseActive)
                    visible: step < 2
                    onClicked: {
                        if (step === 0) {
                            step = 1
                        } else if (step === 1 && Boolean(licenseControllerRef && licenseControllerRef.licenseActive)) {
                            step = 2
                        }
                    }
                }

                Button {
                    text: qsTr("Zamknij")
                    visible: step === 2
                    onClicked: activationStarted = false
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
            const selectedPath = wizard.fileDialogPath(selectedFile)
            if (selectedPath.length === 0 || !licenseControllerRef)
                return
            licenseControllerRef.loadLicenseUrl(selectedFile)
        }
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Eksport fingerprintu")
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("Dokument JSON (*.json)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            const selectedPath = wizard.fileDialogPath(selectedFile)
            if (selectedPath.length === 0 || !activationControllerRef)
                return
            activationControllerRef.exportFingerprint(selectedFile)
        }
    }

    Connections {
        target: activationControllerRef
        ignoreUnknownSignals: true
        function onFingerprintChanged() {
            fingerprintValue = activationControllerRef && activationControllerRef.fingerprint
                               && activationControllerRef.fingerprint.payload
                               ? activationControllerRef.fingerprint.payload.fingerprint || ""
                               : ""
        }
    }

    Connections {
        target: licenseControllerRef
        ignoreUnknownSignals: true
        function onLicenseActiveChanged() {
            if (Boolean(licenseControllerRef && licenseControllerRef.licenseActive)) {
                step = 2
            }
        }
    }
}
