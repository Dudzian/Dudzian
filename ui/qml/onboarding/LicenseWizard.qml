import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

import "../styles" as Styles

Item {
    id: root
    width: parent ? parent.width : 960
    height: parent ? parent.height : 640

    property var licensingController: (typeof licensingController !== "undefined" ? licensingController : null)
    property var onboardingService: (typeof onboardingService !== "undefined" ? onboardingService : null)
    property int currentStep: 0
    readonly property int totalSteps: 5
    property bool summarySuccess: false
    property string summaryStatusId: "licenseWizard.status.pending"
    property string summaryDetails: ""
    property bool strategySetupReady: false
    property string selectedStrategyTitle: ""
    property string onboardingStatusMessageId: ""
    property string onboardingStatusDetails: ""
    property string lastConfiguredExchangeId: ""

    signal wizardCompleted(bool success)

    function trId(id, fallback) {
        const translated = qsTrId(id)
        return translated === id ? fallback : translated
    }

    function canProceed(step) {
        if (step === 2)
            return licensingController && licensingController.licenseAccepted
        if (step === 3)
            return strategySetupReady
        return true
    }

    function goToNextStep() {
        if (currentStep < totalSteps - 1 && canProceed(currentStep)) {
            currentStep += 1
        }
    }

    function goToPreviousStep() {
        if (currentStep > 0) {
            currentStep -= 1
        }
    }

    function resetSummary() {
        summarySuccess = false
        summaryStatusId = "licenseWizard.status.pending"
        summaryDetails = ""
    }

    function applySummaryFromController() {
        if (!licensingController)
            return
        summaryDetails = licensingController.statusDetails || ""
        updateSummaryState()
    }

    function updateSummaryState() {
        const licenseAccepted = licensingController && licensingController.licenseAccepted
        if (!licenseAccepted) {
            summarySuccess = false
            summaryStatusId = licensingController
                ? (licensingController.statusMessageId || "licenseWizard.status.pending")
                : "licenseWizard.status.pending"
            return
        }
        if (!strategySetupReady) {
            summarySuccess = false
            summaryStatusId = "licenseWizard.status.strategyPending"
            return
        }
        summarySuccess = true
        summaryStatusId = "licenseWizard.status.ok"
    }

    onLicensingControllerChanged: {
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
    }

    Connections {
        target: licensingController
        ignoreUnknownSignals: true
        function onLicenseAcceptedChanged() {
            applySummaryFromController()
            if (licensingController.licenseAccepted) {
                currentStep = Math.min(totalSteps - 1, 3)
            }
        }
        function onStatusMessageIdChanged() {
            applySummaryFromController()
        }
        function onStatusDetailsChanged() {
            applySummaryFromController()
        }
        function onFingerprintChanged() {
            // No-op – UI aktualizuje się poprzez bindingi.
        }
    }

    Connections {
        target: onboardingService
        ignoreUnknownSignals: true
        function onConfigurationReadyChanged() {
            strategySetupReady = onboardingService ? onboardingService.configurationReady : false
            updateSummaryState()
        }
        function onSelectedStrategyChanged() {
            selectedStrategyTitle = onboardingService ? onboardingService.selectedStrategyTitle : ""
        }
        function onStatusMessageIdChanged() {
            onboardingStatusMessageId = onboardingService ? onboardingService.statusMessageId : ""
        }
        function onStatusDetailsChanged() {
            onboardingStatusDetails = onboardingService ? onboardingService.statusDetails : ""
        }
        function onLastSavedExchangeChanged() {
            lastConfiguredExchangeId = onboardingService ? onboardingService.lastSavedExchange : ""
        }
        function onStrategiesChanged() {
            updateSummaryState()
        }
    }

    Component.onCompleted: {
        resetSummary()
        applySummaryFromController()
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
        if (onboardingService && onboardingService.refreshStrategies)
            onboardingService.refreshStrategies()
        strategySetupReady = onboardingService ? onboardingService.configurationReady : false
        selectedStrategyTitle = onboardingService ? onboardingService.selectedStrategyTitle : ""
        onboardingStatusMessageId = onboardingService ? onboardingService.statusMessageId : ""
        onboardingStatusDetails = onboardingService ? onboardingService.statusDetails : ""
        lastConfiguredExchangeId = onboardingService ? onboardingService.lastSavedExchange : ""
        updateSummaryState()
    }

    FileDialog {
        id: licenseFileDialog
        title: root.trId("licenseWizard.dialog.selectFile", "Wybierz plik licencji")
        nameFilters: ["JSON (*.json)"]
        onAccepted: {
            if (!licensingController)
                return
            if (selectedFile && selectedFile !== "") {
                const ok = licensingController.applyLicenseFile(selectedFile)
                applySummaryFromController()
                if (ok) {
                    currentStep = Math.min(totalSteps - 1, currentStep + 1)
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingLg
        padding: Styles.AppTheme.spacingLg

        Label {
            objectName: "licenseWizardTitle"
            text: root.trId("licenseWizard.title", "Kreator aktywacji licencji")
            font.pixelSize: Styles.AppTheme.fontSizeHeadline
            font.bold: true
            color: Styles.AppTheme.textPrimary
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm
            Repeater {
                model: totalSteps
                delegate: Rectangle {
                    readonly property bool visited: index <= currentStep
                    Layout.fillWidth: true
                    Layout.preferredHeight: 6
                    radius: 3
                    color: visited ? Styles.AppTheme.accent : Qt.rgba(Styles.AppTheme.surfaceSubtle.r,
                                                                       Styles.AppTheme.surfaceSubtle.g,
                                                                       Styles.AppTheme.surfaceSubtle.b,
                                                                       0.35)
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height - 200
            background: Rectangle {
                color: Styles.AppTheme.cardBackground(0.95)
                radius: Styles.AppTheme.radiusLarge
            }

            StackLayout {
                id: wizardStepStack
                objectName: "licenseWizardStepStack"
                anchors.fill: parent
                currentIndex: Math.max(0, Math.min(currentStep, totalSteps - 1))

                Flickable {
                    contentWidth: parent.width
                    contentHeight: welcomeColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: welcomeColumn
                        width: parent.width
                        spacing: Styles.AppTheme.spacingMd
                        padding: Styles.AppTheme.spacingXl

                        Label {
                            text: root.trId("licenseWizard.step.welcomeTitle", "Witamy w kreatorze licencji")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: root.trId(
                                "licenseWizard.step.welcomeDescription",
                                "Ten kreator pomoże Ci zweryfikować fingerprint urządzenia oraz aktywować licencję OEM."
                            )
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }
                    }
                }

                Flickable {
                    contentWidth: parent.width
                    contentHeight: hardwareColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: hardwareColumn
                        width: parent.width
                        spacing: Styles.AppTheme.spacingMd
                        padding: Styles.AppTheme.spacingXl

                        Label {
                            text: root.trId("licenseWizard.step.hardwareTitle", "Krok 1 – sprawdzenie fingerprintu")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: root.trId(
                                "licenseWizard.step.hardwareDescription",
                                "Zweryfikuj, czy odcisk sprzętowy (HWID) zgadza się z danymi licencyjnymi."
                            )
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            radius: Styles.AppTheme.radiusMedium
                            color: Styles.AppTheme.surfaceSubtle
                            border.width: 1
                            border.color: Styles.AppTheme.surfaceBorder

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: Styles.AppTheme.spacingLg
                                spacing: Styles.AppTheme.spacingSm

                                Label {
                                    text: root.trId("licenseWizard.label.fingerprint", "Fingerprint urządzenia")
                                    font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                                    color: Styles.AppTheme.textPrimary
                                }

                                Label {
                                    objectName: "licenseWizardFingerprintValue"
                                    Layout.fillWidth: true
                                    wrapMode: Text.WrapAnywhere
                                    text: {
                                        if (!licensingController)
                                            return root.trId("licenseWizard.label.fingerprintUnavailable", "Brak danych o fingerprintcie")
                                        const value = licensingController.fingerprint
                                        if (value)
                                            return value
                                        const errorId = licensingController.fingerprintErrorMessageId || "licenseWizard.error.fingerprintUnavailable"
                                        return root.trId(errorId, "Nie udało się odczytać fingerprintu.")
                                    }
                                    color: Styles.AppTheme.textPrimary
                                }

                                Button {
                                    objectName: "licenseWizardRefreshButton"
                                    text: root.trId("licenseWizard.action.refreshFingerprint", "Odśwież fingerprint")
                                    enabled: licensingController
                                    onClicked: {
                                        if (licensingController && licensingController.refreshFingerprint)
                                            licensingController.refreshFingerprint()
                                    }
                                }
                            }
                        }
                    }
                }

                Flickable {
                    contentWidth: parent.width
                    contentHeight: licenseColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: licenseColumn
                        width: parent.width
                        spacing: Styles.AppTheme.spacingMd
                        padding: Styles.AppTheme.spacingXl

                        Label {
                            text: root.trId("licenseWizard.step.licenseTitle", "Krok 2 – wprowadzenie licencji")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: root.trId(
                                "licenseWizard.step.licenseDescription",
                                "Wklej treść pliku licencyjnego (JSON) lub załaduj ją z pliku."
                            )
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        TextArea {
                            id: licenseInput
                            objectName: "licenseWizardInput"
                            Layout.fillWidth: true
                            Layout.minimumHeight: 200
                            placeholderText: root.trId(
                                "licenseWizard.placeholder.licenseInput",
                                "Wklej treść pliku licencji (*.json)"
                            )
                            wrapMode: TextEdit.WrapAnywhere
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm

                            Button {
                                objectName: "licenseWizardFileButton"
                                text: root.trId("licenseWizard.action.loadFromFile", "Wybierz plik…")
                                enabled: licensingController
                                onClicked: licenseFileDialog.open()
                            }

                            Button {
                                objectName: "licenseWizardApplyButton"
                                text: root.trId("licenseWizard.action.applyLicense", "Zastosuj licencję")
                                enabled: licensingController && licenseInput.text.length > 0
                                onClicked: {
                                    if (!licensingController)
                                        return
                                    const ok = licensingController.applyLicenseText(licenseInput.text)
                                    applySummaryFromController()
                                    if (ok) {
                                        currentStep = Math.min(totalSteps - 1, currentStep + 1)
                                    }
                                }
                            }
                        }

                        Label {
                            objectName: "licenseWizardStatusLabel"
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: (licensingController && licensingController.licenseAccepted)
                                   ? Styles.AppTheme.success
                                   : Styles.AppTheme.textSecondary
                            text: {
                                if (!licensingController)
                                    return root.trId("licenseWizard.status.pending", "Oczekiwanie na weryfikację licencji")
                                const id = licensingController.statusMessageId || "licenseWizard.status.pending"
                                return root.trId(id, "Oczekiwanie na weryfikację licencji")
                            }
                        }

                        Label {
                            objectName: "licenseWizardStatusDetails"
                            visible: licensingController && licensingController.statusDetails.length > 0
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: Styles.AppTheme.textSecondary
                            text: licensingController ? licensingController.statusDetails : ""
                        }
                    }
                }

                StrategySetupStep {
                    id: strategySetupStep
                    objectName: "licenseWizardStrategyStep"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    onboardingService: root.onboardingService
                    onCompletionStateChanged: function(ready) {
                        strategySetupReady = ready
                        updateSummaryState()
                    }
                }

                Flickable {
                    contentWidth: parent.width
                    contentHeight: summaryColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: summaryColumn
                        width: parent.width
                        spacing: Styles.AppTheme.spacingMd
                        padding: Styles.AppTheme.spacingXl

                        Label {
                            text: root.trId("licenseWizard.step.summaryTitle", "Podsumowanie aktywacji")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: summarySuccess
                                  ? root.trId(
                                      "licenseWizard.step.summaryDescriptionSuccess",
                                      "Licencja została aktywowana. Możesz przejść do dalszej konfiguracji."
                                    )
                                  : root.trId(
                                      "licenseWizard.step.summaryDescriptionFailure",
                                      "Licencja nie została aktywowana. Wróć do poprzednich kroków, aby poprawić dane."
                                    )
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            radius: Styles.AppTheme.radiusMedium
                            color: summarySuccess ? Styles.AppTheme.successMuted : Styles.AppTheme.surfaceSubtle
                            border.width: 1
                            border.color: summarySuccess ? Styles.AppTheme.success : Styles.AppTheme.surfaceBorder

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: Styles.AppTheme.spacingLg
                                spacing: Styles.AppTheme.spacingSm

                                Label {
                                    text: root.trId("licenseWizard.summary.status", "Status weryfikacji")
                                    font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                                    color: Styles.AppTheme.textPrimary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryStatus"
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: root.trId(summaryStatusId, summarySuccess
                                            ? "Licencja aktywowana pomyślnie"
                                            : "Weryfikacja licencji zakończyła się błędem")
                                    color: summarySuccess ? Styles.AppTheme.success : Styles.AppTheme.textPrimary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryDetails"
                                    visible: summaryDetails.length > 0
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: summaryDetails
                                    color: Styles.AppTheme.textSecondary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryLicenseId"
                                    visible: licensingController && licensingController.licenseId.length > 0
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: licensingController
                                          ? root.trId("licenseWizard.summary.licenseId", "Identyfikator licencji: %1").arg(licensingController.licenseId)
                                          : ""
                                    color: Styles.AppTheme.textSecondary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryStrategy"
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: root.trId("licenseWizard.summary.strategy", "Wybrana strategia: %1").arg(selectedStrategyTitle || root.trId("licenseWizard.strategy.selected.none", "nie wybrano"))
                                    color: Styles.AppTheme.textSecondary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryExchange"
                                    visible: lastConfiguredExchangeId.length > 0
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: root.trId("licenseWizard.summary.exchange", "Skonfigurowana giełda: %1").arg(lastConfiguredExchangeId)
                                    color: Styles.AppTheme.textSecondary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryOnboardingStatus"
                                    visible: onboardingStatusMessageId.length > 0
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: root.trId(onboardingStatusMessageId, root.trId("licenseWizard.strategy.status.ready", "Konfiguracja strategii zakończona."))
                                    color: summarySuccess ? Styles.AppTheme.success : Styles.AppTheme.textPrimary
                                }

                                Label {
                                    objectName: "licenseWizardSummaryOnboardingDetails"
                                    visible: onboardingStatusDetails.length > 0
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: onboardingStatusDetails
                                    color: Styles.AppTheme.textSecondary
                                }
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            Button {
                objectName: "licenseWizardPrevButton"
                text: root.trId("licenseWizard.action.previous", "Wstecz")
                enabled: currentStep > 0
                onClicked: {
                    if (currentStep > 0) {
                        currentStep -= 1
                        if (currentStep < totalSteps - 1)
                            resetSummary()
                    }
                }
            }

            Item { Layout.fillWidth: true }

            Button {
                objectName: "licenseWizardNextButton"
                visible: currentStep < totalSteps - 1
                enabled: currentStep === 0
                         || currentStep === 1
                         || (currentStep === 2 && licensingController && licensingController.licenseAccepted)
                         || (currentStep === 3 && strategySetupReady)
                text: root.trId("licenseWizard.action.next", "Dalej")
                onClicked: {
                    if (currentStep === 2 && (!licensingController || !licensingController.licenseAccepted))
                        return
                    if (currentStep === 3 && !strategySetupReady)
                        return
                    if (currentStep < totalSteps - 1)
                        currentStep += 1
                }
            }

            Button {
                objectName: "licenseWizardFinishButton"
                visible: currentStep === totalSteps - 1
                text: root.trId("licenseWizard.action.finish", "Zakończ")
                onClicked: {
                    wizardCompleted(summarySuccess)
                    if (licensingController && licensingController.finalizeOnboarding)
                        licensingController.finalizeOnboarding(
                            summarySuccess,
                            selectedStrategyTitle,
                            lastConfiguredExchangeId,
                            onboardingStatusMessageId
                        )
                }
            }
        }
    }
}
