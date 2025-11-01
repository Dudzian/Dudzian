import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../styles" as Styles

Item {
    id: root
    width: parent ? parent.width : 960
    height: parent ? parent.height : 640

    property var onboardingService: (typeof onboardingService !== "undefined" ? onboardingService : null)
    property string selectedStrategyName: ""
    property string statusMessageId: ""
    property string statusDetails: ""
    property string lastSavedExchange: ""

    signal completionStateChanged(bool ready)

    function trId(id, fallback) {
        const translated = qsTrId(id)
        return translated === id ? fallback : translated
    }

    function _updateSelectionFromService() {
        if (!onboardingService)
            return
        selectedStrategyName = onboardingService.selectedStrategyTitle || onboardingService.selectedStrategy || ""
        const strategies = onboardingService.strategies || []
        const name = onboardingService.selectedStrategy || ""
        var index = -1
        for (var i = 0; i < strategies.length; ++i) {
            const candidate = strategies[i]
            if (candidate.name === name || candidate.engine === name) {
                index = i
                break
            }
        }
        strategyListView.currentIndex = index
    }

    function _syncStatus() {
        if (!onboardingService)
            return
        statusMessageId = onboardingService.statusMessageId || ""
        statusDetails = onboardingService.statusDetails || ""
        lastSavedExchange = onboardingService.lastSavedExchange || ""
    }

    function _emitCompletion() {
        const ready = onboardingService && onboardingService.configurationReady
        completionStateChanged(!!ready)
    }

    Component.onCompleted: {
        if (onboardingService && onboardingService.refreshStrategies)
            onboardingService.refreshStrategies()
        _updateSelectionFromService()
        _syncStatus()
        _emitCompletion()
    }

    Connections {
        target: onboardingService
        ignoreUnknownSignals: true
        function onStrategiesChanged() {
            strategyListView.model = onboardingService ? onboardingService.strategies : []
            _updateSelectionFromService()
        }
        function onSelectedStrategyChanged() {
            _updateSelectionFromService()
        }
        function onStatusMessageIdChanged() {
            _syncStatus()
        }
        function onStatusDetailsChanged() {
            _syncStatus()
        }
        function onLastSavedExchangeChanged() {
            _syncStatus()
        }
        function onConfigurationReadyChanged() {
            _emitCompletion()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingLg
        padding: Styles.AppTheme.spacingXl

        Label {
            text: root.trId("licenseWizard.step.strategyTitle", "Krok 3 – konfiguracja strategii")
            font.pixelSize: Styles.AppTheme.fontSizeTitle
            font.bold: true
            color: Styles.AppTheme.textPrimary
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: root.trId(
                "licenseWizard.step.strategyDescription",
                "Wybierz preferowaną strategię, a następnie skonfiguruj połączenie z giełdą, dodając klucze API."
            )
            font.pixelSize: Styles.AppTheme.fontSizeBody
            color: Styles.AppTheme.textSecondary
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 360

            ColumnLayout {
                Layout.fillWidth: true
                Layout.preferredWidth: parent ? parent.width * 0.5 : 480
                spacing: Styles.AppTheme.spacingMd

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Styles.AppTheme.spacingSm

                    Label {
                        text: root.trId("licenseWizard.strategy.available", "Dostępne strategie")
                        font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                        color: Styles.AppTheme.textPrimary
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        objectName: "strategySetupRefreshButton"
                        text: root.trId("licenseWizard.strategy.action.refresh", "Odśwież")
                        enabled: onboardingService
                        onClicked: {
                            if (onboardingService && onboardingService.refreshStrategies)
                                onboardingService.refreshStrategies()
                        }
                    }
                }

                ListView {
                    id: strategyListView
                    objectName: "strategySetupList"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: Styles.AppTheme.spacingSm
                    model: onboardingService ? onboardingService.strategies : []
                    onCurrentIndexChanged: {
                        if (!onboardingService || currentIndex < 0)
                            return
                        const items = onboardingService.strategies || []
                        if (currentIndex >= items.length)
                            return
                        const entry = items[currentIndex]
                        if (entry && onboardingService.selectStrategy)
                            onboardingService.selectStrategy(entry.name || entry.engine)
                    }
                    delegate: Frame {
                        required property var modelData
                        property bool selected: index === strategyListView.currentIndex
                        Layout.fillWidth: true
                        background: Rectangle {
                            color: selected ? Styles.AppTheme.accentSubtle : Styles.AppTheme.surfaceSubtle
                            radius: Styles.AppTheme.radiusMedium
                            border.width: selected ? 2 : 1
                            border.color: selected ? Styles.AppTheme.accent : Styles.AppTheme.surfaceBorder
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: Styles.AppTheme.spacingMd
                            spacing: Styles.AppTheme.spacingXs

                            Label {
                                text: modelData.title || modelData.name
                                font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                                font.bold: true
                                color: Styles.AppTheme.textPrimary
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                text: root.trId("licenseWizard.strategy.licenseTier", "Poziom licencji: %1").arg(modelData.licenseTier || "-")
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textSecondary
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                text: root.trId("licenseWizard.strategy.riskClasses", "Klasy ryzyka: %1").arg((modelData.riskClasses || []).join(", "))
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textSecondary
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                text: root.trId("licenseWizard.strategy.requiredData", "Wymagane dane: %1").arg((modelData.requiredData || []).join(", "))
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textSecondary
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: (modelData.tags || []).length > 0
                                wrapMode: Text.WordWrap
                                text: root.trId("licenseWizard.strategy.tags", "Tagi: %1").arg((modelData.tags || []).join(", "))
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textSecondary
                            }
                        }

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                strategyListView.currentIndex = index
                                if (onboardingService && onboardingService.selectStrategy)
                                    onboardingService.selectStrategy(modelData.name || modelData.engine)
                            }
                        }
                    }

                    footer: Label {
                        visible: strategyListView.count === 0
                        text: root.trId("licenseWizard.strategy.selectHint", "Brak zarejestrowanych strategii – spróbuj odświeżyć listę.")
                        color: Styles.AppTheme.textSecondary
                        horizontalAlignment: Text.AlignHCenter
                        Layout.fillWidth: true
                        padding: Styles.AppTheme.spacingMd
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredWidth: parent ? parent.width * 0.5 : 480
                color: Styles.AppTheme.surfaceSubtle
                radius: Styles.AppTheme.radiusMedium
                border.width: 1
                border.color: Styles.AppTheme.surfaceBorder

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingLg
                    spacing: Styles.AppTheme.spacingMd

                    Label {
                        text: root.trId("licenseWizard.strategy.selected", "Wybrana strategia: %1").arg(selectedStrategyName || root.trId("licenseWizard.strategy.selected.none", "nie wybrano"))
                        font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                        color: Styles.AppTheme.textPrimary
                    }

                    ComboBox {
                        id: exchangeSelector
                        objectName: "strategySetupExchangeCombo"
                        Layout.fillWidth: true
                        model: onboardingService ? onboardingService.availableExchanges : []
                        editable: false
                        currentIndex: model.length > 0 ? 0 : -1
                        textRole: ""
                        displayText: currentText || root.trId("licenseWizard.strategy.input.exchange", "Wybierz giełdę")
                    }

                    TextField {
                        id: apiKeyInput
                        objectName: "strategySetupApiKeyField"
                        Layout.fillWidth: true
                        placeholderText: root.trId("licenseWizard.strategy.input.apiKey", "Klucz API")
                    }

                    TextField {
                        id: apiSecretInput
                        objectName: "strategySetupApiSecretField"
                        Layout.fillWidth: true
                        echoMode: TextInput.Password
                        placeholderText: root.trId("licenseWizard.strategy.input.apiSecret", "Sekretny klucz API")
                    }

                    TextField {
                        id: passphraseInput
                        objectName: "strategySetupApiPassphraseField"
                        Layout.fillWidth: true
                        placeholderText: root.trId("licenseWizard.strategy.input.apiPassphrase", "Hasło/Passphrase (opcjonalnie)")
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Styles.AppTheme.spacingSm

                        Button {
                            objectName: "strategySetupSaveButton"
                            Layout.fillWidth: true
                            text: root.trId("licenseWizard.strategy.action.saveCredentials", "Zapisz klucze API")
                            enabled: onboardingService && apiKeyInput.text.length > 0 && apiSecretInput.text.length > 0 && exchangeSelector.currentIndex >= 0
                            onClicked: {
                                if (!onboardingService)
                                    return
                                const exchangeId = exchangeSelector.currentText || onboardingService.availableExchanges[exchangeSelector.currentIndex]
                                onboardingService.importApiKey(exchangeId, apiKeyInput.text, apiSecretInput.text, passphraseInput.text)
                                apiSecretInput.text = ""
                                passphraseInput.text = ""
                            }
                        }
                    }

                    Label {
                        objectName: "strategySetupStatusLabel"
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        color: statusMessageId === "onboarding.strategy.credentials.saved"
                               ? Styles.AppTheme.success
                               : Styles.AppTheme.textSecondary
                        text: {
                            if (!statusMessageId)
                                return root.trId("licenseWizard.strategy.status.missingSelection", "Wybierz strategię i dodaj klucze API, aby kontynuować.")
                            const translated = root.trId(statusMessageId, statusMessageId)
                            if (statusMessageId === "onboarding.strategy.credentials.saved" && lastSavedExchange.length > 0)
                                return translated.arg(lastSavedExchange)
                            return translated
                        }
                    }

                    Label {
                        objectName: "strategySetupStatusDetails"
                        visible: statusDetails.length > 0
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        text: root.trId(statusDetails, statusDetails)
                        color: Styles.AppTheme.textSecondary
                    }
                }
            }
        }
    }
}
