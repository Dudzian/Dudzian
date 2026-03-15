import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../components" as Components
import "../views" as Views
import "../styles" as Styles

Item {
    id: root
    width: parent ? parent.width : 960
    height: parent ? parent.height : 600

    property var appControllerContext: (typeof appController !== "undefined" ? appController : null)
    property var strategyControllerContext: (typeof strategyController !== "undefined" ? strategyController : null)
    property var workbenchControllerContext: (typeof workbenchController !== "undefined" ? workbenchController : null)
    property var licenseControllerContext: (typeof licenseController !== "undefined" ? licenseController : null)
    property var riskModelContext: (typeof riskModel !== "undefined" ? riskModel : null)
    property var wizardControllerContext: (typeof configurationWizard !== "undefined" ? configurationWizard : null)
    property var alertsModelContext: (typeof alertsModel !== "undefined" ? alertsModel : null)

    property var appControllerRef: appControllerContext
    property var strategyControllerRef: strategyControllerContext
    property var workbenchControllerRef: workbenchControllerContext
    property var licenseControllerRef: licenseControllerContext
    property var riskModelRef: riskModelContext
    property var wizardController: wizardControllerContext
    property var schedulerItems: strategyControllerRef ? strategyControllerRef.schedulerList() : []
    property var securityAlertsModel: alertsModelContext
    property int currentStep: 0
    property var exchangeOptions: []
    property string selectedExchange: ""
    property var instruments: []
    property int selectedInstrumentIndex: -1
    property var personalization: ({})

    signal wizardCompleted()

    readonly property int totalSteps: 4

    function refreshExchangeOptions() {
        if (!appControllerRef)
            return
        exchangeOptions = appControllerRef.supportedExchanges ? appControllerRef.supportedExchanges() : []
        if (exchangeOptions.length > 0 && (!selectedExchange || exchangeOptions.indexOf(selectedExchange) === -1)) {
            selectedExchange = exchangeOptions[0]
            loadInstruments()
        }
    }

    function loadInstruments() {
        if (!appControllerRef || !selectedExchange)
            return
        var result = appControllerRef.listTradableInstruments(selectedExchange) || []
        instruments = result
        selectedInstrumentIndex = instruments.length > 0 ? 0 : -1
    }

    function applySelectedInstrument() {
        if (!appControllerRef || selectedInstrumentIndex < 0 || selectedInstrumentIndex >= instruments.length)
            return false
        var item = instruments[selectedInstrumentIndex] || {}
        if (!item.config)
            return false
        var cfg = item.config
        return appControllerRef.updateInstrument(cfg.exchange || selectedExchange,
                                              cfg.symbol || "",
                                              cfg.venueSymbol || "",
                                              cfg.quoteCurrency || "",
                                              cfg.baseCurrency || "",
                                              cfg.granularityIso8601 || "PT1M")
    }

    function refreshPersonalization() {
        if (!appControllerRef)
            return
        personalization = appControllerRef.personalizationSnapshot ? appControllerRef.personalizationSnapshot() : {}
    }

    Component.onCompleted: {
        refreshExchangeOptions()
        refreshPersonalization()
        if (wizardController)
            wizardController.start("default")
    }

    Connections {
        target: appControllerRef
        ignoreUnknownSignals: true
        function onUiThemeChanged() { refreshPersonalization() }
        function onUiLayoutModeChanged() { refreshPersonalization() }
        function onAlertToastsEnabledChanged() { refreshPersonalization() }
    }

    Connections {
        target: strategyControllerRef
        ignoreUnknownSignals: true
        function onSchedulerListChanged() { schedulerItems = strategyControllerRef.schedulerList() }
    }

    ColumnLayout {
        anchors.fill: parent
        objectName: "setupWizardRootColumn"
        spacing: Styles.AppTheme.spacingLg
        anchors.margins: Styles.AppTheme.spacingLg

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm
            Repeater {
                model: totalSteps
                delegate: Rectangle {
                    readonly property bool visited: index <= currentStep
                    Layout.fillWidth: true
                    Layout.preferredHeight: 8
                    radius: 4
                    color: visited ? Styles.AppTheme.accent : Qt.rgba(Styles.AppTheme.surfaceSubtle.r, Styles.AppTheme.surfaceSubtle.g, Styles.AppTheme.surfaceSubtle.b, 0.4)
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            objectName: "setupWizardFrame"
            background: Rectangle {
                color: Styles.AppTheme.cardBackground(0.9)
                radius: Styles.AppTheme.radiusLarge
            }

            StackLayout {
                id: stepStack
                anchors.fill: parent
                currentIndex: Math.max(0, Math.min(currentStep, totalSteps - 1))

                // Step 0: License
                Flickable {
                    contentWidth: parent.width
                    contentHeight: licenseColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: licenseColumn
                        x: Styles.AppTheme.spacingLg
                        width: Math.max(0, parent.width - (2 * Styles.AppTheme.spacingLg))
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Krok 1 z 4 – aktywacja licencji")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Zweryfikuj licencję OEM i fingerprint urządzenia. Bez aktywnej licencji interfejs będzie działać w trybie ograniczonym.")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Components.FirstRunWizard {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 420
                            activationControllerRef: root.wizardController && root.wizardController.activationController ? root.wizardController.activationController : null
                            licenseControllerRef: root.licenseControllerRef
                            visible: true
                            enabled: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: Styles.AppTheme.textSecondary
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            text: Boolean(licenseControllerRef && licenseControllerRef.licenseActive) ?
                                  qsTr("Licencja została poprawnie aktywowana. Możesz przejść dalej.") :
                                  qsTr("Po pomyślnej aktywacji licencji przycisk 'Dalej' odblokuje kolejne kroki kreatora.")
                        }
                    }
                }

                // Step 1: Exchange connection
                Flickable {
                    contentWidth: parent.width
                    contentHeight: exchangeColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: exchangeColumn
                        x: Styles.AppTheme.spacingLg
                        width: Math.max(0, parent.width - (2 * Styles.AppTheme.spacingLg))
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Krok 2 z 4 – konfiguracja giełdy")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Wybierz domyślną giełdę i instrument, z którym aplikacja wystartuje po uruchomieniu. Pełne klucze API możesz uzupełnić w panelu bezpieczeństwa.")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm

                            ComboBox {
                                id: exchangeCombo
                                objectName: "setupWizardExchangeCombo"
                                Layout.preferredWidth: 260
                                model: exchangeOptions
                                currentIndex: Math.max(0, exchangeOptions.indexOf(selectedExchange))
                                onCurrentIndexChanged: {
                                    if (currentIndex >= 0 && currentIndex < exchangeOptions.length) {
                                        selectedExchange = exchangeOptions[currentIndex]
                                        loadInstruments()
                                    }
                                }
                                enabled: exchangeOptions.length > 0
                            }

                            Button {
                                text: qsTr("Odśwież instrumenty")
                                enabled: selectedExchange && selectedExchange.length > 0
                                onClicked: loadInstruments()
                            }
                        }

                        ListView {
                            id: instrumentView
                            objectName: "setupWizardInstrumentView"
                            Layout.fillWidth: true
                            Layout.preferredHeight: 260
                            clip: true
                            model: instruments
                            currentIndex: selectedInstrumentIndex
                            delegate: Frame {
                                required property int index
                                required property var modelData
                                width: ListView.view.width
                                padding: Styles.AppTheme.spacingSm
                                background: Rectangle {
                                    radius: Styles.AppTheme.radiusMedium
                                    color: ListView.isCurrentItem ? Styles.AppTheme.accentMuted : Styles.AppTheme.cardBackground(0.85)
                                }
                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: Styles.AppTheme.spacingXs
                                    Label {
                                        text: (modelData.config && modelData.config.symbol) ? modelData.config.symbol : qsTr("Instrument")
                                        font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                                        font.family: Styles.AppTheme.fontFamily
                                        font.bold: true
                                        color: Styles.AppTheme.textPrimary
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        color: Styles.AppTheme.textSecondary
                                        font.family: Styles.AppTheme.fontFamily
                                        font.pixelSize: Styles.AppTheme.fontSizeBody
                                        text: qsTr("Krok: %1 | Min. notional: %2").arg(modelData.priceStep || "-\u2009?").arg(modelData.minNotional || "—")
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        color: Styles.AppTheme.textSecondary
                                        font.family: Styles.AppTheme.fontFamily
                                        font.pixelSize: Styles.AppTheme.fontSizeBody
                                        text: qsTr("Venue: %1").arg(modelData.config && modelData.config.venueSymbol ? modelData.config.venueSymbol : qsTr("brak"))
                                    }
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        instrumentView.currentIndex = index
                                        selectedInstrumentIndex = index
                                    }
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm
                            Button {
                                text: qsTr("Ustaw jako domyślny")
                                enabled: selectedInstrumentIndex >= 0
                                onClicked: {
                                    var ok = applySelectedInstrument()
                                    if (!ok)
                                        console.warn("Nie udało się zapisać instrumentu startowego")
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                visible: selectedInstrumentIndex >= 0
                                color: Styles.AppTheme.textSecondary
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeBody
                                text: selectedInstrumentIndex >= 0 && instruments[selectedInstrumentIndex] ?
                                      qsTr("Wybrany instrument: %1").arg(instruments[selectedInstrumentIndex].config.symbol) : ""
                            }
                        }
                    }
                }

                // Step 2: Strategies
                Flickable {
                    contentWidth: parent.width
                    contentHeight: strategyColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: strategyColumn
                        x: Styles.AppTheme.spacingLg
                        width: Math.max(0, parent.width - (2 * Styles.AppTheme.spacingLg))
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Krok 3 z 4 – strategie handlowe")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Możesz szybko przejrzeć katalog strategii i dostosować parametry jeszcze przed uruchomieniem bota.")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        Views.StrategyConfigurator {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 420
                            appController: root.appControllerRef
                            strategyController: root.strategyControllerRef
                            workbenchController: root.workbenchControllerRef
                            riskModel: root.riskModelRef
                            licenseController: root.licenseControllerRef
                        }

                        GroupBox {
                            Layout.fillWidth: true
                            title: qsTr("Harmonogramy automatyzacji")
                            background: Rectangle { radius: 10; color: Qt.rgba(1,1,1,0.04) }

                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 8

                                Label {
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: qsTr("Zarządzaj harmonogramami wykonania strategii – możesz je uruchamiać ręcznie oraz usuwać bezpośrednio z kreatora.")
                                }

                                Repeater {
                                    model: schedulerItems || []
                                    delegate: Frame {
                                        Layout.fillWidth: true
                                        background: Rectangle { radius: 8; color: Qt.rgba(1,1,1,0.06) }
                                        ColumnLayout {
                                            anchors.fill: parent
                                            anchors.margins: 8
                                            spacing: 4
                                            Label {
                                                text: (modelData.name || qsTr("harmonogram")) + (modelData.enabled === false ? qsTr(" (wyłączony)") : "")
                                                font.bold: true
                                            }
                                            Label {
                                                Layout.fillWidth: true
                                                wrapMode: Text.WordWrap
                                                color: palette.mid
                                                text: modelData.cron || modelData.expression || qsTr("Brak zdefiniowanej reguły.")
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                Label {
                                                    text: modelData.next_run || modelData.nextRun || ""
                                                    color: palette.mid
                                                }
                                                Item { Layout.fillWidth: true }
                                                Button {
                                                    text: qsTr("Uruchom teraz")
                                                    enabled: strategyControllerRef && !strategyControllerRef.busy
                                                    onClicked: strategyControllerRef.runSchedulerNow(modelData.name || "")
                                                }
                                                Button {
                                                    text: qsTr("Usuń")
                                                    enabled: strategyControllerRef && !strategyControllerRef.busy
                                                    onClicked: strategyControllerRef.removeSchedulerConfig(modelData.name || "")
                                                }
                                            }
                                        }
                                    }
                                }

                                Button {
                                    text: qsTr("Odśwież harmonogramy")
                                    enabled: strategyControllerRef && !strategyControllerRef.busy
                                    onClicked: strategyControllerRef.refresh()
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: Styles.AppTheme.textSecondary
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            text: qsTr("Zmiany zapisują się automatycznie po kliknięciu przycisków w konfiguratorze. Możesz wrócić do tego kroku w każdej chwili.")
                        }
                    }
                }

                // Step 3: Personalization
                Flickable {
                    contentWidth: parent.width
                    contentHeight: personalizationColumn.implicitHeight
                    clip: true
                    ScrollBar.vertical: ScrollBar {}

                    ColumnLayout {
                        id: personalizationColumn
                        x: Styles.AppTheme.spacingLg
                        width: Math.max(0, parent.width - (2 * Styles.AppTheme.spacingLg))
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Krok 4 z 4 – personalizacja interfejsu")
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Dostosuj wygląd aplikacji do swoich preferencji. Ustawienia zostaną zapisane lokalnie w ~/.dudzian/ui_settings.json (lub w ścieżce wskazanej przez zmienną BOT_CORE_UI_SETTINGS_PATH).")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm
                            Label {
                                text: qsTr("Motyw kolorystyczny")
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeBody
                                color: Styles.AppTheme.textPrimary
                            }
                            ComboBox {
                                id: themeCombo
                                objectName: "setupWizardThemeCombo"
                                Layout.preferredWidth: 220
                                model: [qsTr("Ciemny"), qsTr("Jasny"), qsTr("Midnight")]
                                property var themeValues: ["dark", "light", "midnight"]
                                currentIndex: Math.max(0, themeValues.indexOf(personalization.theme || (appControllerRef ? appControllerRef.uiTheme : "dark")))
                                onCurrentIndexChanged: {
                                    if (!appControllerRef)
                                        return
                                    var value = themeValues[currentIndex] || "dark"
                                    if (appControllerRef.setUiTheme)
                                        appControllerRef.setUiTheme(value)
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm
                            Label {
                                text: qsTr("Układ paneli")
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeBody
                                color: Styles.AppTheme.textPrimary
                            }
                            ComboBox {
                                id: layoutCombo
                                objectName: "setupWizardLayoutCombo"
                                Layout.preferredWidth: 220
                                model: [qsTr("Klasyczny"), qsTr("Kompaktowy"), qsTr("Zaawansowany")]
                                property var layoutValues: ["classic", "compact", "advanced"]
                                currentIndex: Math.max(0, layoutValues.indexOf(personalization.layout || (appControllerRef ? appControllerRef.uiLayoutMode : "classic")))
                                onCurrentIndexChanged: {
                                    if (!appControllerRef)
                                        return
                                    var value = layoutValues[currentIndex] || "classic"
                                    if (appControllerRef.setUiLayoutMode)
                                        appControllerRef.setUiLayoutMode(value)
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm
                            Label {
                                text: qsTr("Powiadomienia w postaci toasts")
                                Layout.fillWidth: true
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeBody
                                color: Styles.AppTheme.textPrimary
                            }
                            Switch {
                                id: toastSwitch
                                objectName: "setupWizardToastSwitch"
                                checked: personalization.alert_toasts !== undefined ? Boolean(personalization.alert_toasts) : (appControllerRef ? Boolean(appControllerRef.alertToastsEnabled) : true)
                                onCheckedChanged: {
                                    if (appControllerRef && appControllerRef.setAlertToastsEnabled)
                                        appControllerRef.setAlertToastsEnabled(checked)
                                }
                            }
                        }

                        GroupBox {
                            Layout.fillWidth: true
                            title: qsTr("Alerty bezpieczeństwa")
                            background: Rectangle { radius: 10; color: Qt.rgba(1,1,1,0.04) }

                            ListView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 160
                                clip: true
                                model: securityAlertsModel
                                delegate: Frame {
                                    width: ListView.view.width
                                    background: Rectangle { radius: 6; color: Qt.rgba(1,1,1,0.05) }
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 8
                                        spacing: 4
                                        Label {
                                            text: model.title || qsTr("Alert")
                                            font.bold: true
                                            color: model.severity === 2 ? "#c0392b" : (model.severity === 1 ? "#d68910" : palette.text)
                                        }
                                        Label {
                                            Layout.fillWidth: true
                                            wrapMode: Text.WordWrap
                                            text: model.description || ""
                                        }
                                        Label {
                                            text: Qt.formatDateTime(model.timestamp, Qt.ISODate)
                                            color: palette.mid
                                        }
                                    }
                                }
                                ScrollBar.vertical: ScrollBar {}
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: Styles.AppTheme.textSecondary
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            text: qsTr("Zmiany są zapisywane automatycznie i obowiązują dla całego interfejsu. Możesz je później zmienić w ustawieniach." )
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            Button {
                text: qsTr("Wstecz")
                enabled: currentStep > 0
                onClicked: currentStep = Math.max(0, currentStep - 1)
            }

            Item { Layout.fillWidth: true }

            Button {
                text: currentStep >= totalSteps - 1 ? qsTr("Zakończ") : qsTr("Dalej")
                enabled: stepCanAdvance()
                onClicked: {
                    if (currentStep >= totalSteps - 1) {
                        wizardCompleted()
                    } else {
                        currentStep += 1
                    }
                }
            }
        }
    }

    function stepCanAdvance() {
        if (currentStep === 0)
            return licenseControllerRef ? Boolean(licenseControllerRef.licenseActive) : false
        if (currentStep === 1)
            return selectedInstrumentIndex >= 0 && instruments.length > 0
        return true
    }
}
