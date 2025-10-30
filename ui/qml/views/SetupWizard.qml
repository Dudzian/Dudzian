import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../components" as Components
import "../views" as Views

Item {
    id: root
    width: parent ? parent.width : 960
    height: parent ? parent.height : 600

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var strategyController: (typeof strategyController !== "undefined" ? strategyController : null)
    property var workbenchController: (typeof workbenchController !== "undefined" ? workbenchController : null)
    property var licenseController: (typeof licenseController !== "undefined" ? licenseController : null)
    property var riskModel: (typeof riskModel !== "undefined" ? riskModel : null)
    property int currentStep: 0
    property var exchangeOptions: []
    property string selectedExchange: ""
    property var instruments: []
    property int selectedInstrumentIndex: -1
    property var personalization: ({})

    signal wizardCompleted()

    readonly property int totalSteps: 4

    function refreshExchangeOptions() {
        if (!appController)
            return
        exchangeOptions = appController.supportedExchanges ? appController.supportedExchanges() : []
        if (exchangeOptions.length > 0 && (!selectedExchange || exchangeOptions.indexOf(selectedExchange) === -1)) {
            selectedExchange = exchangeOptions[0]
            loadInstruments()
        }
    }

    function loadInstruments() {
        if (!appController || !selectedExchange)
            return
        var result = appController.listTradableInstruments(selectedExchange) || []
        instruments = result
        selectedInstrumentIndex = instruments.length > 0 ? 0 : -1
    }

    function applySelectedInstrument() {
        if (!appController || selectedInstrumentIndex < 0 || selectedInstrumentIndex >= instruments.length)
            return false
        var item = instruments[selectedInstrumentIndex] || {}
        if (!item.config)
            return false
        var cfg = item.config
        return appController.updateInstrument(cfg.exchange || selectedExchange,
                                              cfg.symbol || "",
                                              cfg.venueSymbol || "",
                                              cfg.quoteCurrency || "",
                                              cfg.baseCurrency || "",
                                              cfg.granularityIso8601 || "PT1M")
    }

    function refreshPersonalization() {
        if (!appController)
            return
        personalization = appController.personalizationSnapshot ? appController.personalizationSnapshot() : {}
    }

    Component.onCompleted: {
        refreshExchangeOptions()
        refreshPersonalization()
    }

    Connections {
        target: appController
        ignoreUnknownSignals: true
        function onUiThemeChanged() { refreshPersonalization() }
        function onUiLayoutModeChanged() { refreshPersonalization() }
        function onAlertToastsEnabledChanged() { refreshPersonalization() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 18
        padding: 18

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Repeater {
                model: totalSteps
                delegate: Rectangle {
                    readonly property bool visited: index <= currentStep
                    Layout.fillWidth: true
                    Layout.preferredHeight: 8
                    radius: 4
                    color: visited ? palette.highlight : Qt.rgba(1,1,1,0.18)
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 10 }

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
                        width: parent.width
                        spacing: 12
                        padding: 16

                        Label {
                            text: qsTr("Krok 1 z 4 – aktywacja licencji")
                            font.pixelSize: 22
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Zweryfikuj licencję OEM i fingerprint urządzenia. Bez aktywnej licencji interfejs będzie działać w trybie ograniczonym.")
                        }

                        Components.FirstRunWizard {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: parent.width - 32
                            visible: true
                            enabled: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: palette.mid
                            text: licenseController && licenseController.licenseActive ?
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
                        width: parent.width
                        spacing: 12
                        padding: 16

                        Label {
                            text: qsTr("Krok 2 z 4 – konfiguracja giełdy")
                            font.pixelSize: 22
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Wybierz domyślną giełdę i instrument, z którym aplikacja wystartuje po uruchomieniu. Pełne klucze API możesz uzupełnić w panelu bezpieczeństwa.")
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

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
                                padding: 10
                                background: Rectangle {
                                    radius: 8
                                    color: ListView.isCurrentItem ? palette.highlight : Qt.darker(palette.base, 1.02)
                                }
                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: 4
                                    Label {
                                        text: (modelData.config && modelData.config.symbol) ? modelData.config.symbol : qsTr("Instrument")
                                        font.pixelSize: 16
                                        font.bold: true
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        color: palette.mid
                                        text: qsTr("Krok: %1 | Min. notional: %2").arg(modelData.priceStep || "-\u2009?").arg(modelData.minNotional || "—")
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        color: palette.mid
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
                            spacing: 12
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
                                color: palette.mid
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
                        width: parent.width
                        spacing: 12
                        padding: 16

                        Label {
                            text: qsTr("Krok 3 z 4 – strategie handlowe")
                            font.pixelSize: 22
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Możesz szybko przejrzeć katalog strategii i dostosować parametry jeszcze przed uruchomieniem bota.")
                        }

                        Views.StrategyConfigurator {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 420
                            appController: root.appController
                            strategyController: root.strategyController
                            workbenchController: root.workbenchController
                            riskModel: root.riskModel
                            licenseController: root.licenseController
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: palette.mid
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
                        width: parent.width
                        spacing: 12
                        padding: 16

                        Label {
                            text: qsTr("Krok 4 z 4 – personalizacja interfejsu")
                            font.pixelSize: 22
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: qsTr("Dostosuj wygląd aplikacji do swoich preferencji. Ustawienia zostaną zapisane lokalnie w pliku config/ui_prefs.json.")
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Label { text: qsTr("Motyw kolorystyczny") }
                            ComboBox {
                                id: themeCombo
                                objectName: "setupWizardThemeCombo"
                                Layout.preferredWidth: 220
                                model: [qsTr("Ciemny"), qsTr("Jasny"), qsTr("Midnight")]
                                property var themeValues: ["dark", "light", "midnight"]
                                currentIndex: Math.max(0, themeValues.indexOf(personalization.theme || (appController ? appController.uiTheme : "dark")))
                                onCurrentIndexChanged: {
                                    if (!appController)
                                        return
                                    var value = themeValues[currentIndex] || "dark"
                                    if (appController.setUiTheme)
                                        appController.setUiTheme(value)
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Label { text: qsTr("Układ paneli") }
                            ComboBox {
                                id: layoutCombo
                                objectName: "setupWizardLayoutCombo"
                                Layout.preferredWidth: 220
                                model: [qsTr("Klasyczny"), qsTr("Kompaktowy"), qsTr("Zaawansowany")]
                                property var layoutValues: ["classic", "compact", "advanced"]
                                currentIndex: Math.max(0, layoutValues.indexOf(personalization.layout || (appController ? appController.uiLayoutMode : "classic")))
                                onCurrentIndexChanged: {
                                    if (!appController)
                                        return
                                    var value = layoutValues[currentIndex] || "classic"
                                    if (appController.setUiLayoutMode)
                                        appController.setUiLayoutMode(value)
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Label {
                                text: qsTr("Powiadomienia w postaci toasts")
                                Layout.fillWidth: true
                            }
                            Switch {
                                id: toastSwitch
                                objectName: "setupWizardToastSwitch"
                                checked: personalization.alert_toasts !== undefined ? personalization.alert_toasts : (appController ? appController.alertToastsEnabled : true)
                                onCheckedChanged: {
                                    if (appController && appController.setAlertToastsEnabled)
                                        appController.setAlertToastsEnabled(checked)
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: palette.mid
                            text: qsTr("Zmiany są zapisywane automatycznie i obowiązują dla całego interfejsu. Możesz je później zmienić w ustawieniach." )
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

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
            return licenseController ? licenseController.licenseActive : false
        if (currentStep === 1)
            return selectedInstrumentIndex >= 0 && instruments.length > 0
        return true
    }
}
