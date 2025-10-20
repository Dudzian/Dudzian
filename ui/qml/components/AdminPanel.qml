import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import QtCore

Drawer {
    id: adminPanel
    width: Math.min(parent ? parent.width * 0.42 : 480, 560)
    implicitHeight: parent ? parent.height : 720
    edge: Qt.RightEdge
    modal: false
    interactive: true
    closePolicy: Popup.CloseOnEscape

    property var instrumentForm: ({})
    property var guardForm: ({})
    property var riskRefreshForm: ({})
    property var riskSchedule: ({})
    property string statusMessage: ""
    property color statusColor: palette.highlight
    property string riskStatusMessage: ""
    property color riskStatusColor: palette.highlight
    property string riskHistoryStatusMessage: ""
    property color riskHistoryStatusColor: palette.highlight
    property bool riskHistoryExportLimitEnabled: false
    property int riskHistoryExportLimitValue: 50
    property url riskHistoryExportLastDirectory: ""
    property bool riskHistoryAutoExportEnabled: false
    property int riskHistoryAutoExportIntervalMinutes: 15
    property string riskHistoryAutoExportBasename: "risk-history"
    property bool riskHistoryAutoExportUseLocalTime: false
    property var riskHistoryLastAutoExportAt: null
    property string riskHistoryLastAutoExportPath: ""

    function updateRiskSchedule() {
        if (typeof appController === "undefined")
            return
        riskSchedule = appController.riskRefreshSchedule
    }

    function syncForms() {
        if (typeof appController === "undefined")
            return
        instrumentForm = appController.instrumentConfigSnapshot()
        guardForm = appController.performanceGuardSnapshot()
        riskRefreshForm = appController.riskRefreshSnapshot()
        updateRiskSchedule()
        riskStatusMessage = ""
        riskHistoryStatusMessage = ""
        if (typeof appController !== "undefined") {
            riskHistoryExportLimitEnabled = appController.riskHistoryExportLimitEnabled
            riskHistoryExportLimitValue = appController.riskHistoryExportLimitValue
            riskHistoryExportLastDirectory = appController.riskHistoryExportLastDirectory
            riskHistoryAutoExportEnabled = appController.riskHistoryAutoExportEnabled
            riskHistoryAutoExportIntervalMinutes = appController.riskHistoryAutoExportIntervalMinutes
            riskHistoryAutoExportBasename = appController.riskHistoryAutoExportBasename
            riskHistoryAutoExportUseLocalTime = appController.riskHistoryAutoExportUseLocalTime
            var autoExportAt = appController.riskHistoryLastAutoExportAt
            riskHistoryLastAutoExportAt = autoExportAt && autoExportAt.isValid && autoExportAt.isValid() ? autoExportAt : null
            var lastPathUrl = appController.riskHistoryLastAutoExportPath
            riskHistoryLastAutoExportPath = lastPathUrl && lastPathUrl.toLocalFile ? lastPathUrl.toLocalFile() : ""
        }
    }

    function refreshData() {
        syncForms()
        if (typeof securityController !== "undefined")
            securityController.refresh()
        if (typeof reportController !== "undefined")
            reportController.refresh()
    }

    function defaultExportFolder() {
        var path = StandardPaths.writableLocation(StandardPaths.DocumentsLocation)
        if (!path || path.length === 0)
            path = StandardPaths.writableLocation(StandardPaths.HomeLocation)
        if (!path || path.length === 0)
            return ""
        path = path.replace(/\\/g, "/")
        if (path.startsWith("file:"))
            return path
        if (path.startsWith("/"))
            return "file://" + path
        return "file:///" + path
    }

    function formatTimestamp(value) {
        if (!value || value.length === 0)
            return qsTr("—")
        var date = new Date(value)
        if (isNaN(date.getTime()))
            return value
        return Qt.formatDateTime(date, Qt.DefaultLocaleShortDate)
    }

    function formatCountdown(value) {
        if (value === undefined || value === null || value < 0)
            return qsTr("wstrzymane")
        var remaining = Math.max(0, Math.floor(value))
        var minutes = Math.floor(remaining / 60)
        var seconds = remaining % 60
        if (minutes > 0) {
            var secondsText = seconds < 10 ? "0" + seconds : seconds
            return qsTr("%1 min %2 s").arg(minutes).arg(secondsText)
        }
        return qsTr("%1 s").arg(seconds)
    }

    onOpened: refreshData()

    Timer {
        interval: 1000
        running: adminPanel.visible
        repeat: true
        onTriggered: adminPanel.updateRiskSchedule()
    }

    background: Rectangle {
        color: Qt.darker(adminPanel.palette.window, 1.2)
    }

    contentItem: TabView {
        id: tabs
        anchors.fill: parent
        currentIndex: 0

        Tab {
            title: qsTr("Strategia")

            Flickable {
                anchors.fill: parent
                contentWidth: width
                contentHeight: strategyLayout.implicitHeight
                clip: true

                ColumnLayout {
                    id: strategyLayout
                    width: parent.width
                    spacing: 16
                    padding: 16

                    GroupBox {
                        title: qsTr("Instrument i rynek")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 10
                            Layout.fillWidth: true

                            Label { text: qsTr("Giełda") }
                            TextField {
                                text: instrumentForm.exchange || ""
                                onEditingFinished: instrumentForm.exchange = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Symbol logiczny") }
                            TextField {
                                text: instrumentForm.symbol || ""
                                onEditingFinished: instrumentForm.symbol = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Symbol na giełdzie") }
                            TextField {
                                text: instrumentForm.venueSymbol || ""
                                onEditingFinished: instrumentForm.venueSymbol = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Waluta kwotowana") }
                            TextField {
                                text: instrumentForm.quoteCurrency || ""
                                onEditingFinished: instrumentForm.quoteCurrency = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Waluta bazowa") }
                            TextField {
                                text: instrumentForm.baseCurrency || ""
                                onEditingFinished: instrumentForm.baseCurrency = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Interwał (ISO8601)") }
                            TextField {
                                text: instrumentForm.granularity || ""
                                onEditingFinished: instrumentForm.granularity = text
                                Layout.fillWidth: true
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Performance guard")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 10
                            Layout.fillWidth: true

                            Label { text: qsTr("Docelowy FPS") }
                            SpinBox {
                                value: guardForm.fpsTarget || 60
                                from: 15
                                to: 240
                                stepSize: 1
                                onValueModified: guardForm.fpsTarget = value
                            }

                            Label { text: qsTr("Reduce motion po (s)") }
                            SpinBox {
                                id: reduceMotionSpin
                                from: 0
                                to: 1000
                                stepSize: 5
                                editable: true
                                onValueModified: guardForm.reduceMotionAfter = value / 100
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value / 100, 'f', 2, locale)
                                }
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    var scaled = Math.round(number * 100)
                                    return Math.max(from, Math.min(to, scaled))
                                }
                                Binding {
                                    target: reduceMotionSpin
                                    property: "value"
                                    value: Math.round((guardForm.reduceMotionAfter !== undefined
                                                       ? guardForm.reduceMotionAfter
                                                       : 1) * 100)
                                    when: !reduceMotionSpin.activeFocus
                                }
                            }

                            Label { text: qsTr("Budżet janku (ms)") }
                            SpinBox {
                                id: jankBudgetSpin
                                from: 100
                                to: 10000
                                stepSize: 5
                                editable: true
                                onValueModified: guardForm.jankThresholdMs = value / 100
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value / 100, 'f', 2, locale)
                                }
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    var scaled = Math.round(number * 100)
                                    return Math.max(from, Math.min(to, scaled))
                                }
                                Binding {
                                    target: jankBudgetSpin
                                    property: "value"
                                    value: Math.round((guardForm.jankThresholdMs !== undefined
                                                       ? guardForm.jankThresholdMs
                                                       : 18) * 100)
                                    when: !jankBudgetSpin.activeFocus
                                }
                            }

                            Label { text: qsTr("Limit nakładek") }
                            SpinBox {
                                value: guardForm.maxOverlayCount || 3
                                from: 0
                                to: 12
                                onValueModified: guardForm.maxOverlayCount = value
                            }

                            Label { text: qsTr("Wyłącz nakładki <FPS") }
                            SpinBox {
                                value: guardForm.disableSecondaryWhenBelow || 0
                                from: 0
                                to: 120
                                onValueModified: guardForm.disableSecondaryWhenBelow = value
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Button {
                            text: qsTr("Przywróć aktualne")
                            onClicked: syncForms()
                        }

                        Item { Layout.fillWidth: true }

                        Button {
                            text: qsTr("Zapisz zmiany")
                            highlighted: true
                            onClicked: {
                                const okInstrument = appController.updateInstrument(
                                            instrumentForm.exchange || "",
                                            instrumentForm.symbol || "",
                                            instrumentForm.venueSymbol || "",
                                            instrumentForm.quoteCurrency || "",
                                            instrumentForm.baseCurrency || "",
                                            instrumentForm.granularity || "")
                                const okGuard = appController.updatePerformanceGuard(
                                            guardForm.fpsTarget || 60,
                                            guardForm.reduceMotionAfter || 1,
                                            guardForm.jankThresholdMs || 18,
                                            guardForm.maxOverlayCount || 3,
                                            guardForm.disableSecondaryWhenBelow || 0)
                                if (okInstrument && okGuard) {
                                    statusMessage = qsTr("Zapisano konfigurację strategii")
                                    statusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                } else {
                                    statusMessage = qsTr("Nie udało się zapisać konfiguracji")
                                    statusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                }
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: statusMessage.length > 0
                        text: statusMessage
                        color: statusColor
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }

        Tab {
            title: qsTr("Alerty")

            AlertCenterPanel {
                anchors.fill: parent
                summaryModel: alertsModel
                listModel: alertsFilterModel
            }
        }

        Tab {
            title: qsTr("Ryzyko")

            ColumnLayout {
                anchors.fill: parent
                spacing: 12
                padding: 12

                GroupBox {
                    title: qsTr("Odświeżanie stanu ryzyka")
                    Layout.fillWidth: true

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 10

                        Switch {
                            id: riskRefreshSwitch
                            text: qsTr("Aktywne odpytywanie serwisu ryzyka")
                            checked: riskRefreshForm && riskRefreshForm.enabled !== false
                            onToggled: {
                                riskRefreshForm.enabled = checked
                            }
                            Binding {
                                target: riskRefreshSwitch
                                property: "checked"
                                value: riskRefreshForm && riskRefreshForm.enabled !== false
                                when: !riskRefreshSwitch.down && !riskRefreshSwitch.activeFocus
                            }
                        }

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 8
                            Layout.fillWidth: true

                            Label {
                                text: qsTr("Interwał odświeżania (s)")
                            }

                            SpinBox {
                                id: riskIntervalSpin
                                from: 1000
                                to: 300000
                                stepSize: 500
                                editable: true
                                Layout.fillWidth: true
                                enabled: riskRefreshSwitch.checked
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    return Math.max(from, Math.min(to, Math.round(number * 1000)))
                                }
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value / 1000, 'f', 2, locale)
                                }
                                onValueModified: {
                                    riskRefreshForm.intervalSeconds = value / 1000
                                }
                                Binding {
                                    target: riskIntervalSpin
                                    property: "value"
                                    value: Math.round((riskRefreshForm && riskRefreshForm.intervalSeconds
                                                       ? riskRefreshForm.intervalSeconds
                                                       : 5) * 1000)
                                    when: !riskIntervalSpin.activeFocus
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4

                            Label {
                                text: qsTr("Status harmonogramu: %1")
                                      .arg(riskSchedule && riskSchedule.active
                                               ? qsTr("aktywny")
                                               : qsTr("wstrzymany"))
                                color: riskSchedule && riskSchedule.active
                                       ? Qt.rgba(0.35, 0.75, 0.45, 1)
                                       : Qt.rgba(0.95, 0.68, 0.26, 1)
                            }

                            Label {
                                text: qsTr("Ostatnie zapytanie: %1")
                                      .arg(formatTimestamp(riskSchedule.lastRequestAt))
                                color: palette.midlight
                            }

                            Label {
                                text: qsTr("Ostatnia aktualizacja: %1")
                                      .arg(formatTimestamp(riskSchedule.lastUpdateAt))
                                color: palette.midlight
                            }

                            Label {
                                text: qsTr("Kolejne odświeżenie za: %1")
                                      .arg(formatCountdown(riskSchedule.nextRefreshInSeconds))
                                color: palette.highlight
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            Button {
                                text: qsTr("Przywróć aktualne")
                                onClicked: {
                                    riskRefreshForm = appController.riskRefreshSnapshot()
                                    riskStatusMessage = ""
                                    adminPanel.updateRiskSchedule()
                                }
                            }

                            Button {
                                text: qsTr("Odśwież teraz")
                                onClicked: {
                                    const okManual = appController.triggerRiskRefreshNow()
                                    if (okManual) {
                                        riskStatusMessage = qsTr("Zainicjowano ręczne odświeżenie ryzyka")
                                        riskStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                        adminPanel.updateRiskSchedule()
                                    } else {
                                        riskStatusMessage = qsTr("Nie udało się zainicjować odświeżenia ryzyka")
                                        riskStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                    }
                                }
                            }

                            Item { Layout.fillWidth: true }

                            Button {
                                text: qsTr("Zapisz odświeżanie")
                                highlighted: true
                                onClicked: {
                                    const ok = appController.updateRiskRefresh(
                                                riskRefreshForm && riskRefreshForm.enabled !== false,
                                                riskRefreshForm && riskRefreshForm.intervalSeconds
                                                    ? riskRefreshForm.intervalSeconds
                                                    : 0)
                                    if (ok) {
                                        riskStatusMessage = qsTr("Zapisano konfigurację odświeżania ryzyka")
                                        riskStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                        riskRefreshForm = appController.riskRefreshSnapshot()
                                        adminPanel.updateRiskSchedule()
                                    } else {
                                        riskStatusMessage = qsTr("Nie udało się zaktualizować harmonogramu ryzyka")
                                        riskStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                    }
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: riskStatusMessage.length > 0
                            text: riskStatusMessage
                            color: riskStatusColor
                        }
                    }
                }

                GroupBox {
                    title: qsTr("Historia ryzyka")
                    Layout.fillWidth: true

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 10

                        Label {
                            text: qsTr("Limit przechowywanych próbek")
                        }

                        SpinBox {
                            id: historyLimitSpin
                            from: 1
                            to: 500
                            stepSize: 1
                            editable: true
                            Layout.fillWidth: true
                            valueFromText: function(text, locale) {
                                var number = Number.fromLocaleString(locale, text)
                                if (isNaN(number))
                                    number = parseFloat(text)
                                if (isNaN(number))
                                    return value
                                return Math.max(from, Math.min(to, Math.round(number)))
                            }
                            textFromValue: function(value, locale) {
                                return Qt.formatLocaleNumber(value, 'f', 0, locale)
                            }
                            onValueModified: {
                                if (typeof appController === "undefined")
                                    return
                                const ok = appController.updateRiskHistoryLimit(value)
                                if (ok) {
                                    riskHistoryStatusMessage = qsTr("Ustawiono limit historii na %1 próbek").arg(value)
                                    riskHistoryStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                } else {
                                    riskHistoryStatusMessage = qsTr("Nie udało się ustawić limitu historii ryzyka")
                                    riskHistoryStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                    if (riskHistoryModel)
                                        historyLimitSpin.value = riskHistoryModel.maximumEntries
                                }
                            }
                            Binding {
                                target: historyLimitSpin
                                property: "value"
                                value: riskHistoryModel ? riskHistoryModel.maximumEntries : 50
                                when: !historyLimitSpin.activeFocus
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            Button {
                                text: qsTr("Wyczyść historię")
                                onClicked: {
                                    if (typeof appController === "undefined")
                                        return
                                    appController.clearRiskHistory()
                                    riskHistoryStatusMessage = qsTr("Wyczyszczono zapisane próbki ryzyka")
                                    riskHistoryStatusColor = Qt.rgba(0.9, 0.55, 0.25, 1)
                                }
                            }

                            Button {
                                text: qsTr("Eksportuj do CSV…")
                                enabled: riskHistoryModel && riskHistoryModel.hasSamples
                                onClicked: historyExportDialog.open()
                            }

                            Item { Layout.fillWidth: true }

                            Label {
                                text: riskHistoryModel && riskHistoryModel.hasSamples
                                      ? qsTr("Zapisanych próbek: %1").arg(riskHistoryModel.entryCount)
                                      : qsTr("Brak zapisanych próbek")
                                color: palette.midlight
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            CheckBox {
                                id: exportLimitCheckbox
                                text: qsTr("Eksportuj tylko ostatnie")
                                checked: riskHistoryExportLimitEnabled
                                enabled: riskHistoryModel && riskHistoryModel.hasSamples
                                onToggled: {
                                    riskHistoryExportLimitEnabled = checked
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryExportLimitEnabled(checked)
                                }
                            }

                            SpinBox {
                                id: exportLimitSpin
                                from: 1
                                to: riskHistoryModel && riskHistoryModel.hasSamples
                                        ? Math.max(riskHistoryModel.entryCount, 1)
                                        : Math.max(riskHistoryExportLimitValue, 1)
                                stepSize: 1
                                enabled: exportLimitCheckbox.checked && riskHistoryModel && riskHistoryModel.hasSamples
                                Layout.preferredWidth: 120
                                onValueModified: {
                                    var normalized = Math.max(1, Math.round(value))
                                    if (value !== normalized)
                                        value = normalized
                                    riskHistoryExportLimitValue = normalized
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryExportLimitValue(normalized)
                                }
                                Component.onCompleted: value = Math.max(1, riskHistoryExportLimitValue)
                            }

                            Label {
                                text: qsTr("próbek")
                                visible: exportLimitCheckbox.checked
                            }

                            Item { Layout.fillWidth: true }

                            data: [
                                Binding {
                                    target: exportLimitSpin
                                    property: "value"
                                    value: Math.max(1, Math.min(riskHistoryExportLimitValue, exportLimitSpin.to))
                                    when: !exportLimitSpin.activeFocus
                                }
                            ]
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            CheckBox {
                                id: autoExportCheckbox
                                text: qsTr("Automatyczny eksport")
                                checked: riskHistoryAutoExportEnabled
                                enabled: typeof appController !== "undefined"
                                onToggled: {
                                    riskHistoryAutoExportEnabled = checked
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryAutoExportEnabled(checked)
                                    riskHistoryStatusMessage = checked
                                            ? qsTr("Automatyczny eksport historii ryzyka włączony")
                                            : qsTr("Automatyczny eksport historii ryzyka wyłączony")
                                    riskHistoryStatusColor = checked
                                            ? Qt.rgba(0.3, 0.7, 0.4, 1)
                                            : Qt.rgba(0.9, 0.55, 0.25, 1)
                                }
                            }

                            Label {
                                text: qsTr("co")
                                visible: autoExportCheckbox.checked
                            }

                            SpinBox {
                                id: autoExportIntervalSpin
                                from: 1
                                to: 1440
                                stepSize: 1
                                enabled: autoExportCheckbox.checked
                                Layout.preferredWidth: 120
                                value: Math.max(1, riskHistoryAutoExportIntervalMinutes)
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    return Math.max(from, Math.min(to, Math.round(number)))
                                }
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value, 'f', 0, locale)
                                }
                                onValueModified: {
                                    var normalized = Math.max(1, Math.round(value))
                                    if (normalized !== value)
                                        value = normalized
                                    riskHistoryAutoExportIntervalMinutes = normalized
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryAutoExportIntervalMinutes(normalized)
                                }
                                Binding {
                                    target: autoExportIntervalSpin
                                    property: "value"
                                    value: Math.max(1, Math.min(riskHistoryAutoExportIntervalMinutes, autoExportIntervalSpin.to))
                                    when: !autoExportIntervalSpin.activeFocus
                                }
                            }

                            Label {
                                text: qsTr("min")
                                visible: autoExportCheckbox.checked
                            }

                            Item { Layout.fillWidth: true }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            visible: autoExportCheckbox.checked

                            Label {
                                text: qsTr("Prefiks pliku")
                                Layout.preferredWidth: 140
                            }

                            TextField {
                                id: autoExportBasenameField
                                Layout.fillWidth: true
                                text: riskHistoryAutoExportBasename
                                placeholderText: qsTr("np. risk-history")
                                inputMethodHints: Qt.ImhPreferLowercase | Qt.ImhNoPredictiveText
                                enabled: autoExportCheckbox.checked
                                onEditingFinished: {
                                    var requested = text
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryAutoExportBasename(requested)
                                    riskHistoryAutoExportBasename = appController
                                            ? appController.riskHistoryAutoExportBasename
                                            : riskHistoryAutoExportBasename
                                    if (text !== riskHistoryAutoExportBasename)
                                        text = riskHistoryAutoExportBasename
                                }
                                Binding {
                                    target: autoExportBasenameField
                                    property: "text"
                                    value: riskHistoryAutoExportBasename
                                    when: !autoExportBasenameField.activeFocus
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            visible: autoExportCheckbox.checked

                            CheckBox {
                                id: autoExportLocalTimeCheckbox
                                text: qsTr("Użyj czasu lokalnego w nazwach plików")
                                checked: riskHistoryAutoExportUseLocalTime
                                enabled: typeof appController !== "undefined"
                                onToggled: {
                                    riskHistoryAutoExportUseLocalTime = checked
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryAutoExportUseLocalTime(checked)
                                    riskHistoryStatusMessage = checked
                                            ? qsTr("Autoeksport będzie używał czasu lokalnego i znacznika strefy w nazwach plików")
                                            : qsTr("Autoeksport będzie używał znaczników czasu UTC w nazwach plików")
                                    riskHistoryStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                }
                            }

                            Item { Layout.fillWidth: true }
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: riskHistoryLastAutoExportAt && riskHistoryLastAutoExportAt.isValid && riskHistoryLastAutoExportAt.isValid()
                            text: {
                                if (!riskHistoryLastAutoExportAt || !riskHistoryLastAutoExportAt.isValid || !riskHistoryLastAutoExportAt.isValid())
                                    return ""
                                var timestampText = formatTimestamp(riskHistoryLastAutoExportAt)
                                var pathText = riskHistoryLastAutoExportPath ? riskHistoryLastAutoExportPath : ""
                                if (pathText.length > 0)
                                    return qsTr("Ostatni auto-eksport: %1\nPlik: %2").arg(timestampText).arg(pathText)
                                return qsTr("Ostatni auto-eksport: %1").arg(timestampText)
                            }
                            color: palette.midlight
                            wrapMode: Text.WordWrap
                        }

                        FileDialog {
                            id: historyExportDialog
                            title: qsTr("Zapisz historię ryzyka jako CSV")
                            fileMode: FileDialog.SaveFile
                            defaultSuffix: "csv"
                            folder: riskHistoryExportLastDirectory.length > 0
                                    ? riskHistoryExportLastDirectory
                                    : defaultExportFolder()
                            nameFilters: [qsTr("Pliki CSV (*.csv)"), qsTr("Wszystkie pliki (*)")]
                            onAccepted: {
                                if (typeof appController === "undefined")
                                    return
                                const requestedLimit = exportLimitCheckbox.checked
                                        ? Math.max(1, Math.round(exportLimitSpin.value))
                                        : -1
                                const ok = appController.exportRiskHistoryToCsv(selectedFile, requestedLimit)
                                if (ok) {
                                    var folderUrl = historyExportDialog.currentFolder || historyExportDialog.folder
                                    if (folderUrl && folderUrl.length > 0) {
                                        riskHistoryExportLastDirectory = folderUrl
                                        appController.setRiskHistoryExportLastDirectory(folderUrl)
                                    }
                                    if (exportLimitCheckbox.checked && riskHistoryModel) {
                                        const exported = Math.min(requestedLimit, riskHistoryModel.entryCount)
                                        riskHistoryStatusMessage = qsTr("Wyeksportowano %1 najnowszych próbek ryzyka do %2")
                                                                     .arg(exported)
                                                                     .arg(selectedFile.toString())
                                    } else {
                                        riskHistoryStatusMessage = qsTr("Wyeksportowano historię ryzyka do %1")
                                                                       .arg(selectedFile.toString())
                                    }
                                    riskHistoryStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                } else {
                                    riskHistoryStatusMessage = qsTr("Nie udało się wyeksportować historii ryzyka")
                                    riskHistoryStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: riskHistoryStatusMessage.length > 0
                            text: riskHistoryStatusMessage
                            color: riskHistoryStatusColor
                            wrapMode: Text.WordWrap
                        }

                        Connections {
                            target: riskHistoryModel
                            enabled: typeof riskHistoryModel !== "undefined" && riskHistoryModel !== null
                            function onMaximumEntriesChanged() {
                                if (typeof riskHistoryModel === "undefined" || riskHistoryModel === null)
                                    return
                                historyLimitSpin.value = riskHistoryModel.maximumEntries
                            }
                            function onHistoryChanged() {
                                if (typeof riskHistoryModel === "undefined" || riskHistoryModel === null)
                                    return
                                if (!riskHistoryModel.hasSamples)
                                    return
                                if (riskHistoryExportLimitValue > riskHistoryModel.entryCount) {
                                    var newValue = Math.max(1, riskHistoryModel.entryCount)
                                    riskHistoryExportLimitValue = newValue
                                    if (typeof appController !== "undefined")
                                        appController.setRiskHistoryExportLimitValue(newValue)
                                }
                            }
                        }

                        Connections {
                            target: appController
                            enabled: typeof appController !== "undefined"
                            function onRiskHistoryExportLimitEnabledChanged() {
                                riskHistoryExportLimitEnabled = appController.riskHistoryExportLimitEnabled
                            }
                            function onRiskHistoryExportLimitValueChanged() {
                                if (!exportLimitSpin.activeFocus)
                                    riskHistoryExportLimitValue = appController.riskHistoryExportLimitValue
                            }
                            function onRiskHistoryExportLastDirectoryChanged() {
                                riskHistoryExportLastDirectory = appController.riskHistoryExportLastDirectory
                            }
                            function onRiskHistoryAutoExportEnabledChanged() {
                                riskHistoryAutoExportEnabled = appController.riskHistoryAutoExportEnabled
                            }
                            function onRiskHistoryAutoExportIntervalMinutesChanged() {
                                riskHistoryAutoExportIntervalMinutes = appController.riskHistoryAutoExportIntervalMinutes
                            }
                            function onRiskHistoryAutoExportBasenameChanged() {
                                riskHistoryAutoExportBasename = appController.riskHistoryAutoExportBasename
                            }
                            function onRiskHistoryAutoExportUseLocalTimeChanged() {
                                riskHistoryAutoExportUseLocalTime = appController.riskHistoryAutoExportUseLocalTime
                                riskHistoryStatusMessage = riskHistoryAutoExportUseLocalTime
                                        ? qsTr("Autoeksport będzie używał czasu lokalnego i znacznika strefy w nazwach plików")
                                        : qsTr("Autoeksport będzie używał znaczników czasu UTC w nazwach plików")
                                riskHistoryStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                            }
                            function onRiskHistoryLastAutoExportAtChanged() {
                                var updated = appController.riskHistoryLastAutoExportAt
                                riskHistoryLastAutoExportAt = updated && updated.isValid && updated.isValid() ? updated : null
                                if (riskHistoryLastAutoExportAt && riskHistoryLastAutoExportAt.isValid && riskHistoryLastAutoExportAt.isValid()) {
                                    riskHistoryStatusMessage = qsTr("Automatycznie wyeksportowano historię ryzyka o %1")
                                                               .arg(formatTimestamp(riskHistoryLastAutoExportAt))
                                    riskHistoryStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                }
                            }
                            function onRiskHistoryLastAutoExportPathChanged() {
                                var updatedPath = appController.riskHistoryLastAutoExportPath
                                riskHistoryLastAutoExportPath = updatedPath && updatedPath.toLocalFile ? updatedPath.toLocalFile() : ""
                            }
                        }
                    }
                }

                RiskMonitorPanel {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: riskModel
                    historyModel: riskHistoryModel
                }
            }
        }

        Tab {
            title: qsTr("Monitorowanie")

            ReportBrowser {
                anchors.fill: parent
            }
        }

        Tab {
            title: qsTr("Licencje i profile")

            Flickable {
                anchors.fill: parent
                contentWidth: width
                contentHeight: securityLayout.implicitHeight
                clip: true

                ColumnLayout {
                    id: securityLayout
                    width: parent.width
                    spacing: 16
                    padding: 16

                    GroupBox {
                        title: qsTr("Licencja OEM")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 8
                            Layout.fillWidth: true

                            Label { text: qsTr("Status") }
                            Label { text: securityController && securityController.licenseInfo.status || qsTr("n/d") }

                            Label { text: qsTr("Fingerprint") }
                            Label {
                                text: securityController && securityController.licenseInfo.fingerprint || qsTr("n/d")
                                wrapMode: Text.WrapAnywhere
                            }

                            Label { text: qsTr("Ważna od") }
                            Label { text: securityController && securityController.licenseInfo.valid_from || qsTr("n/d") }

                            Label { text: qsTr("Ważna do") }
                            Label { text: securityController && securityController.licenseInfo.valid_to || qsTr("n/d") }
                        }
                    }

                    GroupBox {
                        title: qsTr("Profile użytkowników")
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 12

                            ListView {
                                id: profileList
                                Layout.fillWidth: true
                                Layout.preferredHeight: 180
                                model: securityController ? securityController.userProfiles : []
                                clip: true

                                delegate: Frame {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    padding: 8
                                    background: Rectangle {
                                        color: Qt.rgba(0.2, 0.3, 0.5, 0.2)
                                        radius: 6
                                    }

                                    ColumnLayout {
                                        anchors.fill: parent
                                        spacing: 4

                                        Label {
                                            text: (modelData.display_name || modelData.user_id)
                                                  + " (" + (modelData.user_id || "-") + ")"
                                            font.bold: true
                                        }
                                        Label {
                                            text: qsTr("Role: %1").arg((modelData.roles || []).join(", "))
                                        }
                                        Label {
                                            text: qsTr("Aktualizacja: %1").arg(modelData.updated_at || "-")
                                            color: palette.mid
                                        }
                                    }

                                    TapHandler {
                                        acceptedButtons: Qt.LeftButton
                                        onTapped: {
                                            userField.text = modelData.user_id || ""
                                            nameField.text = modelData.display_name || ""
                                            rolesField.text = (modelData.roles || []).join(", ")
                                        }
                                    }
                                }
                            }

                            GridLayout {
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: qsTr("Użytkownik") }
                                TextField {
                                    id: userField
                                    Layout.fillWidth: true
                                    placeholderText: qsTr("Identyfikator użytkownika")
                                }

                                Label { text: qsTr("Nazwa wyświetlana") }
                                TextField {
                                    id: nameField
                                    Layout.fillWidth: true
                                }

                                Label { text: qsTr("Role (CSV)") }
                                TextField {
                                    id: rolesField
                                    Layout.fillWidth: true
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Button {
                                    text: qsTr("Zapisz profil")
                                    enabled: securityController && !securityController.busy
                                    onClicked: {
                                        const roles = rolesField.text.split(",").map(r => r.trim()).filter(r => r.length > 0)
                                        const ok = securityController.assignProfile(
                                                    userField.text,
                                                    roles,
                                                    nameField.text)
                                        if (ok) {
                                            userField.text = ""
                                            nameField.text = ""
                                            rolesField.text = ""
                                        }
                                    }
                                }

                                Button {
                                    text: qsTr("Usuń profil")
                                    enabled: securityController && !securityController.busy && userField.text.length > 0
                                    onClicked: securityController && securityController.removeProfile(userField.text)
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTr("Odśwież")
                                    onClicked: securityController && securityController.refresh()
                                }
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }

    Connections {
        target: appController
        function onInstrumentChanged() { adminPanel.syncForms() }
        function onPerformanceGuardChanged() { adminPanel.syncForms() }
        function onRiskRefreshScheduleChanged() { adminPanel.updateRiskSchedule() }
    }
}
