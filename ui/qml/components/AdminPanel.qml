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
    property var decisionForm: ({})
    property var decisionOverrides: []
    property string decisionStatusMessage: ""
    property color decisionStatusColor: palette.highlight
    property var schedulerListModel: []
    property string selectedSchedulerName: ""
    property var schedulerForm: ({})
    property var schedulerSchedules: []
    property int selectedScheduleIndex: -1
    property string schedulerStatusMessage: ""
    property color schedulerStatusColor: palette.highlight
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

    function clone(value) {
        if (value === undefined || value === null)
            return value
        return JSON.parse(JSON.stringify(value))
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
        decisionStatusMessage = ""
        schedulerStatusMessage = ""
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
        if (typeof strategyController !== "undefined") {
            var decisionSnapshot = strategyController.decisionConfigSnapshot()
            decisionForm = clone(decisionSnapshot) || ({})
            decisionOverrides = decisionForm.profile_overrides ? clone(decisionForm.profile_overrides) : []
            schedulerListModel = strategyController.schedulerList() || []
            if (schedulerListModel.length > 0) {
                if (!selectedSchedulerName || schedulerListModel.findIndex(function(item) { return item.name === selectedSchedulerName }) === -1)
                    selectedSchedulerName = schedulerListModel[0].name
                var schedulerSnapshot = strategyController.schedulerConfigSnapshot(selectedSchedulerName)
                schedulerForm = clone(schedulerSnapshot) || ({})
                schedulerSchedules = schedulerForm.schedules ? clone(schedulerForm.schedules) : []
                if (schedulerSchedules.length > 0) {
                    if (selectedScheduleIndex < 0 || selectedScheduleIndex >= schedulerSchedules.length)
                        selectedScheduleIndex = 0
                } else {
                    selectedScheduleIndex = -1
                }
            } else {
                schedulerForm = ({})
                schedulerSchedules = []
                selectedSchedulerName = ""
                selectedScheduleIndex = -1
            }
        }

        Tab {
            title: qsTr("Wsparcie")

            property string pendingPathTarget: ""

            Flickable {
                anchors.fill: parent
                contentWidth: width
                contentHeight: supportLayout.implicitHeight
                clip: true

                ColumnLayout {
                    id: supportLayout
                    width: parent.width
                    spacing: 16
                    padding: 16

                    GroupBox {
                        title: qsTr("Zakres pakietu")
                        Layout.fillWidth: true

                        ColumnLayout {
                            spacing: 12

                            Repeater {
                                model: [
                                    {
                                        label: qsTr("Logi runtime"),
                                        enabledProperty: "includeLogs",
                                        pathProperty: "logsPath",
                                        target: "logs"
                                    },
                                    {
                                        label: qsTr("Raporty i eksporty"),
                                        enabledProperty: "includeReports",
                                        pathProperty: "reportsPath",
                                        target: "reports"
                                    },
                                    {
                                        label: qsTr("Licencje OEM"),
                                        enabledProperty: "includeLicenses",
                                        pathProperty: "licensesPath",
                                        target: "licenses"
                                    },
                                    {
                                        label: qsTr("Telemetria / metryki"),
                                        enabledProperty: "includeMetrics",
                                        pathProperty: "metricsPath",
                                        target: "metrics"
                                    },
                                    {
                                        label: qsTr("Artefakty audytu"),
                                        enabledProperty: "includeAudit",
                                        pathProperty: "auditPath",
                                        target: "audit"
                                    }
                                ]

                                delegate: RowLayout {
                                    required property string label
                                    required property string enabledProperty
                                    required property string pathProperty
                                    required property string target

                                    Layout.fillWidth: true
                                    spacing: 12

                                    CheckBox {
                                        text: label
                                        checked: supportController && supportController[enabledProperty]
                                        enabled: supportController && !supportController.busy
                                        onToggled: {
                                            if (!supportController)
                                                return
                                            supportController[enabledProperty] = checked
                                        }
                                    }

                                    TextField {
                                        Layout.fillWidth: true
                                        readOnly: true
                                        text: supportController ? supportController[pathProperty] : ""
                                        placeholderText: qsTr("Ścieżka nieustawiona")
                                        color: enabled ? palette.text : palette.mid
                                    }

                                    Button {
                                        text: qsTr("Wybierz…")
                                        enabled: supportController && !supportController.busy
                                        onClicked: {
                                            if (!supportController)
                                                return
                                            pendingPathTarget = target
                                            const basePath = supportController[pathProperty]
                                                    ? QUrl.fromLocalFile(supportController[pathProperty])
                                                    : Qt.resolvedUrl(".")
                                            supportFolderDialog.folder = basePath
                                            supportFolderDialog.open()
                                        }
                                    }
                                }
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Eksport")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 3
                            columnSpacing: 12
                            rowSpacing: 8
                            Layout.fillWidth: true

                            Label { text: qsTr("Format") }
                            ComboBox {
                                id: supportFormatCombo
                                Layout.fillWidth: true
                                model: ["tar.gz", "zip"]
                                enabled: supportController && !supportController.busy
                                onActivated: {
                                    if (supportController)
                                        supportController.format = model[index]
                                }
                                Component.onCompleted: {
                                    if (supportController) {
                                        const idx = model.indexOf(supportController.format)
                                        currentIndex = idx >= 0 ? idx : 0
                                    }
                                }
                                Binding {
                                    target: supportFormatCombo
                                    property: "currentIndex"
                                    value: {
                                        if (!supportController)
                                            return 0
                                        const idx = model.indexOf(supportController.format)
                                        return idx >= 0 ? idx : 0
                                    }
                                    when: supportController && !supportFormatCombo.activeFocus
                                }
                            }
                            Item {}

                            Label { text: qsTr("Katalog wyjściowy") }
                            TextField {
                                Layout.fillWidth: true
                                readOnly: true
                                text: supportController ? supportController.outputDirectory : ""
                                placeholderText: qsTr("Domyślnie var/support")
                            }
                            Button {
                                text: qsTr("Wybierz…")
                                enabled: supportController && !supportController.busy
                                onClicked: {
                                    pendingPathTarget = "output"
                                    const dir = supportController && supportController.outputDirectory
                                            ? QUrl.fromLocalFile(supportController.outputDirectory)
                                            : Qt.resolvedUrl(".")
                                    supportFolderDialog.folder = dir
                                    supportFolderDialog.open()
                                }
                            }

                            Label { text: qsTr("Bazowa nazwa pliku") }
                            TextField {
                                Layout.fillWidth: true
                                enabled: supportController && !supportController.busy
                                text: supportController ? supportController.defaultBasename : "support-bundle"
                                onEditingFinished: {
                                    if (supportController && text.length > 0)
                                        supportController.defaultBasename = text.trim()
                                }
                            }
                            Item {}

                            Item { Layout.columnSpan: 3; Layout.fillWidth: true }
                        }
                    }

                    GroupBox {
                        title: qsTr("Status")
                        Layout.fillWidth: true

                        ColumnLayout {
                            spacing: 8

                            RowLayout {
                                spacing: 12
                                Label {
                                    text: supportController && supportController.busy
                                            ? qsTr("Trwa eksport pakietu wsparcia…")
                                            : qsTr("Gotowe do eksportu")
                                    font.bold: true
                                }
                                BusyIndicator {
                                    running: supportController && supportController.busy
                                    visible: running
                                }
                            }

                            Label {
                                visible: supportController && supportController.lastStatusMessage.length > 0
                                text: supportController ? supportController.lastStatusMessage : ""
                                color: Qt.rgba(0.3, 0.7, 0.4, 1)
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                visible: supportController && supportController.lastErrorMessage.length > 0
                                text: supportController ? supportController.lastErrorMessage : ""
                                color: Qt.rgba(0.9, 0.4, 0.3, 1)
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                visible: supportController && supportController.lastBundlePath.length > 0
                                spacing: 8

                                Label { text: qsTr("Ostatni pakiet:") }
                                TextField {
                                    Layout.fillWidth: true
                                    readOnly: true
                                    text: supportController ? supportController.lastBundlePath : ""
                                }
                                Button {
                                    text: qsTr("Otwórz")
                                    onClicked: {
                                        if (supportController && supportController.lastBundlePath.length > 0)
                                            Qt.openUrlExternally(QUrl.fromLocalFile(supportController.lastBundlePath))
                                    }
                                }
                            }

                            RowLayout {
                                spacing: 12

                                Button {
                                    text: qsTr("Eksportuj…")
                                    enabled: supportController && !supportController.busy
                                    onClicked: supportBundleDialog.open()
                                }

                                Button {
                                    text: qsTr("Szybki eksport")
                                    enabled: supportController && !supportController.busy
                                    onClicked: {
                                        if (supportController)
                                            supportController.exportBundle()
                                    }
                                }
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Ostatni manifest")
                        Layout.fillWidth: true

                        ColumnLayout {
                            spacing: 6

                            ListView {
                                id: supportEntriesView
                                Layout.fillWidth: true
                                Layout.preferredHeight: Math.min(contentHeight, 240)
                                clip: true
                                model: supportController && supportController.lastResult
                                       && supportController.lastResult.entries
                                       ? supportController.lastResult.entries
                                       : []
                                delegate: Frame {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    padding: 8
                                    background: Rectangle {
                                        color: Qt.rgba(0.18, 0.24, 0.32, 0.4)
                                        radius: 6
                                    }

                                    ColumnLayout {
                                        anchors.fill: parent
                                        spacing: 4

                                        Label {
                                            text: modelData.label || qsTr("nieznany zasób")
                                            font.bold: true
                                        }

                                        Label {
                                            text: modelData.source || ""
                                            wrapMode: Text.WordWrap
                                            color: palette.midlight
                                        }

                                        Label {
                                            text: qsTr("Pliki: %1, rozmiar: %2 B").arg(modelData.file_count || 0)
                                                      .arg(modelData.size_bytes || 0)
                                        }

                                        Label {
                                            visible: modelData.exists === false
                                            text: qsTr("Uwaga: ścieżka nie istnieje")
                                            color: Qt.rgba(0.95, 0.45, 0.3, 1)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            FolderDialog {
                id: supportFolderDialog
                title: qsTr("Wybierz katalog")
                onAccepted: {
                    if (!supportController)
                        return
                    const url = supportFolderDialog.selectedFolder
                                 ? supportFolderDialog.selectedFolder
                                 : (currentFolder || folder)
                    if (!url)
                        return
                    const localPath = url.toLocalFile ? url.toLocalFile() : url.toString().replace("file://", "")
                    switch (pendingPathTarget) {
                    case "logs": supportController.logsPath = localPath; break
                    case "reports": supportController.reportsPath = localPath; break
                    case "licenses": supportController.licensesPath = localPath; break
                    case "metrics": supportController.metricsPath = localPath; break
                    case "audit": supportController.auditPath = localPath; break
                    case "output": supportController.outputDirectory = localPath; break
                    }
                }
            }

            FileDialog {
                id: supportBundleDialog
                title: qsTr("Zapisz pakiet wsparcia")
                fileMode: FileDialog.SaveFile
                defaultSuffix: supportController && supportController.format === "zip" ? "zip" : "tar.gz"
                nameFilters: [qsTr("Archiwa (*.tar.gz *.zip)"), qsTr("Wszystkie pliki (*)")]
                onAccepted: {
                    if (supportController)
                        supportController.exportBundle(selectedFile)
                }
            }
        }
    }

    function refreshData() {
        if (typeof strategyController !== "undefined")
            strategyController.refresh()
        syncForms()
        if (typeof securityController !== "undefined")
            securityController.refresh()
        if (typeof reportController !== "undefined")
            reportController.refresh()
    }

    function currentSchedule() {
        if (!schedulerSchedules || schedulerSchedules.length === 0)
            return null
        if (selectedScheduleIndex < 0 || selectedScheduleIndex >= schedulerSchedules.length)
            return null
        return schedulerSchedules[selectedScheduleIndex]
    }

    function updateDecisionField(key, value) {
        var copy = clone(decisionForm) || {}
        copy[key] = value
        decisionForm = copy
    }

    function updateDecisionOverrideField(index, key, value) {
        if (!decisionOverrides || index < 0 || index >= decisionOverrides.length)
            return
        var overrides = clone(decisionOverrides)
        if (!overrides[index])
            overrides[index] = {}
        overrides[index][key] = value
        decisionOverrides = overrides
    }

    function updateSchedulerField(key, value) {
        var copy = clone(schedulerForm) || {}
        copy[key] = value
        schedulerForm = copy
    }

    function updateScheduleField(index, key, value) {
        if (!schedulerSchedules || index < 0 || index >= schedulerSchedules.length)
            return
        var schedules = clone(schedulerSchedules)
        if (!schedules[index])
            schedules[index] = {}
        schedules[index][key] = value
        schedulerSchedules = schedules
    }

    function selectScheduler(name) {
        if (typeof strategyController === "undefined" || !name)
            return
        selectedSchedulerName = name
        var snapshot = strategyController.schedulerConfigSnapshot(name)
        schedulerForm = clone(snapshot) || ({})
        schedulerSchedules = schedulerForm.schedules ? clone(schedulerForm.schedules) : []
        selectedScheduleIndex = schedulerSchedules.length > 0 ? 0 : -1
        schedulerStatusMessage = ""
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

                    GroupBox {
                        title: qsTr("Tryb offline i automatyzacja")
                        Layout.fillWidth: true

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            Label {
                                wrapMode: Text.WordWrap
                                text: appController && appController.offlineMode
                                      ? qsTr("Status: %1").arg(appController.offlineDaemonStatus)
                                      : qsTr("Tryb offline jest nieaktywny. Uruchom aplikację z parametrem --offline-mode, aby korzystać z lokalnego daemona REST.")
                                color: appController && appController.offlineMode ? palette.text : palette.mid
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                visible: appController && appController.offlineMode
                                spacing: 8

                                Label {
                                    text: qsTr("Konfiguracja")
                                    Layout.alignment: Qt.AlignVCenter
                                }

                                TextField {
                                    Layout.fillWidth: true
                                    readOnly: true
                                    text: appController && appController.offlineStrategyPath.length > 0
                                          ? appController.offlineStrategyPath
                                          : qsTr("(brak pliku)")
                                    selectByMouse: true
                                }
                            }

                            Label {
                                visible: appController && appController.offlineMode
                                text: appController && appController.offlineAutomationRunning
                                      ? qsTr("Auto-run: aktywny")
                                      : qsTr("Auto-run: zatrzymany")
                                color: palette.text
                            }

                            RowLayout {
                                Layout.alignment: Qt.AlignLeft
                                spacing: 12
                                visible: appController && appController.offlineMode

                                Button {
                                    text: qsTr("Start auto-run")
                                    enabled: appController && !appController.offlineAutomationRunning
                                    onClicked: appController.startOfflineAutomation()
                                }

                                Button {
                                    text: qsTr("Stop auto-run")
                                    enabled: appController && appController.offlineAutomationRunning
                                    onClicked: appController.stopOfflineAutomation()
                                }
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("DecisionOrchestrator")
                        Layout.fillWidth: true

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            GridLayout {
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: qsTr("Maks. koszt (bps)") }
                                TextField {
                                    id: decisionMaxCostField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.max_cost_bps !== undefined ? String(decisionForm.max_cost_bps) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_cost_bps", value)
                                    }
                                    Binding {
                                        target: decisionMaxCostField
                                        property: "text"
                                        value: decisionForm.max_cost_bps !== undefined ? String(decisionForm.max_cost_bps) : ""
                                        when: !decisionMaxCostField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Minimalny edge netto (bps)") }
                                TextField {
                                    id: decisionMinEdgeField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.min_net_edge_bps !== undefined ? String(decisionForm.min_net_edge_bps) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("min_net_edge_bps", value)
                                    }
                                    Binding {
                                        target: decisionMinEdgeField
                                        property: "text"
                                        value: decisionForm.min_net_edge_bps !== undefined ? String(decisionForm.min_net_edge_bps) : ""
                                        when: !decisionMinEdgeField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Maks. dzienny drawdown (%)") }
                                TextField {
                                    id: decisionDailyLossField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.max_daily_loss_pct !== undefined ? String(decisionForm.max_daily_loss_pct) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_daily_loss_pct", value)
                                    }
                                    Binding {
                                        target: decisionDailyLossField
                                        property: "text"
                                        value: decisionForm.max_daily_loss_pct !== undefined ? String(decisionForm.max_daily_loss_pct) : ""
                                        when: !decisionDailyLossField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Maks. drawdown (%)") }
                                TextField {
                                    id: decisionDrawdownField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.max_drawdown_pct !== undefined ? String(decisionForm.max_drawdown_pct) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_drawdown_pct", value)
                                    }
                                    Binding {
                                        target: decisionDrawdownField
                                        property: "text"
                                        value: decisionForm.max_drawdown_pct !== undefined ? String(decisionForm.max_drawdown_pct) : ""
                                        when: !decisionDrawdownField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Maks. ekspozycja (%)") }
                                TextField {
                                    id: decisionPositionField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.max_position_ratio !== undefined ? String(decisionForm.max_position_ratio) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_position_ratio", value)
                                    }
                                    Binding {
                                        target: decisionPositionField
                                        property: "text"
                                        value: decisionForm.max_position_ratio !== undefined ? String(decisionForm.max_position_ratio) : ""
                                        when: !decisionPositionField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Maks. liczba pozycji") }
                                TextField {
                                    id: decisionOpenPositionsField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhDigitsOnly
                                    text: decisionForm.max_open_positions !== undefined ? String(decisionForm.max_open_positions) : ""
                                    onEditingFinished: {
                                        var value = parseInt(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_open_positions", value)
                                    }
                                    Binding {
                                        target: decisionOpenPositionsField
                                        property: "text"
                                        value: decisionForm.max_open_positions !== undefined ? String(decisionForm.max_open_positions) : ""
                                        when: !decisionOpenPositionsField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Budżet latencji (ms)") }
                                TextField {
                                    id: decisionLatencyField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.max_latency_ms !== undefined ? String(decisionForm.max_latency_ms) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("max_latency_ms", value)
                                    }
                                    Binding {
                                        target: decisionLatencyField
                                        property: "text"
                                        value: decisionForm.max_latency_ms !== undefined ? String(decisionForm.max_latency_ms) : ""
                                        when: !decisionLatencyField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Min. prawdopodobieństwo") }
                                TextField {
                                    id: decisionProbabilityField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.min_probability !== undefined ? String(decisionForm.min_probability) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value)) {
                                            value = Math.max(0.0, Math.min(1.0, value))
                                            updateDecisionField("min_probability", value)
                                            text = String(value)
                                        }
                                    }
                                    Binding {
                                        target: decisionProbabilityField
                                        property: "text"
                                        value: decisionForm.min_probability !== undefined ? String(decisionForm.min_probability) : ""
                                        when: !decisionProbabilityField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Kara kosztowa (bps)") }
                                TextField {
                                    id: decisionPenaltyField
                                    Layout.fillWidth: true
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    text: decisionForm.penalty_cost_bps !== undefined ? String(decisionForm.penalty_cost_bps) : ""
                                    onEditingFinished: {
                                        var value = parseFloat(text)
                                        if (!isNaN(value))
                                            updateDecisionField("penalty_cost_bps", value)
                                    }
                                    Binding {
                                        target: decisionPenaltyField
                                        property: "text"
                                        value: decisionForm.penalty_cost_bps !== undefined ? String(decisionForm.penalty_cost_bps) : ""
                                        when: !decisionPenaltyField.activeFocus
                                    }
                                }
                            }

                            Switch {
                                id: requireCostSwitch
                                text: qsTr("Wymagaj danych kosztowych Decision Engine")
                                checked: decisionForm.require_cost_data === undefined ? true : decisionForm.require_cost_data
                                onToggled: updateDecisionField("require_cost_data", checked)
                                Binding {
                                    target: requireCostSwitch
                                    property: "checked"
                                    value: decisionForm.require_cost_data === undefined ? true : decisionForm.require_cost_data
                                    when: !requireCostSwitch.down && !requireCostSwitch.activeFocus
                                }
                            }

                            Repeater {
                                model: decisionOverrides.length
                                delegate: GroupBox {
                                    title: qsTr("Profil %1").arg(decisionOverrides[index] && decisionOverrides[index].profile ? decisionOverrides[index].profile : qsTr("nieznany"))
                                    Layout.fillWidth: true

                                    GridLayout {
                                        columns: 2
                                        columnSpacing: 12
                                        rowSpacing: 8
                                        Layout.fillWidth: true

                                        Label { text: qsTr("Maks. koszt (bps)") }
                                        TextField {
                                            id: overrideMaxCostField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_cost_bps !== undefined ? String(decisionOverrides[index].max_cost_bps) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_cost_bps", value)
                                            }
                                            Binding {
                                                target: overrideMaxCostField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_cost_bps !== undefined ? String(decisionOverrides[index].max_cost_bps) : ""
                                                when: !overrideMaxCostField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Minimalny edge netto (bps)") }
                                        TextField {
                                            id: overrideMinEdgeField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].min_net_edge_bps !== undefined ? String(decisionOverrides[index].min_net_edge_bps) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "min_net_edge_bps", value)
                                            }
                                            Binding {
                                                target: overrideMinEdgeField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].min_net_edge_bps !== undefined ? String(decisionOverrides[index].min_net_edge_bps) : ""
                                                when: !overrideMinEdgeField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Maks. dzienny drawdown (%)") }
                                        TextField {
                                            id: overrideDailyLossField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_daily_loss_pct !== undefined ? String(decisionOverrides[index].max_daily_loss_pct) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_daily_loss_pct", value)
                                            }
                                            Binding {
                                                target: overrideDailyLossField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_daily_loss_pct !== undefined ? String(decisionOverrides[index].max_daily_loss_pct) : ""
                                                when: !overrideDailyLossField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Maks. drawdown (%)") }
                                        TextField {
                                            id: overrideDrawdownField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_drawdown_pct !== undefined ? String(decisionOverrides[index].max_drawdown_pct) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_drawdown_pct", value)
                                            }
                                            Binding {
                                                target: overrideDrawdownField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_drawdown_pct !== undefined ? String(decisionOverrides[index].max_drawdown_pct) : ""
                                                when: !overrideDrawdownField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Maks. ekspozycja (%)") }
                                        TextField {
                                            id: overridePositionField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_position_ratio !== undefined ? String(decisionOverrides[index].max_position_ratio) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_position_ratio", value)
                                            }
                                            Binding {
                                                target: overridePositionField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_position_ratio !== undefined ? String(decisionOverrides[index].max_position_ratio) : ""
                                                when: !overridePositionField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Maks. liczba pozycji") }
                                        TextField {
                                            id: overrideOpenPositionsField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhDigitsOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_open_positions !== undefined ? String(decisionOverrides[index].max_open_positions) : ""
                                            onEditingFinished: {
                                                var value = parseInt(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_open_positions", value)
                                            }
                                            Binding {
                                                target: overrideOpenPositionsField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_open_positions !== undefined ? String(decisionOverrides[index].max_open_positions) : ""
                                                when: !overrideOpenPositionsField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Budżet latencji (ms)") }
                                        TextField {
                                            id: overrideLatencyField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_latency_ms !== undefined ? String(decisionOverrides[index].max_latency_ms) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_latency_ms", value)
                                            }
                                            Binding {
                                                target: overrideLatencyField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_latency_ms !== undefined ? String(decisionOverrides[index].max_latency_ms) : ""
                                                when: !overrideLatencyField.activeFocus
                                            }
                                        }

                                        Label { text: qsTr("Limit notional (USD)") }
                                        TextField {
                                            id: overrideNotionalField
                                            Layout.fillWidth: true
                                            inputMethodHints: Qt.ImhFormattedNumbersOnly
                                            text: decisionOverrides[index] && decisionOverrides[index].max_trade_notional !== undefined ? String(decisionOverrides[index].max_trade_notional) : ""
                                            onEditingFinished: {
                                                var value = parseFloat(text)
                                                if (!isNaN(value))
                                                    updateDecisionOverrideField(index, "max_trade_notional", value)
                                            }
                                            Binding {
                                                target: overrideNotionalField
                                                property: "text"
                                                value: decisionOverrides[index] && decisionOverrides[index].max_trade_notional !== undefined ? String(decisionOverrides[index].max_trade_notional) : ""
                                                when: !overrideNotionalField.activeFocus
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Button {
                                    text: qsTr("Przywróć")
                                    onClicked: syncForms()
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTr("Zapisz DecisionOrchestrator")
                                    highlighted: true
                                    enabled: typeof strategyController !== "undefined"
                                    onClicked: {
                                        if (typeof strategyController === "undefined")
                                            return
                                        var payload = clone(decisionForm) || ({})
                                        payload.profile_overrides = decisionOverrides || []
                                        if (strategyController.saveDecisionConfig(payload)) {
                                            decisionStatusMessage = qsTr("Zapisano konfigurację DecisionOrchestratora")
                                            decisionStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                            syncForms()
                                        } else {
                                            decisionStatusMessage = strategyController.lastError || qsTr("Nie udało się zapisać konfiguracji DecisionOrchestratora")
                                            decisionStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                        }
                                    }
                                }
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: decisionStatusMessage.length > 0
                                text: decisionStatusMessage
                                color: decisionStatusColor
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Scheduler strategii/AI")
                        Layout.fillWidth: true

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: qsTr("Scheduler")
                                    enabled: schedulerListModel && schedulerListModel.length > 0
                                }

                                ComboBox {
                                    id: schedulerCombo
                                    Layout.fillWidth: true
                                    model: schedulerListModel
                                    textRole: "name"
                                    enabled: schedulerListModel && schedulerListModel.length > 0
                                    onActivated: selectScheduler(model[index].name)
                                    Binding {
                                        target: schedulerCombo
                                        property: "currentIndex"
                                        value: schedulerListModel && schedulerListModel.length > 0 ? schedulerListModel.findIndex(function(item) { return item.name === selectedSchedulerName }) : -1
                                        when: !schedulerCombo.popup.visible
                                    }
                                }
                            }

                            GridLayout {
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: qsTr("Namespace telemetrii") }
                                TextField {
                                    id: schedulerTelemetryField
                                    Layout.fillWidth: true
                                    text: schedulerForm.telemetry_namespace !== undefined ? schedulerForm.telemetry_namespace : ""
                                    onEditingFinished: updateSchedulerField("telemetry_namespace", text)
                                    Binding {
                                        target: schedulerTelemetryField
                                        property: "text"
                                        value: schedulerForm.telemetry_namespace !== undefined ? schedulerForm.telemetry_namespace : ""
                                        when: !schedulerTelemetryField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Kategoria decision logu") }
                                TextField {
                                    id: schedulerDecisionLogField
                                    Layout.fillWidth: true
                                    text: schedulerForm.decision_log_category !== undefined ? schedulerForm.decision_log_category : ""
                                    onEditingFinished: updateSchedulerField("decision_log_category", text)
                                    Binding {
                                        target: schedulerDecisionLogField
                                        property: "text"
                                        value: schedulerForm.decision_log_category !== undefined ? schedulerForm.decision_log_category : ""
                                        when: !schedulerDecisionLogField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Interwał health-check (s)") }
                                SpinBox {
                                    id: schedulerHealthSpin
                                    from: 30
                                    to: 3600
                                    stepSize: 10
                                    value: schedulerForm.health_check_interval !== undefined ? schedulerForm.health_check_interval : 300
                                    onValueModified: updateSchedulerField("health_check_interval", value)
                                    Binding {
                                        target: schedulerHealthSpin
                                        property: "value"
                                        value: schedulerForm.health_check_interval !== undefined ? schedulerForm.health_check_interval : 300
                                        when: !schedulerHealthSpin.activeFocus
                                    }
                                }

                                Label { text: qsTr("Portfolio Governor") }
                                TextField {
                                    id: schedulerGovernorField
                                    Layout.fillWidth: true
                                    text: schedulerForm.portfolio_governor !== undefined ? schedulerForm.portfolio_governor : ""
                                    onEditingFinished: updateSchedulerField("portfolio_governor", text)
                                    Binding {
                                        target: schedulerGovernorField
                                        property: "text"
                                        value: schedulerForm.portfolio_governor !== undefined ? schedulerForm.portfolio_governor : ""
                                        when: !schedulerGovernorField.activeFocus
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: qsTr("Zadanie harmonogramu")
                                    enabled: schedulerSchedules && schedulerSchedules.length > 0
                                }

                                ComboBox {
                                    id: scheduleCombo
                                    Layout.fillWidth: true
                                    model: schedulerSchedules
                                    textRole: "name"
                                    enabled: schedulerSchedules && schedulerSchedules.length > 0
                                    onActivated: selectedScheduleIndex = index
                                    Binding {
                                        target: scheduleCombo
                                        property: "currentIndex"
                                        value: selectedScheduleIndex
                                        when: !scheduleCombo.popup.visible
                                    }
                                }
                            }

                            GridLayout {
                                visible: currentSchedule() !== null
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: qsTr("Strategia") }
                                TextField {
                                    id: scheduleStrategyField
                                    Layout.fillWidth: true
                                    enabled: currentSchedule() !== null
                                    text: currentSchedule() ? (currentSchedule().strategy || "") : ""
                                    onEditingFinished: updateScheduleField(selectedScheduleIndex, "strategy", text)
                                    Binding {
                                        target: scheduleStrategyField
                                        property: "text"
                                        value: currentSchedule() ? (currentSchedule().strategy || "") : ""
                                        when: !scheduleStrategyField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Ryzyko (profil)") }
                                TextField {
                                    id: scheduleRiskField
                                    Layout.fillWidth: true
                                    enabled: currentSchedule() !== null
                                    text: currentSchedule() ? (currentSchedule().risk_profile || "") : ""
                                    onEditingFinished: updateScheduleField(selectedScheduleIndex, "risk_profile", text)
                                    Binding {
                                        target: scheduleRiskField
                                        property: "text"
                                        value: currentSchedule() ? (currentSchedule().risk_profile || "") : ""
                                        when: !scheduleRiskField.activeFocus
                                    }
                                }

                                Label { text: qsTr("Cadence (s)") }
                                SpinBox {
                                    id: scheduleCadenceSpin
                                    from: 1
                                    to: 3600
                                    stepSize: 5
                                    enabled: currentSchedule() !== null
                                    value: currentSchedule() ? (currentSchedule().cadence_seconds || 60) : 60
                                    onValueModified: updateScheduleField(selectedScheduleIndex, "cadence_seconds", value)
                                    Binding {
                                        target: scheduleCadenceSpin
                                        property: "value"
                                        value: currentSchedule() ? (currentSchedule().cadence_seconds || 60) : 60
                                        when: !scheduleCadenceSpin.activeFocus
                                    }
                                }

                                Label { text: qsTr("Maks. dryf (s)") }
                                SpinBox {
                                    id: scheduleDriftSpin
                                    from: 0
                                    to: 3600
                                    stepSize: 5
                                    enabled: currentSchedule() !== null
                                    value: currentSchedule() ? (currentSchedule().max_drift_seconds || 0) : 0
                                    onValueModified: updateScheduleField(selectedScheduleIndex, "max_drift_seconds", value)
                                    Binding {
                                        target: scheduleDriftSpin
                                        property: "value"
                                        value: currentSchedule() ? (currentSchedule().max_drift_seconds || 0) : 0
                                        when: !scheduleDriftSpin.activeFocus
                                    }
                                }

                                Label { text: qsTr("Warmup (bary)") }
                                SpinBox {
                                    id: scheduleWarmupSpin
                                    from: 0
                                    to: 2000
                                    stepSize: 1
                                    enabled: currentSchedule() !== null
                                    value: currentSchedule() ? (currentSchedule().warmup_bars || 0) : 0
                                    onValueModified: updateScheduleField(selectedScheduleIndex, "warmup_bars", value)
                                    Binding {
                                        target: scheduleWarmupSpin
                                        property: "value"
                                        value: currentSchedule() ? (currentSchedule().warmup_bars || 0) : 0
                                        when: !scheduleWarmupSpin.activeFocus
                                    }
                                }

                                Label { text: qsTr("Limit sygnałów") }
                                SpinBox {
                                    id: scheduleSignalsSpin
                                    from: 1
                                    to: 100
                                    stepSize: 1
                                    enabled: currentSchedule() !== null
                                    value: currentSchedule() ? (currentSchedule().max_signals || 10) : 10
                                    onValueModified: updateScheduleField(selectedScheduleIndex, "max_signals", value)
                                    Binding {
                                        target: scheduleSignalsSpin
                                        property: "value"
                                        value: currentSchedule() ? (currentSchedule().max_signals || 10) : 10
                                        when: !scheduleSignalsSpin.activeFocus
                                    }
                                }

                                Label { text: qsTr("Interwał cron") }
                                TextField {
                                    id: scheduleIntervalField
                                    Layout.fillWidth: true
                                    enabled: currentSchedule() !== null
                                    text: currentSchedule() ? (currentSchedule().interval || "") : ""
                                    onEditingFinished: updateScheduleField(selectedScheduleIndex, "interval", text)
                                    Binding {
                                        target: scheduleIntervalField
                                        property: "text"
                                        value: currentSchedule() ? (currentSchedule().interval || "") : ""
                                        when: !scheduleIntervalField.activeFocus
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Button {
                                    text: qsTr("Przywróć scheduler")
                                    onClicked: syncForms()
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTr("Zapisz scheduler")
                                    highlighted: true
                                    enabled: typeof strategyController !== "undefined" && selectedSchedulerName.length > 0
                                    onClicked: {
                                        if (typeof strategyController === "undefined" || selectedSchedulerName.length === 0)
                                            return
                                        var payload = clone(schedulerForm) || ({})
                                        payload.schedules = schedulerSchedules || []
                                        if (strategyController.saveSchedulerConfig(selectedSchedulerName, payload)) {
                                            schedulerStatusMessage = qsTr("Zapisano konfigurację schedulera %1").arg(selectedSchedulerName)
                                            schedulerStatusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                            syncForms()
                                        } else {
                                            schedulerStatusMessage = strategyController.lastError || qsTr("Nie udało się zapisać konfiguracji schedulera")
                                            schedulerStatusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                        }
                                    }
                                }
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: schedulerStatusMessage.length > 0
                                text: schedulerStatusMessage
                                color: schedulerStatusColor
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

            Item {
                anchors.fill: parent

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    GroupBox {
                        title: qsTr("Status backendu")
                        Layout.fillWidth: true

                        ColumnLayout {
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                BusyIndicator {
                                    running: healthController && healthController.busy
                                    visible: running
                                }

                                Label {
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: healthController ? (healthController.statusMessage || qsTr("Brak danych o HealthService"))
                                                          : qsTr("Brak danych o HealthService")
                                    color: healthController && healthController.healthy ? Qt.rgba(0.3, 0.7, 0.4, 1)
                                                                                         : Qt.rgba(0.86, 0.35, 0.35, 1)
                                }

                                Button {
                                    text: qsTr("Odśwież")
                                    enabled: healthController && !healthController.busy
                                    onClicked: healthController && healthController.refresh()
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                CheckBox {
                                    text: qsTr("Auto-odświeżanie")
                                    checked: healthController && healthController.autoRefreshEnabled
                                    enabled: !!healthController
                                    onToggled: {
                                        if (!healthController)
                                            return
                                        healthController.setAutoRefreshEnabled(checked)
                                    }
                                }

                                Label { text: qsTr("Interwał (s)") }

                                SpinBox {
                                    id: healthIntervalSpin
                                    from: 5
                                    to: 3600
                                    stepSize: 5
                                    value: healthController ? healthController.refreshIntervalSeconds : 60
                                    enabled: healthController && healthController.autoRefreshEnabled
                                    onValueModified: {
                                        if (!healthController)
                                            return
                                        healthController.setRefreshIntervalSeconds(value)
                                    }
                                    Binding {
                                        target: healthIntervalSpin
                                        property: "value"
                                        value: healthController ? healthController.refreshIntervalSeconds : 60
                                        when: !!healthController && !healthIntervalSpin.activeFocus
                                    }
                                }
                            }

                            GridLayout {
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 6
                                Layout.fillWidth: true

                                Label { text: qsTr("Wersja") }
                                Label {
                                    wrapMode: Text.WrapAnywhere
                                    text: healthController && healthController.version.length > 0
                                          ? healthController.version
                                          : qsTr("n/d")
                                }

                                Label { text: qsTr("Commit") }
                                Label {
                                    wrapMode: Text.WrapAnywhere
                                    text: healthController && healthController.gitCommit.length > 0
                                          ? healthController.gitCommitShort
                                          : qsTr("n/d")
                                }

                                Label { text: qsTr("Start (UTC)") }
                                Label {
                                    text: healthController && healthController.startedAt.length > 0
                                          ? healthController.startedAt
                                          : qsTr("n/d")
                                }

                                Label { text: qsTr("Start (lokalny)") }
                                Label {
                                    text: healthController && healthController.startedAtLocal.length > 0
                                          ? healthController.startedAtLocal
                                          : qsTr("n/d")
                                }

                                Label { text: qsTr("Czas działania") }
                                Label {
                                    text: healthController && healthController.uptime.length > 0
                                          ? healthController.uptime
                                          : qsTr("n/d")
                                }

                                Label { text: qsTr("Ostatnie sprawdzenie") }
                                Label {
                                    text: healthController && healthController.lastCheckedAt.length > 0
                                          ? healthController.lastCheckedAt
                                          : qsTr("n/d")
                                }
                            }
                        }
                    }

                    ReportBrowser {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                }
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

                            Label { text: qsTr("Edycja") }
                            Label { text: securityController && securityController.licenseInfo.edition || qsTr("n/d") }

                            Label { text: qsTr("Utrzymanie do") }
                            Label { text: securityController && securityController.licenseInfo.maintenance_until || qsTr("n/d") }

                            Label { text: qsTr("Trial") }
                            Label {
                                text: securityController && securityController.licenseInfo.trial_active
                                      ? (securityController.licenseInfo.trial_expires_at || qsTr("aktywny"))
                                      : qsTr("nieaktywny")
                            }

                            Label { text: qsTr("Odbiorca") }
                            Label {
                                wrapMode: Text.WrapAnywhere
                                text: {
                                    if (!securityController || !securityController.licenseInfo.holder)
                                        return qsTr("n/d");
                                    const holder = securityController.licenseInfo.holder;
                                    let base = holder.name || qsTr("n/d");
                                    if (holder.email)
                                        base += " (" + holder.email + ")";
                                    return base;
                                }
                            }

                            Label { text: qsTr("Seats") }
                            Label {
                                text: securityController && securityController.licenseInfo.seats !== undefined
                                      ? securityController.licenseInfo.seats
                                      : qsTr("n/d")
                            }

                            Label { text: qsTr("Fingerprint") }
                            Label {
                                text: securityController && securityController.licenseInfo.fingerprint || qsTr("n/d")
                                wrapMode: Text.WrapAnywhere
                            }

                            Label { text: qsTr("Moduły") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.modules)
                                        return qsTr("brak");
                                    const modules = securityController.licenseInfo.modules;
                                    return modules.length > 0 ? modules.join(", ") : qsTr("brak");
                                }
                            }

                            Label { text: qsTr("Środowiska") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.environments)
                                        return qsTr("brak");
                                    const envs = securityController.licenseInfo.environments;
                                    return envs.length > 0 ? envs.join(", ") : qsTr("brak");
                                }
                            }

                            Label { text: qsTr("Runtime") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.runtime)
                                        return qsTr("brak");
                                    const runtime = securityController.licenseInfo.runtime;
                                    return runtime.length > 0 ? runtime.join(", ") : qsTr("brak");
                                }
                            }
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
