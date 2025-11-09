import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../design-system" as DesignSystem
import "../design-system/components" as DesignComponents

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var reportController: (typeof reportController !== "undefined" ? reportController : null)

    property var savedPresets: []
    property var presetPreview: null
    property int selectedPresetIndex: -1
    property var pendingClonePayload: null
    property string actionStatus: ""
    property var championSummary: ({})
    property var reportsSummary: ({})
    property var championReports: []
    property string selectedReportPath: ""
    property var archivePreviewInfo: null
    property var performanceIndicators: ({})
    property var strategyHistory: []
    property var exchangeAllocations: []
    property var exchangeHistory: []
    property var exchangeSelected: null
    property string runtimeSnapshotTimestamp: ""

    onSelectedReportPathChanged: {
        if (typeof reportsList === "undefined")
            return
        if (!championReports || championReports.length === 0) {
            reportsList.currentIndex = -1
            return
        }
        for (var i = 0; i < championReports.length; ++i) {
            var entry = championReports[i]
            if (entry && entry.relative_path === selectedReportPath) {
                reportsList.currentIndex = i
                return
            }
        }
        reportsList.currentIndex = -1
    }
    property string pendingPromotionModel: ""
    property string pendingPromotionVersion: ""
    property string pendingPromotionReason: ""

    function resetPresetPreview() {
        presetPreview = null
        if (typeof presetList !== "undefined")
            presetList.currentIndex = -1
        selectedPresetIndex = -1
    }

    function resetArchivePreview() {
        archivePreviewInfo = null
    }

    function refreshPresets() {
        if (!runtimeService || !runtimeService.listStrategyPresets) {
            savedPresets = []
            resetPresetPreview()
            actionStatus = qsTr("Mostek runtime nie jest dostępny")
            return
        }
        var entries = runtimeService.listStrategyPresets()
        savedPresets = entries || []
        if (!entries || entries.length === 0) {
            resetPresetPreview()
            actionStatus = qsTr("Brak zapisanych presetów")
            return
        }
        selectedPresetIndex = 0
        if (typeof presetList !== "undefined")
            presetList.currentIndex = selectedPresetIndex
        if (savedPresets.length > 0)
            previewPresetEntry(savedPresets[0])
        actionStatus = qsTr("Załadowano %1 preset(ów)").arg(entries.length)
    }

    function formatDiffValue(value, isPercent) {
        if (value === undefined || value === null || value === "")
            return qsTr("brak")
        if (typeof value === "number" && !isNaN(value)) {
            var locale = Qt.locale()
            var scaled = isPercent ? value * 100 : value
            var precision = isPercent ? 2 : 3
            var formatted = Number(scaled).toLocaleString(locale, 'f', precision)
            return isPercent ? formatted + " %" : formatted
        }
        if (typeof value === "string" && value.length > 0)
            return value
        return value
    }

    function cloneValue(value) {
        try {
            return JSON.parse(JSON.stringify(value))
        } catch (err) {
            console.warn("StrategyManagement clone fallback", err)
            return value
        }
    }

    function formatNumber(value, digits) {
        if (value === undefined || value === null)
            return "—"
        var precision = digits !== undefined ? digits : 2
        return Number(value).toLocaleString(Qt.locale(), "f", precision)
    }

    function formatPercent(value, digits) {
        if (value === undefined || value === null)
            return "—"
        var precision = digits !== undefined ? digits : 1
        return Number(value * 100).toLocaleString(Qt.locale(), "f", precision) + "%"
    }

    function formatTimestamp(value) {
        if (!value)
            return qsTr("brak")
        var date = new Date(value)
        if (isNaN(date.getTime()))
            return value
        return Qt.formatDateTime(date, Qt.DefaultLocaleShortDate)
    }

    function refreshRuntimeIndicators() {
        if (!runtimeService || !runtimeService.autoModeSnapshot) {
            performanceIndicators = {}
            strategyHistory = []
            exchangeAllocations = []
            exchangeHistory = []
            exchangeSelected = null
            runtimeSnapshotTimestamp = ""
            return
        }
        var snapshot = runtimeService.autoModeSnapshot()
        runtimeSnapshotTimestamp = snapshot && snapshot.timestamp ? snapshot.timestamp : ""
        var indicators = snapshot.performanceIndicators || snapshot.performance_indicators || {}
        performanceIndicators = cloneValue(indicators || {})
        var strategySection = indicators && indicators.strategy ? indicators.strategy : {}
        strategyHistory = strategySection && strategySection.history ? cloneValue(strategySection.history) : []
        var exchangeSection = indicators && indicators.exchange ? indicators.exchange : {}
        exchangeAllocations = exchangeSection && exchangeSection.allocations ? cloneValue(exchangeSection.allocations) : []
        exchangeHistory = exchangeSection && exchangeSection.history ? cloneValue(exchangeSection.history) : []
        exchangeSelected = exchangeSection && exchangeSection.selected !== undefined ? cloneValue(exchangeSection.selected) : null
    }

    onPerformanceIndicatorsChanged: {
        var strategySection = performanceIndicators && performanceIndicators.strategy ? performanceIndicators.strategy : {}
        strategyHistory = strategySection && strategySection.history ? cloneValue(strategySection.history) : []
        var exchangeSection = performanceIndicators && performanceIndicators.exchange ? performanceIndicators.exchange : {}
        exchangeAllocations = exchangeSection && exchangeSection.allocations ? cloneValue(exchangeSection.allocations) : []
        exchangeHistory = exchangeSection && exchangeSection.history ? cloneValue(exchangeSection.history) : []
        exchangeSelected = exchangeSection && exchangeSection.selected !== undefined ? cloneValue(exchangeSection.selected) : null
    }

    Timer {
        id: runtimeIndicatorsTimer
        interval: 5000
        running: runtimeService && runtimeService.autoModeSnapshot
        repeat: true
        onTriggered: refreshRuntimeIndicators()
    }

    Component.onCompleted: refreshRuntimeIndicators()

    function previewPresetEntry(entry) {
        if (!runtimeService || !runtimeService.previewStrategyPreset) {
            actionStatus = qsTr("Mostek runtime nie obsługuje podglądu presetów")
            presetPreview = null
            return
        }
        if (!entry) {
            actionStatus = qsTr("Nie wybrano presetu do porównania")
            presetPreview = null
            return
        }
        var selector = {}
        if (entry.slug)
            selector.slug = entry.slug
        if (entry.path)
            selector.path = entry.path
        if (entry.name && !selector.slug)
            selector.name = entry.name
        var response = runtimeService.previewStrategyPreset(selector)
        if (!response || response.ok === false) {
            actionStatus = response && response.error ? response.error : qsTr("Podgląd presetu nie jest dostępny")
            presetPreview = response || null
            return
        }
        presetPreview = response
        var label = entry.label || entry.name || entry.slug || qsTr("preset")
        actionStatus = qsTr("Zbudowano podgląd presetu: %1").arg(label)
    }

    function requestClonePreset() {
        if (!presetPreview || !presetPreview.preset_payload) {
            actionStatus = qsTr("Brak danych podglądu do zapisania")
            return
        }
        pendingClonePayload = JSON.parse(JSON.stringify(presetPreview.preset_payload))
        var defaultName = ""
        if (presetPreview.preset)
            defaultName = presetPreview.preset.name || presetPreview.preset.slug || ""
        if (!defaultName || defaultName.length === 0)
            defaultName = qsTr("Nowy preset")
        cloneNameField.text = defaultName + " (kopia)"
        cloneDialog.open()
    }

    function saveClonePreset(name) {
        if (!pendingClonePayload) {
            actionStatus = qsTr("Brak danych presetu do zapisania")
            return
        }
        if (!runtimeService || !runtimeService.saveStrategyPreset) {
            actionStatus = qsTr("Mostek runtime nie obsługuje zapisu presetów")
            return
        }
        var payload = JSON.parse(JSON.stringify(pendingClonePayload))
        var newName = name && name.length > 0 ? name : (payload.name || qsTr("Nowy preset"))
        payload.name = newName
        if (payload.metadata === undefined || payload.metadata === null)
            payload.metadata = {}
        payload.metadata.cloned_from = presetPreview && presetPreview.preset ? (presetPreview.preset.slug || presetPreview.preset.name || "") : ""
        payload.slug = ""
        payload.id = ""
        var response = runtimeService.saveStrategyPreset(payload)
        if (!response || response.ok === false) {
            actionStatus = response && response.error ? response.error : qsTr("Nie udało się zapisać nowego presetu")
            return
        }
        actionStatus = qsTr("Zapisano nowy preset: %1").arg(response.name || newName)
        pendingClonePayload = null
        cloneDialog.close()
        refreshPresets()
    }

    function loadPreset(entry) {
        if (!runtimeService || !runtimeService.loadStrategyPreset) {
            actionStatus = qsTr("Mostek runtime nie obsługuje wczytywania presetów")
            return
        }
        if (!entry) {
            actionStatus = qsTr("Nie wybrano presetu")
            return
        }
        var request = {}
        if (entry.slug)
            request.slug = entry.slug
        if (entry.path)
            request.path = entry.path
        var response = runtimeService.loadStrategyPreset(request)
        if (!response || response.error) {
            actionStatus = response && response.error ? response.error : qsTr("Nie udało się wczytać presetu")
            return
        }
        var label = entry.label || entry.name || entry.slug || qsTr("preset")
        actionStatus = qsTr("Wczytano preset: %1").arg(label)
    }

    function deletePreset(entry) {
        if (!runtimeService || !runtimeService.deleteStrategyPreset) {
            actionStatus = qsTr("Mostek runtime nie obsługuje usuwania presetów")
            return
        }
        if (!entry) {
            actionStatus = qsTr("Nie wybrano presetu do usunięcia")
            return
        }
        var request = {}
        if (entry.slug)
            request.slug = entry.slug
        if (entry.path)
            request.path = entry.path
        var response = runtimeService.deleteStrategyPreset(request)
        if (!response || response.error) {
            actionStatus = response && response.error ? response.error : qsTr("Nie udało się usunąć presetu")
            return
        }
        refreshPresets()
        var label = entry.label || entry.name || entry.slug || qsTr("preset")
        actionStatus = qsTr("Usunięto preset: %1").arg(label)
    }

    function refreshChampion() {
        if (reportController && reportController.refreshChampionOverview)
            reportController.refreshChampionOverview()
    }

    function refreshReports() {
        if (reportController && reportController.refresh)
            reportController.refresh()
    }

    function extractReportMetrics(summary) {
        if (!summary || typeof summary !== "object")
            return []

        var metricsSource = null
        if (summary.metrics && typeof summary.metrics === "object")
            metricsSource = summary.metrics
        else if (summary.performance && typeof summary.performance === "object")
            metricsSource = summary.performance
        else if (summary.summary && typeof summary.summary === "object")
            metricsSource = summary.summary

        if (!metricsSource)
            return []

        var prioritized = [
            { key: "score", label: qsTr("Score"), isPercent: false },
            { key: "directional_accuracy", label: qsTr("Skuteczność"), isPercent: true },
            { key: "mae", label: qsTr("MAE"), isPercent: false },
            { key: "rmse", label: qsTr("RMSE"), isPercent: false },
            { key: "net_return_pct", label: qsTr("Zwrot"), isPercent: true },
            { key: "max_drawdown_pct", label: qsTr("Max DD"), isPercent: true },
            { key: "sharpe_ratio", label: qsTr("Sharpe"), isPercent: false },
            { key: "sortino_ratio", label: qsTr("Sortino"), isPercent: false }
        ]

        var results = []
        for (var idx = 0; idx < prioritized.length; ++idx) {
            var entry = prioritized[idx]
            if (metricsSource.hasOwnProperty(entry.key)) {
                results.push({
                    label: entry.label,
                    value: metricsSource[entry.key],
                    isPercent: entry.isPercent
                })
            }
        }

        if (results.length === 0) {
            var keys = Object.keys(metricsSource)
            keys.sort()
            for (var i = 0; i < keys.length && results.length < 3; ++i) {
                var key = keys[i]
                var rawValue = metricsSource[key]
                if (rawValue === undefined || rawValue === null)
                    continue
                if (typeof rawValue === "object")
                    continue
                results.push({
                    label: key,
                    value: rawValue,
                    isPercent: key.indexOf("_pct") >= 0
                })
            }
        }

        return results
    }

    function previewArchiveReportsAction() {
        if (!reportController || !reportController.previewArchiveReports) {
            actionStatus = qsTr("Mostek raportów nie obsługuje podglądu archiwizacji")
            return
        }
        actionStatus = qsTr("Przygotowywanie podglądu archiwizacji raportów…")
        resetArchivePreview()
        reportController.previewArchiveReports()
    }

    function archiveReportsAction() {
        if (!reportController || !reportController.archiveReports) {
            actionStatus = qsTr("Mostek raportów nie obsługuje archiwizacji")
            return
        }
        actionStatus = qsTr("Rozpoczęto archiwizację raportów dla bieżących filtrów")
        reportController.archiveReports()
    }

    function openReportLocation(entry) {
        if (!reportController || !reportController.openReportLocation) {
            actionStatus = qsTr("Mostek raportów nie obsługuje otwierania lokalizacji")
            return
        }
        if (!entry || !entry.relative_path) {
            actionStatus = qsTr("Brak ścieżki raportu do otwarcia")
            return
        }
        var ok = reportController.openReportLocation(entry.relative_path)
        if (!ok)
            actionStatus = reportController.lastError ? reportController.lastError : qsTr("Nie udało się otworzyć raportu")
    }

    function startPromotion(modelName, version, defaultReason) {
        if (!reportController || !reportController.promoteChampion) {
            actionStatus = qsTr("Mostek raportów nie obsługuje promocji championów")
            return
        }
        if (!modelName || !version) {
            actionStatus = qsTr("Brak danych challengera do promocji")
            return
        }
        pendingPromotionModel = modelName
        pendingPromotionVersion = version
        pendingPromotionReason = defaultReason || qsTr("Ręczna promocja challengera %1").arg(version)
        promotionDialog.open()
    }

    Connections {
        target: reportController
        ignoreUnknownSignals: true

        function onChampionOverviewChanged() {
            championSummary = reportController && reportController.championOverview ? reportController.championOverview : ({})
        }

        function onOverviewStatsChanged() {
            reportsSummary = reportController && reportController.overviewStats ? reportController.overviewStats : ({})
        }

        function onReportsChanged() {
            if (!reportController || !reportController.reports) {
                championReports = []
                selectedReportPath = ""
                return
            }
            championReports = reportController.reports
            if (!selectedReportPath || selectedReportPath.length === 0) {
                if (championReports.length > 0)
                    selectedReportPath = championReports[0].relative_path || ""
            } else {
                var exists = false
                for (var i = 0; i < championReports.length; ++i) {
                    if (championReports[i] && championReports[i].relative_path === selectedReportPath) {
                        exists = true
                        break
                    }
                }
                if (!exists && championReports.length > 0)
                    selectedReportPath = championReports[0].relative_path || ""
                if (!exists && championReports.length === 0)
                    selectedReportPath = ""
            }
        }

        function onLastErrorChanged() {
            if (reportController && reportController.lastError)
                actionStatus = reportController.lastError
        }

        function onLastNotificationChanged() {
            if (reportController && reportController.lastNotification)
                actionStatus = reportController.lastNotification
        }

        function onArchivePreviewReady(destination, overwrite, format, result) {
            archivePreviewInfo = result || {}
            if (!result) {
                actionStatus = qsTr("Podgląd archiwizacji nie zwrócił danych")
                return
            }
            var status = result.status || ""
            if (status === "error") {
                actionStatus = result.error ? result.error : qsTr("Nie udało się przygotować podglądu archiwizacji")
                return
            }
            if (status === "empty") {
                actionStatus = qsTr("Brak raportów do archiwizacji dla aktywnych filtrów")
                return
            }
            if (status === "preview") {
                var destinationLabel = ""
                if (destination && destination.length > 0)
                    destinationLabel = destination
                else if (reportController && reportController.defaultArchiveDestination)
                    destinationLabel = reportController.defaultArchiveDestination()
                else
                    destinationLabel = qsTr("domyślny katalog")
                var planned = 0
                if (result.copied_count !== undefined && result.copied_count !== null)
                    planned = result.copied_count
                else if (result.planned_count !== undefined && result.planned_count !== null)
                    planned = result.planned_count
                actionStatus = qsTr("Podgląd: %1 raport(ów) trafi do „%2” (%3 plików, %4 katalogów)").arg(planned)
                    .arg(destinationLabel)
                    .arg(result.copied_files !== undefined ? result.copied_files : 0)
                    .arg(result.copied_directories !== undefined ? result.copied_directories : 0)
                return
            }
            actionStatus = qsTr("Podgląd archiwizacji zakończony statusem: %1").arg(status)
        }

        function onArchiveFinished(success) {
            if (success) {
                if (reportController && reportController.lastNotification)
                    actionStatus = reportController.lastNotification
                else
                    actionStatus = qsTr("Archiwizacja raportów zakończona pomyślnie")
            } else {
                if (reportController && reportController.lastError)
                    actionStatus = reportController.lastError
                else
                    actionStatus = qsTr("Archiwizacja raportów zakończyła się błędem")
            }
        }
    }

    Component.onCompleted: {
        refreshPresets()
        refreshChampion()
        refreshReports()
        if (reportController && reportController.reports)
            championReports = reportController.reports
    }

    Dialog {
        id: promotionDialog
        modal: true
        focus: true
        title: qsTr("Promocja championa: %1 (%2)").arg(pendingPromotionModel || qsTr("model")).arg(pendingPromotionVersion || qsTr("wersja"))
        standardButtons: Dialog.Ok | Dialog.Cancel
        onOpened: reasonField.text = pendingPromotionReason
        onAccepted: {
            if (reportController && reportController.promoteChampion) {
                reportController.promoteChampion(pendingPromotionModel, pendingPromotionVersion, reasonField.text)
            }
        }
        onClosed: {
            pendingPromotionModel = ""
            pendingPromotionVersion = ""
            pendingPromotionReason = ""
            reasonField.text = ""
        }

        contentItem: ColumnLayout {
            spacing: 8
            width: Math.max(360, implicitWidth)

            Label {
                text: qsTr("Potwierdź awans challengera do roli championa.")
                wrapMode: Text.WordWrap
            }

            TextField {
                id: reasonField
                Layout.fillWidth: true
                placeholderText: qsTr("Uzasadnienie promocji")
                selectByMouse: true
            }
        }

        footer: DialogButtonBox {
            standardButtons: Dialog.Ok | Dialog.Cancel
            enabled: !(reportController && reportController.busy)
            onAccepted: promotionDialog.accept()
            onRejected: promotionDialog.reject()
        }
    }

    Dialog {
        id: cloneDialog
        objectName: "cloneDialog"
        modal: true
        focus: true
        title: qsTr("Zapisz jako nowy preset")
        standardButtons: Dialog.Ok | Dialog.Cancel
        onAccepted: saveClonePreset(cloneNameField.text)
        onRejected: {
            pendingClonePayload = null
            cloneNameField.text = ""
        }

        contentItem: ColumnLayout {
            spacing: 8
            width: Math.max(360, implicitWidth)

            Label {
                text: qsTr("Podaj nazwę nowego presetu przygotowanego na podstawie wybranego porównania.")
                wrapMode: Text.WordWrap
            }

            TextField {
                id: cloneNameField
                objectName: "cloneNameField"
                Layout.fillWidth: true
                placeholderText: qsTr("Nazwa nowego presetu")
                selectByMouse: true
            }
        }

        footer: DialogButtonBox {
            standardButtons: Dialog.Ok | Dialog.Cancel
            onAccepted: cloneDialog.accept()
            onRejected: cloneDialog.reject()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        GroupBox {
            id: indicatorsGroup
            title: qsTr("Wskaźniki automatyzacji i alokacji")
            Layout.fillWidth: true
            Layout.preferredHeight: 280

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: runtimeSnapshotTimestamp && runtimeSnapshotTimestamp.length > 0
                              ? qsTr("Ostatnia migawka: %1").arg(formatTimestamp(runtimeSnapshotTimestamp))
                              : qsTr("Migawka runtime nie jest dostępna")
                        color: palette.mid
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: {
                            var strategySection = performanceIndicators && performanceIndicators.strategy ? performanceIndicators.strategy : {}
                            var state = strategySection.state || qsTr("baseline")
                            return qsTr("Stan strategii: %1").arg(state)
                        }
                    }
                    Button {
                        text: qsTr("Odśwież")
                        icon.name: "view-refresh"
                        onClicked: refreshRuntimeIndicators()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 24

                    ColumnLayout {
                        spacing: 2
                        Label { text: qsTr("Rolling P&L"); font.bold: true }
                        Label {
                            text: performanceIndicators && performanceIndicators.rolling_pnl !== undefined && performanceIndicators.rolling_pnl !== null
                                  ? formatNumber(performanceIndicators.rolling_pnl, 2)
                                  : "—"
                            font.pointSize: font.pointSize + 1
                        }
                        Label {
                            text: qsTr("Max DD: %1").arg(
                                      performanceIndicators && performanceIndicators.max_drawdown_pct !== undefined && performanceIndicators.max_drawdown_pct !== null
                                      ? formatPercent(performanceIndicators.max_drawdown_pct, 2) : "—")
                            color: palette.mid
                        }
                        Label {
                            text: qsTr("Win rate: %1").arg(
                                      performanceIndicators && performanceIndicators.win_rate !== undefined && performanceIndicators.win_rate !== null
                                      ? formatPercent(performanceIndicators.win_rate, 2) : "—")
                            color: palette.mid
                        }
                    }

                    ColumnLayout {
                        spacing: 2
                        Label { text: qsTr("Parametry strategii"); font.bold: true }
                        Label {
                            text: {
                                var strategySection = performanceIndicators && performanceIndicators.strategy ? performanceIndicators.strategy : {}
                                var name = strategySection.current || qsTr("brak")
                                return qsTr("Strategia: %1").arg(name)
                            }
                        }
                        Label {
                            text: {
                                var strategySection = performanceIndicators && performanceIndicators.strategy ? performanceIndicators.strategy : {}
                                var leverage = strategySection.leverage !== undefined && strategySection.leverage !== null ? formatNumber(strategySection.leverage, 2) : "—"
                                return qsTr("Lewar: %1").arg(leverage)
                            }
                            color: palette.mid
                        }
                        Label {
                            text: {
                                var strategySection = performanceIndicators && performanceIndicators.strategy ? performanceIndicators.strategy : {}
                                var stopLoss = strategySection.stop_loss_pct !== undefined && strategySection.stop_loss_pct !== null ? formatPercent(strategySection.stop_loss_pct, 2) : "—"
                                var takeProfit = strategySection.take_profit_pct !== undefined && strategySection.take_profit_pct !== null ? formatPercent(strategySection.take_profit_pct, 2) : "—"
                                return qsTr("SL %1 • TP %2").arg(stopLoss).arg(takeProfit)
                            }
                            color: palette.mid
                        }
                    }

                    ColumnLayout {
                        spacing: 2
                        Label { text: qsTr("Alokacja giełd"); font.bold: true }
                        Label {
                            text: {
                                var selected = exchangeSelected || {}
                                var exchange = selected.exchange || qsTr("brak")
                                var segment = selected.segment ? selected.segment : "default"
                                return qsTr("Giełda: %1 / %2").arg(exchange).arg(segment)
                            }
                        }
                        Label {
                            text: {
                                var selected = exchangeSelected || {}
                                if (selected && selected.allocation !== undefined && selected.allocation !== null)
                                    return qsTr("Alokacja: %1").arg(formatPercent(selected.allocation, 2))
                                return qsTr("Alokacja: —")
                            }
                            color: palette.mid
                        }
                        Label {
                            text: {
                                var selected = exchangeSelected || {}
                                if (selected && selected.weight !== undefined && selected.weight !== null)
                                    return qsTr("Waga: %1").arg(formatNumber(selected.weight, 2))
                                return qsTr("Waga: —")
                            }
                            color: palette.mid
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 12

                    Frame {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 6

                            Label { text: qsTr("Historia adaptacji strategii"); font.bold: true }

                            ListView {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                clip: true
                                spacing: 6
                                model: strategyHistory
                                delegate: ColumnLayout {
                                    width: parent ? parent.width : implicitWidth
                                    spacing: 2
                                    Label {
                                        text: {
                                            var state = modelData.state || modelData.reason || ""
                                            var ts = modelData.timestamp ? formatTimestamp(modelData.timestamp) : qsTr("brak")
                                            return qsTr("%1 • %2").arg(ts).arg(state)
                                        }
                                        font.weight: Font.Medium
                                    }
                                    Label {
                                        text: {
                                            var strategy = modelData.strategy || qsTr("brak")
                                            var reason = modelData.reason || qsTr("brak powodu")
                                            return qsTr("Strategia: %1 – %2").arg(strategy).arg(reason)
                                        }
                                        color: indicatorsGroup.palette.mid
                                        elide: Text.ElideRight
                                    }
                                    Label {
                                        text: {
                                            var pnl = modelData.rolling_pnl !== undefined && modelData.rolling_pnl !== null ? formatNumber(modelData.rolling_pnl, 2) : "—"
                                            var winRate = modelData.win_rate !== undefined && modelData.win_rate !== null ? formatPercent(modelData.win_rate, 2) : "—"
                                            var drawdown = modelData.max_drawdown_pct !== undefined && modelData.max_drawdown_pct !== null ? formatPercent(modelData.max_drawdown_pct, 2) : "—"
                                            return qsTr("P&L %1 • Win %2 • DD %3").arg(pnl).arg(winRate).arg(drawdown)
                                        }
                                        color: indicatorsGroup.palette.mid
                                    }
                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 1
                                        color: Qt.darker(indicatorsGroup.palette.window, 1.2)
                                        visible: index < strategyHistory.length - 1
                                    }
                                }
                            }
                        }
                    }

                    Frame {
                        Layout.preferredWidth: 340
                        Layout.fillHeight: true

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 6

                            Label { text: qsTr("Aktualne wagi giełd"); font.bold: true }
                            ListView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 120
                                clip: true
                                model: exchangeAllocations
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Label {
                                        Layout.fillWidth: true
                                        text: {
                                            var exchange = modelData.exchange || qsTr("brak")
                                            var segment = modelData.segment ? modelData.segment : "default"
                                            return exchange + " / " + segment
                                        }
                                    }
                                    Label {
                                        text: formatPercent(modelData.allocation !== undefined && modelData.allocation !== null ? modelData.allocation : 0, 2)
                                        color: indicatorsGroup.palette.mid
                                    }
                                }
                            }

                            Label {
                                text: qsTr("Historia przełączeń")
                                font.bold: true
                                visible: exchangeHistory.length > 0
                            }
                            ListView {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                clip: true
                                visible: exchangeHistory.length > 0
                                model: exchangeHistory
                                delegate: ColumnLayout {
                                    width: parent ? parent.width : implicitWidth
                                    spacing: 2
                                    Label {
                                        text: {
                                            var ts = modelData.timestamp ? formatTimestamp(modelData.timestamp) : qsTr("brak")
                                            var exchange = modelData.exchange || qsTr("brak")
                                            var segment = modelData.segment ? modelData.segment : "default"
                                            return qsTr("%1 • %2 / %3").arg(ts).arg(exchange).arg(segment)
                                        }
                                        font.weight: Font.Medium
                                    }
                                    Label {
                                        text: {
                                            var allocation = modelData.allocation !== undefined && modelData.allocation !== null ? formatPercent(modelData.allocation, 2) : "—"
                                            var weight = modelData.weight !== undefined && modelData.weight !== null ? formatNumber(modelData.weight, 2) : "—"
                                            return qsTr("Alokacja %1 • Waga %2").arg(allocation).arg(weight)
                                        }
                                        color: indicatorsGroup.palette.mid
                                    }
                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 1
                                        color: Qt.darker(indicatorsGroup.palette.window, 1.2)
                                        visible: index < exchangeHistory.length - 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            GroupBox {
                title: qsTr("Status modeli (Champion / Challenger)")
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            text: qsTr("Odśwież status")
                            icon.name: "view-refresh"
                            enabled: reportController && reportController.refreshChampionOverview
                            onClicked: refreshChampion()
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: championSummary && championSummary.base_directory
                                  ? qsTr("Katalog: %1").arg(championSummary.base_directory)
                                  : qsTr("Domyślna ścieżka jakości modeli")
                            color: palette.mid
                            font.pointSize: font.pointSize - 1
                            Layout.alignment: Qt.AlignVCenter
                        }
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true

                        ListView {
                            id: championList
                            anchors.fill: parent
                            spacing: 10
                            model: championSummary && championSummary.models ? championSummary.models : []
                            delegate: Frame {
                                Layout.fillWidth: true
                                property var championData: modelData || ({})
                                background: Rectangle {
                                    color: Qt.darker(palette.window, 1.05)
                                    radius: 6
                                }
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 6

                                    Label {
                                        text: {
                                            var champion = championData.champion || {}
                                            var version = champion.version || qsTr("brak")
                                            var modelName = championData.model_name || qsTr("model")
                                            return qsTr("Model %1 – champion: %2").arg(modelName).arg(version)
                                        }
                                        font.bold: true
                                    }

                                    Label {
                                        text: {
                                            var meta = championData.champion_metadata || {}
                                            var reason = meta.reason || qsTr("bez uzasadnienia")
                                            var decided = meta.decided_at ? Qt.formatDateTime(new Date(meta.decided_at), Qt.DefaultLocaleShortDate) : qsTr("brak daty")
                                            return qsTr("Aktualizacja: %1 • %2").arg(decided).arg(reason)
                                        }
                                        color: palette.mid
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        visible: championData.challengers && championData.challengers.length > 0
                                        spacing: 8

                                        Button {
                                            text: qsTr("Przywróć poprzedniego championa")
                                            icon.name: "go-previous"
                                            enabled: reportController && reportController.promoteChampion
                                            onClicked: {
                                                var challenger = championData.challengers && championData.challengers.length > 0 ? championData.challengers[0] : null
                                                var report = challenger && challenger.report ? challenger.report : {}
                                                var version = report.version || ""
                                                if (!version) {
                                                    actionStatus = qsTr("Brak wersji do przywrócenia")
                                                    return
                                                }
                                                startPromotion(championData.model_name || "", version, qsTr("Przywrócenie championa %1").arg(version))
                                            }
                                        }

                                        Item { Layout.fillWidth: true }
                                    }

                                    Repeater {
                                        model: championData.challengers || []
                                        delegate: ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 2

                                            Rectangle {
                                                Layout.fillWidth: true
                                                height: 1
                                                color: Qt.darker(palette.window, 1.2)
                                            }

                                            Label {
                                                text: {
                                                    var challenger = modelData.report || {}
                                                    var version = challenger.version || qsTr("n/a")
                                                    return qsTr("Challenger: %1").arg(version)
                                                }
                                                font.weight: Font.Medium
                                            }

                                            Label {
                                                text: {
                                                    var meta = modelData.metadata || {}
                                                    var decided = meta.decided_at ? Qt.formatDateTime(new Date(meta.decided_at), Qt.DefaultLocaleShortDate) : qsTr("brak daty")
                                                    var reason = meta.reason || qsTr("brak uzasadnienia")
                                                    return qsTr("• %1 – %2").arg(decided).arg(reason)
                                                }
                                                color: palette.mid
                                            }

                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                Button {
                                                    text: qsTr("Promuj")
                                                    icon.name: "go-up"
                                                    enabled: reportController && reportController.promoteChampion
                                                    onClicked: {
                                                        var report = modelData.report || {}
                                                        var version = report.version || ""
                                                        if (!version) {
                                                            actionStatus = qsTr("Brak wersji challengera")
                                                            return
                                                        }
                                                        startPromotion(championData.model_name || "", version, qsTr("Ręczna promocja challengera %1").arg(version))
                                                    }
                                                }
                                                Item { Layout.fillWidth: true }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            GroupBox {
                title: qsTr("Biblioteka presetów strategii")
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            text: qsTr("Odśwież")
                            icon.name: "view-refresh"
                            onClicked: refreshPresets()
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: qsTr("Łącznie: %1").arg(savedPresets.length)
                            color: palette.mid
                        }
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true

                        ListView {
                            id: presetList
                            objectName: "presetList"
                            anchors.fill: parent
                            spacing: 8
                            model: savedPresets
                            onCurrentIndexChanged: {
                                selectedPresetIndex = currentIndex
                                if (currentIndex >= 0 && currentIndex < savedPresets.length)
                                    previewPresetEntry(savedPresets[currentIndex])
                            }
                            delegate: Frame {
                                Layout.fillWidth: true
                                property var presetData: modelData || ({})
                                background: Rectangle {
                                    color: Qt.darker(palette.window, 1.08)
                                    radius: 6
                                }
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 10
                                    spacing: 6

                                    Label {
                                        text: presetData.label || presetData.name || presetData.slug || qsTr("Preset bez nazwy")
                                        font.bold: true
                                    }
                                    Label {
                                        text: presetData.description || presetData.source || presetData.path || qsTr("Lokalny preset")
                                        color: palette.mid
                                        wrapMode: Text.Wrap
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 8
                                        Button {
                                            text: qsTr("Wczytaj")
                                            icon.name: "document-open"
                                            onClicked: loadPreset(presetData)
                                        }
                                        Button {
                                            text: qsTr("Porównaj")
                                            icon.name: "view-list-details"
                                            onClicked: {
                                                presetList.currentIndex = index
                                                previewPresetEntry(presetData)
                                            }
                                        }
                                        Button {
                                            text: qsTr("Usuń")
                                            icon.name: "edit-delete"
                                            onClicked: deletePreset(presetData)
                                        }
                                        Item { Layout.fillWidth: true }
                                        Label {
                                            text: presetData.updated_at ? Qt.formatDateTime(new Date(presetData.updated_at), Qt.DefaultLocaleShortDate) : ""
                                            color: palette.mid
                                            visible: !!presetData.updated_at
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Label {
                        text: actionStatus
                        color: palette.highlight
                        visible: actionStatus && actionStatus.length > 0
                        wrapMode: Text.Wrap
                    }

                    GroupBox {
                        title: qsTr("Porównanie z championem")
                        Layout.fillWidth: true
                        visible: presetPreview && presetPreview.ok

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 6

                            Label {
                                text: {
                                    if (!presetPreview || !presetPreview.champion)
                                        return qsTr("Brak danych champion")
                                    var version = presetPreview.champion.version || qsTr("brak")
                                    return qsTr("Champion: wersja %1").arg(version)
                                }
                                font.bold: true
                            }

                            Label {
                                text: {
                                    var status = presetPreview && presetPreview.validation ? (presetPreview.validation.status || "") : ""
                                    if (!status)
                                        status = qsTr("brak informacji o walidacji")
                                    return qsTr("Status walidacji: %1").arg(status)
                                }
                                color: palette.mid
                            }

                            Repeater {
                                model: presetPreview && presetPreview.diff ? presetPreview.diff : []
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        Layout.fillWidth: true
                                        text: modelData.label || modelData.parameter
                                    }

                                    Label {
                                        Layout.preferredWidth: 120
                                        horizontalAlignment: Text.AlignRight
                                        text: formatDiffValue(modelData.preset_value, modelData.is_percent)
                                    }

                                    Label {
                                        Layout.preferredWidth: 120
                                        horizontalAlignment: Text.AlignRight
                                        text: formatDiffValue(modelData.champion_value, modelData.is_percent)
                                        color: palette.mid
                                    }

                                    Label {
                                        Layout.preferredWidth: 90
                                        horizontalAlignment: Text.AlignRight
                                        text: {
                                            var delta = modelData.delta
                                            if (delta === undefined || delta === null || isNaN(delta))
                                                return ""
                                            var formatted = formatDiffValue(delta, modelData.is_percent)
                                            return delta > 0 ? "+" + formatted : formatted
                                        }
                                        color: modelData.delta > 0 ? palette.highlight : palette.mid
                                    }
                                }
                            }

                            Label {
                                text: {
                                    if (!presetPreview || !presetPreview.simulation)
                                        return ""
                                    var net = presetPreview.simulation.net_return_pct
                                    var dd = presetPreview.simulation.max_drawdown_pct
                                    if (net === undefined && dd === undefined)
                                        return ""
                                    var netLabel = net !== undefined ? formatDiffValue(net, true) : qsTr("n/d")
                                    var ddLabel = dd !== undefined ? formatDiffValue(dd, true) : qsTr("n/d")
                                    return qsTr("Symulacja: zwrot %1 • max DD %2").arg(netLabel).arg(ddLabel)
                                }
                                color: palette.mid
                            }

                            RowLayout {
                                Layout.fillWidth: true

                                Button {
                                    text: qsTr("Zapisz jako nowy preset")
                                    icon.name: "document-save"
                                    enabled: presetPreview && presetPreview.preset_payload && runtimeService && runtimeService.saveStrategyPreset
                                    onClicked: requestClonePreset()
                                }
                                Item { Layout.fillWidth: true }
                            }
                        }
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Raporty i automatyzacja jakości")
            Layout.fillWidth: true
            Layout.preferredHeight: 260
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        text: qsTr("Odśwież raporty")
                        icon.name: "view-refresh"
                        enabled: reportController && reportController.refresh
                        onClicked: refreshReports()
                    }
                    Button {
                        text: qsTr("Podgląd archiwizacji")
                        icon.name: "document-preview"
                        enabled: reportController && reportController.previewArchiveReports && !(reportController && reportController.busy)
                        onClicked: previewArchiveReportsAction()
                    }
                    Button {
                        text: qsTr("Archiwizuj")
                        icon.name: "document-save"
                        enabled: reportController && reportController.archiveReports && !(reportController && reportController.busy)
                        onClicked: archiveReportsAction()
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: reportController && reportController.busy ? qsTr("Przetwarzanie...") : ""
                        color: palette.mid
                        visible: reportController && reportController.busy
                    }
                }

                Label {
                    text: {
                        if (!reportsSummary)
                            return qsTr("Brak danych podsumowania")
                        var total = reportsSummary.total_reports || reportsSummary.total_count || 0
                        var categories = reportsSummary.categories ? reportsSummary.categories.length : 0
                        return qsTr("Łącznie raportów: %1 • Kategorie: %2").arg(total).arg(categories)
                    }
                    wrapMode: Text.Wrap
                }

                Label {
                    text: reportController && reportController.lastNotification ? reportController.lastNotification : ""
                    visible: reportController && reportController.lastNotification
                    color: palette.mid
                    wrapMode: Text.Wrap
                }

                Label {
                    Layout.fillWidth: true
                    text: qsTr("Brak raportów championów w katalogu jakości")
                    visible: championReports.length === 0 && !(reportController && reportController.busy)
                    color: palette.mid
                    wrapMode: Text.Wrap
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    visible: championReports.length > 0

                    ListView {
                        id: reportsList
                        anchors.fill: parent
                        spacing: 10
                        clip: true
                        model: championReports

                        delegate: Frame {
                            Layout.fillWidth: true
                            property var reportData: modelData || ({})
                            property bool isSelected: reportData.relative_path && reportData.relative_path === selectedReportPath
                            background: Rectangle {
                                color: isSelected ? Qt.darker(palette.highlight, 1.5) : Qt.darker(palette.window, 1.05)
                                radius: 6
                                border.color: isSelected ? palette.highlight : Qt.darker(palette.window, 1.2)
                                border.width: isSelected ? 2 : 1
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 6

                                Label {
                                    text: reportData.display_name || reportData.relative_path || qsTr("Raport jakości")
                                    font.bold: true
                                }

                                Label {
                                    text: {
                                        var category = reportData.category || qsTr("brak kategorii")
                                        var updated = reportData.updated_at ? Qt.formatDateTime(new Date(reportData.updated_at), Qt.DefaultLocaleShortDate) : qsTr("brak daty")
                                        return qsTr("Kategoria: %1 • Aktualizacja: %2").arg(category).arg(updated)
                                    }
                                    color: palette.mid
                                }

                                Flow {
                                    id: metricFlow
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Repeater {
                                        id: metricRepeater
                                        model: extractReportMetrics(reportData.summary)
                                        delegate: Rectangle {
                                            radius: 4
                                            color: Qt.darker(palette.window, 1.15)
                                            border.color: Qt.darker(palette.window, 1.3)
                                            border.width: 1
                                            anchors.verticalCenter: undefined
                                            implicitHeight: metricLabel.implicitHeight + 8
                                            implicitWidth: metricLabel.implicitWidth + 12

                                            Label {
                                                id: metricLabel
                                                anchors.centerIn: parent
                                                text: (modelData.label || "") + ": " + formatDiffValue(modelData.value, modelData.isPercent)
                                                font.pointSize: font.pointSize - 1
                                            }
                                        }
                                    }
                                    visible: metricRepeater.count > 0
                                }

                                Label {
                                    text: {
                                        if (!reportData.summary || !reportData.summary.report_date)
                                            return ""
                                        return qsTr("Data raportu: %1").arg(reportData.summary.report_date)
                                    }
                                    color: palette.mid
                                    visible: reportData.summary && reportData.summary.report_date
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    Button {
                                        text: qsTr("Otwórz lokalizację")
                                        icon.name: "folder"
                                        onClicked: openReportLocation(reportData)
                                    }
                                    Item { Layout.fillWidth: true }
                                    Label {
                                        text: reportData.summary && reportData.summary.status ? reportData.summary.status : ""
                                        visible: reportData.summary && reportData.summary.status
                                        color: palette.mid
                                    }
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: selectedReportPath = reportData.relative_path || ""
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                            }
                        }
                    }
                }
            }
        }
    }
}
