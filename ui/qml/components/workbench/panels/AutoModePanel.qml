import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../.." as Components

Frame {
    id: root
    objectName: "autoModePanel"

    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var automation: ({ enabled: false, running: false, trusted: false })
    property var metrics: ({})
    property var strategyInfo: ({})
    property var scheduleInfo: ({})
    property var guardrailSummary: ({})
    property var decisionSummary: ({})
    property var presets: []
    property var equityPoints: []
    property var heatmapCells: []
    property var reasons: []
    property var performanceSummary: ({})
    property var recentPerformanceSummary: ({})
    property var controllerHistory: []
    property var recalibrations: []
    property var riskProfile: ({})
    property var portfolio: ({})
    property var environment: ""
    property var symbol: ""
    property var alertDraft: ({ severity: "warning", maxDrawdown: 0.07, notifyOnSwitch: true })
    property var severityOptions: ["info", "warning", "error", "critical"]
    property int historyPreviewLimit: 6

    signal snapshotRefreshed()

    Layout.fillWidth: true
    Layout.columnSpan: 2

    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    function cloneObject(value) {
        try {
            return JSON.parse(JSON.stringify(value))
        } catch (err) {
            console.warn("AutoModePanel clone fallback", err)
            return value
        }
    }

    function applySnapshot(payload) {
        if (!payload)
            return
        automation = payload.automation || automation
        metrics = cloneObject(payload.metrics || {})
        strategyInfo = cloneObject(payload.strategy || {})
        scheduleInfo = cloneObject(payload.schedule || payload.scheduleInfo || {})
        guardrailSummary = cloneObject(payload.guardrailSummary || payload.guardrail_summary || {})
        decisionSummary = cloneObject(payload.decisionSummary || payload.decision_summary || {})
        presets = payload.presets || []
        equityPoints = payload.equityCurve || payload.equity_curve || []
        heatmapCells = payload.riskHeatmap || payload.risk_heatmap || []
        reasons = payload.reasons || []
        controllerHistory = cloneObject(payload.controllerHistory || payload.controller_history || [])
        recalibrations = cloneObject(payload.recalibrations || [])
        performanceSummary = payload.performance ? cloneObject(payload.performance) : {}
        recentPerformanceSummary = payload.performanceWindow ? cloneObject(payload.performanceWindow) :
                                   (payload.performance_window ? cloneObject(payload.performance_window) : {})
        riskProfile = payload.riskProfile || payload.risk_profile || null
        portfolio = payload.portfolio ? cloneObject(payload.portfolio) : null
        if (payload.environment !== undefined && payload.environment !== null)
            environment = payload.environment
        if (payload.symbol !== undefined && payload.symbol !== null)
            symbol = payload.symbol
        if (payload.alerts)
            alertDraft = cloneObject(payload.alerts)
        snapshotRefreshed()
    }

    function formatTimestamp(value) {
        if (!value)
            return qsTr("brak")
        var date = new Date(value)
        if (isNaN(date.getTime()))
            return value
        return Qt.formatDateTime(date, Qt.DefaultLocaleShortDate)
    }

    function formatDuration(seconds) {
        if (seconds === null || seconds === undefined)
            return "—"
        var total = Math.floor(Number(seconds))
        if (!isFinite(total))
            return "—"
        var hours = Math.floor(total / 3600)
        var minutes = Math.floor((total % 3600) / 60)
        var secs = Math.max(0, total % 60)
        var parts = []
        if (hours > 0)
            parts.push(qsTr("%1 h").arg(hours))
        if (minutes > 0)
            parts.push(qsTr("%1 min").arg(minutes))
        if (parts.length === 0 || secs > 0)
            parts.push(qsTr("%1 s").arg(secs))
        return parts.join(" ")
    }

    function formatBoolean(value) {
        if (value === null || value === undefined)
            return "—"
        return value ? qsTr("tak") : qsTr("nie")
    }

    function formatPercent(value) {
        if (value === null || value === undefined)
            return "—"
        return Number(value * 100).toLocaleString(Qt.locale(), "f", 1) + "%"
    }

    function formatNumber(value, digits) {
        if (value === null || value === undefined)
            return "—"
        var precision = digits !== undefined ? digits : 2
        return Number(value).toLocaleString(Qt.locale(), "f", precision)
    }

    function stringify(value) {
        if (value === null || value === undefined)
            return "—"
        if (typeof value === "object") {
            try {
                return JSON.stringify(value)
            } catch (err) {
                console.warn("AutoModePanel stringify", err)
                return value
            }
        }
        return value
    }

    function objectEntriesOrdered(obj) {
        if (!obj)
            return []
        var entries = []
        for (var key in obj) {
            if (!Object.prototype.hasOwnProperty.call(obj, key))
                continue
            entries.push({ key: key, value: obj[key] })
        }
        entries.sort(function(a, b) { return a.key.localeCompare(b.key) })
        return entries
    }

    function breakdownEntries(summary, key) {
        if (!summary)
            return []
        var map = summary[key]
        if (!map && key.indexOf("_") >= 0) {
            var camel = key.replace(/_([a-z])/g, function(match, letter) { return letter.toUpperCase() })
            map = summary[camel]
        }
        return objectEntriesOrdered(map)
    }

    function controllerHistoryPreview() {
        if (!controllerHistory || controllerHistory.length === 0)
            return []
        return controllerHistory.slice(0, historyPreviewLimit)
    }

    function recalibrationPreview() {
        if (!recalibrations || recalibrations.length === 0)
            return []
        return recalibrations.slice(0, historyPreviewLimit)
    }

    function scheduleWindowDescription(window) {
        if (!window)
            return "—"
        var mode = window.mode || scheduleInfo.mode || ""
        var start = formatTimestamp(window.start)
        var end = formatTimestamp(window.end)
        var label = window.label ? window.label + " • " : ""
        return label + qsTr("%1 → %2 (%3)").arg(start).arg(end).arg(mode || qsTr("brak"))
    }

    function refreshSnapshot() {
        if (!root.runtimeService || !root.runtimeService.autoModeSnapshot)
            return
        var payload = root.runtimeService.autoModeSnapshot()
        applySnapshot(payload)
    }

    function toggleAutomation(enabled) {
        automation = Object.assign({}, automation, { enabled: enabled })
        if (!root.runtimeService)
            return
        if (enabled) {
            if (root.runtimeService.startAutomation)
                root.runtimeService.startAutomation()
        } else {
            if (root.runtimeService.stopAutomation)
                root.runtimeService.stopAutomation()
        }
        if (root.runtimeService.toggleAutoMode)
            root.runtimeService.toggleAutoMode(enabled)
    }

    function updateAlertPreference(key, value) {
        var updated = cloneObject(alertDraft)
        updated[key] = value
        alertDraft = updated
        alertDebounce.restart()
    }

    Timer {
        id: refreshTimer
        interval: 6000
        repeat: true
        running: true
        onTriggered: refreshSnapshot()
    }

    Timer {
        id: alertDebounce
        interval: 450
        repeat: false
        onTriggered: {
            if (root.runtimeService && root.runtimeService.updateAlertPreferences)
                root.runtimeService.updateAlertPreferences(alertDraft)
        }
    }

    Component.onCompleted: refreshSnapshot()
    onRuntimeServiceChanged: refreshSnapshot()

    Connections {
        target: root.runtimeService
        function onAutomationStateChanged(running) {
            automation = Object.assign({}, automation, { running: running })
        }
        function onAlertPreferencesChanged(preferences) {
            if (!preferences)
                return
            alertDraft = cloneObject(preferences)
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Tryb auto-tradingu")
                font.pointSize: 15
                font.bold: true
            }

            Rectangle {
                width: 10
                height: width
                radius: width / 2
                color: automation.running ? "#27AE60" : "#7F8C8D"
                Layout.alignment: Qt.AlignVCenter
            }

            Label {
                text: automation.running ? qsTr("Aktywny") : (automation.enabled ? qsTr("Czeka na start") : qsTr("Wyłączony"))
                color: automation.running ? "#27AE60" : palette.text
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            Switch {
                id: automationSwitch
                text: checked ? qsTr("Automatyczny") : qsTr("Manualny")
                checked: automation.enabled
                onToggled: toggleAutomation(checked)
            }

            Button {
                text: qsTr("Odśwież")
                onClicked: refreshSnapshot()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Symbol: %1").arg(symbol ? symbol : qsTr("brak"))
                color: palette.mid
            }

            Label {
                text: qsTr("Środowisko: %1").arg(environment ? environment : qsTr("brak"))
                color: palette.mid
            }

            Label {
                visible: !!portfolio
                text: {
                    if (!portfolio)
                        return ""
                    if (typeof portfolio === "object") {
                        var identifier = portfolio.name || portfolio.id || portfolio.title
                        if (identifier)
                            return qsTr("Portfel: %1").arg(identifier)
                        try {
                            return qsTr("Portfel: %1").arg(JSON.stringify(portfolio))
                        } catch (err) {
                            console.warn("AutoModePanel portfolio stringify", err)
                        }
                    }
                    return qsTr("Portfel: %1").arg(portfolio)
                }
                color: palette.mid
            }

            Label {
                text: {
                    if (!riskProfile)
                        return qsTr("Profil ryzyka: brak")
                    if (typeof riskProfile === "object") {
                        var label = riskProfile.name || riskProfile.label || riskProfile.risk_profile
                        if (label)
                            return qsTr("Profil ryzyka: %1").arg(label)
                        try {
                            return qsTr("Profil ryzyka: %1").arg(JSON.stringify(riskProfile))
                        } catch (err) {
                            console.warn("AutoModePanel riskProfile stringify", err)
                        }
                    }
                    return qsTr("Profil ryzyka: %1").arg(riskProfile)
                }
                color: palette.mid
            }

            Item { Layout.fillWidth: true }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.EquityCurveDashboard {
                Layout.fillWidth: true
                Layout.preferredWidth: parent.width / 2
                Layout.fillHeight: true
                points: equityPoints
                title: qsTr("Krzywa kapitału (lokalnie)")
            }

            Components.AssetHeatmapDashboard {
                Layout.fillWidth: true
                Layout.preferredWidth: parent.width / 2
                Layout.fillHeight: true
                cells: heatmapCells
                title: qsTr("Heatmapa ryzyka (offline)")
            }
        }

        GroupBox {
            title: qsTr("Harmonogram automatyzacji")
            Layout.fillWidth: true

            GridLayout {
                columns: 2
                columnSpacing: 12
                rowSpacing: 6
                Layout.fillWidth: true

                Label { text: qsTr("Tryb") }
                Label { text: scheduleInfo.mode || "—" }

                Label { text: qsTr("Okno otwarte") }
                Label { text: formatBoolean(scheduleInfo.is_open) }

                Label { visible: !!scheduleInfo.window; text: qsTr("Bieżące okno") }
                Label {
                    visible: !!scheduleInfo.window
                    text: scheduleWindowDescription(scheduleInfo.window)
                    wrapMode: Text.WordWrap
                }

                Label { visible: scheduleInfo.time_until_transition_s !== undefined; text: qsTr("Do zmiany trybu") }
                Label {
                    visible: scheduleInfo.time_until_transition_s !== undefined
                    text: formatDuration(scheduleInfo.time_until_transition_s)
                    color: palette.mid
                }

                Label { visible: !!scheduleInfo.next_transition; text: qsTr("Następne przejście") }
                Label {
                    visible: !!scheduleInfo.next_transition
                    text: formatTimestamp(scheduleInfo.next_transition)
                    color: palette.mid
                }

                Label { visible: !!scheduleInfo.override; text: qsTr("Aktywne nadpisanie") }
                Label {
                    visible: !!scheduleInfo.override
                    text: scheduleWindowDescription(scheduleInfo.override)
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }

                Label { visible: !!scheduleInfo.next_override; text: qsTr("Następna nadpiska") }
                Label {
                    visible: !!scheduleInfo.next_override
                    text: scheduleWindowDescription(scheduleInfo.next_override)
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }

                Label { visible: scheduleInfo.override_active !== undefined; text: qsTr("Nadpisanie aktywne") }
                Label {
                    visible: scheduleInfo.override_active !== undefined
                    text: formatBoolean(scheduleInfo.override_active)
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Parametry strategii")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 6

                GridLayout {
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 6
                    Layout.fillWidth: true

                    Label { text: qsTr("Strategia aktywna") }
                    Label { text: strategyInfo.current || "—" }

                    Label { text: qsTr("Dźwignia") }
                    Label { text: strategyInfo.leverage !== undefined ? formatNumber(strategyInfo.leverage, 2) : "—" }

                    Label { text: qsTr("Stop loss") }
                    Label { text: strategyInfo.stop_loss_pct !== undefined ? formatPercent(strategyInfo.stop_loss_pct) : "—" }

                    Label { text: qsTr("Take profit") }
                    Label { text: strategyInfo.take_profit_pct !== undefined ? formatPercent(strategyInfo.take_profit_pct) : "—" }

                    Label { text: qsTr("Ostatni sygnał") }
                    Label {
                        text: stringify(strategyInfo.last_signal)
                        wrapMode: Text.WordWrap
                        color: palette.mid
                    }

                    Label { text: qsTr("Ostatni reżim") }
                    Label {
                        text: stringify(strategyInfo.last_regime)
                        wrapMode: Text.WordWrap
                        color: palette.mid
                    }
                }

                Label {
                    visible: !!strategyInfo.metadata_summary
                    text: strategyInfo.metadata_summary || ""
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }

                Label {
                    visible: !!strategyInfo.metadata && !strategyInfo.metadata_summary
                    text: stringify(strategyInfo.metadata)
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }

                ColumnLayout {
                    visible: Array.isArray(strategyInfo.recommendations) && strategyInfo.recommendations.length > 0
                    spacing: 4

                    Label {
                        text: qsTr("Rekomendacje")
                        font.bold: true
                    }

                    Repeater {
                        model: strategyInfo.recommendations || []
                        delegate: Label {
                            width: parent.width
                            text: typeof modelData === "object" ? stringify(modelData) : modelData
                            wrapMode: Text.WordWrap
                            color: palette.mid
                        }
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Podsumowanie guardraili")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 4

                Repeater {
                    model: objectEntriesOrdered(guardrailSummary)
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: modelData.key
                            font.bold: true
                            Layout.preferredWidth: 180
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: stringify(modelData.value)
                            color: palette.mid
                        }
                    }
                }

                Label {
                    visible: objectEntriesOrdered(guardrailSummary).length === 0
                    text: qsTr("Brak danych guardraili.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Podsumowanie decyzji ryzyka")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 4

                Repeater {
                    model: objectEntriesOrdered(decisionSummary)
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: modelData.key
                            font.bold: true
                            Layout.preferredWidth: 180
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: stringify(modelData.value)
                            color: palette.mid
                        }
                    }
                }

                Label {
                    visible: objectEntriesOrdered(decisionSummary).length === 0
                    text: qsTr("Brak zarejestrowanych decyzji.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Historia cykli kontrolera")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 6

                Repeater {
                    model: controllerHistoryPreview()
                    delegate: Frame {
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0, 0, 0, 0.08)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            Label {
                                text: qsTr("Cykl #%1").arg(modelData.sequence !== undefined ? modelData.sequence : index + 1)
                                font.bold: true
                            }

                            Label {
                                text: qsTr("Start: %1").arg(formatTimestamp(modelData.started_at))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Koniec: %1").arg(formatTimestamp(modelData.finished_at))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Czas trwania: %1").arg(formatDuration(modelData.duration_s))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Zlecenia: %1").arg(modelData.orders !== undefined ? modelData.orders : "—")
                                color: palette.mid
                            }

                            Label {
                                visible: !!modelData.signals
                                text: qsTr("Sygnały: %1").arg(stringify(modelData.signals))
                                wrapMode: Text.WordWrap
                                color: palette.mid
                            }

                            Label {
                                visible: !!modelData.results
                                text: qsTr("Wyniki: %1").arg(stringify(modelData.results))
                                wrapMode: Text.WordWrap
                                color: palette.mid
                            }
                        }
                    }
                }

                Label {
                    visible: controllerHistory.length === 0
                    text: qsTr("Brak historii cykli kontrolera.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Zaplanowane rekalkibracje")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 6

                Repeater {
                    model: recalibrationPreview()
                    delegate: Frame {
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0, 0, 0, 0.08)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            Label {
                                text: qsTr("Strategia: %1").arg(modelData.strategy || qsTr("nieznana"))
                                font.bold: true
                            }

                            Label {
                                text: qsTr("Następne uruchomienie: %1").arg(formatTimestamp(modelData.next_run))
                                color: palette.mid
                            }

                            Label {
                                visible: typeof modelData === "object" && Object.keys(modelData).length > 2
                                text: stringify(modelData)
                                wrapMode: Text.WordWrap
                                color: palette.mid
                            }
                        }
                    }
                }

                Label {
                    visible: recalibrations.length === 0
                    text: qsTr("Brak oczekujących rekalkibracji.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Presety automatyczne")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 6

                Repeater {
                    model: presets
                    delegate: Frame {
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0, 0, 0, 0.1)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            Label {
                                text: (modelData.name || modelData.id || qsTr("Preset")) + (modelData.version ? qsTr(" • v%1").arg(modelData.version) : "")
                                font.bold: true
                            }

                            Label {
                                visible: !!modelData.summary
                                text: modelData.summary || ""
                                wrapMode: Text.WordWrap
                                color: palette.mid
                            }

                            Flow {
                                width: parent.width
                                spacing: 4
                                Repeater {
                                    model: modelData.tags || []
                                    delegate: Rectangle {
                                        radius: 4
                                        color: Qt.rgba(0.2, 0.2, 0.25, 0.6)
                                        border.color: Qt.rgba(1, 1, 1, 0.08)
                                        height: label.implicitHeight + 6
                                        width: label.implicitWidth + 12
                                        Label {
                                            id: label
                                            anchors.centerIn: parent
                                            text: modelData
                                            font.pixelSize: 11
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Label {
                    visible: presets.length === 0
                    text: qsTr("Brak zainstalowanych presetów Marketplace.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Alerty auto-mode")
            Layout.fillWidth: true

            GridLayout {
                columns: 3
                columnSpacing: 12
                rowSpacing: 8
                Layout.fillWidth: true

                Label { text: qsTr("Minimalna ważność") }
                ComboBox {
                    Layout.fillWidth: true
                    model: root.severityOptions
                    currentIndex: Math.max(0, root.severityOptions.indexOf(alertDraft.severity || "warning"))
                    onActivated: updateAlertPreference("severity", root.severityOptions[index])
                }
                Item { Layout.fillWidth: true }

                Label { text: qsTr("Limit strat dziennych") }
                Slider {
                    id: drawdownSlider
                    Layout.fillWidth: true
                    from: 0.01
                    to: 0.2
                    stepSize: 0.005
                    value: alertDraft.maxDrawdown || 0.07
                    onValueChanged: updateAlertPreference("maxDrawdown", value)
                }
                Label {
                    text: Number(drawdownSlider.value * 100).toLocaleString(Qt.locale(), "f", 1) + "%"
                    color: palette.mid
                }

                Label { text: qsTr("Powiadomienia o zmianie trybu") }
                CheckBox {
                    checked: !!alertDraft.notifyOnSwitch
                    onToggled: updateAlertPreference("notifyOnSwitch", checked)
                }
                Item {}
            }
        }

        GroupBox {
            title: qsTr("Powody ostatnich zmian")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 4

                Repeater {
                    model: reasons
                    delegate: Label {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        text: {
                            const type = modelData.type || "reason"
                            const reason = modelData.reason || modelData.details || "?"
                            const ts = modelData.timestamp || modelData.until || ""
                            if (ts)
                                return qsTr("[%1] %2 — %3").arg(type).arg(reason).arg(ts)
                            return qsTr("[%1] %2").arg(type).arg(reason)
                        }
                    }
                }

                Label {
                    visible: reasons.length === 0
                    text: qsTr("Brak zarejestrowanych zdarzeń.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Metryki skuteczności")
            Layout.fillWidth: true

            ColumnLayout {
                width: parent.width
                spacing: 8

                GridLayout {
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 4
                    Layout.fillWidth: true

                    Label { text: qsTr("Łączna liczba decyzji") }
                    Label { text: performanceSummary.total !== undefined ? performanceSummary.total : "—" }

                    Label {
                        visible: performanceSummary.cycle_count !== undefined
                        text: qsTr("Liczba cykli kontrolera")
                    }
                    Label {
                        visible: performanceSummary.cycle_count !== undefined
                        text: performanceSummary.cycle_count !== undefined ? Number(performanceSummary.cycle_count).toLocaleString(Qt.locale(), "f", 0) : "—"
                    }

                    Label {
                        visible: performanceSummary.avg_confidence !== undefined
                        text: qsTr("Średnia ufność")
                    }
                    Label {
                        visible: performanceSummary.avg_confidence !== undefined
                        text: performanceSummary.avg_confidence !== undefined ? Number(performanceSummary.avg_confidence).toLocaleString(Qt.locale(), "f", 2) : "—"
                    }

                    Label {
                        visible: performanceSummary.avg_latency_ms !== undefined
                        text: qsTr("Średnie opóźnienie [ms]")
                    }
                    Label {
                        visible: performanceSummary.avg_latency_ms !== undefined
                        text: performanceSummary.avg_latency_ms !== undefined ? Number(performanceSummary.avg_latency_ms).toLocaleString(Qt.locale(), "f", 0) : "—"
                    }

                    Label {
                        visible: performanceSummary.p95_latency_ms !== undefined
                        text: qsTr("P95 opóźnienia [ms]")
                    }
                    Label {
                        visible: performanceSummary.p95_latency_ms !== undefined
                        text: performanceSummary.p95_latency_ms !== undefined ? Number(performanceSummary.p95_latency_ms).toLocaleString(Qt.locale(), "f", 0) : "—"
                    }

                    Label {
                        visible: performanceSummary.net_return_pct !== undefined
                        text: qsTr("Stopa zwrotu netto")
                    }
                    Label {
                        visible: performanceSummary.net_return_pct !== undefined
                        text: performanceSummary.net_return_pct !== undefined ? formatPercent(performanceSummary.net_return_pct) : "—"
                    }

                    Label {
                        visible: performanceSummary.avg_return_pct !== undefined
                        text: qsTr("Średnia stopa zwrotu")
                    }
                    Label {
                        visible: performanceSummary.avg_return_pct !== undefined
                        text: performanceSummary.avg_return_pct !== undefined ? formatPercent(performanceSummary.avg_return_pct) : "—"
                    }

                    Label {
                        visible: performanceSummary.volatility_pct !== undefined
                        text: qsTr("Zmienność (σ)")
                    }
                    Label {
                        visible: performanceSummary.volatility_pct !== undefined
                        text: performanceSummary.volatility_pct !== undefined ? formatPercent(performanceSummary.volatility_pct) : "—"
                    }

                    Label {
                        visible: performanceSummary.max_drawdown_pct !== undefined
                        text: qsTr("Maksymalne obsunięcie")
                    }
                    Label {
                        visible: performanceSummary.max_drawdown_pct !== undefined
                        text: performanceSummary.max_drawdown_pct !== undefined ? formatPercent(performanceSummary.max_drawdown_pct) : "—"
                    }

                    Label { text: qsTr("Zablokowane guardraile") }
                    Label {
                        text: metrics.guardrail_blocks_total !== undefined ? Number(metrics.guardrail_blocks_total).toLocaleString(Qt.locale(), "f", 0) : "—"
                    }
                }

                RowLayout {
                    visible: breakdownEntries(performanceSummary, "by_status").length > 0
                    Layout.fillWidth: true
                    spacing: 12

                    ColumnLayout {
                        spacing: 2
                        Label {
                            text: qsTr("Podział wg statusu")
                            font.bold: true
                        }
                        Repeater {
                            model: breakdownEntries(performanceSummary, "by_status")
                            delegate: Label {
                                text: qsTr("%1: %2").arg(modelData.key).arg(modelData.value)
                                color: palette.mid
                            }
                        }
                    }

                    ColumnLayout {
                        visible: breakdownEntries(performanceSummary, "by_symbol").length > 0
                        spacing: 2
                        Label {
                            text: qsTr("Podział wg symbolu")
                            font.bold: true
                        }
                        Repeater {
                            model: breakdownEntries(performanceSummary, "by_symbol")
                            delegate: Label {
                                text: qsTr("%1: %2").arg(modelData.key).arg(modelData.value)
                                color: palette.mid
                            }
                        }
                    }
                }

                Frame {
                    visible: recentPerformanceSummary && Object.keys(recentPerformanceSummary).length > 0
                    Layout.fillWidth: true
                    padding: 8
                    background: Rectangle {
                        radius: 6
                        color: Qt.rgba(0, 0, 0, 0.05)
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 6

                        Label {
                            text: recentPerformanceSummary.label || qsTr("Okno czasowe")
                            font.bold: true
                        }

                        Label {
                            visible: !!recentPerformanceSummary.window
                            text: {
                                var window = recentPerformanceSummary.window || {}
                                var start = formatTimestamp(window.start)
                                var end = formatTimestamp(window.end)
                                if (window.duration_s !== undefined)
                                    return qsTr("%1 → %2 (%3)").arg(start).arg(end).arg(formatDuration(window.duration_s))
                                return qsTr("%1 → %2").arg(start).arg(end)
                            }
                            color: palette.mid
                        }

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 4
                            Layout.fillWidth: true

                            Label { text: qsTr("Łączna liczba decyzji") }
                            Label { text: recentPerformanceSummary.total !== undefined ? recentPerformanceSummary.total : "—" }

                            Label {
                                visible: recentPerformanceSummary.cycle_count !== undefined
                                text: qsTr("Liczba cykli kontrolera")
                            }
                            Label {
                                visible: recentPerformanceSummary.cycle_count !== undefined
                                text: recentPerformanceSummary.cycle_count !== undefined ? Number(recentPerformanceSummary.cycle_count).toLocaleString(Qt.locale(), "f", 0) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.avg_confidence !== undefined
                                text: qsTr("Średnia ufność")
                            }
                            Label {
                                visible: recentPerformanceSummary.avg_confidence !== undefined
                                text: recentPerformanceSummary.avg_confidence !== undefined ? Number(recentPerformanceSummary.avg_confidence).toLocaleString(Qt.locale(), "f", 2) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.avg_latency_ms !== undefined
                                text: qsTr("Średnie opóźnienie [ms]")
                            }
                            Label {
                                visible: recentPerformanceSummary.avg_latency_ms !== undefined
                                text: recentPerformanceSummary.avg_latency_ms !== undefined ? Number(recentPerformanceSummary.avg_latency_ms).toLocaleString(Qt.locale(), "f", 0) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.p95_latency_ms !== undefined
                                text: qsTr("P95 opóźnienia [ms]")
                            }
                            Label {
                                visible: recentPerformanceSummary.p95_latency_ms !== undefined
                                text: recentPerformanceSummary.p95_latency_ms !== undefined ? Number(recentPerformanceSummary.p95_latency_ms).toLocaleString(Qt.locale(), "f", 0) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.net_return_pct !== undefined
                                text: qsTr("Stopa zwrotu netto")
                            }
                            Label {
                                visible: recentPerformanceSummary.net_return_pct !== undefined
                                text: recentPerformanceSummary.net_return_pct !== undefined ? formatPercent(recentPerformanceSummary.net_return_pct) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.avg_return_pct !== undefined
                                text: qsTr("Średnia stopa zwrotu")
                            }
                            Label {
                                visible: recentPerformanceSummary.avg_return_pct !== undefined
                                text: recentPerformanceSummary.avg_return_pct !== undefined ? formatPercent(recentPerformanceSummary.avg_return_pct) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.volatility_pct !== undefined
                                text: qsTr("Zmienność (σ)")
                            }
                            Label {
                                visible: recentPerformanceSummary.volatility_pct !== undefined
                                text: recentPerformanceSummary.volatility_pct !== undefined ? formatPercent(recentPerformanceSummary.volatility_pct) : "—"
                            }

                            Label {
                                visible: recentPerformanceSummary.max_drawdown_pct !== undefined
                                text: qsTr("Maksymalne obsunięcie")
                            }
                            Label {
                                visible: recentPerformanceSummary.max_drawdown_pct !== undefined
                                text: recentPerformanceSummary.max_drawdown_pct !== undefined ? formatPercent(recentPerformanceSummary.max_drawdown_pct) : "—"
                            }
                        }

                        ColumnLayout {
                            visible: breakdownEntries(recentPerformanceSummary, "by_status").length > 0 || breakdownEntries(recentPerformanceSummary, "by_symbol").length > 0
                            spacing: 4

                            ColumnLayout {
                                visible: breakdownEntries(recentPerformanceSummary, "by_status").length > 0
                                spacing: 2
                                Label {
                                    text: qsTr("Statusy w oknie")
                                    font.bold: true
                                }
                                Repeater {
                                    model: breakdownEntries(recentPerformanceSummary, "by_status")
                                    delegate: Label {
                                        text: qsTr("%1: %2").arg(modelData.key).arg(modelData.value)
                                        color: palette.mid
                                    }
                                }
                            }

                            ColumnLayout {
                                visible: breakdownEntries(recentPerformanceSummary, "by_symbol").length > 0
                                spacing: 2
                                Label {
                                    text: qsTr("Symbole w oknie")
                                    font.bold: true
                                }
                                Repeater {
                                    model: breakdownEntries(recentPerformanceSummary, "by_symbol")
                                    delegate: Label {
                                        text: qsTr("%1: %2").arg(modelData.key).arg(modelData.value)
                                        color: palette.mid
                                    }
                                }
                            }
                        }
                    }
                }

                Label {
                    visible: (!performanceSummary || Object.keys(performanceSummary).length === 0) && (!recentPerformanceSummary || Object.keys(recentPerformanceSummary).length === 0)
                    text: qsTr("Brak dostępnych statystyk decyzji.")
                    color: palette.mid
                }
            }
        }
    }
}
