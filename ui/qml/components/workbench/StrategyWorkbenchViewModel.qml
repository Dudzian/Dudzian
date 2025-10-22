import QtQuick

import "./StrategyWorkbenchPresets.js" as Presets

QtObject {
    id: root
    objectName: "strategyWorkbenchViewModel"

    // Zależności przekazywane z okna aplikacji
    property var appController: null
    property var strategyController: null
    property var riskModel: null
    property var riskHistoryModel: null
    property var licenseController: null

    // Dane agregowane dla paneli roboczych
    property var schedulerEntries: []
    property var exchangeConnections: []
    property var aiConfiguration: ({})
    property var portfolioSummary: ({})
    property var riskSnapshot: ({})
    property var runtimeStatus: ({})
    property var licenseStatus: ({})
    property var instrumentDetails: ({})
    property var controlState: defaultControlState()

    // Obsługa trybów demonstracyjnych
    readonly property var demoPresets: Presets.createPresets()
    property bool demoModeActive: false
    property string demoModeId: ""
    property string demoModeTitle: ""
    property string demoModeDescription: ""

    function defaultControlState() {
        return {
            schedulerRunning: false,
            offlineMode: false,
            automationRunning: false,
            lastActionMessage: "",
            lastActionSuccess: true,
            lastActionAt: "",
            lastRiskRefreshAt: "",
            nextRiskRefreshDueAt: "",
            manualRefreshCount: 0,
        }
    }

    function normalizeControlState(state) {
        return Object.assign({}, defaultControlState(), state || {})
    }

    function updateControlState(state) {
        controlState = normalizeControlState(state)
    }

    function mergeControlState(overrides) {
        const base = normalizeControlState(controlState)
        updateControlState(Object.assign({}, base, overrides || {}))
    }

    function updateDemoRuntimeStatus(overrides) {
        const current = runtimeStatus || {}
        const next = Object.assign({}, current, overrides || {})
        if (overrides && overrides.riskRefresh) {
            const baseRefresh = current.riskRefresh || {}
            next.riskRefresh = Object.assign({}, baseRefresh, overrides.riskRefresh)
        }
        runtimeStatus = next
    }

    function activateDemoMode(id) {
        const preset = Presets.findPreset(demoPresets, id)
        if (!preset)
            return false
        applyDemoPreset(Presets.clonePreset(preset))
        return true
    }

    function applyDemoPreset(preset) {
        demoModeActive = true
        demoModeId = preset.id
        demoModeTitle = preset.title
        demoModeDescription = preset.description

        schedulerEntries = preset.schedulerEntries || []
        exchangeConnections = preset.exchangeConnections || []
        aiConfiguration = preset.aiConfiguration || ({})
        portfolioSummary = preset.portfolioSummary || ({})
        riskSnapshot = preset.riskSnapshot || ({})
        runtimeStatus = preset.runtimeStatus || ({})
        licenseStatus = preset.licenseStatus || ({})
        instrumentDetails = preset.instrumentDetails || computeInstrumentDetails()
        updateControlState(preset.controlState)
    }

    function disableDemoMode() {
        if (!demoModeActive)
            return
        demoModeActive = false
        demoModeId = ""
        demoModeTitle = ""
        demoModeDescription = ""
        updateControlState(defaultControlState())
        refreshFromLive()
    }

    function refreshFromLive() {
        if (demoModeActive)
            return
        schedulerEntries = computeSchedulerEntries()
        exchangeConnections = computeExchangeConnections()
        aiConfiguration = computeAiConfiguration()
        portfolioSummary = computePortfolioSummary()
        riskSnapshot = computeRiskSnapshot()
        runtimeStatus = computeRuntimeStatus()
        licenseStatus = computeLicenseStatus()
        instrumentDetails = computeInstrumentDetails()
        controlState = computeControlState()
    }

    function computeSchedulerEntries() {
        if (!strategyController || typeof strategyController.schedulerList !== "function")
            return []
        const list = strategyController.schedulerList() || []
        return list.map(function(item) {
            const schedules = item.schedules || []
            return {
                name: item.name || qsTr("Strategia"),
                enabled: item.enabled !== false,
                scheduleCount: schedules.length,
                timezone: item.timezone || "",
                nextRun: item.next_run_at || item.nextRun || "",
                notes: item.notes || "",
            }
        })
    }

    function computeExchangeConnections() {
        if (!appController || typeof appController.instrumentConfigSnapshot !== "function")
            return []
        const instrument = appController.instrumentConfigSnapshot() || {}
        const guard = typeof appController.performanceGuardSnapshot === "function"
            ? (appController.performanceGuardSnapshot() || {})
            : {}
        return [
            {
                exchange: instrument.exchange || "",
                symbol: instrument.symbol || "",
                venueSymbol: instrument.venueSymbol || "",
                status: appController.connectionStatus || "",
                offline: !!appController.offlineMode,
                automationRunning: !!appController.offlineAutomationRunning,
                fpsTarget: guard.fpsTarget || 0,
            }
        ]
    }

    function computeAiConfiguration() {
        if (!strategyController || typeof strategyController.decisionConfigSnapshot !== "function")
            return {}
        return strategyController.decisionConfigSnapshot() || {}
    }

    function computePortfolioSummary() {
        const summary = {}
        if (riskHistoryModel) {
            summary.entryCount = riskHistoryModel.entryCount || 0
            summary.minValue = riskHistoryModel.minPortfolioValue || 0
            summary.maxValue = riskHistoryModel.maxPortfolioValue || 0
            summary.maxDrawdown = riskHistoryModel.maxDrawdown || 0
            summary.averageDrawdown = riskHistoryModel.averageDrawdown || 0
            summary.maxLeverage = riskHistoryModel.maxLeverage || 0
            summary.averageLeverage = riskHistoryModel.averageLeverage || 0
            summary.anyBreach = !!riskHistoryModel.anyExposureBreached
            summary.totalBreaches = riskHistoryModel.totalBreachCount || 0
            summary.maxExposureUtilization = riskHistoryModel.maxExposureUtilization || 0
        }
        if (riskModel && riskModel.hasData) {
            summary.latestValue = riskModel.portfolioValue
            summary.profileLabel = riskModel.profileLabel
        }
        return summary
    }

    function computeRiskSnapshot() {
        if (!riskModel || !riskModel.hasData)
            return {}
        const snapshot = {
            profileLabel: riskModel.profileLabel,
            currentDrawdown: riskModel.currentDrawdown,
            maxDailyLoss: riskModel.maxDailyLoss,
            usedLeverage: riskModel.usedLeverage,
            generatedAt: riskModel.generatedAt ? riskModel.generatedAt.toString() : "",
            exposures: [],
        }
        if (typeof riskModel.count === "number") {
            for (let i = 0; i < riskModel.count; ++i) {
                const row = riskModel.get(i)
                snapshot.exposures.push({
                    code: row.code,
                    current: row.currentValue,
                    max: row.maxValue,
                    threshold: row.thresholdValue,
                    breach: row.breach,
                })
            }
        }
        return snapshot
    }

    function computeRuntimeStatus() {
        if (!appController)
            return {}
        const runtime = {
            connection: appController.connectionStatus || "",
            reduceMotion: !!appController.reduceMotionActive,
            offlineMode: !!appController.offlineMode,
            offlineDaemonStatus: appController.offlineDaemonStatus || "",
            automationRunning: !!appController.offlineAutomationRunning,
        }
        if (typeof appController.performanceGuardSnapshot === "function")
            runtime.performanceGuard = appController.performanceGuardSnapshot() || {}
        if (typeof appController.riskRefreshSnapshot === "function")
            runtime.riskRefresh = appController.riskRefreshSnapshot() || {}
        return runtime
    }

    function computeLicenseStatus() {
        if (!licenseController)
            return {}
        return {
            active: !!licenseController.licenseActive,
            edition: licenseController.licenseEdition || "",
            licenseId: licenseController.licenseLicenseId || "",
            issuedAt: licenseController.licenseIssuedAt || "",
            maintenanceUntil: licenseController.licenseMaintenanceUntil || "",
            maintenanceActive: !!licenseController.licenseMaintenanceActive,
            holderName: licenseController.licenseHolderName || "",
            holderEmail: licenseController.licenseHolderEmail || "",
            seats: licenseController.licenseSeats || 0,
            trialActive: !!licenseController.licenseTrialActive,
            trialExpiresAt: licenseController.licenseTrialExpiresAt || "",
            modules: licenseController.licenseModules || [],
            environments: licenseController.licenseEnvironments || [],
            runtime: licenseController.licenseRuntime || [],
        }
    }

    function computeInstrumentDetails() {
        if (!appController || typeof appController.instrumentConfigSnapshot !== "function")
            return {}
        const snapshot = appController.instrumentConfigSnapshot() || {}
        return {
            exchange: snapshot.exchange || "",
            symbol: snapshot.symbol || "",
            venueSymbol: snapshot.venueSymbol || "",
            quoteCurrency: snapshot.quoteCurrency || "",
            baseCurrency: snapshot.baseCurrency || "",
            granularity: snapshot.granularity || "",
        }
    }

    function computeControlState() {
        const base = normalizeControlState(controlState)
        let schedulerRunning = base.schedulerRunning
        if (strategyController) {
            if (typeof strategyController.schedulerRunning === "boolean")
                schedulerRunning = strategyController.schedulerRunning
            else if (typeof strategyController.schedulerRunning === "function")
                schedulerRunning = !!strategyController.schedulerRunning()
            else if (typeof strategyController.isSchedulerRunning === "function")
                schedulerRunning = !!strategyController.isSchedulerRunning()
        }

        let automationRunning = base.automationRunning
        if (appController && typeof appController.offlineAutomationRunning !== "undefined")
            automationRunning = !!appController.offlineAutomationRunning

        let offlineMode = base.offlineMode
        if (appController && typeof appController.offlineMode !== "undefined")
            offlineMode = !!appController.offlineMode

        let refreshSnapshot = null
        if (appController && typeof appController.riskRefreshSnapshot === "function")
            refreshSnapshot = appController.riskRefreshSnapshot() || {}
        else if (runtimeStatus && runtimeStatus.riskRefresh)
            refreshSnapshot = runtimeStatus.riskRefresh

        const next = Object.assign({}, base, {
            schedulerRunning: schedulerRunning,
            automationRunning: automationRunning,
            offlineMode: offlineMode,
        })

        if (refreshSnapshot) {
            if (refreshSnapshot.lastUpdateAt !== undefined)
                next.lastRiskRefreshAt = refreshSnapshot.lastUpdateAt
            if (refreshSnapshot.nextRefreshDueAt !== undefined)
                next.nextRiskRefreshDueAt = refreshSnapshot.nextRefreshDueAt
        }

        return next
    }

    function startScheduler() {
        const base = normalizeControlState(controlState)
        let success = false
        if (demoModeActive) {
            success = true
            updateDemoRuntimeStatus({ automationRunning: true })
        } else if (strategyController && typeof strategyController.startScheduler === "function") {
            const result = strategyController.startScheduler()
            success = result !== false
        }
        const nowIso = new Date().toISOString()
        const next = Object.assign({}, base, {
            schedulerRunning: success ? true : base.schedulerRunning,
            lastActionSuccess: success,
            lastActionMessage: success ? qsTr("Uruchomiono harmonogram")
                                       : qsTr("Nie udało się uruchomić harmonogramu"),
            lastActionAt: nowIso,
        })
        controlState = next
        if (!demoModeActive)
            refreshFromLive()
        return success
    }

    function stopScheduler() {
        const base = normalizeControlState(controlState)
        let success = false
        if (demoModeActive) {
            success = true
            updateDemoRuntimeStatus({ automationRunning: false })
        } else if (strategyController && typeof strategyController.stopScheduler === "function") {
            const result = strategyController.stopScheduler()
            success = result !== false
        }
        const nowIso = new Date().toISOString()
        const next = Object.assign({}, base, {
            schedulerRunning: success ? false : base.schedulerRunning,
            lastActionSuccess: success,
            lastActionMessage: success ? qsTr("Zatrzymano harmonogram")
                                       : qsTr("Nie udało się zatrzymać harmonogramu"),
            lastActionAt: nowIso,
        })
        controlState = next
        if (!demoModeActive)
            refreshFromLive()
        return success
    }

    function triggerRiskRefresh() {
        const base = normalizeControlState(controlState)
        let success = false
        let nextDueIso = base.nextRiskRefreshDueAt
        if (demoModeActive) {
            const now = new Date()
            const nextDue = new Date(now.getTime() + 30000)
            nextDueIso = nextDue.toISOString()
            updateDemoRuntimeStatus({
                riskRefresh: {
                    lastUpdateAt: now.toISOString(),
                    lastRequestAt: now.toISOString(),
                    nextRefreshDueAt: nextDueIso,
                    nextRefreshInSeconds: Math.max(0, Math.round((nextDue.getTime() - now.getTime()) / 1000)),
                },
            })
            success = true
        } else if (appController) {
            if (typeof appController.triggerRiskRefresh === "function")
                success = appController.triggerRiskRefresh() !== false
            else if (typeof appController.requestRiskRefresh === "function")
                success = appController.requestRiskRefresh() !== false
        }

        const nowIso = new Date().toISOString()
        const next = Object.assign({}, base, {
            lastActionSuccess: success,
            lastActionMessage: success ? qsTr("Zainicjowano odświeżenie ryzyka")
                                       : qsTr("Nie udało się odświeżyć ryzyka"),
            lastActionAt: nowIso,
            manualRefreshCount: success ? base.manualRefreshCount + 1 : base.manualRefreshCount,
        })
        if (success) {
            next.lastRiskRefreshAt = nowIso
            if (demoModeActive)
                next.nextRiskRefreshDueAt = nextDueIso
        }
        controlState = next
        if (!demoModeActive)
            refreshFromLive()
        return success
    }

    onAppControllerChanged: {
        runtimeConnections.target = appController
        refreshFromLive()
    }
    onStrategyControllerChanged: {
        strategyConnections.target = strategyController
        refreshFromLive()
    }
    onRiskModelChanged: {
        riskConnections.target = riskModel
        refreshFromLive()
    }
    onRiskHistoryModelChanged: {
        riskHistoryConnections.target = riskHistoryModel
        refreshFromLive()
    }
    onLicenseControllerChanged: {
        licenseConnections.target = licenseController
        refreshFromLive()
    }

    Component.onCompleted: refreshFromLive()

    Connections {
        id: runtimeConnections
        target: root.appController
        function onConnectionStatusChanged() { root.refreshFromLive() }
        function onPerformanceGuardChanged() { root.refreshFromLive() }
        function onRiskRefreshScheduleChanged() { root.refreshFromLive() }
        function onInstrumentChanged() { root.refreshFromLive() }
        function onOfflineDaemonStatusChanged() { root.refreshFromLive() }
        function onOfflineAutomationRunningChanged() { root.refreshFromLive() }
        function onReduceMotionActiveChanged() { root.refreshFromLive() }
    }

    Connections {
        id: strategyConnections
        target: root.strategyController
        function onSchedulerListChanged() { root.refreshFromLive() }
        function onDecisionConfigChanged() { root.refreshFromLive() }
        function onSchedulerStateChanged() { root.refreshFromLive() }
    }

    Connections {
        id: riskConnections
        target: root.riskModel
        function onRiskStateChanged() { root.refreshFromLive() }
    }

    Connections {
        id: riskHistoryConnections
        target: root.riskHistoryModel
        function onSummaryChanged() { root.refreshFromLive() }
        function onHistoryChanged() { root.refreshFromLive() }
    }

    Connections {
        id: licenseConnections
        target: root.licenseController
        function onLicenseActiveChanged() { root.refreshFromLive() }
        function onLicenseDataChanged() { root.refreshFromLive() }
    }
}
