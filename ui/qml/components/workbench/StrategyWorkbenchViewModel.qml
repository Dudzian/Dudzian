import QtQuick

QtObject {
    id: root
    objectName: "strategyWorkbenchViewModel"

    // External dependencies injected from parent QML
    property var appController: null
    property var strategyController: null
    property var workbenchController: null
    property var riskModel: null
    property var riskHistoryModel: null
    property var licenseController: null

    property bool catalogReady: false
    property var catalogEngines: []
    property var catalogDefinitions: []
    property var catalogMetadata: ({})
    property var catalogBlockedDefinitions: ({})
    property var catalogRegimeTemplates: ({})
    property var presetDraft: ({ name: "", metadata: ({}), entries: [] })
    property var presetValidation: ({ ok: false, errors: [], issues: [], preset: null, regime_map: ({}) })
    property var presetIssues: []
    property var regimeMappings: ({})
    property var schedulerHistory: []
    property bool schedulerHistoryInitialized: false
    property var presetHistorySnapshots: []

    function cloneValue(value) {
        try {
            return JSON.parse(JSON.stringify(value))
        } catch (error) {
            console.warn("StrategyWorkbenchViewModel: cloneValue fallback", error)
            return value
        }
    }

    function deriveRegimeMapFromPreset(preset) {
        var map = {}
        if (!preset)
            return map
        var items = preset.strategies || []
        for (var i = 0; i < items.length; ++i) {
            var entry = items[i]
            if (!entry)
                continue
            var profile = entry.risk_profile
            if (!profile && entry.metadata)
                profile = entry.metadata.risk_profile
            if (!profile)
                continue
            var key = String(profile)
            if (!map[key])
                map[key] = []
            map[key].push(entry.name || ("strategy_" + i))
        }
        return map
    }

    function syncCatalog() {
        if (!workbenchController)
            return
        catalogEngines = cloneValue(workbenchController.catalogEngines || [])
        catalogDefinitions = cloneValue(workbenchController.catalogDefinitions || [])
        catalogMetadata = cloneValue(workbenchController.catalogMetadata || {})
        catalogBlockedDefinitions = cloneValue(workbenchController.catalogBlockedDefinitions || {})
        catalogRegimeTemplates = cloneValue(workbenchController.catalogRegimeTemplates || {})
        catalogReady = catalogDefinitions.length > 0
    }

    function syncValidation(result) {
        var payload = result
        if (payload === undefined)
            payload = workbenchController ? workbenchController.lastValidation : null
        if (!payload)
            payload = { ok: false, errors: [], issues: [], preset: null }
        presetValidation = payload
        presetIssues = cloneValue(payload.issues || [])
        if (payload.regime_map)
            regimeMappings = cloneValue(payload.regime_map)
        else if (payload.ok && payload.preset)
            regimeMappings = deriveRegimeMapFromPreset(payload.preset)
        else if (!payload.ok)
            regimeMappings = {}
    }

    function syncPresetHistory() {
        if (!workbenchController)
            return
        presetHistorySnapshots = cloneValue(workbenchController.presetHistory || [])
    }

    function ensureSchedulerHistoryInitialized() {
        if (schedulerHistoryInitialized)
            return
        schedulerHistory = []
        schedulerHistoryInitialized = true
    }

    function findCatalogDefinition(name) {
        if (!name)
            return null
        for (var i = 0; i < catalogDefinitions.length; ++i) {
            var entry = catalogDefinitions[i]
            if (entry && entry.name === name)
                return entry
        }
        return null
    }

    function updatePresetDraft(newDraft) {
        if (!newDraft)
            return
        presetDraft = newDraft
    }

    function startPresetDraftFromDefinition(name) {
        var definition = findCatalogDefinition(name)
        if (!definition)
            return false
        var entry = {
            name: definition.name,
            engine: definition.engine,
            risk_profile: definition.risk_profile || (definition.metadata ? definition.metadata.risk_profile : ""),
            parameters: cloneValue(definition.parameters || {}),
            tags: cloneValue(definition.tags || []),
            license_tier: definition.license_tier,
            risk_classes: cloneValue(definition.risk_classes || []),
            required_data: cloneValue(definition.required_data || []),
            metadata: cloneValue(definition.metadata || {})
        }
        updatePresetDraft({
            name: definition.name + "-preset",
            metadata: { source: "catalog" },
            entries: [entry]
        })
        return true
    }

    function updatePresetDraftEntry(index, field, value) {
        if (!presetDraft || !presetDraft.entries || index < 0 || index >= presetDraft.entries.length)
            return false
        var draft = cloneValue(presetDraft)
        draft.entries[index][field] = value
        updatePresetDraft(draft)
        return true
    }

    function requestPresetValidation() {
        if (!workbenchController)
            return false
        var payload = presetDraft || { name: "", entries: [] }
        var result = workbenchController.validatePreset(payload)
        if (result)
            syncValidation(result)
        return result && result.ok
    }

    function applyPresetDraft() {
        if (!presetValidation || !presetValidation.ok || !presetValidation.preset)
            return false
        ensureSchedulerHistoryInitialized()
        schedulerHistory.push(cloneValue(schedulerEntries))
        var sanitized = presetValidation.preset
        var entries = sanitized.strategies || []
        var normalized = []
        for (var i = 0; i < entries.length; ++i) {
            var entry = entries[i]
            normalized.push({
                name: entry.name || ("strategy_" + i),
                engine: entry.engine || "",
                risk_profile: entry.risk_profile || "",
                license_tier: entry.license_tier || "",
                tags: cloneValue(entry.tags || []),
                metadata: cloneValue(entry.metadata || {}),
                parameters: cloneValue(entry.parameters || {})
            })
        }
        schedulerEntries = normalized
        if (presetValidation.regime_map)
            regimeMappings = cloneValue(presetValidation.regime_map)
        else
            regimeMappings = deriveRegimeMapFromPreset(sanitized)
        if (workbenchController)
            workbenchController.savePresetSnapshot(presetValidation)
        syncPresetHistory()
        return true
    }

    function rollbackSchedulerConfig() {
        if (!schedulerHistoryInitialized || schedulerHistory.length === 0)
            return false
        var previous = schedulerHistory.pop()
        schedulerEntries = previous || []
        var restore = workbenchController ? workbenchController.restorePreviousPreset() : null
        if (restore && restore.ok) {
            if (restore.regime_map)
                regimeMappings = cloneValue(restore.regime_map)
            else if (restore.preset)
                regimeMappings = deriveRegimeMapFromPreset(restore.preset)
        } else if (restore && restore.preset) {
            regimeMappings = deriveRegimeMapFromPreset(restore.preset)
        } else if (!restore) {
            regimeMappings = {}
        }
        syncPresetHistory()
        return true
    }

    // Aggregated state exposed to the UI
    property var schedulerEntries: []
    property var exchangeConnections: []
    property var aiConfiguration: ({})
    property var portfolioSummary: ({})
    property var riskSnapshot: ({})
    property var runtimeStatus: ({})
    property var licenseStatus: ({})

    // Demo mode metadata
    property bool demoModeActive: false
    property string demoModeId: ""
    property string demoModeTitle: ""
    property string demoModeDescription: ""
    property var demoPresets: [
        {
            id: "momentum",
            title: qsTr("Momentum Pro"),
            description: qsTr("Zestaw presetów momentum dla rynku krypto – automatyczne balansowanie ryzyka i szybkie przełączanie profili."),
            schedulerEntries: [
                {
                    name: "Momentum Alpha",
                    enabled: true,
                    scheduleCount: 4,
                    timezone: "UTC",
                    nextRun: "2024-05-18T09:00:00Z",
                    notes: qsTr("Aktywne okna 1m/5m z trailing stop")
                },
                {
                    name: "Momentum Beta",
                    enabled: true,
                    scheduleCount: 2,
                    timezone: "UTC",
                    nextRun: "2024-05-18T10:15:00Z",
                    notes: qsTr("Agregacja multi-venue, target delta 0.3")
                }
            ],
            exchangeConnections: [
                {
                    exchange: "BINANCE",
                    symbol: "BTC/USDT",
                    venueSymbol: "BTCUSDT",
                    status: qsTr("Demo: połączenie symulacyjne"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 120
                },
                {
                    exchange: "COINBASE",
                    symbol: "ETH/USD",
                    venueSymbol: "ETHUSD",
                    status: qsTr("Demo: monitorowanie danych order book"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 90
                }
            ],
            aiConfiguration: {
                policy: "momentum",
                decision_window: "PT15S",
                risk_profile: "growth",
                overrides: [
                    { profile: "intraday", maxDrawdown: 0.04, takeProfit: 0.012 },
                    { profile: "swing", maxDrawdown: 0.06, takeProfit: 0.028 }
                ],
                features: ["ema12", "ema26", "vwap", "macro_feed"],
                modelRevision: "2024.04-momentum"
            },
            portfolioSummary: {
                entryCount: 64,
                minValue: 92500,
                maxValue: 132400,
                latestValue: 129800,
                profileLabel: "Momentum",
                maxDrawdown: 0.083,
                averageDrawdown: 0.031,
                maxLeverage: 2.4,
                averageLeverage: 1.7,
                anyBreach: false,
                totalBreaches: 0
            },
            riskSnapshot: {
                profileLabel: "Momentum",
                currentDrawdown: 0.021,
                maxDailyLoss: 0.055,
                usedLeverage: 1.68,
                generatedAt: "2024-05-18T08:55:12Z",
                exposures: [
                    { code: "XBT_PERP", current: 0.62, max: 0.8, threshold: 0.75, breach: false },
                    { code: "ETH_PERP", current: 0.48, max: 0.7, threshold: 0.65, breach: false },
                    { code: "ALT_BASKET", current: 0.31, max: 0.55, threshold: 0.5, breach: false }
                ]
            },
            runtimeStatus: {
                connection: qsTr("Demo: połączenie symulacyjne"),
                reduceMotion: false,
                offlineMode: false,
                offlineDaemonStatus: qsTr("Aktywny symulator"),
                automationRunning: true,
                performanceGuard: { fpsTarget: 120, jankThresholdMs: 12.0, maxOverlayCount: 3 },
                riskRefresh: { enabled: true, intervalSeconds: 30, nextRefreshInSeconds: 18 }
            },
            licenseStatus: {
                active: true,
                edition: "OEM Demo",
                licenseId: "DEMO-MOMENTUM",
                holderName: "Core Labs QA",
                maintenanceActive: true,
                modules: ["momentum", "portfolio", "scheduler"],
                runtime: ["desktop-shell", "scheduler-engine"]
            }
        },
        {
            id: "hedge",
            title: qsTr("Market Neutral"),
            description: qsTr("Tryb prezentacyjny dla funduszy market-neutral – pokazuje balancing delta i zabezpieczenia krzyżowe."),
            schedulerEntries: [
                {
                    name: "Delta Hedge",
                    enabled: true,
                    scheduleCount: 6,
                    timezone: "Europe/Warsaw",
                    nextRun: "2024-05-18T09:05:00+02:00",
                    notes: qsTr("Rebalans co 10 minut z limitem slippage 2 bps")
                },
                {
                    name: "Gamma Overlay",
                    enabled: false,
                    scheduleCount: 1,
                    timezone: "Europe/Warsaw",
                    nextRun: "2024-05-18T12:00:00+02:00",
                    notes: qsTr("Opcjonalna noga opcyjna do pokazu")
                }
            ],
            exchangeConnections: [
                {
                    exchange: "OKX",
                    symbol: "BTC/USDT",
                    venueSymbol: "BTC-USDT-SWAP",
                    status: qsTr("Demo: kanał hedge aktywny"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 75
                },
                {
                    exchange: "DERIBIT",
                    symbol: "BTC-PERP",
                    venueSymbol: "BTC-PERPETUAL",
                    status: qsTr("Demo: opcje zabezpieczające"),
                    offline: false,
                    automationRunning: false,
                    fpsTarget: 60
                }
            ],
            aiConfiguration: {
                policy: "market_neutral",
                decision_window: "PT1M",
                risk_profile: "conservative",
                overrides: [
                    { profile: "hedge", maxDrawdown: 0.025, takeProfit: 0.006 }
                ],
                features: ["zscore", "spread_mean", "inventory_skew"],
                modelRevision: "2024.03-neutral"
            },
            portfolioSummary: {
                entryCount: 48,
                minValue: 198000,
                maxValue: 205600,
                latestValue: 203450,
                profileLabel: "Neutral",
                maxDrawdown: 0.019,
                averageDrawdown: 0.011,
                maxLeverage: 1.2,
                averageLeverage: 0.9,
                anyBreach: false,
                totalBreaches: 0
            },
            riskSnapshot: {
                profileLabel: "Neutral",
                currentDrawdown: 0.008,
                maxDailyLoss: 0.02,
                usedLeverage: 0.95,
                generatedAt: "2024-05-18T07:42:10Z",
                exposures: [
                    { code: "BTC_DELTA", current: 0.05, max: 0.1, threshold: 0.08, breach: false },
                    { code: "ETH_DELTA", current: -0.01, max: 0.08, threshold: 0.08, breach: false },
                    { code: "BASIS_SPREAD", current: 0.12, max: 0.25, threshold: 0.2, breach: false }
                ]
            },
            runtimeStatus: {
                connection: qsTr("Demo: tryb market neutral"),
                reduceMotion: false,
                offlineMode: false,
                offlineDaemonStatus: qsTr("Stabilny"),
                automationRunning: true,
                performanceGuard: { fpsTarget: 75, jankThresholdMs: 16.0, maxOverlayCount: 2 },
                riskRefresh: { enabled: true, intervalSeconds: 45, nextRefreshInSeconds: 22 }
            },
            licenseStatus: {
                active: true,
                edition: "OEM Enterprise",
                licenseId: "DEMO-HEDGE",
                holderName: "Core Labs Hedge Demo",
                maintenanceActive: true,
                modules: ["hedge", "risk", "scheduler"],
                runtime: ["desktop-shell", "risk-service"]
            }
        }
    ]

    function activateDemoMode(id) {
        var preset = findDemoPreset(id)
        if (!preset)
            return false
        demoModeActive = true
        demoModeId = preset.id
        demoModeTitle = preset.title
        demoModeDescription = preset.description
        schedulerEntries = preset.schedulerEntries
        exchangeConnections = preset.exchangeConnections
        aiConfiguration = preset.aiConfiguration
        portfolioSummary = preset.portfolioSummary
        riskSnapshot = preset.riskSnapshot
        runtimeStatus = preset.runtimeStatus
        licenseStatus = preset.licenseStatus
        return true
    }

    function disableDemoMode() {
        if (!demoModeActive)
            return
        demoModeActive = false
        demoModeId = ""
        demoModeTitle = ""
        demoModeDescription = ""
        refreshFromLive()
    }

    function findDemoPreset(id) {
        if (!id)
            return null
        for (var i = 0; i < demoPresets.length; ++i) {
            if (demoPresets[i].id === id)
                return demoPresets[i]
        }
        return null
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
    }

    function computeSchedulerEntries() {
        if (!strategyController || typeof strategyController.schedulerList !== "function")
            return []
        var list = strategyController.schedulerList() || []
        return list.map(function(item) {
            var schedules = item.schedules || []
            return {
                name: item.name || qsTr("Strategia"),
                enabled: item.enabled !== false,
                scheduleCount: schedules.length,
                timezone: item.timezone || "",
                nextRun: item.next_run_at || item.nextRun || "",
                notes: item.notes || ""
            }
        })
    }

    function computeExchangeConnections() {
        if (!appController || typeof appController.instrumentConfigSnapshot !== "function")
            return []
        var instrument = appController.instrumentConfigSnapshot() || {}
        var guard = appController.performanceGuardSnapshot ? appController.performanceGuardSnapshot() : {}
        return [
            {
                exchange: instrument.exchange || "",
                symbol: instrument.symbol || "",
                venueSymbol: instrument.venueSymbol || "",
                status: appController.connectionStatus || "",
                offline: !!appController.offlineMode,
                automationRunning: !!appController.offlineAutomationRunning,
                fpsTarget: guard.fpsTarget || 0
            }
        ]
    }

    function computeAiConfiguration() {
        if (!strategyController || typeof strategyController.decisionConfigSnapshot !== "function")
            return {}
        var snapshot = strategyController.decisionConfigSnapshot() || {}
        return snapshot
    }

    function computePortfolioSummary() {
        var summary = {}
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
        var snapshot = {
            profileLabel: riskModel.profileLabel,
            currentDrawdown: riskModel.currentDrawdown,
            maxDailyLoss: riskModel.maxDailyLoss,
            usedLeverage: riskModel.usedLeverage,
            generatedAt: riskModel.generatedAt ? riskModel.generatedAt.toString() : ""
        }
        var exposures = []
        if (typeof riskModel.count === "number") {
            for (var i = 0; i < riskModel.count; ++i) {
                var row = riskModel.get(i)
                exposures.push({
                    code: row.code,
                    current: row.currentValue,
                    max: row.maxValue,
                    threshold: row.thresholdValue,
                    breach: row.breach
                })
            }
        }
        snapshot.exposures = exposures
        return snapshot
    }

    function computeRuntimeStatus() {
        if (!appController)
            return {}
        var runtime = {
            connection: appController.connectionStatus || "",
            reduceMotion: !!appController.reduceMotionActive,
            offlineMode: !!appController.offlineMode,
            offlineDaemonStatus: appController.offlineDaemonStatus || "",
            automationRunning: !!appController.offlineAutomationRunning
        }
        if (typeof appController.performanceGuardSnapshot === "function")
            runtime.performanceGuard = appController.performanceGuardSnapshot()
        if (typeof appController.riskRefreshSnapshot === "function")
            runtime.riskRefresh = appController.riskRefreshSnapshot()
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
            runtime: licenseController.licenseRuntime || []
        }
    }

    onAppControllerChanged: {
        runtimeConnections.target = appController
        refreshFromLive()
    }
    onStrategyControllerChanged: {
        strategyConnections.target = strategyController
        refreshFromLive()
    }
    onWorkbenchControllerChanged: {
        workbenchConnections.target = workbenchController
        syncCatalog()
        syncValidation()
        syncPresetHistory()
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

    Component.onCompleted: {
        refreshFromLive()
        syncCatalog()
        syncValidation()
        syncPresetHistory()
    }

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
    }

    Connections {
        id: workbenchConnections
        target: root.workbenchController
        function onCatalogChanged() { root.syncCatalog() }
        function onLastValidationChanged() { root.syncValidation() }
        function onPresetHistoryChanged() { root.syncPresetHistory() }
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
