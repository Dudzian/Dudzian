function createPresets() {
    return [
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
                    notes: qsTr("Aktywne okna 1m/5m z trailing stop"),
                },
                {
                    name: "Momentum Beta",
                    enabled: true,
                    scheduleCount: 2,
                    timezone: "UTC",
                    nextRun: "2024-05-18T10:15:00Z",
                    notes: qsTr("Agregacja multi-venue, target delta 0.3"),
                },
            ],
            exchangeConnections: [
                {
                    exchange: "BINANCE",
                    symbol: "BTC/USDT",
                    venueSymbol: "BTCUSDT",
                    status: qsTr("Demo: połączenie symulacyjne"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 120,
                },
                {
                    exchange: "COINBASE",
                    symbol: "ETH/USD",
                    venueSymbol: "ETHUSD",
                    status: qsTr("Demo: monitorowanie danych order book"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 90,
                },
            ],
            instrumentDetails: {
                exchange: "BINANCE",
                symbol: "BTC/USDT",
                venueSymbol: "BTCUSDT",
                quoteCurrency: "USDT",
                baseCurrency: "BTC",
                granularity: "PT1M",
            },
            aiConfiguration: {
                policy: "momentum",
                decision_window: "PT15S",
                risk_profile: "growth",
                overrides: [
                    { profile: "intraday", maxDrawdown: 0.04, takeProfit: 0.012 },
                    { profile: "swing", maxDrawdown: 0.06, takeProfit: 0.028 },
                ],
                features: ["ema12", "ema26", "vwap", "macro_feed"],
                modelRevision: "2024.04-momentum",
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
                maxExposureUtilization: 0.82,
                anyBreach: false,
                totalBreaches: 0,
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
                    { code: "ALT_BASKET", current: 0.31, max: 0.55, threshold: 0.5, breach: false },
                ],
            },
            runtimeStatus: {
                connection: qsTr("Demo: połączenie symulacyjne"),
                reduceMotion: false,
                offlineMode: false,
                offlineDaemonStatus: qsTr("Aktywny symulator"),
                automationRunning: true,
                performanceGuard: { fpsTarget: 120, jankThresholdMs: 12.0, maxOverlayCount: 3 },
                riskRefresh: {
                    enabled: true,
                    intervalSeconds: 30,
                    nextRefreshInSeconds: 18,
                    nextRefreshDueAt: "2024-05-18T09:00:18Z",
                    lastUpdateAt: "2024-05-18T08:59:48Z",
                    lastRequestAt: "2024-05-18T08:59:45Z",
                },
            },
            controlState: {
                schedulerRunning: true,
                automationRunning: true,
                offlineMode: false,
                lastActionMessage: qsTr("Tryb demo gotowy"),
                lastActionSuccess: true,
                lastActionAt: "2024-05-18T08:59:30Z",
                lastRiskRefreshAt: "2024-05-18T08:59:48Z",
                nextRiskRefreshDueAt: "2024-05-18T09:00:18Z",
                manualRefreshCount: 2,
            },
            licenseStatus: {
                active: true,
                edition: "OEM Demo",
                licenseId: "DEMO-MOMENTUM",
                holderName: "Core Labs QA",
                maintenanceActive: true,
                modules: ["momentum", "portfolio", "scheduler"],
                runtime: ["desktop-shell", "scheduler-engine"],
            },
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
                    notes: qsTr("Rebalans co 10 minut z limitem slippage 2 bps"),
                },
                {
                    name: "Gamma Overlay",
                    enabled: false,
                    scheduleCount: 1,
                    timezone: "Europe/Warsaw",
                    nextRun: "2024-05-18T12:00:00+02:00",
                    notes: qsTr("Opcjonalna noga opcyjna do pokazu"),
                },
            ],
            exchangeConnections: [
                {
                    exchange: "OKX",
                    symbol: "BTC/USDT",
                    venueSymbol: "BTC-USDT-SWAP",
                    status: qsTr("Demo: kanał hedge aktywny"),
                    offline: false,
                    automationRunning: true,
                    fpsTarget: 75,
                },
                {
                    exchange: "DERIBIT",
                    symbol: "BTC-PERP",
                    venueSymbol: "BTC-PERPETUAL",
                    status: qsTr("Demo: opcje zabezpieczające"),
                    offline: false,
                    automationRunning: false,
                    fpsTarget: 60,
                },
            ],
            instrumentDetails: {
                exchange: "OKX",
                symbol: "BTC/USDT",
                venueSymbol: "BTC-USDT-SWAP",
                quoteCurrency: "USDT",
                baseCurrency: "BTC",
                granularity: "PT5M",
            },
            aiConfiguration: {
                policy: "market_neutral",
                decision_window: "PT1M",
                risk_profile: "conservative",
                overrides: [
                    { profile: "hedge", maxDrawdown: 0.025, takeProfit: 0.006 },
                ],
                features: ["zscore", "spread_mean", "inventory_skew"],
                modelRevision: "2024.03-neutral",
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
                maxExposureUtilization: 0.54,
                anyBreach: false,
                totalBreaches: 0,
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
                    { code: "BASIS_SPREAD", current: 0.12, max: 0.25, threshold: 0.2, breach: false },
                ],
            },
            runtimeStatus: {
                connection: qsTr("Demo: tryb market neutral"),
                reduceMotion: false,
                offlineMode: false,
                offlineDaemonStatus: qsTr("Stabilny"),
                automationRunning: true,
                performanceGuard: { fpsTarget: 75, jankThresholdMs: 16.0, maxOverlayCount: 2 },
                riskRefresh: {
                    enabled: true,
                    intervalSeconds: 45,
                    nextRefreshInSeconds: 22,
                    nextRefreshDueAt: "2024-05-18T09:05:22+02:00",
                    lastUpdateAt: "2024-05-18T09:04:37+02:00",
                },
            },
            controlState: {
                schedulerRunning: true,
                automationRunning: true,
                offlineMode: false,
                lastActionMessage: qsTr("Hedge demo aktywne"),
                lastActionSuccess: true,
                lastActionAt: "2024-05-18T09:04:10+02:00",
                lastRiskRefreshAt: "2024-05-18T09:04:37+02:00",
                nextRiskRefreshDueAt: "2024-05-18T09:05:22+02:00",
                manualRefreshCount: 1,
            },
            licenseStatus: {
                active: true,
                edition: "OEM Enterprise",
                licenseId: "DEMO-HEDGE",
                holderName: "Core Labs Hedge Demo",
                maintenanceActive: true,
                modules: ["hedge", "risk", "scheduler"],
                runtime: ["desktop-shell", "risk-service"],
            },
        },
    ];
}

function findPreset(presets, id) {
    if (!id)
        return null;
    for (let i = 0; i < presets.length; ++i) {
        if (presets[i].id === id)
            return presets[i];
    }
    return null;
}

function clonePreset(preset) {
    return JSON.parse(JSON.stringify(preset));
}
