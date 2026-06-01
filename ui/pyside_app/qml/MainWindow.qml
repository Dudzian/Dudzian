import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "components" as Components
import "components/layout" as LayoutComponents
import Styles 1.0 as StylesModule
import "views" as Views

ApplicationWindow {
    id: root
    width: 1280
    height: 720
    visible: true
    title: qsTr("Dudzian Product Preview — safe dry-run")
    color: designSystem.color("background")
    property var contextGrpcBridge: (typeof grpcBridge !== "undefined" ? grpcBridge : null)
    property var runtimeService: contextGrpcBridge && contextGrpcBridge.runtimeService ? contextGrpcBridge.runtimeService : null
    property var contextRuntimeState: (typeof runtimeState !== "undefined" ? runtimeState : null)
    property string defaultPanelId: "sidePanel"
    property string currentPanelId: defaultPanelId
    readonly property var rootDesignSystem: designSystem

    // UI-PREVIEW-7.4 local-only product preview state. Safe dry-run/paper preview only: live trading disabled, exchange route disabled, order submission disabled, API keys not required.
    property var selectedExchanges: ["Paper Preview Catalog"]
    property var selectedPairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    property var whitelistPairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    property var blacklistPairs: []
    property var activeStrategies: ["Momentum Guard", "Range Guard"]
    property string paperSessionState: "stopped"
    property int paperTicks: 0
    property int decisionSequence: 0
    property int telemetryTick: 0
    property real previewEquity: 100000.0
    property real previewPnl: 0.0
    property alias mockEquity: root.previewEquity
    property alias mockPnl: root.previewPnl
    property int paperOrdersCount: 0
    property int blockedOrdersCount: 0
    property int noOrderCount: 1
    property int simulatedOrdersCount: 0
    property var paperOrdersPreview: [
        ({ timestamp: "12:00:01Z", pair: "BTC/USDT", action: "HOLD", status: "no order", confidence: "0.81", reason: "Preview guard held the setup; order submission disabled." })
    ]
    property var openPaperPositions: [
        ({ pair: "BTC/USDT", side: "paper long", size: "0.012", pnl: "+42.10", label: "simulated" }),
        ({ pair: "SOL/USDT", side: "watch", size: "0", pnl: "0.00", label: "no order" })
    ]
    property var closedPaperTrades: [
        ({ pair: "ETH/USDT", side: "paper sell", pnl: "+18.44", label: "simulated" })
    ]
    property string lastGovernorDecision: "BTC/USDT HOLD • confidence 0.81 • NO ORDER — preview only"
    property string autonomyMode: "Supervised dry-run"
    property int autonomyLevel: 2
    property int modelReadiness: 72
    property int trainingCoverage: 68
    property int dataCoverage: 74
    property int confidenceThreshold: 75
    property string decisionPolicyPreview: "Balanced policy"
    property bool riskLocked: true
    property string riskProfile: "Balanced"
    property string riskState: "guarded preview • kill-switch armed • live blocked"
    property string maxPosition: "2,500 USDT"
    property int maxOpenPositions: 4
    property string stopLoss: "2.4%"
    property string takeProfit: "4.8%"
    property string maxSlippage: "0.20%"
    property string maxDrawdown: "6.0%"
    property string dailyLossLimit: "1,200 USDT"
    property bool liveTradingDisabled: true
    property bool exchangeIoDisabled: true
    property bool orderSubmissionDisabled: true
    property bool apiKeysRequired: false
    property bool runtimeLoopStarted: false
    property bool marketsImported: false
    property string marketSearch: ""
    property string marketQuoteFilter: "All"
    property string marketCategoryFilter: "All"
    property bool marketSelectedOnly: false
    property bool marketAiCandidatesOnly: false
    property bool marketExcludedOnly: false
    property string decisionFilter: "all"
    property string decisionPairFilter: "All pairs"
    property var previewMarketPairs: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT",
        "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "APT/USDT",
        "SUI/USDT", "TON/USDT", "PEPE/USDT", "WIF/USDT", "FET/USDT", "RENDER/USDT",
        "TAO/USDT", "UNI/USDT", "AAVE/USDT", "ETC/USDT", "FIL/USDT", "ICP/USDT",
        "MATIC/USDT", "SEI/USDT", "TIA/USDT", "JUP/USDT", "PYTH/USDT", "WLD/USDT",
        "GRT/USDT", "RNDR/USDT", "MKR/USDT", "LDO/USDT", "CRV/USDT", "SAND/USDT",
        "MANA/USDT", "AXS/USDT", "GALA/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT",
        "ENA/USDT", "JTO/USDT", "STX/USDT", "KAS/USDT", "ALGO/USDT", "VET/USDT",
        "HBAR/USDT", "EGLD/USDT", "RUNE/USDT", "FTM/USDT", "IMX/USDT", "STRK/USDT",
        "ZK/USDT", "MANTA/USDT", "BLUR/USDT", "DYDX/USDT", "SNX/USDT", "COMP/USDT",
        "YFI/USDT", "SUSHI/USDT", "1INCH/USDT", "CAKE/USDT", "PENDLE/USDT", "ONDO/USDT",
        "BEAM/USDT", "ROSE/USDT", "MINA/USDT", "AR/USDT", "ENS/USDT", "MASK/USDT",
        "CHZ/USDT", "APE/USDT", "BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC",
        "XRP/USDC", "ADA/USDC", "DOGE/USDC", "AVAX/USDC", "DOT/USDC", "LINK/USDC",
        "LTC/USDC", "BCH/USDC", "ATOM/USDC", "NEAR/USDC", "ARB/USDC", "OP/USDC",
        "INJ/USDC", "APT/USDC", "SUI/USDC", "TON/USDC", "PEPE/USDC", "WIF/USDC",
        "FET/USDC", "RENDER/USDC", "TAO/USDC", "UNI/USDC", "AAVE/USDC", "ETC/USDC",
        "FIL/USDC", "ICP/USDC", "MATIC/USDC", "SEI/USDC", "TIA/USDC", "JUP/USDC",
        "PYTH/USDC", "WLD/USDC", "GRT/USDC", "RNDR/USDC", "MKR/USDC", "LDO/USDC",
        "CRV/USDC", "SAND/USDC", "MANA/USDC", "AXS/USDC", "GALA/USDC", "SHIB/USDC",
        "FLOKI/USDC", "BONK/USDC"
    ]
    property var previewExchanges: [
        "Binance", "Bybit", "OKX", "KuCoin", "Coinbase", "Kraken", "Bitget", "Gate.io", "MEXC", "Paper Preview Catalog"
    ]
    property var decisionPreviewRows: [
        ({ timestamp: "12:04:18Z", symbol: "BTC/USDT", action: "HOLD", confidence: "0.81", reason: "Momentum neutral; confidence floor not reached.", riskReason: "Position cap unused; drawdown guard inside preview limit.", strategy: "Momentum Guard", safety: "NO ORDER — preview only", paperState: "stopped" }),
        ({ timestamp: "12:03:42Z", symbol: "ETH/USDT", action: "WAIT", confidence: "0.74", reason: "Coverage check asks for a fresher candle batch.", riskReason: "Risk governor waits for lower slippage.", strategy: "Range Guard", safety: "Exchange route disabled", paperState: "stopped" }),
        ({ timestamp: "12:02:57Z", symbol: "SOL/USDT", action: "BLOCKED LIVE", confidence: "0.69", reason: "Live bridge is intentionally unavailable in preview.", riskReason: "Execution guard blocks the order route.", strategy: "Volatility Breakout Preview", safety: "Live trading disabled • Order submission disabled", paperState: "stopped" }),
        ({ timestamp: "12:01:33Z", symbol: "BNB/USDT", action: "NO ORDER", confidence: "0.61", reason: "Advisory preview rejected low confidence setup.", riskReason: "Below confidence threshold.", strategy: "Strategy governor", safety: "Preview only / no order", paperState: "stopped" })
    ]
    property string telemetryHeartbeat: "12:04:18Z"
    property int telemetryReconnects: 0
    property string telemetryDowntime: "0 ms"
    property string telemetryFreshness: "freshness status: ready • safe preview feed"
    property var telemetryRows: [
        ({ timestamp: "12:04:18Z", message: "heartbeat #0 • feed: safe preview • runtime loop: not started" }),
        ({ timestamp: "12:03:58Z", message: "exchange route: disabled • paper bridge: not connected / planned" })
    ]
    property string diagnosticsBundleStatus: "Last bundle path/status: not generated"

    function hasValue(list, value) {
        return list && list.indexOf(value) >= 0
    }

    function toggledList(list, value) {
        var copy = list ? list.slice() : []
        var idx = copy.indexOf(value)
        if (idx >= 0) copy.splice(idx, 1); else copy.push(value)
        return copy
    }

    function pairQuote(pair) { return pair.split("/")[1] || "" }
    function pairBase(pair) { return pair.split("/")[0] || pair }
    function pairCategory(pair) {
        var base = pairBase(pair)
        if (["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"].indexOf(base) >= 0) return "Major"
        if (["FET", "RENDER", "RNDR", "TAO", "WLD", "GRT", "AI"].indexOf(base) >= 0) return "AI"
        if (["DOGE", "SHIB", "PEPE", "WIF", "FLOKI", "BONK"].indexOf(base) >= 0) return "Meme"
        if (["UNI", "AAVE", "MKR", "LDO", "CRV", "SNX", "COMP", "YFI", "SUSHI", "1INCH", "CAKE", "PENDLE"].indexOf(base) >= 0) return "DeFi"
        if (["SOL", "AVAX", "DOT", "NEAR", "APT", "SUI", "TON", "ATOM", "SEI", "TIA", "KAS", "ALGO", "VET", "HBAR", "EGLD", "RUNE", "FTM"].indexOf(base) >= 0) return "Layer1"
        if (["ARB", "OP", "MATIC", "IMX", "STRK", "ZK", "MANTA"].indexOf(base) >= 0) return "Layer2"
        if (["BTC", "ETH", "SOL", "BNB", "XRP", "LINK", "LTC", "BCH", "DOGE", "ADA", "AVAX", "TON"].indexOf(base) >= 0) return "High volume"
        return "High volume"
    }
    function isAiCandidate(pair) { return ["BTC", "ETH", "SOL", "INJ", "TAO", "RENDER", "RNDR", "FET", "ARB", "OP", "PYTH", "WLD", "GRT"].indexOf(pairBase(pair)) >= 0 }
    function filteredMarketPairs() {
        var term = marketSearch.toLowerCase()
        var out = []
        for (var i = 0; i < previewMarketPairs.length; ++i) {
            var pair = previewMarketPairs[i]
            if (term.length > 0 && pair.toLowerCase().indexOf(term) < 0) continue
            if (marketQuoteFilter !== "All" && pairQuote(pair) !== marketQuoteFilter) continue
            if (marketCategoryFilter !== "All" && pairCategory(pair) !== marketCategoryFilter) continue
            if (marketSelectedOnly && !hasValue(selectedPairs, pair)) continue
            if (marketAiCandidatesOnly && !isAiCandidate(pair)) continue
            if (marketExcludedOnly && !hasValue(blacklistPairs, pair)) continue
            out.push(pair)
        }
        return out
    }

    function toggleExchange(exchange) { selectedExchanges = toggledList(selectedExchanges, exchange) }
    function togglePair(pair) { selectedPairs = toggledList(selectedPairs, pair); whitelistPairs = selectedPairs.slice() }
    function toggleBlacklist(pair) { blacklistPairs = toggledList(blacklistPairs, pair) }
    function importMarketsPreview() { marketsImported = true }
    function selectAllVisiblePairs() { marketsImported = true; selectedPairs = filteredMarketPairs().slice(); whitelistPairs = selectedPairs.slice() }
    function selectAllPairs() { selectAllVisiblePairs() }
    function clearSelectedPairs() { selectedPairs = []; whitelistPairs = [] }
    function selectTop20Pairs() { marketsImported = true; selectedPairs = filteredMarketPairs().slice(0, 20); whitelistPairs = selectedPairs.slice() }
    function blacklistSelectedPairs() { blacklistPairs = selectedPairs.slice() }
    function whitelistSelectedPairs() { whitelistPairs = selectedPairs.slice(); blacklistPairs = blacklistPairs.filter(function(pair) { return selectedPairs.indexOf(pair) < 0 }) }
    function setAutonomyMode(mode) { autonomyMode = mode; autonomyLevel = mode === "Advisory" ? 1 : (mode === "Supervised dry-run" ? 2 : (mode === "Autonomous paper" ? 4 : 0)) }
    function setDecisionPolicy(policy) { decisionPolicyPreview = policy }
    function setConfidenceThreshold(value) { confidenceThreshold = value }
    function setRiskProfile(profile) {
        riskProfile = profile
        if (profile === "Conservative") { maxPosition = "1,000 USDT"; maxOpenPositions = 2; stopLoss = "1.5%"; takeProfit = "3.0%"; maxSlippage = "0.10%"; maxDrawdown = "3.0%"; dailyLossLimit = "500 USDT" }
        else if (profile === "Aggressive") { maxPosition = "5,000 USDT"; maxOpenPositions = 8; stopLoss = "4.0%"; takeProfit = "8.0%"; maxSlippage = "0.35%"; maxDrawdown = "10.0%"; dailyLossLimit = "2,500 USDT" }
        else { maxPosition = "2,500 USDT"; maxOpenPositions = 4; stopLoss = "2.4%"; takeProfit = "4.8%"; maxSlippage = "0.20%"; maxDrawdown = "6.0%"; dailyLossLimit = "1,200 USDT" }
        riskState = profile + " preview • daily loss limit active • live blocked"
        riskLocked = true
    }
    function setStrategyActive(name, enabled) {
        var active = activeStrategies.slice()
        var idx = active.indexOf(name)
        if (enabled && idx < 0) active.push(name)
        if (!enabled && idx >= 0) active.splice(idx, 1)
        activeStrategies = active
    }
    function previewTime(offset) {
        var total = 12 * 3600 + 5 * 60 + paperTicks * 7 + decisionSequence * 3 + telemetryTick + (offset || 0)
        var h = Math.floor(total / 3600) % 24
        var m = Math.floor((total % 3600) / 60)
        var sec = total % 60
        return (h < 10 ? "0" + h : h) + ":" + (m < 10 ? "0" + m : m) + ":" + (sec < 10 ? "0" + sec : sec) + "Z"
    }
    function actionStatus(action) {
        if (action === "BLOCKED LIVE") return "blocked"
        if (action === "NO ORDER" || action === "HOLD" || action === "WAIT") return "no order"
        return "simulated"
    }
    function recountOrderCounters() {
        var paper = 0, blocked = 0, none = 0, simulated = 0
        for (var i = 0; i < paperOrdersPreview.length; ++i) {
            var status = paperOrdersPreview[i].status
            if (status === "blocked") blocked += 1
            else if (status === "no order") none += 1
            else if (status === "simulated") simulated += 1
        }
        paperOrdersCount = paperOrdersPreview.length
        blockedOrdersCount = blocked
        noOrderCount = none
        simulatedOrdersCount = simulated
    }
    function addDecision(action, reason) {
        decisionSequence += 1
        var pair = selectedPairs.length > 0 ? selectedPairs[decisionSequence % selectedPairs.length] : "BTC/USDT"
        var strategy = activeStrategies.length > 0 ? activeStrategies[decisionSequence % activeStrategies.length] : "Strategy governor"
        var confidence = (0.58 + ((decisionSequence % 11) * 0.033)).toFixed(2)
        var row = ({ timestamp: previewTime(decisionSequence), symbol: pair, action: action, confidence: confidence, reason: reason, riskReason: riskState, strategy: strategy, safety: "Live trading disabled • Exchange route disabled • Order submission disabled", paperState: paperSessionState })
        var rows = decisionPreviewRows.slice()
        rows.unshift(row)
        decisionPreviewRows = rows.slice(0, 12)
        lastGovernorDecision = pair + " " + action + " • confidence " + confidence + " • " + reason
        return row
    }
    function generateGovernorRecommendation() {
        var actions = ["PAPER BUY", "PAPER SELL", "HOLD", "WAIT", "NO ORDER", "BLOCKED LIVE"]
        var action = actions[decisionSequence % actions.length]
        addDecision(action, "Governor recommendation updated the preview stream; sandbox/testnet bridge remains planned and disabled.")
    }
    function generateNextDecision() {
        var actions = ["PAPER BUY", "PAPER SELL", "HOLD", "WAIT", "NO ORDER", "BLOCKED LIVE"]
        addDecision(actions[decisionSequence % actions.length], "Generated from selected pairs and active strategy preview state.")
    }
    function generatePaperTick() {
        paperTicks += 1
        paperSessionState = paperSessionState === "stopped" ? "running" : paperSessionState
        var actions = ["PAPER BUY", "PAPER SELL", "HOLD", "WAIT", "NO ORDER", "BLOCKED LIVE"]
        var action = actions[paperTicks % actions.length]
        var status = actionStatus(action)
        var pair = selectedPairs.length > 0 ? selectedPairs[paperTicks % selectedPairs.length] : "BTC/USDT"
        var confidence = (0.60 + ((paperTicks % 10) * 0.031)).toFixed(2)
        previewPnl = Number((previewPnl + (status === "simulated" ? 18.5 : (status === "blocked" ? 0 : -2.25))).toFixed(2))
        previewEquity = Number((100000 + previewPnl).toFixed(2))
        var order = ({ timestamp: previewTime(1), pair: pair, action: action, status: status, confidence: confidence, reason: status === "simulated" ? "Paper preview fill recorded locally; no real route used." : (status === "blocked" ? "Live order route blocked by preview guard." : "No order emitted by governor policy.") })
        var orders = paperOrdersPreview.slice()
        orders.unshift(order)
        paperOrdersPreview = orders.slice(0, 10)
        recountOrderCounters()
        addDecision(action, order.reason)
    }
    function runTenMockTicks() { for (var i = 0; i < 10; ++i) generatePaperTick() }
    function startPaperPreview() { paperSessionState = "running"; generatePaperTick() }
    function pausePaperPreview() { paperSessionState = "paused" }
    function stopPaperPreview() { paperSessionState = "stopped" }
    function resetPaperPreview() { paperSessionState = "stopped"; paperTicks = 0; decisionSequence += 1; previewEquity = 100000.0; previewPnl = 0.0; paperOrdersPreview = []; recountOrderCounters(); addDecision("NO ORDER", "Paper preview reset; order submission remains disabled.") }
    function pingTelemetryFeed() {
        telemetryTick += 1
        telemetryHeartbeat = previewTime(telemetryTick)
        telemetryFreshness = "freshness status: heartbeat #" + telemetryTick + " • updated " + telemetryHeartbeat
        var messages = [
            "heartbeat #" + telemetryTick + " • feed: safe preview • runtime loop: not started",
            "market catalog scan #" + telemetryTick + " • exchange route: disabled",
            "paper bridge check #" + telemetryTick + " • not connected / planned",
            "decision stream pulse #" + telemetryTick + " • order submission disabled"
        ]
        var rows = telemetryRows.slice()
        rows.unshift(({ timestamp: telemetryHeartbeat, message: messages[telemetryTick % messages.length] }))
        telemetryRows = rows.slice(0, 10)
    }
    function generateDiagnosticBundle() { diagnosticsBundleStatus = "Last bundle path/status: var/tmp/preview-diagnostic-bundle-ui-only.zip • generated local diagnostic status • included UI smoke metadata • excluded secrets, env files, keychain and real exchange state" }

    function showPanel(panelId) {
        if (!panelId)
            return
        currentPanelId = panelId
        if (layoutController)
            layoutController.setPanelVisibility(panelId, true)
    }

    function showOperatorDashboard() {
        showPanel(defaultPanelId)
    }

    function selectedPanelComponent() {
        if (panelRegistry && panelRegistry[currentPanelId] && panelRegistry[currentPanelId].component)
            return panelRegistry[currentPanelId].component
        return sidePanelComponent
    }

    property var panelMetadata: [
        ({ panelId: "sidePanel", title: qsTr("Dashboard"), icon: "fingerprint", defaultColumn: 0, defaultOrder: 0 }),
        ({ panelId: "aiCenterPanel", title: qsTr("AI Center"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 1 }),
        ({ panelId: "tradingUniversePanel", title: qsTr("Trading Universe"), icon: "cloud", defaultColumn: 0, defaultOrder: 2 }),
        ({ panelId: "strategiesPanel", title: qsTr("Strategie"), icon: "strategy_manager", defaultColumn: 0, defaultOrder: 3 }),
        ({ panelId: "riskControlsPanel", title: qsTr("Ryzyko"), icon: "shield", defaultColumn: 0, defaultOrder: 4 }),
        ({ panelId: "aiDecisionsPanel", title: qsTr("Decyzje"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 5 }),
        ({ panelId: "telemetryPanel", title: qsTr("Telemetria"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 6 }),
        ({ panelId: "diagnosticsPanel", title: qsTr("Diagnostyka"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 7 })
    ]

    property var productTabs: panelMetadata

    property var panelRegistry: ({
        "sidePanel": { title: qsTr("Dashboard"), icon: "fingerprint", component: sidePanelComponent },
        "aiCenterPanel": { title: qsTr("AI Center"), icon: "mode_wizard", component: aiCenterPanelComponent },
        "tradingUniversePanel": { title: qsTr("Trading Universe"), icon: "cloud", component: tradingUniversePanelComponent },
        "strategiesPanel": { title: qsTr("Strategie"), icon: "strategy_manager", component: strategiesPanelComponent },
        "riskControlsPanel": { title: qsTr("Ryzyko"), icon: "shield", component: riskControlsPanelComponent },
        "aiDecisionsPanel": { title: qsTr("Decyzje"), icon: "mode_wizard", component: aiDecisionsPanelComponent },
        "telemetryPanel": { title: qsTr("Telemetria"), icon: "diagnostics", component: telemetryPanelComponent },
        "diagnosticsPanel": { title: qsTr("Diagnostyka"), icon: "diagnostics", component: diagnosticsPanelComponent },
        "chartView": { title: qsTr("Strumień decyzji"), icon: "cloud", component: chartViewComponent },
        "strategyWorkbench": { title: qsTr("Warsztat strategii"), icon: "package", component: strategyWorkbenchComponent },
        "modeWizardPanel": { title: qsTr("Tryby pracy"), icon: "mode_wizard", component: modeWizardPanelComponent },
        "strategyManagerPanel": { title: qsTr("Menedżer strategii"), icon: "strategy_manager", component: strategyManagerPanelComponent }
    })

    StylesModule.DesignSystem {
        id: designSystem
        themeBridge: theme
    }

    Dialog {
        id: startupDialog
        modal: true
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        title: qsTr("Stan backendu")
        property string body: ""
        onAccepted: visible = false

        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 12

            Label {
                id: statusBody
                text: startupDialog.body
                wrapMode: Text.WordWrap
                color: designSystem.color("textPrimary")
                Layout.preferredWidth: 420
            }

            Label {
                text: qsTr("Jeśli problem dotyczy konfiguracji, sprawdź plik runtime.yaml lub flagę cloud.")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
                visible: startupDialog.title.indexOf(qsTr("Błąd")) !== -1
            }
        }
    }

    Connections {
        target: runtimeService
        function onErrorMessageChanged() {
            if (!runtimeService)
                return
            if (runtimeService.errorMessage && runtimeService.errorMessage.length > 0) {
                startupDialog.title = qsTr("Błąd uruchomienia runtime")
                startupDialog.body = runtimeService.errorMessage
                startupDialog.open()
            }
        }
        function onCloudRuntimeStatusChanged() {
            if (!runtimeService)
                return
            const status = runtimeService.cloudRuntimeStatus || {}
            if (status.status === "ready") {
                const targetLabel = status.target || (cloudRuntimeEnabled ? qsTr("profil cloud") : qsTr("tryb lokalny"))
                startupDialog.title = qsTr("Runtime gotowy")
                startupDialog.body = qsTr("Połączenie z backendem %1 aktywne.").arg(targetLabel)
                startupDialog.open()
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
            GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
        }
        z: -2
    }

    header: ToolBar {
        id: toolbar
        implicitHeight: 64
        background: Item {
            anchors.fill: parent
            Rectangle {
                id: toolbarGradient
                anchors.fill: parent
                gradient: Gradient {
                    GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
                    GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
                }
                opacity: 0.9
            }
            MultiEffect {
                anchors.fill: parent
                source: toolbarGradient
                blurEnabled: true
                blur: 1.0
                blurMax: 24
                saturation: 0.95
                brightness: 0.05
            }
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12

            ColumnLayout {
                Layout.alignment: Qt.AlignVCenter
                spacing: 2
                Label {
                    text: qsTr("Dudzian Bot Preview")
                    font.bold: true
                    font.pixelSize: 16
                    color: designSystem.color("textPrimary")
                }
                Label {
                    text: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled")
                    color: designSystem.color("textSecondary")
                    font.pixelSize: 11
                }
            }

            Rectangle { width: 1; height: parent.height * 0.6; color: designSystem.color("border"); opacity: 0.45 }

            Flickable {
                id: tabOverflow
                objectName: "productPreviewTabBar"
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                Layout.alignment: Qt.AlignVCenter
                clip: true
                contentWidth: browserTabBar.implicitWidth
                boundsBehavior: Flickable.StopAtBounds
                flickableDirection: Flickable.HorizontalFlick

                Row {
                    id: browserTabBar
                    height: parent.height
                    spacing: 6
                    Repeater {
                        model: root.productTabs
                        delegate: Rectangle {
                            required property var modelData
                            property bool hovered: false
                            readonly property bool active: root.currentPanelId === modelData.panelId
                            width: Math.max(120, tabLabel.implicitWidth + 34)
                            height: 42
                            radius: 14
                            color: active ? designSystem.color("surface") : (hovered ? Qt.rgba(0.33, 0.78, 1.0, 0.12) : Qt.rgba(0, 0, 0, 0.20))
                            border.color: active ? designSystem.color("accent") : (hovered ? designSystem.color("textSecondary") : designSystem.color("border"))
                            border.width: active ? 2 : 1
                            Rectangle {
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                height: active ? 3 : 0
                                radius: 2
                                color: designSystem.color("accent")
                            }
                            Label {
                                id: tabLabel
                                anchors.centerIn: parent
                                text: modelData.title
                                font.bold: active
                                color: active ? designSystem.color("textPrimary") : designSystem.color("textSecondary")
                            }
                            MouseArea {
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onEntered: parent.hovered = true
                                onExited: parent.hovered = false
                                onPressed: parent.opacity = 0.82
                                onReleased: parent.opacity = 1.0
                                onClicked: root.showPanel(modelData.panelId)
                            }
                        }
                    }
                }
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: qsTr("Odśwież preview")
                iconName: "refresh"
                backgroundColor: designSystem.color("accent")
                foregroundColor: designSystem.color("surface")
                onClicked: runtimeService && runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
            }
        }
    }

    Rectangle {
        id: centralContentRoot
        objectName: "centralContentRoot"
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: parent.top
        anchors.margins: 16
        radius: 28
        color: Qt.rgba(0, 0, 0, 0.08)
        border.color: designSystem.color("border")
        border.width: 1

        Loader {
            id: centralContentLoader
            objectName: "centralContentLoader"
            anchors.fill: parent
            anchors.margins: 0
            active: true
            sourceComponent: root.selectedPanelComponent()
        }
    }

    LayoutComponents.DockManager {
        id: dockManager
        anchors.fill: centralContentRoot
        layoutController: layoutController
        panelRegistry: panelRegistry
        designSystem: rootDesignSystem
        visible: false
    }

    Component {
        id: sidePanelComponent
        Views.OperatorDashboard {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: aiCenterPanelComponent
        Views.AiControlCenter {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: tradingUniversePanelComponent
        Views.TradingUniverse {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: telemetryPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "telemetryFeedPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "telemetryTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "telemetryFeedPreviewTitle"; text: qsTr("Telemetria"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: qsTr("Safe preview telemetry feed with heartbeat/tick source state, freshness status and bounded 8–12 visible rows."); wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: width > 900 ? 4 : 2
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Feed status"); description: qsTr("feed: safe preview"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("runtime loop"); description: qsTr("not started"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("exchange route"); description: qsTr("disabled"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("paper bridge"); description: qsTr("not connected / planned"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Reconnects"); description: String(root.telemetryReconnects); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Downtime"); description: root.telemetryDowntime; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Last heartbeat"); description: root.telemetryHeartbeat; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Data freshness"); description: root.telemetryFreshness; Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Telemetry heartbeat feed")
                    description: qsTr("Ping feed increments heartbeat/tick, appends varied rows, avoids duplicate timestamps and caps the visible feed.")
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Ping feed"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.pingTelemetryFeed() }
                        Label { text: qsTr("heartbeat/tick source state: %1 • %2").arg(root.telemetryTick).arg(root.telemetryFreshness); color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                    ListView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 330
                        clip: true
                        spacing: 8
                        model: root.telemetryRows
                        delegate: Rectangle {
                            required property var modelData
                            width: ListView.view ? ListView.view.width : 900
                            height: telemetryRow.implicitHeight + 16
                            radius: 12
                            color: designSystem.color("surfaceMuted")
                            border.color: designSystem.color("border")
                            ColumnLayout {
                                id: telemetryRow
                                anchors.fill: parent
                                anchors.margins: 10
                                Label { text: modelData.timestamp; color: designSystem.color("textPrimary"); font.bold: true }
                                Label { text: modelData.message; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: aiDecisionsPanelComponent
        Views.AiDecisionsView {
            previewState: root
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: chartViewComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "decisionStreamPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label {
                    objectName: "decisionStreamPreviewTitle"
                    text: qsTr("Strumień decyzji i dziennik governora")
                    font.bold: true
                    font.pixelSize: 22
                    color: designSystem.color("textPrimary")
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Wykres confidence oraz dziennik zdarzeń w trybie demo/offline. Order execution disabled.")
                    wrapMode: Text.WordWrap
                    color: designSystem.color("textSecondary")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Confidence preview")
                    description: qsTr("Canvas zachowany, z poprawionym paddingiem i pustym stanem.")
                    Canvas {
                        id: chartCanvas
                        Layout.fillWidth: true
                        Layout.preferredHeight: 180
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            ctx.fillStyle = designSystem.color("surfaceMuted")
                            ctx.fillRect(0, 0, width, height)
                            var data = runtimeService ? runtimeService.decisions || [] : []
                            if (data.length === 0) {
                                ctx.strokeStyle = designSystem.color("border")
                                ctx.lineWidth = 1
                                for (var g = 1; g < 4; ++g) {
                                    ctx.beginPath(); ctx.moveTo(0, height * g / 4); ctx.lineTo(width, height * g / 4); ctx.stroke()
                                }
                                return
                            }
                            var windowSize = Math.min(40, data.length)
                            var step = width / Math.max(windowSize - 1, 1)
                            ctx.strokeStyle = designSystem.color("accent")
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            for (var i = 0; i < windowSize; ++i) {
                                var entry = data[data.length - windowSize + i]
                                var confidence = entry.decision && entry.decision.confidence !== undefined ? Number(entry.decision.confidence) : 0.35
                                confidence = Math.max(0.05, Math.min(confidence, 1.0))
                                var x = i * step
                                var y = height - (confidence * height)
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                            }
                            ctx.stroke()
                        }
                    }
                }
                Connections {
                    target: runtimeService
                    function onDecisionsChanged() { chartCanvas.requestPaint() }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Zdarzenia governora")
                    description: (!runtimeService || !runtimeService.decisions || runtimeService.decisions.length === 0)
                                 ? qsTr("Brak danych live — pokazuję pusty stan demo/offline.")
                                 : qsTr("Ostatnie decyzje z lokalnego preview bridge.")
                    ListView {
                        id: decisionList
                        Layout.fillWidth: true
                        Layout.preferredHeight: 260
                        model: runtimeService ? runtimeService.decisions : []
                        clip: true
                        spacing: 8
                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                            width: 10
                            background: Rectangle { radius: 5; color: Qt.rgba(1, 1, 1, 0.04) }
                            contentItem: Rectangle { radius: 4; color: designSystem.color("surfaceElevated"); border.color: designSystem.color("border"); border.width: 1 }
                        }
                        delegate: Rectangle {
                            width: ListView.view.width
                            color: designSystem.color("surfaceMuted")
                            height: column.implicitHeight + 18
                            radius: 12
                            border.color: designSystem.color("border")
                            border.width: 1
                            Column {
                                id: column
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 4
                                Label {
                                    text: qsTr("%1 • %2 • %3").arg(modelData.timestamp || "-").arg(modelData.portfolio || "-").arg(modelData.marketRegime && modelData.marketRegime.label ? modelData.marketRegime.label : "")
                                    font.bold: true
                                    color: designSystem.color("textPrimary")
                                    wrapMode: Text.Wrap
                                }
                                Label {
                                    text: modelData.decision && modelData.decision.shouldTrade ? qsTr("Decyzja: %1 %2 @ %3").arg(modelData.symbol || "-").arg(modelData.side || "").arg(modelData.price || "") : qsTr("Decyzja: brak transakcji")
                                    wrapMode: Text.Wrap
                                    color: designSystem.color("textPrimary")
                                }
                                Label {
                                    text: modelData.ai && modelData.ai.strategy ? qsTr("Governor: %1").arg(modelData.ai.strategy) : ""
                                    color: designSystem.color("textSecondary")
                                    visible: text.length > 0
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: strategyWorkbenchComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "strategyWorkbenchPreviewPanel"
            contentWidth: availableWidth
            clip: true
            property var strategies: []
            property var marketplacePresets: strategyManagementController ? strategyManagementController.presets : []
            function rebuild() {
                var data = runtimeService ? runtimeService.decisions || [] : []
                var stats = {}
                for (var i = 0; i < data.length; ++i) {
                    var entry = data[i]
                    var strategy = entry.ai && entry.ai.strategy ? entry.ai.strategy : qsTr("Nieznana strategia")
                    if (!stats[strategy]) stats[strategy] = { count: 0, lastSymbol: entry.symbol || "-" }
                    stats[strategy].count += 1
                    stats[strategy].lastSymbol = entry.symbol || stats[strategy].lastSymbol
                }
                var collection = []
                for (var key in stats) collection.push({ name: key, count: stats[key].count, symbol: stats[key].lastSymbol })
                collection.sort(function(a, b) { return b.count - a.count })
                strategies = collection
            }
            Component.onCompleted: rebuild()
            Connections { target: runtimeService; function onDecisionsChanged() { rebuild() } }
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label { objectName: "strategyWorkbenchPreviewTitle"; text: qsTr("Warsztat strategii"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary") }
                Label { text: qsTr("Demo/offline workspace do analizy strategii bez uruchamiania live tradingu ani order execution."); wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Sygnały"); description: strategies.length > 0 ? qsTr("%1 strategii z decyzji preview").arg(strategies.length) : qsTr("Brak live danych — statyczny empty state demo/offline"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Marketplace"); description: marketplacePresets.length > 0 ? qsTr("Presety dostępne lokalnie") : qsTr("Marketplace unavailable w tym preview"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety"); description: qsTr("Runtime loop not started, API keys not required"); Layout.fillWidth: true }
                }
            }
        }
    }

    Component {
        id: strategiesPanelComponent
        Views.Strategies { previewState: root; Layout.fillWidth: true; Layout.fillHeight: true; runtimeService: runtimeService; designSystem: rootDesignSystem }
    }

    Component {
        id: riskControlsPanelComponent
        Views.RiskControls { previewState: root; Layout.fillWidth: true; Layout.fillHeight: true; runtimeService: runtimeService; designSystem: rootDesignSystem }
    }

    Component {
        id: modeWizardPanelComponent
        Views.ModeWizard { width: parent ? parent.width : 900; height: parent ? parent.height : 620; designSystem: rootDesignSystem; modeWizardController: modeWizardController; compact: true; onLaunchWizardRequested: modeWizardDialog.open(); layoutController: layoutController; strategyManagementController: strategyManagementController }
    }

    Component {
        id: strategyManagerPanelComponent
        Views.StrategyManager { width: parent ? parent.width : 900; height: parent ? parent.height : 620; designSystem: rootDesignSystem; strategyManagementController: strategyManagementController; layoutController: layoutController }
    }

    Component {
        id: diagnosticsPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "diagnosticsPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "diagnosticsTitleAccentBar"; Layout.preferredWidth: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "diagnosticsPreviewTitle"; text: qsTr("Diagnostyka"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: qsTr("Preview diagnostics readiness panel. Generate local diagnostic status updates UI text only and does not read secrets, env files, keychain or real environment values."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: width > 900 ? 3 : 1
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Preview diagnostics readiness"); description: qsTr("ready for safe dry-run audit • UI-only bundle status"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Last bundle path/status"); description: root.diagnosticsBundleStatus; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety boundary"); description: qsTr("Live trading disabled • Exchange route disabled • Order submission disabled • API keys not required"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Included • included"); description: qsTr("UI smoke metadata • visible panel state • telemetry heartbeat • governor preview rows"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Excluded • excluded"); description: qsTr("secrets • env files • keychain • real environment values • exchange state"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("generate local diagnostic status"); description: qsTr("local UI status only; no filesystem secret scan and no exchange connection"); Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Generate diagnostic bundle")
                    description: qsTr("Generate diagnostic bundle records a local status line for preview diagnostics; it never starts live/cloud/runtime services.")
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton { designSystem: rootDesignSystem; iconName: "diagnostics"; text: qsTr("Generate diagnostic bundle"); backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.generateDiagnosticBundle() }
                        Label { text: root.diagnosticsBundleStatus; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        if (runtimeService && runtimeService.loadRecentDecisions)
            runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
        if (layoutController && layoutController.registerPanels) {
            layoutController.registerPanels(panelMetadata)
            showOperatorDashboard()
        }
    }

    Timer {
        interval: 15000
        repeat: true
        running: true
        onTriggered: runtimeService && runtimeService.loadRecentDecisions(0)
    }

    Dialog {
        id: modeWizardDialog
        modal: true
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.8, 1100)
        height: Math.min(parent.height * 0.85, 780)
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        Overlay.modal: Rectangle {
            color: Qt.rgba(0, 0, 0, 0.62)
        }
        background: Rectangle {
            anchors.fill: parent
            radius: 24
            color: designSystem.color("surface")
            border.color: designSystem.color("border")
            border.width: 1
        }
        contentItem: Views.ModeWizard {
            anchors.fill: parent
            anchors.margins: 16
            designSystem: rootDesignSystem
            modeWizardController: modeWizardController
            compact: false
            onLaunchWizardRequested: modeWizardDialog.close()
        }
    }
}
