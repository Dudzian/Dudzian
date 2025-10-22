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
    property var performanceComparison: ({})
    property var capitalAllocation: []
    property var runtimeStatus: ({})
    property var licenseStatus: ({})
    property var instrumentDetails: ({})
    property var riskTimeline: []
    property var activityLog: []
    property var openPositions: []
    property var pendingOrders: []
    property var tradeHistory: []
    property var signalAlerts: []
    property var scenarioTests: []
    property var automationRules: []
    property var executionDiagnostics: ({})
    property var marketSentiment: ({})
    property var newsHeadlines: []
    property var controlState: defaultControlState()
    property var complianceSummary: ({})
    property var pendingActivityEvents: []

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

    function normalizeActivityEntry(entry) {
        if (!entry)
            return null
        const timestamp = entry.timestamp || entry.createdAt || entry.time || entry.date || ""
        let normalizedTimestamp = timestamp
        if (timestamp instanceof Date)
            normalizedTimestamp = timestamp.toISOString()
        else if (typeof timestamp === "number") {
            const fromEpoch = new Date(timestamp)
            if (!isNaN(fromEpoch.getTime()))
                normalizedTimestamp = fromEpoch.toISOString()
        } else if (typeof timestamp === "string" && timestamp.length === 0) {
            normalizedTimestamp = new Date().toISOString()
        }
        return {
            timestamp: normalizedTimestamp,
            type: entry.type || entry.category || entry.level || "",
            message: entry.message || entry.title || "",
            success: entry.success !== undefined ? !!entry.success : (entry.level ? entry.level !== "error" : true),
            source: entry.source || "external",
            details: entry.details || entry.meta || ({})
        }
    }

    function normalizeActivityLog(list) {
        const normalized = []
        const items = Array.isArray(list) ? list : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeActivityEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setActivityLog(entries) {
        activityLog = normalizeActivityLog(entries)
    }

    function normalizeRiskTimelineEntry(entry) {
        if (!entry)
            return null

        const timestamp = entry.timestamp || entry.generatedAt || entry.time || entry.date || ""
        let normalizedTimestamp = timestamp
        if (timestamp && typeof timestamp.toISOString === "function")
            normalizedTimestamp = timestamp.toISOString()
        else if (timestamp instanceof Date)
            normalizedTimestamp = timestamp.toISOString()
        else if (typeof timestamp === "number") {
            const fromEpoch = new Date(timestamp)
            if (!isNaN(fromEpoch.getTime()))
                normalizedTimestamp = fromEpoch.toISOString()
        } else if (typeof timestamp === "string" && timestamp.length === 0)
            normalizedTimestamp = new Date().toISOString()
        else if (timestamp && typeof Qt !== "undefined" && Qt.formatDateTime) {
            const formatted = Qt.formatDateTime(timestamp, Qt.ISODateWithMs)
            if (formatted && formatted.length > 0)
                normalizedTimestamp = formatted
        }

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        function asInteger(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : Math.round(number)
        }

        const exposures = Array.isArray(entry.exposures) ? entry.exposures : []

        return {
            timestamp: normalizedTimestamp,
            portfolioValue: asNumber(entry.portfolioValue !== undefined ? entry.portfolioValue : (entry.value !== undefined ? entry.value : 0), 0),
            drawdown: asNumber(entry.drawdown !== undefined ? entry.drawdown : (entry.maxDrawdown !== undefined ? entry.maxDrawdown : (entry.drawdownPct !== undefined ? entry.drawdownPct : 0)), 0),
            leverage: asNumber(entry.leverage !== undefined ? entry.leverage : (entry.maxLeverage !== undefined ? entry.maxLeverage : (entry.averageLeverage !== undefined ? entry.averageLeverage : 0)), 0),
            exposureUtilization: asNumber(entry.exposureUtilization !== undefined ? entry.exposureUtilization : (entry.maxExposureUtilization !== undefined ? entry.maxExposureUtilization : (entry.utilization !== undefined ? entry.utilization : 0)), 0),
            breach: entry.breach !== undefined ? !!entry.breach : (entry.hasBreach !== undefined ? !!entry.hasBreach : !!entry.anyBreach),
            breachCount: asInteger(entry.breachCount !== undefined ? entry.breachCount : (entry.totalBreaches !== undefined ? entry.totalBreaches : (entry.breach ? 1 : 0)), 0),
            profileLabel: entry.profileLabel || entry.profile || "",
            source: entry.source || "live",
            notes: entry.notes || entry.comment || "",
            exposures: exposures,
        }
    }

    function normalizeRiskTimeline(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeRiskTimelineEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setRiskTimeline(entries) {
        riskTimeline = normalizeRiskTimeline(entries)
        recalculateComplianceSummary()
    }

    function normalizePositionEntry(entry) {
        if (!entry)
            return null

        const symbol = entry.symbol || entry.instrument || entry.pair || ""
        const exchange = entry.exchange || entry.venue || entry.market || ""
        const venueSymbol = entry.venueSymbol || entry.marketSymbol || entry.instrumentId || ""
        const account = entry.account || entry.subAccount || ""
        const profile = entry.profile || entry.bucket || ""
        const id = entry.id || entry.positionId || entry.uuid || (symbol && exchange ? exchange + ":" + symbol : symbol)

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        const rawQuantity = asNumber(entry.quantity !== undefined ? entry.quantity : (entry.size !== undefined ? entry.size : (entry.positionSize !== undefined ? entry.positionSize : (entry.contracts !== undefined ? entry.contracts : 0))), 0)
        let side = entry.side || entry.positionSide || entry.direction || ""
        if (!side) {
            if (rawQuantity > 0)
                side = qsTr("Long")
            else if (rawQuantity < 0)
                side = qsTr("Short")
        }
        const quantity = Math.abs(rawQuantity)
        const entryPrice = asNumber(entry.entryPrice !== undefined ? entry.entryPrice : (entry.avgEntryPrice !== undefined ? entry.avgEntryPrice : (entry.averagePrice !== undefined ? entry.averagePrice : (entry.openPrice !== undefined ? entry.openPrice : (entry.price !== undefined ? entry.price : 0)))), 0)
        const markPrice = asNumber(entry.markPrice !== undefined ? entry.markPrice : (entry.lastPrice !== undefined ? entry.lastPrice : (entry.currentPrice !== undefined ? entry.currentPrice : (entry.price !== undefined ? entry.price : entryPrice)))), entryPrice)
        const unrealizedPnl = asNumber(entry.unrealizedPnl !== undefined ? entry.unrealizedPnl : (entry.unrealized !== undefined ? entry.unrealized : (entry.pnl !== undefined ? entry.pnl : 0)), 0)
        let leverage = entry.leverage !== undefined ? Number(entry.leverage) : NaN
        const notional = asNumber(entry.notional !== undefined ? entry.notional : (quantity * markPrice), quantity * markPrice)
        const margin = asNumber(entry.margin !== undefined ? entry.margin : (entry.maintenanceMargin !== undefined ? entry.maintenanceMargin : (entry.initialMargin !== undefined ? entry.initialMargin : 0)), 0)
        if (isNaN(leverage) && margin > 0)
            leverage = notional / margin
        const normalizedLeverage = isNaN(leverage) ? 0 : leverage

        let lastUpdate = entry.lastUpdate || entry.updatedAt || entry.timestamp || entry.time || ""
        if (lastUpdate instanceof Date)
            lastUpdate = lastUpdate.toISOString()
        else if (typeof lastUpdate === "number") {
            const date = new Date(lastUpdate)
            if (!isNaN(date.getTime()))
                lastUpdate = date.toISOString()
        }

        return {
            id: id,
            symbol: symbol,
            exchange: exchange,
            venueSymbol: venueSymbol,
            account: account,
            profile: profile,
            side: side,
            quantity: quantity,
            signedQuantity: rawQuantity,
            entryPrice: entryPrice,
            markPrice: markPrice,
            notional: notional,
            margin: margin,
            leverage: normalizedLeverage,
            unrealizedPnl: unrealizedPnl,
            lastUpdate: lastUpdate,
        }
    }

    function normalizeOpenPositions(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizePositionEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setOpenPositions(entries) {
        openPositions = normalizeOpenPositions(entries)
    }

    function normalizeTradeEntry(entry) {
        if (!entry)
            return null

        const id = entry.id || entry.tradeId || entry.executionId || ""
        const orderId = entry.orderId || entry.order || entry.parentOrderId || ""
        const symbol = entry.symbol || entry.instrument || entry.pair || ""
        const exchange = entry.exchange || entry.venue || entry.market || ""
        const venueSymbol = entry.venueSymbol || entry.marketSymbol || ""
        const account = entry.account || entry.subAccount || ""

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        const rawQuantity = asNumber(entry.quantity !== undefined ? entry.quantity : (entry.size !== undefined ? entry.size : (entry.volume !== undefined ? entry.volume : 0)), 0)
        let side = entry.side || entry.direction || ""
        if (!side) {
            if (rawQuantity > 0)
                side = qsTr("Kupno")
            else if (rawQuantity < 0)
                side = qsTr("Sprzedaż")
        }
        const quantity = Math.abs(rawQuantity)
        const price = asNumber(entry.price !== undefined ? entry.price : (entry.executionPrice !== undefined ? entry.executionPrice : (entry.fillPrice !== undefined ? entry.fillPrice : 0)), 0)
        const pnl = asNumber(entry.pnl !== undefined ? entry.pnl : (entry.realizedPnl !== undefined ? entry.realizedPnl : (entry.realized !== undefined ? entry.realized : 0)), 0)
        const fee = asNumber(entry.fee !== undefined ? entry.fee : (entry.commission !== undefined ? entry.commission : 0), 0)
        const status = entry.status || entry.state || entry.result || ""
        const notes = entry.notes || entry.comment || ""

        let timestamp = entry.timestamp || entry.executedAt || entry.time || entry.date || ""
        if (timestamp instanceof Date)
            timestamp = timestamp.toISOString()
        else if (typeof timestamp === "number") {
            const date = new Date(timestamp)
            if (!isNaN(date.getTime()))
                timestamp = date.toISOString()
        }

        return {
            id: id,
            orderId: orderId,
            symbol: symbol,
            exchange: exchange,
            venueSymbol: venueSymbol,
            account: account,
            side: side,
            quantity: quantity,
            signedQuantity: rawQuantity,
            price: price,
            pnl: pnl,
            fee: fee,
            status: status,
            notes: notes,
            timestamp: timestamp,
        }
    }

    function normalizeTradeHistory(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeTradeEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setTradeHistory(entries) {
        tradeHistory = normalizeTradeHistory(entries)
    }

    function normalizeOrderEntry(entry) {
        if (!entry)
            return null

        const id = entry.id || entry.orderId || entry.clientOrderId || entry.clOrdId || ""
        const clientOrderId = entry.clientOrderId || entry.clOrdId || entry.clientId || ""
        const symbol = entry.symbol || entry.instrument || entry.market || entry.pair || ""
        const exchange = entry.exchange || entry.venue || entry.market || ""
        const venueSymbol = entry.venueSymbol || entry.marketSymbol || entry.exchangeSymbol || ""
        const account = entry.account || entry.subAccount || entry.accountId || ""

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let type = entry.type || entry.orderType || entry.kind || ""
        if (!type && entry.postOnly)
            type = qsTr("Limit")

        let side = entry.side || entry.direction || ""
        const rawQuantity = asNumber(entry.quantity !== undefined ? entry.quantity : (entry.size !== undefined ? entry.size : (entry.amount !== undefined ? entry.amount : 0)), 0)
        const filledQuantity = asNumber(entry.filled !== undefined ? entry.filled : (entry.executedQuantity !== undefined ? entry.executedQuantity : (entry.filledSize !== undefined ? entry.filledSize : 0)), 0)
        const remainingQuantity = asNumber(entry.remaining !== undefined ? entry.remaining : (entry.leavesQuantity !== undefined ? entry.leavesQuantity : (entry.remainingSize !== undefined ? entry.remainingSize : (rawQuantity - filledQuantity))), 0)
        const price = asNumber(entry.price !== undefined ? entry.price : (entry.limitPrice !== undefined ? entry.limitPrice : (entry.stopPrice !== undefined ? entry.stopPrice : 0)), 0)
        const averagePrice = asNumber(entry.averagePrice !== undefined ? entry.averagePrice : (entry.avgPrice !== undefined ? entry.avgPrice : 0), 0)
        const status = entry.status || entry.state || entry.orderStatus || entry.executionStatus || ""
        const timeInForce = entry.timeInForce || entry.tif || ""
        const reduceOnly = !!(entry.reduceOnly !== undefined ? entry.reduceOnly : entry.isReduceOnly)
        const postOnly = !!(entry.postOnly !== undefined ? entry.postOnly : entry.isPostOnly)

        function normalizeDate(value) {
            if (!value)
                return ""
            if (value instanceof Date)
                return value.toISOString()
            if (typeof value === "number") {
                const date = new Date(value)
                if (!isNaN(date.getTime()))
                    return date.toISOString()
            }
            if (typeof value === "string" && value.length === 0)
                return ""
            if (value && typeof Qt !== "undefined" && Qt.formatDateTime) {
                const formatted = Qt.formatDateTime(value, Qt.ISODateWithMs)
                if (formatted && formatted.length > 0)
                    return formatted
            }
            return value
        }

        const createdAt = normalizeDate(entry.createdAt || entry.timestamp || entry.submittedAt || entry.time || entry.placedAt || "")
        const updatedAt = normalizeDate(entry.updatedAt || entry.lastUpdate || entry.lastUpdatedAt || entry.filledAt || entry.doneAt || "")
        const expiresAt = normalizeDate(entry.expiresAt || entry.expireTime || entry.validUntil || "")

        if (!side && rawQuantity !== 0) {
            if (rawQuantity > 0)
                side = qsTr("Kupno")
            else if (rawQuantity < 0)
                side = qsTr("Sprzedaż")
        }

        return {
            id: id,
            clientOrderId: clientOrderId,
            symbol: symbol,
            exchange: exchange,
            venueSymbol: venueSymbol,
            account: account,
            type: type,
            side: side,
            quantity: Math.abs(rawQuantity),
            signedQuantity: rawQuantity,
            filledQuantity: filledQuantity,
            remainingQuantity: remainingQuantity,
            price: price,
            averagePrice: averagePrice,
            status: status,
            timeInForce: timeInForce,
            reduceOnly: reduceOnly,
            postOnly: postOnly,
            createdAt: createdAt,
            updatedAt: updatedAt,
            expiresAt: expiresAt,
        }
    }

    function normalizePendingOrders(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeOrderEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setPendingOrders(entries) {
        pendingOrders = normalizePendingOrders(entries)
    }

    function normalizeSignalEntry(entry) {
        if (!entry)
            return null

        const id = entry.id || entry.signalId || entry.alertId || ""
        const category = entry.category || entry.channel || entry.type || ""
        const symbol = entry.symbol || entry.instrument || entry.market || ""
        let direction = entry.direction || entry.bias || entry.side || ""

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        const confidence = asNumber(entry.confidence !== undefined ? entry.confidence : (entry.score !== undefined ? entry.score : (entry.strength !== undefined ? entry.strength : 0)), 0)
        const impact = asNumber(entry.impact !== undefined ? entry.impact : (entry.weight !== undefined ? entry.weight : 0), 0)
        const message = entry.message || entry.note || entry.description || ""
        const ttl = entry.ttl || entry.expiry || entry.validUntil || entry.expiresAt || ""
        const priority = entry.priority || entry.severity || ""
        const tags = Array.isArray(entry.tags) ? entry.tags : (typeof entry.tag === "string" && entry.tag.length > 0 ? [entry.tag] : [])

        function normalizeDate(value) {
            if (!value)
                return ""
            if (value instanceof Date)
                return value.toISOString()
            if (typeof value === "number") {
                const date = new Date(value)
                if (!isNaN(date.getTime()))
                    return date.toISOString()
            }
            if (typeof value === "string" && value.length === 0)
                return ""
            if (value && typeof Qt !== "undefined" && Qt.formatDateTime) {
                const formatted = Qt.formatDateTime(value, Qt.ISODateWithMs)
                if (formatted && formatted.length > 0)
                    return formatted
            }
            return value
        }

        const generatedAt = normalizeDate(entry.generatedAt || entry.createdAt || entry.timestamp || entry.time || "")
        const expiresAt = normalizeDate(ttl)

        if (!direction && confidence !== 0) {
            direction = confidence >= 0 ? qsTr("Long") : qsTr("Short")
        }

        return {
            id: id,
            category: category,
            symbol: symbol,
            direction: direction,
            confidence: confidence,
            impact: impact,
            message: message,
            priority: priority,
            generatedAt: generatedAt,
            expiresAt: expiresAt,
            tags: tags,
        }
    }

    function normalizeSignalAlerts(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeSignalEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setSignalAlerts(entries) {
        signalAlerts = normalizeSignalAlerts(entries)
        recalculateComplianceSummary()
    }

    function normalizeScenarioTestEntry(entry) {
        if (!entry)
            return null

        const id = entry.id || entry.scenarioId || entry.testId || entry.name || ""
        const name = entry.name || entry.title || entry.scenario || ""
        const category = entry.category || entry.type || entry.segment || ""
        const severity = entry.severity || entry.level || entry.impactLevel || ""
        const description = entry.description || entry.summary || entry.note || ""
        let status = entry.status || entry.outcome || entry.result || ""
        const success = entry.success !== undefined
            ? !!entry.success
            : (typeof status === "string" ? status.toLowerCase().indexOf("fail") === -1 : true)

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        const pnlImpact = asNumber(entry.pnlImpact !== undefined ? entry.pnlImpact
            : (entry.expectedPnl !== undefined ? entry.expectedPnl
                : (entry.pnlDelta !== undefined ? entry.pnlDelta
                    : (entry.pnl !== undefined ? entry.pnl : 0))), 0)
        const maxDrawdown = asNumber(entry.maxDrawdown !== undefined ? entry.maxDrawdown
            : (entry.drawdown !== undefined ? entry.drawdown
                : (entry.drawdownImpact !== undefined ? entry.drawdownImpact : 0)), 0)
        const liquidityImpact = asNumber(entry.liquidityImpact !== undefined ? entry.liquidityImpact
            : (entry.slippageImpact !== undefined ? entry.slippageImpact
                : (entry.orderBookImpact !== undefined ? entry.orderBookImpact : 0)), 0)
        const marginUsage = asNumber(entry.marginUsage !== undefined ? entry.marginUsage
            : (entry.marginUtilization !== undefined ? entry.marginUtilization
                : (entry.marginImpact !== undefined ? entry.marginImpact : 0)), 0)
        const durationSeconds = asNumber(entry.durationSeconds !== undefined ? entry.durationSeconds
            : (entry.duration !== undefined ? entry.duration
                : (entry.elapsedSeconds !== undefined ? entry.elapsedSeconds : 0)), 0)

        function normalizeDate(value) {
            if (!value)
                return ""
            if (value instanceof Date)
                return value.toISOString()
            if (typeof value === "number") {
                const date = new Date(value)
                if (!isNaN(date.getTime()))
                    return date.toISOString()
            }
            if (typeof value === "string") {
                if (value.length === 0)
                    return ""
                return value
            }
            if (value && typeof Qt !== "undefined" && Qt.formatDateTime) {
                const formatted = Qt.formatDateTime(value, Qt.ISODateWithMs)
                if (formatted && formatted.length > 0)
                    return formatted
            }
            return ""
        }

        const startedAt = normalizeDate(entry.startedAt || entry.startTime || entry.launchedAt || "")
        const completedAt = normalizeDate(entry.completedAt || entry.endTime || entry.finishedAt || entry.lastRunAt || "")
        const lastRunAt = normalizeDate(entry.lastRunAt || entry.completedAt || entry.endTime || "")
        const runCountRaw = entry.runCount !== undefined ? entry.runCount
            : (entry.executions !== undefined ? entry.executions
                : (entry.timesRun !== undefined ? entry.timesRun : 0))
        const runCountNumber = Number(runCountRaw)
        const runCount = isNaN(runCountNumber) ? 0 : Math.max(0, Math.round(runCountNumber))

        if (!status)
            status = success ? qsTr("Zakończony") : qsTr("Nieudany")

        const notes = []
        const appendNote = function(value) {
            if (value === null || value === undefined)
                return
            const asString = value.toString()
            if (asString.length > 0)
                notes.push(asString)
        }
        if (Array.isArray(entry.notes)) {
            for (let i = 0; i < entry.notes.length; ++i)
                appendNote(entry.notes[i])
        } else if (Array.isArray(entry.messages)) {
            for (let i = 0; i < entry.messages.length; ++i)
                appendNote(entry.messages[i])
        } else if (entry.message) {
            appendNote(entry.message)
        } else if (entry.note) {
            appendNote(entry.note)
        } else if (entry.description && description !== entry.description) {
            appendNote(entry.description)
        }

        const metrics = Array.isArray(entry.metrics) ? entry.metrics : []

        return {
            id: id,
            name: name,
            category: category,
            severity: severity,
            description: description,
            status: status,
            success: success,
            pnlImpact: pnlImpact,
            maxDrawdown: maxDrawdown,
            liquidityImpact: liquidityImpact,
            marginUsage: marginUsage,
            durationSeconds: durationSeconds,
            startedAt: startedAt,
            completedAt: completedAt,
            lastRunAt: lastRunAt,
            runCount: runCount,
            notes: notes,
            metrics: metrics,
        }
    }

    function normalizeScenarioTests(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeScenarioTestEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setScenarioTests(entries) {
        scenarioTests = normalizeScenarioTests(entries)
        recalculateComplianceSummary()
    }

    function normalizeAutomationRule(entry) {
        if (!entry)
            return null

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        function asInteger(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : Math.round(number)
        }

        function normalizeTimestamp(value) {
            if (!value)
                return ""
            if (value instanceof Date)
                return value.toISOString()
            if (typeof value === "number") {
                const fromEpoch = new Date(value)
                if (!isNaN(fromEpoch.getTime()))
                    return fromEpoch.toISOString()
            }
            if (value && typeof Qt !== "undefined" && Qt.formatDateTime) {
                const formatted = Qt.formatDateTime(value, Qt.ISODateWithMs)
                if (formatted && formatted.length > 0)
                    return formatted
            }
            return value.toString()
        }

        const tags = Array.isArray(entry.tags) ? entry.tags : []

        return {
            id: entry.id || entry.ruleId || "",
            name: entry.name || entry.title || "",
            description: entry.description || entry.summary || "",
            enabled: entry.enabled !== undefined ? !!entry.enabled : (entry.active !== undefined ? !!entry.active : true),
            type: entry.type || entry.category || "",
            trigger: entry.trigger || entry.condition || entry.threshold || "",
            action: entry.action || entry.response || entry.effect || "",
            lastTriggeredAt: normalizeTimestamp(entry.lastTriggeredAt || entry.lastExecutedAt || entry.triggeredAt || ""),
            successRate: asNumber(entry.successRate !== undefined ? entry.successRate : (entry.winRate !== undefined ? entry.winRate : 0), 0),
            errorCount: asInteger(entry.errorCount !== undefined ? entry.errorCount : (entry.failures !== undefined ? entry.failures : 0), 0),
            critical: entry.critical !== undefined ? !!entry.critical : (entry.criticality !== undefined ? entry.criticality === "high" : false),
            tags: tags,
        }
    }

    function normalizeAutomationRules(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeAutomationRule(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setAutomationRules(entries) {
        automationRules = normalizeAutomationRules(entries)
        recalculateComplianceSummary()
    }

    function normalizeExecutionIncident(entry) {
        if (!entry)
            return null

        function normalizeTimestamp(value) {
            if (!value)
                return ""
            if (value instanceof Date)
                return value.toISOString()
            if (typeof value === "number") {
                const fromEpoch = new Date(value)
                if (!isNaN(fromEpoch.getTime()))
                    return fromEpoch.toISOString()
            }
            if (value && typeof Qt !== "undefined" && Qt.formatDateTime) {
                const formatted = Qt.formatDateTime(value, Qt.ISODateWithMs)
                if (formatted && formatted.length > 0)
                    return formatted
            }
            return value.toString()
        }

        return {
            type: entry.type || entry.category || "",
            message: entry.message || entry.description || "",
            timestamp: normalizeTimestamp(entry.timestamp || entry.time || entry.occurredAt || entry.loggedAt || ""),
            resolved: entry.resolved !== undefined ? !!entry.resolved : (entry.closed !== undefined ? !!entry.closed : false),
        }
    }

    function normalizeExecutionIncidents(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeExecutionIncident(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function normalizeExecutionDiagnostics(entry) {
        if (!entry) {
            return {
                provider: "",
                avgLatencyMs: 0,
                p95LatencyMs: 0,
                maxLatencyMs: 0,
                fillRate: 0,
                rejectRate: 0,
                slippageBps: 0,
                lastUpdated: "",
                notes: "",
                recentIncidents: [],
            }
        }

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let lastUpdated = entry.lastUpdated || entry.updatedAt || entry.timestamp || ""
        if (lastUpdated instanceof Date)
            lastUpdated = lastUpdated.toISOString()
        else if (typeof lastUpdated === "number") {
            const date = new Date(lastUpdated)
            if (!isNaN(date.getTime()))
                lastUpdated = date.toISOString()
        }

        return {
            provider: entry.provider || entry.source || "",
            avgLatencyMs: asNumber(entry.avgLatencyMs !== undefined ? entry.avgLatencyMs : (entry.averageLatencyMs !== undefined ? entry.averageLatencyMs : entry.latencyAvgMs), 0),
            p95LatencyMs: asNumber(entry.p95LatencyMs !== undefined ? entry.p95LatencyMs : (entry.latencyP95Ms !== undefined ? entry.latencyP95Ms : entry.highLatencyMs), 0),
            maxLatencyMs: asNumber(entry.maxLatencyMs !== undefined ? entry.maxLatencyMs : (entry.latencyMaxMs !== undefined ? entry.latencyMaxMs : entry.worstLatencyMs), 0),
            fillRate: asNumber(entry.fillRate !== undefined ? entry.fillRate : (entry.executionRate !== undefined ? entry.executionRate : entry.successRate), 0),
            rejectRate: asNumber(entry.rejectRate !== undefined ? entry.rejectRate : (entry.rejectionRate !== undefined ? entry.rejectionRate : 0), 0),
            slippageBps: asNumber(entry.slippageBps !== undefined ? entry.slippageBps : (entry.slippage !== undefined ? entry.slippage : 0), 0),
            lastUpdated: lastUpdated,
            notes: entry.notes || entry.comment || "",
            recentIncidents: normalizeExecutionIncidents(entry.recentIncidents || entry.incidents || entry.events),
        }
    }

    function setExecutionDiagnostics(entry) {
        executionDiagnostics = normalizeExecutionDiagnostics(entry)
    }

    function normalizeSentimentSource(entry) {
        if (!entry)
            return null

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let updatedAt = entry.updatedAt || entry.timestamp || entry.time || ""
        if (updatedAt instanceof Date)
            updatedAt = updatedAt.toISOString()
        else if (typeof updatedAt === "number") {
            const date = new Date(updatedAt)
            if (!isNaN(date.getTime()))
                updatedAt = date.toISOString()
        }

        return {
            name: entry.name || entry.label || entry.source || "",
            score: asNumber(entry.score !== undefined ? entry.score : (entry.value !== undefined ? entry.value : 0), 0),
            weight: asNumber(entry.weight !== undefined ? entry.weight : (entry.importance !== undefined ? entry.importance : 0), 0),
            sentiment: entry.sentiment || entry.bias || "",
            updatedAt: updatedAt,
            notes: entry.notes || entry.comment || "",
        }
    }

    function normalizeSentimentSources(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeSentimentSource(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function normalizeMarketSentiment(entry) {
        if (!entry) {
            return {
                trend: "",
                globalScore: 0,
                confidence: 0,
                volatilityIndex: 0,
                onChainFlow: 0,
                newsScore: 0,
                derivedAt: "",
                sources: [],
            }
        }

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let derivedAt = entry.derivedAt || entry.generatedAt || entry.updatedAt || entry.timestamp || ""
        if (derivedAt instanceof Date)
            derivedAt = derivedAt.toISOString()
        else if (typeof derivedAt === "number") {
            const date = new Date(derivedAt)
            if (!isNaN(date.getTime()))
                derivedAt = date.toISOString()
        }

        return {
            trend: entry.trend || entry.bias || entry.direction || "",
            globalScore: asNumber(entry.globalScore !== undefined ? entry.globalScore : (entry.score !== undefined ? entry.score : 0), 0),
            confidence: asNumber(entry.confidence !== undefined ? entry.confidence : (entry.reliability !== undefined ? entry.reliability : 0), 0),
            volatilityIndex: asNumber(entry.volatilityIndex !== undefined ? entry.volatilityIndex : (entry.volatility !== undefined ? entry.volatility : 0), 0),
            onChainFlow: asNumber(entry.onChainFlow !== undefined ? entry.onChainFlow : (entry.flow !== undefined ? entry.flow : 0), 0),
            newsScore: asNumber(entry.newsScore !== undefined ? entry.newsScore : (entry.newsSentiment !== undefined ? entry.newsSentiment : 0), 0),
            derivedAt: derivedAt,
            sources: normalizeSentimentSources(entry.sources || entry.components || entry.contributors)
        }
    }

    function setMarketSentiment(entry) {
        marketSentiment = normalizeMarketSentiment(entry)
    }

    function normalizeNewsHeadline(entry) {
        if (!entry)
            return null

        const id = entry.id || entry.uuid || entry.articleId || ""
        const title = entry.title || entry.headline || ""
        const summary = entry.summary || entry.description || entry.body || ""
        const sentiment = entry.sentiment || entry.bias || ""
        const source = entry.source || entry.publisher || entry.feed || ""
        const url = entry.url || entry.link || ""

        let publishedAt = entry.publishedAt || entry.time || entry.timestamp || entry.date || ""
        if (publishedAt instanceof Date)
            publishedAt = publishedAt.toISOString()
        else if (typeof publishedAt === "number") {
            const date = new Date(publishedAt)
            if (!isNaN(date.getTime()))
                publishedAt = date.toISOString()
        }

        return {
            id: id,
            title: title,
            summary: summary,
            sentiment: sentiment,
            source: source,
            url: url,
            publishedAt: publishedAt,
        }
    }

    function normalizeNewsHeadlines(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeNewsHeadline(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setNewsHeadlines(entries) {
        newsHeadlines = normalizeNewsHeadlines(entries)
    }

    function normalizePerformanceComparison(entry) {
        if (!entry)
            return {}

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let since = entry.since || entry.sinceDate || entry.startDate || ""
        if (since instanceof Date)
            since = since.toISOString()
        else if (typeof since === "number") {
            const date = new Date(since)
            if (!isNaN(date.getTime()))
                since = date.toISOString()
        }

        let updatedAt = entry.updatedAt || entry.lastUpdated || entry.calculatedAt || ""
        if (updatedAt instanceof Date)
            updatedAt = updatedAt.toISOString()
        else if (typeof updatedAt === "number") {
            const date = new Date(updatedAt)
            if (!isNaN(date.getTime()))
                updatedAt = date.toISOString()
        }

        let strategyReturn = asNumber(entry.strategyReturn !== undefined ? entry.strategyReturn : (entry.strategyPerformance !== undefined ? entry.strategyPerformance : 0), 0)
        let benchmarkReturn = asNumber(entry.benchmarkReturn !== undefined ? entry.benchmarkReturn : (entry.benchmarkPerformance !== undefined ? entry.benchmarkPerformance : 0), 0)

        if (strategyReturn > 1.5)
            strategyReturn = strategyReturn / 100
        if (benchmarkReturn > 1.5)
            benchmarkReturn = benchmarkReturn / 100

        let alpha = asNumber(entry.alpha !== undefined ? entry.alpha : (entry.excessReturn !== undefined ? entry.excessReturn : 0), 0)
        if (alpha > 1.5)
            alpha = alpha / 100

        const beta = asNumber(entry.beta !== undefined ? entry.beta : (entry.betaCoefficient !== undefined ? entry.betaCoefficient : 0), 0)
        const sharpe = asNumber(entry.sharpe !== undefined ? entry.sharpe : (entry.sharpeRatio !== undefined ? entry.sharpeRatio : 0), 0)
        const sortino = asNumber(entry.sortino !== undefined ? entry.sortino : (entry.sortinoRatio !== undefined ? entry.sortinoRatio : 0), 0)
        let volatility = asNumber(entry.volatility !== undefined ? entry.volatility : (entry.strategyVolatility !== undefined ? entry.strategyVolatility : 0), 0)
        if (volatility > 1.5)
            volatility = volatility / 100
        let trackingError = asNumber(entry.trackingError !== undefined ? entry.trackingError : (entry.tracking !== undefined ? entry.tracking : 0), 0)
        if (trackingError > 1.5)
            trackingError = trackingError / 100
        let maxDrawdown = asNumber(entry.maxDrawdown !== undefined ? entry.maxDrawdown : (entry.drawdown !== undefined ? entry.drawdown : 0), 0)
        if (maxDrawdown > 1.5)
            maxDrawdown = maxDrawdown / 100

        return {
            benchmarkName: entry.benchmarkName || entry.benchmark || "",
            since: since,
            updatedAt: updatedAt,
            strategyReturn: strategyReturn,
            benchmarkReturn: benchmarkReturn,
            alpha: alpha,
            beta: beta,
            sharpe: sharpe,
            sortino: sortino,
            volatility: volatility,
            trackingError: trackingError,
            maxDrawdown: maxDrawdown,
            notes: entry.notes || entry.comment || "",
        }
    }

    function setPerformanceComparison(entry) {
        performanceComparison = normalizePerformanceComparison(entry)
    }

    function normalizeAllocationEntry(entry) {
        if (!entry)
            return null

        function asNumber(value, fallback) {
            if (value === null || value === undefined)
                return fallback
            const number = Number(value)
            return isNaN(number) ? fallback : number
        }

        let weight = asNumber(entry.weight !== undefined ? entry.weight : (entry.share !== undefined ? entry.share : (entry.percentage !== undefined ? entry.percentage : 0)), 0)
        let targetWeight = asNumber(entry.targetWeight !== undefined ? entry.targetWeight : (entry.targetShare !== undefined ? entry.targetShare : (entry.target !== undefined ? entry.target : 0)), 0)

        if (weight > 1.5)
            weight = weight / 100
        if (targetWeight > 1.5)
            targetWeight = targetWeight / 100

        let delta = entry.deltaWeight !== undefined ? entry.deltaWeight : (entry.delta !== undefined ? entry.delta : (entry.deviation !== undefined ? entry.deviation : (weight - targetWeight)))
        delta = asNumber(delta, weight - targetWeight)
        if (delta > 1.5)
            delta = delta / 100

        const notional = asNumber(entry.notional !== undefined ? entry.notional : (entry.value !== undefined ? entry.value : (entry.currentValue !== undefined ? entry.currentValue : 0)), 0)
        const leverage = asNumber(entry.leverage !== undefined ? entry.leverage : (entry.leverageRatio !== undefined ? entry.leverageRatio : 0), 0)

        return {
            segment: entry.segment || entry.category || entry.bucket || entry.name || "",
            weight: weight,
            targetWeight: targetWeight,
            deltaWeight: delta,
            notional: notional,
            currentValue: notional,
            leverage: leverage,
            hedged: entry.hedged !== undefined ? !!entry.hedged : !!entry.isHedged,
            notes: entry.notes || entry.comment || "",
        }
    }

    function normalizeCapitalAllocation(entries) {
        const normalized = []
        const items = Array.isArray(entries) ? entries : []
        for (let i = 0; i < items.length; ++i) {
            const entry = normalizeAllocationEntry(items[i])
            if (entry)
                normalized.push(entry)
        }
        return normalized
    }

    function setCapitalAllocation(entries) {
        capitalAllocation = normalizeCapitalAllocation(entries)
    }

    function appendRiskTimelineEntry(entry) {
        const normalized = normalizeRiskTimelineEntry(entry)
        if (!normalized)
            return
        const current = Array.isArray(riskTimeline) ? riskTimeline.slice(0) : []
        current.unshift(normalized)
        const limit = 60
        if (current.length > limit)
            current.splice(limit)
        riskTimeline = current
    }

    function appendDemoRiskTimelineEntry() {
        const manualCount = controlState && controlState.manualRefreshCount ? controlState.manualRefreshCount : 0
        const base = Array.isArray(riskTimeline) && riskTimeline.length > 0 ? riskTimeline[0] : null
        const baseValue = base && typeof base.portfolioValue === "number"
            ? base.portfolioValue
            : (portfolioSummary && typeof portfolioSummary.latestValue === "number" ? portfolioSummary.latestValue : 100000)
        const baseDrawdown = base && typeof base.drawdown === "number"
            ? base.drawdown
            : (riskSnapshot && typeof riskSnapshot.currentDrawdown === "number" ? riskSnapshot.currentDrawdown : 0.02)
        const baseExposure = base && typeof base.exposureUtilization === "number"
            ? base.exposureUtilization
            : (portfolioSummary && typeof portfolioSummary.maxExposureUtilization === "number" ? portfolioSummary.maxExposureUtilization : 0.5)
        const baseLeverage = base && typeof base.leverage === "number"
            ? base.leverage
            : (portfolioSummary && typeof portfolioSummary.maxLeverage === "number" ? portfolioSummary.maxLeverage : 1.5)

        const direction = manualCount % 2 === 0 ? -1 : 1
        const drift = (manualCount % 5 + 1) * 0.0025 * direction
        const nextValue = Math.max(0, baseValue * (1 + drift))
        const nextDrawdown = Math.max(0, baseDrawdown * (1 - direction * 0.08))
        const nextExposure = Math.max(0, Math.min(1.5, baseExposure + direction * 0.03))
        const nextLeverage = Math.max(0, baseLeverage + direction * 0.1)
        const breach = nextExposure > 1.0

        appendRiskTimelineEntry({
            timestamp: new Date().toISOString(),
            portfolioValue: nextValue,
            drawdown: nextDrawdown,
            exposureUtilization: nextExposure,
            leverage: nextLeverage,
            breach: breach,
            breachCount: breach ? 1 : 0,
            notes: qsTr("Symulowany odczyt po ręcznym odświeżeniu"),
            source: "demo",
        })

        if (performanceComparison && typeof performanceComparison === "object") {
            const currentComparison = normalizePerformanceComparison(performanceComparison)
            const adjusted = Object.assign({}, currentComparison, {
                strategyReturn: Math.max(-1, currentComparison.strategyReturn + drift),
                alpha: Math.max(-1, currentComparison.alpha + drift * 0.6),
                updatedAt: new Date().toISOString(),
            })
            setPerformanceComparison(adjusted)
        }

        recalculateComplianceSummary()
    }

    function updateDemoExecutionDiagnosticsAfterRiskRefresh() {
        if (!demoModeActive)
            return
        const current = normalizeExecutionDiagnostics(executionDiagnostics)
        const now = new Date().toISOString()
        const incidents = Array.isArray(current.recentIncidents) ? current.recentIncidents.slice(0) : []
        incidents.unshift({
            type: "refresh",
            message: qsTr("Ręczne odświeżenie ryzyka ukończone w trybie demo"),
            timestamp: now,
            resolved: true,
        })
        if (incidents.length > 6)
            incidents.splice(6)

        const next = Object.assign({}, current, {
            avgLatencyMs: Math.max(5, (current.avgLatencyMs || 40) * 0.96),
            p95LatencyMs: Math.max(8, (current.p95LatencyMs || 70) * 0.95),
            maxLatencyMs: Math.max(10, (current.maxLatencyMs || 120) * 0.92),
            fillRate: Math.min(1, (current.fillRate || 0.9) + 0.005),
            rejectRate: Math.max(0, (current.rejectRate || 0.05) * 0.9),
            slippageBps: Math.max(-50, Math.min(50, (current.slippageBps || 3) * 0.9)),
            lastUpdated: now,
            recentIncidents: incidents,
        })
        setExecutionDiagnostics(next)
    }

    function prependActivityEvent(event) {
        const entry = normalizeActivityEntry(event)
        if (!entry)
            return
        const current = Array.isArray(activityLog) ? activityLog.slice(0) : []
        current.unshift(entry)
        const limit = 50
        if (current.length > limit)
            current.splice(limit)
        activityLog = current
    }

    function recordActivityEvent(type, message, overrides) {
        const nowIso = new Date().toISOString()
        const details = overrides && overrides.details ? overrides.details : {}
        const entry = {
            timestamp: overrides && overrides.timestamp ? overrides.timestamp : nowIso,
            type: type || "",
            message: message || "",
            success: overrides && overrides.success !== undefined ? !!overrides.success : true,
            source: overrides && overrides.source ? overrides.source : (demoModeActive ? "demo" : "local"),
            details: details,
        }
        prependActivityEvent(entry)
        if (entry.source === "local") {
            const pending = Array.isArray(pendingActivityEvents) ? pendingActivityEvents.slice(0) : []
            pending.unshift(entry)
            const limit = 10
            if (pending.length > limit)
                pending.splice(limit)
            pendingActivityEvents = pending
        }
    }

    function normalizeControlState(state) {
        return Object.assign({}, defaultControlState(), state || {})
    }

    function updateControlState(state) {
        controlState = normalizeControlState(state)
        recalculateComplianceSummary()
    }

    function mergeControlState(overrides) {
        const base = normalizeControlState(controlState)
        updateControlState(Object.assign({}, base, overrides || {}))
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value))
    }

    function buildComplianceSummaryFromState() {
        const license = licenseStatus || {}
        const timeline = Array.isArray(riskTimeline) ? riskTimeline : []
        const alerts = Array.isArray(signalAlerts) ? signalAlerts : []
        const scenarios = Array.isArray(scenarioTests) ? scenarioTests : []
        const currentControl = normalizeControlState(controlState)
        const runtime = runtimeStatus || {}
        const portfolio = portfolioSummary || {}
        const risk = riskSnapshot || {}

        let outstandingBreaches = 0
        let lastBreachAt = ""
        for (let i = 0; i < timeline.length; ++i) {
            const entry = timeline[i]
            if (!entry || !entry.breach)
                continue
            let entryCount = entry.breachCount !== undefined ? Number(entry.breachCount) : 1
            if (!isFinite(entryCount) || entryCount < 1)
                entryCount = 1
            outstandingBreaches += entryCount
            if (!lastBreachAt && entry.timestamp)
                lastBreachAt = entry.timestamp
        }

        let failingScenarioCount = 0
        let worstScenarioDrawdown = 0
        for (let i = 0; i < scenarios.length; ++i) {
            const scenario = scenarios[i]
            if (!scenario)
                continue
            const success = scenario.success !== undefined ? !!scenario.success : true
            if (!success)
                failingScenarioCount += 1
            const scenarioDrawdown = typeof scenario.maxDrawdown === "number"
                ? scenario.maxDrawdown
                : (typeof scenario.drawdown === "number" ? scenario.drawdown : 0)
            if (scenarioDrawdown > worstScenarioDrawdown)
                worstScenarioDrawdown = scenarioDrawdown
        }

        const latestRisk = timeline.length > 0 ? timeline[0] : null
        const exposureUtilization = latestRisk && typeof latestRisk.exposureUtilization === "number"
            ? latestRisk.exposureUtilization
            : (portfolio && typeof portfolio.maxExposureUtilization === "number"
                ? portfolio.maxExposureUtilization
                : 0)

        const licenseActive = license.active !== undefined ? !!license.active : false
        const automationPaused = !!(currentControl.offlineMode
            || !currentControl.schedulerRunning
            || runtime.offlineMode
            || (runtime.automationRunning === false))

        const baseline = licenseActive ? 0.9 : 0.45
        const exposurePenalty = Math.max(0, exposureUtilization - 0.85) * 0.5
        const breachPenalty = Math.min(outstandingBreaches * 0.1, 0.5)
        const alertPenalty = Math.min(alerts.length * 0.06, 0.3)
        const automationPenalty = automationPaused ? 0.08 : 0
        const automationRuleList = Array.isArray(automationRules) ? automationRules : []
        let disabledAutomationRules = 0
        for (let i = 0; i < automationRuleList.length; ++i) {
            const rule = automationRuleList[i]
            if (!rule || rule.enabled === undefined)
                continue
            if (!rule.enabled)
                disabledAutomationRules += 1
        }
        const automationRulesPenalty = Math.min(disabledAutomationRules * 0.02, 0.1)
        const scenarioPenalty = Math.min(failingScenarioCount * 0.05, 0.2)
        const complianceScore = clamp(baseline - exposurePenalty - breachPenalty - alertPenalty - automationPenalty - automationRulesPenalty - scenarioPenalty, 0, 1)

        let nextRiskReviewAt = currentControl.nextRiskRefreshDueAt || ""
        let lastRiskAssessmentAt = risk.generatedAt || (latestRisk ? latestRisk.timestamp : "")
        let refreshSnapshot = null
        if (!nextRiskReviewAt) {
            if (runtime && runtime.riskRefresh)
                refreshSnapshot = runtime.riskRefresh
            else if (appController && typeof appController.riskRefreshSnapshot === "function")
                refreshSnapshot = appController.riskRefreshSnapshot() || {}
            if (refreshSnapshot) {
                nextRiskReviewAt = refreshSnapshot.nextRefreshDueAt
                    || refreshSnapshot.nextScheduledAt
                    || refreshSnapshot.nextRun
                    || refreshSnapshot.nextRequestAt
                    || ""
                if (!lastRiskAssessmentAt && refreshSnapshot.lastUpdateAt)
                    lastRiskAssessmentAt = refreshSnapshot.lastUpdateAt
            }
        }

        const notes = []
        if (!licenseActive)
            notes.push(qsTr("Brak aktywnej licencji – funkcje zgodności ograniczone."))
        if (outstandingBreaches > 0)
            notes.push(qsTr("Wykryto %1 naruszeń limitów ekspozycji.").arg(outstandingBreaches))
        if (alerts.length > 0)
            notes.push(qsTr("Aktywnych alertów sygnałowych: %1.").arg(alerts.length))
        if (!automationPaused && outstandingBreaches === 0)
            notes.push(qsTr("Automatyzacja aktywna, brak otwartych naruszeń."))
        if (disabledAutomationRules > 0)
            notes.push(qsTr("Wyłączone reguły automatyzacji: %1.").arg(disabledAutomationRules))
        if (failingScenarioCount > 0)
            notes.push(qsTr("Nieudane testy scenariuszy: %1.").arg(failingScenarioCount))
        else if (scenarios.length > 0)
            notes.push(qsTr("Ostatnie testy scenariuszy zakończone powodzeniem."))
        if (worstScenarioDrawdown > 0.15) {
            const drawdownPercentText = (worstScenarioDrawdown * 100).toFixed(1) + "%"
            notes.push(qsTr("Najgorszy scenariusz zakłada obsunięcie %1.").arg(drawdownPercentText))
        }
        if (notes.length === 0)
            notes.push(qsTr("Brak krytycznych odchyleń."))

        return {
            licenseActive: licenseActive,
            complianceScore: complianceScore,
            outstandingBreaches: outstandingBreaches,
            openAlerts: alerts.length,
            lastBreachAt: lastBreachAt,
            lastRiskAssessmentAt: lastRiskAssessmentAt,
            nextRiskReviewAt: nextRiskReviewAt,
            automationPaused: automationPaused,
            disabledAutomationRules: disabledAutomationRules,
            scenarioFailures: failingScenarioCount,
            worstScenarioDrawdown: worstScenarioDrawdown,
            notes: notes,
        }
    }

    function recalculateComplianceSummary() {
        complianceSummary = buildComplianceSummaryFromState()
    }

    function updateDemoRuntimeStatus(overrides) {
        const current = runtimeStatus || {}
        const next = Object.assign({}, current, overrides || {})
        if (overrides && overrides.riskRefresh) {
            const baseRefresh = current.riskRefresh || {}
            next.riskRefresh = Object.assign({}, baseRefresh, overrides.riskRefresh)
        }
        runtimeStatus = next
        recalculateComplianceSummary()
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
        setPerformanceComparison(preset.performanceComparison)
        setCapitalAllocation(preset.capitalAllocation)
        runtimeStatus = preset.runtimeStatus || ({})
        licenseStatus = preset.licenseStatus || ({})
        instrumentDetails = preset.instrumentDetails || computeInstrumentDetails()
        setRiskTimeline(preset.riskTimeline)
        setActivityLog(preset.activityLog)
        setOpenPositions(preset.openPositions)
        setPendingOrders(preset.pendingOrders)
        setTradeHistory(preset.tradeHistory)
        setSignalAlerts(preset.signalAlerts)
        setScenarioTests(preset.scenarioTests)
        setAutomationRules(preset.automationRules)
        setExecutionDiagnostics(preset.executionDiagnostics)
        pendingActivityEvents = []
        setMarketSentiment(preset.marketSentiment)
        setNewsHeadlines(preset.newsHeadlines)
        updateControlState(preset.controlState)
    }

    function disableDemoMode() {
        if (!demoModeActive)
            return
        demoModeActive = false
        demoModeId = ""
        demoModeTitle = ""
        demoModeDescription = ""
        pendingActivityEvents = []
        setRiskTimeline([])
        setOpenPositions([])
        setPendingOrders([])
        setTradeHistory([])
        setSignalAlerts([])
        setScenarioTests([])
        setAutomationRules([])
        setExecutionDiagnostics({})
        setMarketSentiment({})
        setNewsHeadlines([])
        setCapitalAllocation([])
        setPerformanceComparison({})
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
        setPerformanceComparison(computePerformanceComparison())
        setCapitalAllocation(computeCapitalAllocation())
        riskTimeline = computeRiskTimeline()
        runtimeStatus = computeRuntimeStatus()
        licenseStatus = computeLicenseStatus()
        instrumentDetails = computeInstrumentDetails()
        openPositions = computeOpenPositions()
        pendingOrders = computePendingOrders()
        tradeHistory = computeTradeHistory()
        setSignalAlerts(computeSignalAlerts())
        setScenarioTests(computeScenarioTests())
        setAutomationRules(computeAutomationRules())
        setExecutionDiagnostics(computeExecutionDiagnostics())
        marketSentiment = computeMarketSentiment()
        newsHeadlines = computeNewsHeadlines()
        const liveLog = computeActivityLog()
        const pending = Array.isArray(pendingActivityEvents) ? pendingActivityEvents.slice(0) : []
        if (liveLog.length > 0 || pending.length > 0) {
            activityLog = pending.concat(liveLog)
        } else if (!Array.isArray(activityLog) || activityLog.length === 0) {
            activityLog = []
        }
        pendingActivityEvents = pending
        updateControlState(computeControlState())
    }

    function computeRiskTimeline() {
        if (!riskHistoryModel)
            return []

        if (typeof riskHistoryModel.timeline === "function") {
            const timeline = riskHistoryModel.timeline()
            if (Array.isArray(timeline))
                return normalizeRiskTimeline(timeline)
        }

        if (Array.isArray(riskHistoryModel.timeline))
            return normalizeRiskTimeline(riskHistoryModel.timeline)

        if (typeof riskHistoryModel.history === "function") {
            const history = riskHistoryModel.history()
            if (Array.isArray(history))
                return normalizeRiskTimeline(history)
        }

        if (Array.isArray(riskHistoryModel.history))
            return normalizeRiskTimeline(riskHistoryModel.history)

        const count = typeof riskHistoryModel.entryCount === "number"
            ? riskHistoryModel.entryCount
            : (typeof riskHistoryModel.count === "number" ? riskHistoryModel.count : 0)
        if (typeof riskHistoryModel.get === "function" && count > 0) {
            const collected = []
            for (let i = count - 1; i >= 0; --i) {
                const value = riskHistoryModel.get(i)
                if (value)
                    collected.push(value)
            }
            return normalizeRiskTimeline(collected)
        }

        if (Array.isArray(riskHistoryModel.entries))
            return normalizeRiskTimeline(riskHistoryModel.entries)

        return []
    }

    function computeOpenPositions() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.openPositionsSnapshot === "function")
                snapshot = strategyController.openPositionsSnapshot()
            else if (Array.isArray(strategyController.openPositions))
                snapshot = strategyController.openPositions
            else if (typeof strategyController.positionsSnapshot === "function")
                snapshot = strategyController.positionsSnapshot()
        }
        if (!snapshot && appController) {
            if (typeof appController.openPositionsSnapshot === "function")
                snapshot = appController.openPositionsSnapshot()
            else if (Array.isArray(appController.openPositions))
                snapshot = appController.openPositions
        }
        return normalizeOpenPositions(snapshot)
    }

    function computePendingOrders() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.pendingOrdersSnapshot === "function")
                snapshot = strategyController.pendingOrdersSnapshot()
            else if (typeof strategyController.openOrdersSnapshot === "function")
                snapshot = strategyController.openOrdersSnapshot()
            else if (typeof strategyController.orderQueueSnapshot === "function")
                snapshot = strategyController.orderQueueSnapshot()
            else if (Array.isArray(strategyController.pendingOrders))
                snapshot = strategyController.pendingOrders
            else if (Array.isArray(strategyController.openOrders))
                snapshot = strategyController.openOrders
        }
        if (!snapshot && appController) {
            if (typeof appController.pendingOrdersSnapshot === "function")
                snapshot = appController.pendingOrdersSnapshot()
            else if (typeof appController.openOrdersSnapshot === "function")
                snapshot = appController.openOrdersSnapshot()
            else if (typeof appController.orderQueueSnapshot === "function")
                snapshot = appController.orderQueueSnapshot()
            else if (Array.isArray(appController.pendingOrders))
                snapshot = appController.pendingOrders
        }
        return normalizePendingOrders(snapshot)
    }

    function computeTradeHistory() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.tradeHistorySnapshot === "function")
                snapshot = strategyController.tradeHistorySnapshot()
            else if (typeof strategyController.recentTradesSnapshot === "function")
                snapshot = strategyController.recentTradesSnapshot()
            else if (Array.isArray(strategyController.tradeHistory))
                snapshot = strategyController.tradeHistory
        }
        if (!snapshot && appController) {
            if (typeof appController.tradeHistorySnapshot === "function")
                snapshot = appController.tradeHistorySnapshot()
            else if (Array.isArray(appController.tradeHistory))
                snapshot = appController.tradeHistory
        }
        return normalizeTradeHistory(snapshot)
    }

    function computeSignalAlerts() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.signalFeedSnapshot === "function")
                snapshot = strategyController.signalFeedSnapshot()
            else if (typeof strategyController.signalsSnapshot === "function")
                snapshot = strategyController.signalsSnapshot()
            else if (typeof strategyController.recentSignalsSnapshot === "function")
                snapshot = strategyController.recentSignalsSnapshot()
            else if (Array.isArray(strategyController.signalFeed))
                snapshot = strategyController.signalFeed
            else if (Array.isArray(strategyController.signals))
                snapshot = strategyController.signals
        }
        if (!snapshot && appController) {
            if (typeof appController.signalFeedSnapshot === "function")
                snapshot = appController.signalFeedSnapshot()
            else if (typeof appController.signalsSnapshot === "function")
                snapshot = appController.signalsSnapshot()
            else if (Array.isArray(appController.signalFeed))
                snapshot = appController.signalFeed
        }
        return normalizeSignalAlerts(snapshot)
    }

    function computeScenarioTests() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.scenarioTestsSnapshot === "function")
                snapshot = strategyController.scenarioTestsSnapshot()
            else if (typeof strategyController.stressTestResultsSnapshot === "function")
                snapshot = strategyController.stressTestResultsSnapshot()
            else if (typeof strategyController.backtestScenariosSnapshot === "function")
                snapshot = strategyController.backtestScenariosSnapshot()
            else if (Array.isArray(strategyController.scenarioTests))
                snapshot = strategyController.scenarioTests
            else if (Array.isArray(strategyController.stressTests))
                snapshot = strategyController.stressTests
        }
        if (!snapshot && appController) {
            if (typeof appController.scenarioTestsSnapshot === "function")
                snapshot = appController.scenarioTestsSnapshot()
            else if (typeof appController.stressTestResultsSnapshot === "function")
                snapshot = appController.stressTestResultsSnapshot()
            else if (typeof appController.backtestScenariosSnapshot === "function")
                snapshot = appController.backtestScenariosSnapshot()
            else if (Array.isArray(appController.scenarioTests))
                snapshot = appController.scenarioTests
            else if (Array.isArray(appController.stressTests))
                snapshot = appController.stressTests
        }
        return normalizeScenarioTests(snapshot)
    }

    function computeAutomationRules() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.automationRulesSnapshot === "function")
                snapshot = strategyController.automationRulesSnapshot()
            else if (typeof strategyController.automationConfigSnapshot === "function")
                snapshot = strategyController.automationConfigSnapshot()
            else if (Array.isArray(strategyController.automationRules))
                snapshot = strategyController.automationRules
        }
        if (!snapshot && appController) {
            if (typeof appController.automationRulesSnapshot === "function")
                snapshot = appController.automationRulesSnapshot()
            else if (Array.isArray(appController.automationRules))
                snapshot = appController.automationRules
        }
        return normalizeAutomationRules(snapshot)
    }

    function computeExecutionDiagnostics() {
        let snapshot = null
        if (appController) {
            if (typeof appController.executionDiagnosticsSnapshot === "function")
                snapshot = appController.executionDiagnosticsSnapshot()
            else if (appController.executionDiagnostics)
                snapshot = appController.executionDiagnostics
        }
        if (!snapshot && strategyController) {
            if (typeof strategyController.executionDiagnosticsSnapshot === "function")
                snapshot = strategyController.executionDiagnosticsSnapshot()
            else if (strategyController.executionDiagnostics)
                snapshot = strategyController.executionDiagnostics
        }
        return normalizeExecutionDiagnostics(snapshot)
    }

    function computeMarketSentiment() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.marketSentimentSnapshot === "function")
                snapshot = strategyController.marketSentimentSnapshot()
            else if (typeof strategyController.sentimentSnapshot === "function")
                snapshot = strategyController.sentimentSnapshot()
            else if (strategyController.marketSentiment)
                snapshot = strategyController.marketSentiment
        }
        if (!snapshot && appController) {
            if (typeof appController.marketSentimentSnapshot === "function")
                snapshot = appController.marketSentimentSnapshot()
            else if (typeof appController.sentimentSnapshot === "function")
                snapshot = appController.sentimentSnapshot()
            else if (appController.marketSentiment)
                snapshot = appController.marketSentiment
        }
        return normalizeMarketSentiment(snapshot)
    }

    function computeNewsHeadlines() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.newsHeadlinesSnapshot === "function")
                snapshot = strategyController.newsHeadlinesSnapshot()
            else if (typeof strategyController.newsFeedSnapshot === "function")
                snapshot = strategyController.newsFeedSnapshot()
            else if (typeof strategyController.marketNewsSnapshot === "function")
                snapshot = strategyController.marketNewsSnapshot()
            else if (Array.isArray(strategyController.newsHeadlines))
                snapshot = strategyController.newsHeadlines
            else if (Array.isArray(strategyController.marketNews))
                snapshot = strategyController.marketNews
        }
        if (!snapshot && appController) {
            if (typeof appController.newsHeadlinesSnapshot === "function")
                snapshot = appController.newsHeadlinesSnapshot()
            else if (typeof appController.newsFeedSnapshot === "function")
                snapshot = appController.newsFeedSnapshot()
            else if (Array.isArray(appController.newsHeadlines))
                snapshot = appController.newsHeadlines
            else if (Array.isArray(appController.newsFeed))
                snapshot = appController.newsFeed
        }
        return normalizeNewsHeadlines(snapshot)
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

    function computePerformanceComparison() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.performanceComparisonSnapshot === "function")
                snapshot = strategyController.performanceComparisonSnapshot()
            else if (typeof strategyController.performanceMetricsSnapshot === "function")
                snapshot = strategyController.performanceMetricsSnapshot()
            else if (strategyController.performanceComparison)
                snapshot = strategyController.performanceComparison
        }
        if (!snapshot && appController) {
            if (typeof appController.performanceComparisonSnapshot === "function")
                snapshot = appController.performanceComparisonSnapshot()
            else if (typeof appController.performanceMetricsSnapshot === "function")
                snapshot = appController.performanceMetricsSnapshot()
            else if (appController.performanceComparison)
                snapshot = appController.performanceComparison
        }
        return normalizePerformanceComparison(snapshot)
    }

    function computeCapitalAllocation() {
        let snapshot = null
        if (strategyController) {
            if (typeof strategyController.capitalAllocationSnapshot === "function")
                snapshot = strategyController.capitalAllocationSnapshot()
            else if (typeof strategyController.allocationSnapshot === "function")
                snapshot = strategyController.allocationSnapshot()
            else if (Array.isArray(strategyController.capitalAllocation))
                snapshot = strategyController.capitalAllocation
        }
        if (!snapshot && appController) {
            if (typeof appController.capitalAllocationSnapshot === "function")
                snapshot = appController.capitalAllocationSnapshot()
            else if (Array.isArray(appController.capitalAllocation))
                snapshot = appController.capitalAllocation
        }
        return normalizeCapitalAllocation(snapshot)
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

    function computeActivityLog() {
        if (appController && typeof appController.activityLogSnapshot === "function")
            return normalizeActivityLog(appController.activityLogSnapshot())
        if (strategyController && typeof strategyController.activityLogSnapshot === "function")
            return normalizeActivityLog(strategyController.activityLogSnapshot())
        return []
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
        const message = success ? qsTr("Uruchomiono harmonogram")
                                : qsTr("Nie udało się uruchomić harmonogramu")
        const next = Object.assign({}, base, {
            schedulerRunning: success ? true : base.schedulerRunning,
            lastActionSuccess: success,
            lastActionMessage: message,
            lastActionAt: nowIso,
        })
        updateControlState(next)
        recordActivityEvent(success ? "scheduler:start" : "scheduler:start:error", message, {
            success: success,
        })
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
        const message = success ? qsTr("Zatrzymano harmonogram")
                                : qsTr("Nie udało się zatrzymać harmonogramu")
        const next = Object.assign({}, base, {
            schedulerRunning: success ? false : base.schedulerRunning,
            lastActionSuccess: success,
            lastActionMessage: message,
            lastActionAt: nowIso,
        })
        updateControlState(next)
        recordActivityEvent(success ? "scheduler:stop" : "scheduler:stop:error", message, {
            success: success,
        })
        if (!demoModeActive)
            refreshFromLive()
        return success
    }

    function updateDemoScenarioTestsAfterRiskRefresh() {
        const items = Array.isArray(scenarioTests) ? scenarioTests : []
        if (items.length === 0)
            return
        const now = new Date()
        const nowIso = now.toISOString()
        const updated = []
        for (let i = 0; i < items.length; ++i) {
            const base = items[i] || {}
            const baseRunCount = Number(base.runCount !== undefined ? base.runCount : 0)
            const next = Object.assign({}, base)
            next.runCount = isNaN(baseRunCount) ? 1 : baseRunCount + 1
            next.lastRunAt = nowIso
            if (!next.completedAt || next.completedAt.length === 0)
                next.completedAt = nowIso
            if (!next.startedAt || next.startedAt.length === 0)
                next.startedAt = nowIso
            if (typeof next.pnlImpact === "number")
                next.pnlImpact = Math.max(next.pnlImpact, -0.25)
            updated.push(next)
        }
        setScenarioTests(updated)
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
        const message = success ? qsTr("Zainicjowano odświeżenie ryzyka")
                                : qsTr("Nie udało się odświeżyć ryzyka")
        const next = Object.assign({}, base, {
            lastActionSuccess: success,
            lastActionMessage: message,
            lastActionAt: nowIso,
            manualRefreshCount: success ? base.manualRefreshCount + 1 : base.manualRefreshCount,
        })
        if (success) {
            next.lastRiskRefreshAt = nowIso
            if (demoModeActive)
                next.nextRiskRefreshDueAt = nextDueIso
        }
        updateControlState(next)
        if (success && demoModeActive) {
            appendDemoRiskTimelineEntry()
            updateDemoScenarioTestsAfterRiskRefresh()
            updateDemoExecutionDiagnosticsAfterRiskRefresh()
        }
        recordActivityEvent(success ? "risk:refresh" : "risk:refresh:error", message, {
            success: success,
        })
        if (!demoModeActive)
            refreshFromLive()
        return success
    }

    function refreshActivityLog() {
        if (demoModeActive)
            return
        const liveLog = computeActivityLog()
        const pending = Array.isArray(pendingActivityEvents) ? pendingActivityEvents.slice(0) : []
        if (liveLog.length > 0 || pending.length > 0) {
            activityLog = pending.concat(liveLog)
        } else if (!Array.isArray(activityLog) || activityLog.length === 0) {
            activityLog = []
        }
        pendingActivityEvents = pending
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
        function onActivityLogChanged() { root.refreshActivityLog() }
        function onPendingOrdersChanged() { root.refreshFromLive() }
        function onOrderQueueChanged() { root.refreshFromLive() }
        function onSignalFeedChanged() { root.refreshFromLive() }
        function onSignalsChanged() { root.refreshFromLive() }
        function onMarketSentimentChanged() { root.refreshFromLive() }
        function onNewsFeedChanged() { root.refreshFromLive() }
        function onNewsHeadlinesChanged() { root.refreshFromLive() }
        function onCapitalAllocationChanged() { root.refreshFromLive() }
        function onPerformanceComparisonChanged() { root.refreshFromLive() }
        function onPerformanceMetricsChanged() { root.refreshFromLive() }
        function onScenarioTestsChanged() { root.refreshFromLive() }
        function onStressTestsChanged() { root.refreshFromLive() }
        function onBacktestScenariosChanged() { root.refreshFromLive() }
        function onExecutionDiagnosticsChanged() { root.refreshFromLive() }
        function onAutomationRulesChanged() { root.refreshFromLive() }
        ignoreUnknownSignals: true
    }

    Connections {
        id: strategyConnections
        target: root.strategyController
        function onSchedulerListChanged() { root.refreshFromLive() }
        function onDecisionConfigChanged() { root.refreshFromLive() }
        function onSchedulerStateChanged() { root.refreshFromLive() }
        function onActivityLogChanged() { root.refreshActivityLog() }
        function onPositionsChanged() { root.refreshFromLive() }
        function onOpenPositionsChanged() { root.refreshFromLive() }
        function onTradeHistoryChanged() { root.refreshFromLive() }
        function onRecentTradesChanged() { root.refreshFromLive() }
        function onPendingOrdersChanged() { root.refreshFromLive() }
        function onOpenOrdersChanged() { root.refreshFromLive() }
        function onOrderQueueChanged() { root.refreshFromLive() }
        function onSignalsChanged() { root.refreshFromLive() }
        function onSignalFeedChanged() { root.refreshFromLive() }
        function onRecentSignalsChanged() { root.refreshFromLive() }
        function onMarketSentimentChanged() { root.refreshFromLive() }
        function onSentimentChanged() { root.refreshFromLive() }
        function onNewsFeedChanged() { root.refreshFromLive() }
        function onNewsHeadlinesChanged() { root.refreshFromLive() }
        function onCapitalAllocationChanged() { root.refreshFromLive() }
        function onAllocationChanged() { root.refreshFromLive() }
        function onPerformanceComparisonChanged() { root.refreshFromLive() }
        function onPerformanceMetricsChanged() { root.refreshFromLive() }
        function onScenarioTestsChanged() { root.refreshFromLive() }
        function onStressTestsChanged() { root.refreshFromLive() }
        function onBacktestScenariosChanged() { root.refreshFromLive() }
        function onAutomationRulesChanged() { root.refreshFromLive() }
        function onAutomationConfigChanged() { root.refreshFromLive() }
        function onExecutionDiagnosticsChanged() { root.refreshFromLive() }
        ignoreUnknownSignals: true
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
