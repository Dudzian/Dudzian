.pragma library

function extractPnl(entry) {
    if (!entry)
        return 0

    if (entry.pnl !== undefined && entry.pnl !== null) {
        const numeric = Number(entry.pnl)
        return isNaN(numeric) ? 0 : numeric
    }

    if (entry.performance && entry.performance.pnl !== undefined) {
        const numeric = Number(entry.performance.pnl)
        return isNaN(numeric) ? 0 : numeric
    }

    if (entry.performance && entry.performance.realized !== undefined) {
        const numeric = Number(entry.performance.realized)
        return isNaN(numeric) ? 0 : numeric
    }

    if (entry.metrics && entry.metrics.pnl !== undefined) {
        const numeric = Number(entry.metrics.pnl)
        return isNaN(numeric) ? 0 : numeric
    }

    if (entry.decision && entry.decision.shouldTrade)
        return 1

    if (entry.decision && entry.decision.state === "skip")
        return -1

    return 0
}

function bucketLabel(timestamp, fallbackLabel) {
    if (!timestamp || timestamp.length === 0)
        return fallbackLabel

    const date = new Date(timestamp)
    if (isNaN(date.getTime()))
        return fallbackLabel

    return Qt.formatDateTime(date, "yyyy-MM-dd")
}

function normaliseSymbol(entry, fallbackSymbol) {
    if (!entry)
        return fallbackSymbol

    const candidates = [
        entry.symbol,
        entry.asset,
        entry.instrument,
        entry.portfolio
    ]
    for (let i = 0; i < candidates.length; ++i) {
        const value = candidates[i]
        if (value !== undefined && value !== null && String(value).length > 0)
            return String(value)
    }

    return fallbackSymbol
}

function buildHeatmap(records, options) {
    options = options || {}
    const fallbackSymbol = options.noSymbol || "N/A"
    const fallbackBucket = options.noDate || "N/A"

    const matrix = {}
    const bucketOrder = []

    for (let i = 0; i < (records || []).length; ++i) {
        const entry = records[i] || {}
        const symbol = normaliseSymbol(entry, fallbackSymbol)
        const bucket = bucketLabel(entry.timestamp || entry.generated_at || entry.created_at, fallbackBucket)

        if (bucketOrder.indexOf(bucket) === -1)
            bucketOrder.push(bucket)

        if (!Object.prototype.hasOwnProperty.call(matrix, symbol))
            matrix[symbol] = {}

        const numeric = extractPnl(entry)
        const currentValue = matrix[symbol][bucket] || 0
        matrix[symbol][bucket] = currentValue + numeric
    }

    bucketOrder.sort()

    const rows = []
    const symbols = Object.keys(matrix)
    symbols.sort(function(a, b) { return a.localeCompare(b) })

    for (let s = 0; s < symbols.length; ++s) {
        const label = symbols[s]
        const row = { label: label, buckets: [] }
        for (let b = 0; b < bucketOrder.length; ++b) {
            const bucketKey = bucketOrder[b]
            const value = matrix[label][bucketKey] || 0
            row.buckets.push({ label: bucketKey, value: value })
        }
        rows.push(row)
    }

    return rows
}

function toDecisionCount(records) {
    return (records || []).length
}

function aggregateBySymbol(records, options) {
    options = options || {}
    const fallbackSymbol = options.noSymbol || "N/A"

    const totals = {}

    for (let i = 0; i < (records || []).length; ++i) {
        const entry = records[i] || {}
        const symbol = normaliseSymbol(entry, fallbackSymbol)
        const numeric = extractPnl(entry)
        if (!Object.prototype.hasOwnProperty.call(totals, symbol))
            totals[symbol] = 0
        totals[symbol] += numeric
    }

    const rows = []
    const keys = Object.keys(totals)
    keys.sort(function(a, b) { return a.localeCompare(b) })
    for (let j = 0; j < keys.length; ++j) {
        const key = keys[j]
        rows.push({ label: key, value: totals[key] })
    }
    return rows
}

function rankSymbols(records, options) {
    options = options || {}
    const limit = options.limit !== undefined ? Number(options.limit) : 5
    const normalizedLimit = isNaN(limit) || limit <= 0 ? undefined : Math.floor(limit)

    const aggregated = aggregateBySymbol(records, options)
    const sorted = aggregated.slice().sort(function(a, b) { return b.value - a.value })

    const positives = []
    const negatives = []

    for (let i = 0; i < sorted.length; ++i) {
        const entry = sorted[i]
        if (entry.value > 0)
            positives.push(entry)
        else if (entry.value < 0)
            negatives.push(entry)
    }

    const bottomSorted = negatives.slice().sort(function(a, b) { return a.value - b.value })

    const slice = function(items) {
        if (normalizedLimit === undefined)
            return items
        return items.slice(0, normalizedLimit)
    }

    return {
        top: slice(positives),
        bottom: slice(bottomSorted)
    }
}

function latestTimestamp(records) {
    let latest = null
    for (let i = 0; i < (records || []).length; ++i) {
        const entry = records[i] || {}
        const candidate = entry.timestamp || entry.generated_at || entry.created_at || entry.updated_at
        if (!candidate)
            continue
        const date = new Date(candidate)
        if (isNaN(date.getTime()))
            continue
        if (!latest || date.getTime() > latest.getTime())
            latest = date
    }
    return latest
}

var exports = {
    extractPnl: extractPnl,
    buildHeatmap: buildHeatmap,
    normaliseSymbol: normaliseSymbol,
    bucketLabel: bucketLabel,
    toDecisionCount: toDecisionCount,
    aggregateBySymbol: aggregateBySymbol,
    rankSymbols: rankSymbols,
    latestTimestamp: latestTimestamp
}
