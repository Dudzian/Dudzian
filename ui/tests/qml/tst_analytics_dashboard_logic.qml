import QtQuick
import QtTest
import "../../qml/views/AnalyticsUtils.js" as AnalyticsUtils

TestCase {
    name: "AnalyticsDashboardLogic"

    function test_extractPnl_prioritisesTopLevel() {
        compare(AnalyticsUtils.extractPnl({ pnl: "5.25" }), 5.25)
    }

    function test_extractPnl_fallsBackToDecision() {
        compare(AnalyticsUtils.extractPnl({ decision: { shouldTrade: true } }), 1)
        compare(AnalyticsUtils.extractPnl({ decision: { state: "skip" } }), -1)
    }

    function test_buildHeatmap_groupsBySymbolAndBucket() {
        const records = [
            { symbol: "BTC/USDT", timestamp: "2024-02-01T12:00:00Z", pnl: 2.5 },
            { symbol: "BTC/USDT", timestamp: "2024-02-02T08:30:00Z", performance: { pnl: -1.0 } },
            { asset: "ETH/USDT", generated_at: "2024-02-01T03:00:00Z", decision: { shouldTrade: true } },
            { portfolio: "Gamma", created_at: "", metrics: { pnl: 0.75 } }
        ]
        const rows = AnalyticsUtils.buildHeatmap(records, {
            noSymbol: "N/D",
            noDate: "Brak daty"
        })
        compare(rows.length, 3)

        const btcRow = rows.find(function(row) { return row.label === "BTC/USDT" })
        verify(btcRow !== undefined)
        compare(btcRow.buckets.length, 3)
        compare(btcRow.buckets[0].label, "2024-02-01")
        compare(btcRow.buckets[0].value, 2.5)
        compare(btcRow.buckets[1].label, "2024-02-02")
        compare(btcRow.buckets[1].value, -1)

        const ethRow = rows.find(function(row) { return row.label === "ETH/USDT" })
        verify(ethRow !== undefined)
        compare(ethRow.buckets[0].value, 1)

        const fallbackRow = rows.find(function(row) { return row.label === "Gamma" })
        verify(fallbackRow !== undefined)
        compare(fallbackRow.buckets[0].label, "Brak daty")
        compare(fallbackRow.buckets[0].value, 0.75)
    }

    function test_aggregateBySymbol_sumsPnlPerLabel() {
        const records = [
            { symbol: "BTC", pnl: 1.5 },
            { symbol: "BTC", performance: { pnl: -0.5 } },
            { asset: "ETH", decision: { shouldTrade: true } },
            { metrics: { pnl: 3 } }
        ]
        const aggregated = AnalyticsUtils.aggregateBySymbol(records, { noSymbol: "Fallback" })
        compare(aggregated.length, 3)

        const btc = aggregated.find(function(entry) { return entry.label === "BTC" })
        verify(btc !== undefined)
        compare(btc.value, 1.0)

        const eth = aggregated.find(function(entry) { return entry.label === "ETH" })
        verify(eth !== undefined)
        compare(eth.value, 1)

        const fallback = aggregated.find(function(entry) { return entry.label === "Fallback" })
        verify(fallback !== undefined)
        compare(fallback.value, 3)
    }

    function test_rankSymbols_returnsTopAndBottomWithinLimit() {
        const records = [
            { symbol: "AAA", pnl: 4 },
            { symbol: "BBB", pnl: -5 },
            { symbol: "CCC", pnl: 2 },
            { symbol: "DDD", pnl: -1 },
            { symbol: "EEE", pnl: 3 }
        ]
        const rankings = AnalyticsUtils.rankSymbols(records, { limit: 2 })
        compare(rankings.top.length, 2)
        compare(rankings.top[0].label, "AAA")
        compare(rankings.top[1].label, "EEE")
        compare(rankings.bottom.length, 2)
        compare(rankings.bottom[0].label, "BBB")
        compare(rankings.bottom[1].label, "DDD")
    }

    function test_latestTimestamp_prefersMostRecentValidDate() {
        const records = [
            { timestamp: "2024-02-01T00:00:00Z" },
            { generated_at: "2024-02-02T08:00:00Z" },
            { created_at: "", updated_at: "not-a-date" },
            { generated_at: "2024-01-28T12:00:00Z" }
        ]
        const latest = AnalyticsUtils.latestTimestamp(records)
        verify(latest !== null)
        compare(latest.getTime(), new Date("2024-02-02T08:00:00Z").getTime())
    }
}
