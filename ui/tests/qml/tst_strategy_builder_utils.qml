import QtQuick
import QtTest
import "../../qml/views/StrategyBuilderUtils.js" as StrategyUtils

TestCase {
    name: "StrategyBuilderUtils"

    function test_default_params_data_feed() {
        var params = StrategyUtils.defaultParams("data_feed")
        compare(params.symbol, "BTCUSDT")
        compare(params.interval, "1h")
        compare(params.source, "sandbox")
    }

    function test_merge_params_normalizes_types() {
        var params = StrategyUtils.mergeParams("filter", { risk_limit: "3.5", lookback: "120", enabled: "false" })
        compare(params.risk_limit, 3.5)
        compare(params.lookback, 120)
        compare(params.enabled, false)
    }

    function test_merge_params_preserves_unknown_keys() {
        var params = StrategyUtils.mergeParams("signal", { threshold: 0.75, lookback: 10, custom: "value" })
        compare(params.threshold, 0.75)
        compare(params.custom, "value")
    }

    function test_summary_order_matches_template() {
        var order = StrategyUtils.summaryOrder("execution")
        compare(order.length, 4)
        compare(order[0], "venue")
        compare(order[order.length - 1], "quantity")
    }
}
