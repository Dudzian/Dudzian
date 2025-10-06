import QtQuick
import QtQuick.Controls
import QtCharts

ChartView {
    id: chartView
    property var model
    property PerformanceGuard performanceGuard
    property bool reduceMotion: false
    backgroundRoundness: 8
    theme: ChartView.ChartThemeDark
    animationOptions: ChartView.NoAnimation
    legend.visible: false
    dropShadowEnabled: false

    CandlestickSeries {
        id: candleSeries
        increasingColor: Qt.rgba(0.2, 0.8, 0.5, 1)
        decreasingColor: Qt.rgba(0.85, 0.25, 0.3, 1)
        bodyWidth: 0.7
    }

    DateTimeAxis {
        id: axisX
        format: "HH:mm"
        labelsVisible: true
        gridVisible: false
    }

    ValueAxis {
        id: axisY
        labelFormat: "%.2f"
        gridLineColor: Qt.rgba(1, 1, 1, 0.08)
    }

    axisX: axisX
    axisY: axisY

    onModelChanged: rebuild()
    onPerformanceGuardChanged: {
        reduceMotion = performanceGuard.reduceMotionAfterSeconds <= 0.6
                || performanceGuard.fpsTarget < 90
                || performanceGuard.jankThresholdMs < 12.0
    }

    Component.onCompleted: {
        candleSeries.axisX = axisX
        candleSeries.axisY = axisY
        rebuild()
    }

    function rebuild() {
        candleSeries.clear()
        if (!model)
            return
        for (let row = 0; row < model.count; ++row) {
            appendRow(row)
        }
        updateAxisRange()
    }

    function appendRow(row) {
        if (!model)
            return
        const candle = model.candleAt(row)
        if (!candle || candle.timestamp === undefined)
            return
        const set = Qt.createQmlObject('import QtCharts; CandlestickSet {}', candleSeries)
        set.timestamp = candle.timestamp
        set.open = candle.open
        set.high = candle.high
        set.low = candle.low
        set.close = candle.close
        candleSeries.append(set)
    }

    function updateAxisRange() {
        if (candleSeries.count === 0)
            return
        const first = candleSeries.at(0)
        const last = candleSeries.at(candleSeries.count - 1)
        axisX.min = new Date(first.timestamp)
        axisX.max = new Date(last.timestamp)
        var minValue = Number.POSITIVE_INFINITY
        var maxValue = Number.NEGATIVE_INFINITY
        for (let i = 0; i < candleSeries.count; ++i) {
            const set = candleSeries.at(i)
            minValue = Math.min(minValue, set.low)
            maxValue = Math.max(maxValue, set.high)
        }
        if (isFinite(minValue) && isFinite(maxValue)) {
            const padding = Math.max(0.01, (maxValue - minValue) * 0.05)
            axisY.min = minValue - padding
            axisY.max = maxValue + padding
        }
    }

    Connections {
        target: model
        function onModelReset() { chartView.rebuild() }
        function onRowsInserted(parent, first, last) {
            for (let row = first; row <= last; ++row)
                chartView.appendRow(row)
            chartView.updateAxisRange()
        }
        function onDataChanged(topLeft, bottomRight, roles) {
            for (let row = topLeft.row; row <= bottomRight.row; ++row) {
                const candle = model.candleAt(row)
                if (!candle)
                    continue
                if (row < candleSeries.count) {
                    const set = candleSeries.at(row)
                    set.timestamp = candle.timestamp
                    set.open = candle.open
                    set.high = candle.high
                    set.low = candle.low
                    set.close = candle.close
                }
            }
            chartView.updateAxisRange()
        }
    }

    Item {
        anchors.fill: parent
        z: 2
        property bool crosshairVisible: false
        property real crosshairX: 0
        property var crosshairData: ({})

        Behavior on crosshairX {
            enabled: !chartView.reduceMotion
            NumberAnimation {
                duration: chartView.reduceMotion ? 0 : 90
                easing.type: Easing.OutCubic
            }
        }

        Rectangle {
            anchors.fill: parent
            color: Qt.rgba(0, 0, 0, 0)

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onPositionChanged: {
                    crosshairVisible = true
                    crosshairX = mouse.x
                    crosshairData = chartView.sampleAt(mouse.x)
                }
                onExited: crosshairVisible = false
            }
        }

        Rectangle {
            visible: crosshairVisible
            width: 1
            color: Qt.rgba(1, 1, 1, 0.35)
            x: crosshairX
            anchors.top: parent.top
            anchors.bottom: parent.bottom
        }

        Rectangle {
            visible: crosshairVisible && crosshairData.timestamp !== undefined
            color: Qt.rgba(0, 0, 0, 0.65)
            radius: 4
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 12
            padding: 8

            Column {
                spacing: 2
                Label { text: Qt.formatDateTime(new Date(crosshairData.timestamp), "yyyy-MM-dd HH:mm") }
                Label { text: qsTr("O %1 H %2 L %3 C %4").arg(crosshairData.open.toFixed(2)).arg(crosshairData.high.toFixed(2)).arg(crosshairData.low.toFixed(2)).arg(crosshairData.close.toFixed(2)) }
                Label { text: qsTr("Vol %1").arg(crosshairData.volume.toFixed(2)) }
            }
        }
    }

    function sampleAt(x) {
        if (!model || model.count === 0)
            return ({})
        const point = chartView.mapToValue(Qt.point(x, height / 2), candleSeries)
        const timestamp = point.x
        var closest = null
        var bestDelta = Number.POSITIVE_INFINITY
        for (let row = 0; row < model.count; ++row) {
            const candle = model.candleAt(row)
            if (!candle)
                continue
            const delta = Math.abs(candle.timestamp - timestamp)
            if (delta < bestDelta) {
                bestDelta = delta
                closest = candle
            }
        }
        return closest || ({})
    }
}
