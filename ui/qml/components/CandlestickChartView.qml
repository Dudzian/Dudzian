import QtQuick
import QtQuick.Controls
import QtCharts

ChartView {
    id: chartView

    // --- Inputs ---------------------------------------------------------------
    property var model
    property var indicatorModel: null
    property PerformanceGuard performanceGuard
    property bool reduceMotion: false

    // Definicje nakładek (kolejność musi odpowiadać overlaySeriesList)
    property var overlayDefinitions: [
        { key: "ema_fast", label: qsTr("EMA 12"),  color: Qt.rgba(0.96, 0.74, 0.23, 1), secondary: false },
        { key: "ema_slow", label: qsTr("EMA 26"),  color: Qt.rgba(0.62, 0.81, 0.93, 1), secondary: true  },
        { key: "vwap",     label: qsTr("VWAP"),    color: Qt.rgba(0.74, 0.53, 0.96, 1), secondary: true  }
    ]

    // --- Viz defaults ---------------------------------------------------------
    backgroundRoundness: 8
    theme: ChartView.ChartThemeDark
    animationOptions: ChartView.NoAnimation
    legend.visible: false
    dropShadowEnabled: false

    // --- Axes -----------------------------------------------------------------
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

    // --- Price (candles) ------------------------------------------------------
    CandlestickSeries {
        id: candleSeries
        increasingColor: Qt.rgba(0.2, 0.8, 0.5, 1)
        decreasingColor: Qt.rgba(0.85, 0.25, 0.3, 1)
        bodyWidth: 0.7
    }

    // --- Overlays (lines) -----------------------------------------------------
    LineSeries {
        id: emaFastSeries
        objectName: "emaFastSeries"
        color: chartView.overlayDefinitions[0].color
        width: 1.6
        useOpenGL: false
        axisX: axisX
        axisY: axisY
        opacity: visible ? 1 : 0
        Behavior on opacity {
            NumberAnimation { duration: chartView.reduceMotion ? 0 : 180; easing.type: Easing.OutCubic }
        }
    }
    LineSeries {
        id: emaSlowSeries
        objectName: "emaSlowSeries"
        color: chartView.overlayDefinitions[1].color
        width: 1.3
        useOpenGL: false
        axisX: axisX
        axisY: axisY
        opacity: visible ? 1 : 0
        Behavior on opacity {
            NumberAnimation { duration: chartView.reduceMotion ? 0 : 180; easing.type: Easing.OutCubic }
        }
    }
    LineSeries {
        id: vwapSeries
        objectName: "vwapSeries"
        color: chartView.overlayDefinitions[2].color
        width: 1.1
        useOpenGL: false
        axisX: axisX
        axisY: axisY
        opacity: visible ? 1 : 0
        Behavior on opacity {
            NumberAnimation { duration: chartView.reduceMotion ? 0 : 180; easing.type: Easing.OutCubic }
        }
    }

    readonly property var overlaySeriesList: [emaFastSeries, emaSlowSeries, vwapSeries]

    // --- Crosshair + tooltip --------------------------------------------------
    Item {
        anchors.fill: parent
        z: 2
        property bool crosshairVisible: false
        property real crosshairX: 0
        property var crosshairData: ({})

        Behavior on crosshairX {
            enabled: !chartView.reduceMotion
            NumberAnimation { duration: chartView.reduceMotion ? 0 : 90; easing.type: Easing.OutCubic }
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
                Label {
                    text: qsTr("O %1 H %2 L %3 C %4")
                          .arg(crosshairData.open.toFixed(2))
                          .arg(crosshairData.high.toFixed(2))
                          .arg(crosshairData.low.toFixed(2))
                          .arg(crosshairData.close.toFixed(2))
                }
                Label { text: qsTr("Vol %1").arg(crosshairData.volume.toFixed(2)) }
            }
        }
    }

    // --- Lifecycle ------------------------------------------------------------
    Component.onCompleted: {
        candleSeries.axisX = axisX
        candleSeries.axisY = axisY
        rebuild()
        refreshOverlayVisibility()
    }
    onModelChanged: rebuild()
    onPerformanceGuardChanged: refreshOverlayVisibility()
    onReduceMotionChanged: refreshOverlayVisibility()

    // --- Model signal handlers ------------------------------------------------
    Connections {
        target: model
        function onModelReset() { chartView.rebuild() }
        function onRowsInserted(parent, first, last) {
            for (let row = first; row <= last; ++row) chartView.appendRow(row)
            chartView.updateAxisRange()
            chartView.updateOverlays()
        }
        function onDataChanged(topLeft, bottomRight, roles) {
            for (let row = topLeft.row; row <= bottomRight.row; ++row) {
                const candle = model.candleAt(row)
                if (!candle) continue
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
            chartView.updateOverlays()
        }
        function onRowsRemoved(parent, first, last) {
            // Bezpiecznie przebuduj całość (prostota > mikro-optymalizacje)
            chartView.rebuild()
        }
    }

    Connections {
        target: indicatorModel
        function onModelReset() { chartView.updateOverlays() }
        function onRowsInserted() { chartView.updateOverlays() }
        function onRowsRemoved() { chartView.updateOverlays() }
        function onDataChanged() { chartView.updateOverlays() }
    }

    // --- API ------------------------------------------------------------------
    function rebuild() {
        candleSeries.clear()
        if (!model || model.count === undefined) return
        for (let row = 0; row < model.count; ++row) appendRow(row)
        updateAxisRange()
        updateOverlays()
    }

    function appendRow(row) {
        if (!model) return
        const candle = model.candleAt(row)
        if (!candle || candle.timestamp === undefined) return
        const set = Qt.createQmlObject('import QtCharts; CandlestickSet {}', candleSeries)
        set.timestamp = candle.timestamp
        set.open = candle.open
        set.high = candle.high
        set.low = candle.low
        set.close = candle.close
        candleSeries.append(set)
    }

    function updateAxisRange() {
        if (candleSeries.count === 0) return
        const first = candleSeries.at(0)
        const last  = candleSeries.at(candleSeries.count - 1)
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

    function sampleAt(x) {
        if (!model || model.count === 0) return ({})
        const point = chartView.mapToValue(Qt.point(x, height / 2), candleSeries)
        const timestamp = point.x
        var closest = null
        var bestDelta = Number.POSITIVE_INFINITY
        for (let row = 0; row < model.count; ++row) {
            const candle = model.candleAt(row)
            if (!candle) continue
            const delta = Math.abs(candle.timestamp - timestamp)
            if (delta < bestDelta) {
                bestDelta = delta
                closest = candle
            }
        }
        return closest || ({})
    }

    // --- Overlay logic --------------------------------------------------------
    function refreshOverlayVisibility() {
        var guard = performanceGuard
        var allowed = overlaySeriesList.length
        var activeCount = 0

        if (guard) {
            allowed = (guard.maxOverlayCount > 0) ? guard.maxOverlayCount : allowed

            // Ograniczamy animacje / nakładki przy reduceMotion
            if (reduceMotion)
                allowed = Math.min(allowed, 1)

            // Gdy zadany próg FPS dla wyłączenia drugorzędnych <= aktualnego celu FPS
            // (brak aktualnego FPS po stronie QML — stosujemy heurystykę względem celu)
            if (guard.disableSecondaryWhenFpsBelow > 0
                    && guard.fpsTarget < guard.disableSecondaryWhenFpsBelow)
                allowed = Math.min(allowed, 1)
        } else if (reduceMotion) {
            allowed = 1
        }

        for (var i = 0; i < overlaySeriesList.length; ++i) {
            var series = overlaySeriesList[i]
            if (!series) continue
            var def = overlayDefinitions[i]
            var visible = i < allowed
            if (visible && def.secondary && allowed <= 1)
                visible = false
            series.visible = visible
            series.opacity = visible ? 1.0 : 0.0
            if (visible) activeCount++
        }

        if (typeof appController !== "undefined" && appController.notifyOverlayUsage)
            appController.notifyOverlayUsage(activeCount, allowed, reduceMotion)

        updateOverlays()
    }

    function seriesEntry(seriesId) {
        if (!indicatorModel || indicatorModel.count === undefined) return null
        for (let row = 0; row < indicatorModel.count; ++row) {
            const entry = indicatorModel.get(row)
            if (entry && entry.seriesId === seriesId)
                return entry
        }
        return null
    }

    function samplesForSeries(seriesId) {
        const entry = seriesEntry(seriesId)
        if (entry && entry.samples !== undefined)
            return entry.samples
        if (model && typeof model.overlaySeries === "function")
            return model.overlaySeries(seriesId) || []
        return []
    }

    function updateOverlays() {
        if (!model && !indicatorModel) return
        for (var i = 0; i < overlaySeriesList.length; ++i) {
            var series = overlaySeriesList[i]
            if (!series) continue
            series.clear()
            if (!series.visible) continue
            var def = overlayDefinitions[i]
            var entry = seriesEntry(def.key)
            if (entry && entry.color !== undefined)
                series.color = entry.color
            var samples = samplesForSeries(def.key)
            for (var j = 0; j < samples.length; ++j) {
                var sample = samples[j]
                if (!sample || sample.timestamp === undefined || sample.value === undefined) continue
                series.append(new Date(sample.timestamp), sample.value)
            }
        }
    }
}
