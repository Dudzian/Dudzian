import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Pane {
    id: root
    property var points: []
    property string title: qsTr("Krzywa kapitału")
    property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)
    // W CI możemy wyłączyć QtCharts (np. AV w offscreen); flaga przychodzi z context property.
    property bool chartsDisabled: typeof disableQtCharts !== "undefined" && !!disableQtCharts
    readonly property bool chartReady: chartLoader.status === Loader.Ready
    readonly property int chartSeriesCount: chartLoader.item && chartLoader.item.equitySeries
        ? chartLoader.item.equitySeries.count
        : 0
    padding: 12

    background: Rectangle {
        color: Qt.rgba(0, 0, 0, 0.25)
        radius: 8
        border.color: Qt.rgba(1, 1, 1, 0.08)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            Label {
                text: root.title
                font.pixelSize: 16
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Label {
                text: root.latestValueText()
                color: root.accentColor
                font.pixelSize: 14
            }
        }

        Item {
            id: chartHost
            Layout.fillWidth: true
            Layout.fillHeight: true

            Loader {
                id: chartLoader
                anchors.fill: parent
                source: root.chartsDisabled ? "" : "qrc:/qml/components/EquityCurveDashboardChart.qml"
                onLoaded: {
                    if (item) {
                        item.accentColor = root.accentColor
                        root.rebuildSeries()
                    }
                }
                onStatusChanged: {
                    if (status === Loader.Ready) {
                        root.rebuildSeries()
                    }
                }
            }

            Rectangle {
                anchors.fill: parent
                visible: root.chartsDisabled
                color: Qt.rgba(1, 1, 1, 0.03)
                radius: 6
                border.color: Qt.rgba(1, 1, 1, 0.08)
                Label {
                    anchors.centerIn: parent
                    text: qsTr("Charts disabled in CI")
                    color: palette.mid
                }
            }
        }
    }

    function latestValueText() {
        const data = points || []
        if (!data || data.length === 0)
            return qsTr("—")
        if (root.chartsDisabled) {
            const last = data[data.length - 1] || {}
            const rawValue = last.value !== undefined ? last.value : last.y
            const value = Number(rawValue)
            if (isNaN(value))
                return qsTr("—")
            return qsTr("%1").arg(value.toLocaleString(Qt.locale(), "f", 2))
        }
        const chartItem = chartLoader.item
        try {
            if (!chartItem || !chartItem.equitySeries || chartItem.equitySeries.count === 0)
                return qsTr("—")
            const point = chartItem.equitySeries.at(chartItem.equitySeries.count - 1)
            if (!point || point.y === undefined)
                return qsTr("—")
            return qsTr("%1").arg(Number(point.y).toLocaleString(Qt.locale(), "f", 2))
        } catch (err) {
            return qsTr("—")
        }
    }

    function rebuildSeries() {
        const chartItem = chartLoader.item
        if (!chartItem || !chartItem.equitySeries || root.chartsDisabled || !root.chartReady)
            return
        chartItem.equitySeries.clear()
        const data = points || []
        if (!data || data.length === 0) {
            chartItem.axisIndex.visible = true
            chartItem.axisTime.visible = false
            chartItem.axisIndex.min = 0
            chartItem.axisIndex.max = 1
            chartItem.axisY.min = 0
            chartItem.axisY.max = 1
            return
        }

        let minY = Number.POSITIVE_INFINITY
        let maxY = Number.NEGATIVE_INFINITY
        let minX = Number.POSITIVE_INFINITY
        let maxX = Number.NEGATIVE_INFINITY
        let hasTimestamp = false

        for (let i = 0; i < data.length; ++i) {
            const entry = data[i] || {}
            const rawValue = entry.value !== undefined ? entry.value : entry.y
            const value = Number(rawValue)
            if (isNaN(value))
                continue
            let timeValue = Number.NaN
            const rawTimestamp = entry.timestamp
            if (rawTimestamp !== undefined && rawTimestamp !== null) {
                const parsed = new Date(rawTimestamp)
                if (!isNaN(parsed.getTime())) {
                    timeValue = parsed.getTime()
                    hasTimestamp = true
                } else if (!isNaN(Number(rawTimestamp))) {
                    timeValue = Number(rawTimestamp)
                }
            }
            if (isNaN(timeValue))
                timeValue = i
            chartItem.equitySeries.append(timeValue, value)
            if (value < minY)
                minY = value
            if (value > maxY)
                maxY = value
            if (timeValue < minX)
                minX = timeValue
            if (timeValue > maxX)
                maxX = timeValue
        }

        if (chartItem.equitySeries.count === 0) {
            chartItem.axisIndex.visible = true
            chartItem.axisTime.visible = false
            chartItem.axisIndex.min = 0
            chartItem.axisIndex.max = 1
            chartItem.axisY.min = 0
            chartItem.axisY.max = 1
            return
        }

        if (hasTimestamp) {
            chartItem.equitySeries.axisX = chartItem.axisTime
            chartItem.axisTime.visible = true
            chartItem.axisIndex.visible = false
            chartItem.axisTime.min = new Date(minX)
            chartItem.axisTime.max = new Date(maxX)
        } else {
            chartItem.equitySeries.axisX = chartItem.axisIndex
            chartItem.axisIndex.visible = true
            chartItem.axisTime.visible = false
            chartItem.axisIndex.min = minX
            chartItem.axisIndex.max = maxX > minX ? maxX : minX + 1
        }

        chartItem.axisY.min = minY
        chartItem.axisY.max = maxY > minY ? maxY : minY + 1
    }

    onPointsChanged: rebuildSeries()
    Component.onCompleted: rebuildSeries()

    onChartsDisabledChanged: rebuildSeries()
}
