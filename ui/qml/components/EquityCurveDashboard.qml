import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts

Pane {
    id: root
    property var points: []
    property string title: qsTr("Krzywa kapitału")
    property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)
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

        ChartView {
            id: chart
            Layout.fillWidth: true
            Layout.fillHeight: true
            antialiasing: true
            theme: ChartView.ChartThemeDark
            legend.visible: false
            dropShadowEnabled: false
            animationOptions: ChartView.NoAnimation

            DateTimeAxis {
                id: axisTime
                format: "yyyy-MM-dd"
                labelsAngle: -30
                visible: false
            }

            ValueAxis {
                id: axisIndex
                visible: true
                labelFormat: "%.0f"
                min: 0
                max: 1
            }

            ValueAxis {
                id: axisY
                labelFormat: "%.2f"
                gridLineColor: Qt.rgba(1, 1, 1, 0.08)
            }

            LineSeries {
                id: equitySeries
                axisX: axisIndex
                axisY: axisY
                color: root.accentColor
                width: 2
                useOpenGL: false
            }

            Label {
                anchors.centerIn: parent
                visible: equitySeries.count === 0
                text: qsTr("Brak danych do wyświetlenia")
                color: palette.mid
            }
        }
    }

    function latestValueText() {
        if (chart === null || equitySeries.count === undefined || equitySeries.count === 0)
            return qsTr("—")
        const point = equitySeries.at(equitySeries.count - 1)
        if (!point || point.y === undefined)
            return qsTr("—")
        return qsTr("%1").arg(Number(point.y).toLocaleString(Qt.locale(), "f", 2))
    }

    function rebuildSeries() {
        equitySeries.clear()
        const data = points || []
        if (!data || data.length === 0) {
            axisIndex.visible = true
            axisTime.visible = false
            axisIndex.min = 0
            axisIndex.max = 1
            axisY.min = 0
            axisY.max = 1
            return
        }

        let minY = Number.POSITIVE_INFINITY
        let maxY = Number.NEGATIVE_INFINITY
        let minX = Number.POSITIVE_INFINITY
        let maxX = Number.NEGATIVE_INFINITY
        let hasTimestamp = false

        for (let i = 0; i < data.length; ++i) {
            const entry = data[i] || {}
            const value = Number(entry.value)
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
            equitySeries.append(timeValue, value)
            if (value < minY)
                minY = value
            if (value > maxY)
                maxY = value
            if (timeValue < minX)
                minX = timeValue
            if (timeValue > maxX)
                maxX = timeValue
        }

        if (equitySeries.count === 0) {
            axisIndex.visible = true
            axisTime.visible = false
            axisIndex.min = 0
            axisIndex.max = 1
            axisY.min = 0
            axisY.max = 1
            return
        }

        if (hasTimestamp) {
            equitySeries.axisX = axisTime
            axisTime.visible = true
            axisIndex.visible = false
            axisTime.min = new Date(minX)
            axisTime.max = new Date(maxX)
        } else {
            equitySeries.axisX = axisIndex
            axisIndex.visible = true
            axisTime.visible = false
            axisIndex.min = minX
            axisIndex.max = maxX > minX ? maxX : minX + 1
        }

        axisY.min = minY
        axisY.max = maxY > minY ? maxY : minY + 1
    }

    onPointsChanged: rebuildSeries()
    Component.onCompleted: rebuildSeries()
}
