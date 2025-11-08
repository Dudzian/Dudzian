import QtQuick
import QtCharts
import QtQuick.Layouts
import ".." as Design

ChartView {
    id: chart
    property var data: []
    property color lineColor: Design.Palette.accent
    antialiasing: true
    legend.visible: false
    backgroundColor: Design.Palette.surface
    dropShadowEnabled: false
    theme: ChartView.ChartThemeDark
    animationOptions: ChartView.NoAnimation

    ValueAxis {
        id: xAxis
        min: 0
        max: Math.max(1, chart.data.length - 1)
        labelsVisible: false
    }

    ValueAxis {
        id: yAxis
        min: chart.minimumValue
        max: chart.maximumValue
        labelsColor: Design.Palette.textSecondary
        gridLineColor: Design.Palette.border
        lineVisible: true
    }

    property real minimumValue: {
        var minValue = 0
        for (var i = 0; i < chart.data.length; ++i)
            minValue = Math.min(minValue, Number(chart.data[i].value || chart.data[i]))
        return minValue
    }

    property real maximumValue: {
        var maxValue = 0
        for (var i = 0; i < chart.data.length; ++i)
            maxValue = Math.max(maxValue, Number(chart.data[i].value || chart.data[i]))
        return maxValue
    }

    LineSeries {
        axisX: xAxis
        axisY: yAxis
        color: chart.lineColor
        width: 2

        XYPoint {
            id: _dummy
        }

        Component.onCompleted: rebuild()
        onCountChanged: rebuild()

        function rebuild() {
            clear()
            for (var i = 0; i < chart.data.length; ++i) {
                var value = chart.data[i]
                if (value === undefined || value === null)
                    continue
                if (value.value !== undefined)
                    append(i, Number(value.value))
                else if (value.portfolio !== undefined)
                    append(i, Number(value.portfolio))
                else
                    append(i, Number(value))
            }
        }

        Connections {
            target: chart
            function onDataChanged() {
                rebuild()
            }
        }
    }
}
