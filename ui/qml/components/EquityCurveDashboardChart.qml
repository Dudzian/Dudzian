import QtQuick
import QtQuick.Controls
import QtCharts

Item {
    id: chartRoot
    property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)
    property alias axisTime: axisTime
    property alias axisIndex: axisIndex
    property alias axisY: axisY
    property alias equitySeries: equitySeries

    ChartView {
        id: chart
        anchors.fill: parent
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
            color: chartRoot.accentColor
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
