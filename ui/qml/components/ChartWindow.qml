import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.settings
import "."

ApplicationWindow {
    id: chartWindow
    property alias chartView: chartView
    property var ohlcvModel
    property var performanceGuard: ({})
    property string instrumentLabel: qsTr("Chart")

    width: windowSettings.width
    height: windowSettings.height
    visible: true
    title: qsTr("%1 – dodatkowe okno").arg(instrumentLabel)
    flags: Qt.Window | Qt.WindowTitleHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint

    signal windowClosed()

    background: Rectangle {
        color: Qt.darker(chartWindow.palette.window, 1.15)
    }

    Settings {
        id: windowSettings
        category: "bot_trading_shell/chart_window"
        property real xPos: 240
        property real yPos: 180
        property real width: 960
        property real height: 540
    }

    Component.onCompleted: {
        if (!isNaN(windowSettings.xPos)) chartWindow.x = windowSettings.xPos
        if (!isNaN(windowSettings.yPos)) chartWindow.y = windowSettings.yPos
    }

    onXChanged: windowSettings.xPos = chartWindow.x
    onYChanged: windowSettings.yPos = chartWindow.y
    onWidthChanged: windowSettings.width = chartWindow.width
    onHeightChanged: windowSettings.height = chartWindow.height

    onClosing: {
        windowClosed()
        Qt.callLater(function() { chartWindow.destroy() })
    }

    Shortcut {
        sequences: [ StandardKey.Close ]
        onActivated: chartWindow.close()
    }

    Pane {
        anchors.fill: parent
        padding: 12
        background: Rectangle {
            color: Qt.darker(chartWindow.palette.window, 1.25)
            radius: 8
            border.color: Qt.rgba(1, 1, 1, 0.06)
            border.width: 1
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 8

            Label {
                text: instrumentLabel
                font.pixelSize: 18
                font.bold: true
            }

            CandlestickChartView {
                id: chartView
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: chartWindow.ohlcvModel
                performanceGuard: chartWindow.performanceGuard
                reduceMotion: appController.reduceMotionActive
            }
        }
    }
}
