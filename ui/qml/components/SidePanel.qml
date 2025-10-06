import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Pane {
    id: root
    property PerformanceGuard performanceGuard
    padding: 16
    background: Rectangle {
        color: Qt.darker(root.palette.window, 1.2)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        Label {
            text: qsTr("Performance guard")
            font.pixelSize: 18
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 8

            Label { text: qsTr("FPS target") }
            Label { text: performanceGuard.fpsTarget.toString() }

            Label { text: qsTr("Reduce motion after") }
            Label { text: qsTr("%1 s").arg(performanceGuard.reduceMotionAfterSeconds.toFixed(2)) }

            Label { text: qsTr("Jank budget") }
            Label { text: qsTr("%1 ms").arg(performanceGuard.jankThresholdMs.toFixed(1)) }

            Label { text: qsTr("Overlay limit") }
            Label { text: performanceGuard.maxOverlayCount.toString() }
        }

        Label {
            text: qsTr("Latest close: %1").arg(ohlcvModel.latestClose() === undefined ? qsTr("--") : Number(ohlcvModel.latestClose()).toFixed(2))
            font.pixelSize: 16
        }

        Rectangle {
            height: 1
            color: Qt.darker(root.palette.window, 1.4)
            Layout.fillWidth: true
        }

        Label {
            text: qsTr("Connection status: %1").arg(appController.connectionStatus)
        }

        Item { Layout.fillHeight: true }
    }
}
