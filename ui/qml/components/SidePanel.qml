import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Pane {
    id: root
    property PerformanceGuard performanceGuard
    property string instrumentLabel: appController.instrumentLabel
    signal openWindowRequested()

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
            Label { text: performanceGuard ? performanceGuard.fpsTarget.toString() : qsTr("—") }

            Label { text: qsTr("Reduce motion after") }
            Label {
                text: performanceGuard
                      ? qsTr("%1 s").arg(performanceGuard.reduceMotionAfterSeconds.toFixed(2))
                      : qsTr("—")
            }

            Label { text: qsTr("Jank budget") }
            Label {
                text: performanceGuard
                      ? qsTr("%1 ms").arg(performanceGuard.jankThresholdMs.toFixed(1))
                      : qsTr("—")
            }

            Label { text: qsTr("Overlay limit") }
            Label { text: performanceGuard ? performanceGuard.maxOverlayCount.toString() : qsTr("—") }

            Label { text: qsTr("Disable overlays <FPS") }
            Label {
                text: performanceGuard && performanceGuard.disableSecondaryWhenFpsBelow > 0
                      ? performanceGuard.disableSecondaryWhenFpsBelow.toString()
                      : qsTr("—")
            }
        }

        Label {
            text: qsTr("Latest close: %1")
                    .arg(ohlcvModel && ohlcvModel.latestClose() !== undefined
                             ? Number(ohlcvModel.latestClose()).toFixed(2)
                             : qsTr("--"))
            font.pixelSize: 16
        }

        Rectangle {
            height: 1
            color: Qt.darker(root.palette.window, 1.4)
            Layout.fillWidth: true
        }

        Label {
            text: qsTr("Instrument: %1").arg(instrumentLabel)
        }

        Label {
            text: qsTr("Connection status: %1").arg(appController.connectionStatus)
        }

        Button {
            text: qsTr("Otwórz nowe okno")
            Layout.fillWidth: true
            onClicked: root.openWindowRequested()
        }

        Item { Layout.fillHeight: true }
    }

    function currentInstrumentLabel() {
        return instrumentLabel && instrumentLabel.length > 0 ? instrumentLabel : qsTr("Wykres")
    }
}
