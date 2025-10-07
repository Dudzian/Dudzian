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
            Label { text: performanceGuard.fpsTarget.toString() }

            Label { text: qsTr("Reduce motion after") }
            Label { text: qsTr("%1 s").arg(performanceGuard.reduceMotionAfterSeconds.toFixed(2)) }

            Label { text: qsTr("Jank budget") }
            Label { text: qsTr("%1 ms").arg(performanceGuard.jankThresholdMs.toFixed(1)) }

            Label { text: qsTr("Overlay limit") }
            Label { text: performanceGuard.maxOverlayCount.toString() }

            Label { text: qsTr("Disable overlays <FPS") }
            Label {
                text: performanceGuard.disableSecondaryWhenFpsBelow > 0
                    ? performanceGuard.disableSecondaryWhenFpsBelow.toString()
                    : qsTr("—")
            }
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
            text: qsTr("Instrument: %1").arg(instrumentLabel)
        }

        Label {
            text: qsTr("Connection status: %1").arg(appController.connectionStatus)
        }

        Rectangle {
            height: 1
            color: Qt.darker(root.palette.window, 1.4)
            Layout.fillWidth: true
        }

        Label {
            text: qsTr("Profil ryzyka")
            font.pixelSize: 18
            font.bold: true
        }

        Label {
            text: riskModel.hasData ? qsTr("Profil: %1").arg(riskModel.profileLabel) : qsTr("Profil: —")
        }

        Label {
            text: riskModel.hasData
                ? qsTr("Wartość portfela: %1").arg(Number(riskModel.portfolioValue).toLocaleString(Qt.locale(), 'f', 0))
                : qsTr("Wartość portfela: —")
        }

        Label {
            text: riskModel.hasData
                ? qsTr("Drawdown: %1 %").arg((riskModel.currentDrawdown * 100).toFixed(2))
                : qsTr("Drawdown: —")
        }

        Label {
            text: riskModel.hasData
                ? qsTr("Dźwignia: %1x").arg(riskModel.usedLeverage.toFixed(2))
                : qsTr("Dźwignia: —")
        }

        Label {
            text: riskModel.hasData
                ? qsTr("Aktualizacja: %1").arg(riskModel.generatedAt.toString(Qt.ISODate))
                : qsTr("Aktualizacja: —")
        }

        ListView {
            Layout.fillWidth: true
            visible: riskModel.hasData && riskModel.count > 0
            implicitHeight: contentHeight
            model: riskModel
            delegate: Rectangle {
                width: parent.width
                height: 36
                color: model.breached ? Qt.rgba(0.6, 0.0, 0.0, 0.3) : Qt.rgba(0.0, 0.0, 0.0, 0.0)
                radius: 4
                border.color: Qt.rgba(1.0, 1.0, 1.0, 0.1)

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8

                    Label {
                        text: model.code
                        Layout.fillWidth: true
                    }

                    Label {
                        text: qsTr("%1 / %2").arg(Number(model.currentValue).toLocaleString(Qt.locale(), 'f', 0)).arg(Number(model.maxValue).toLocaleString(Qt.locale(), 'f', 0))
                    }
                }
            }
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
