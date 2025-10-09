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

        // --- Performance guard ------------------------------------------------
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
                      ? qsTr("%1 s").arg(Number(performanceGuard.reduceMotionAfterSeconds).toFixed(2))
                      : qsTr("—")
            }

            Label { text: qsTr("Jank budget") }
            Label {
                text: performanceGuard
                      ? qsTr("%1 ms").arg(Number(performanceGuard.jankThresholdMs).toFixed(1))
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

        Rectangle { height: 1; color: Qt.darker(root.palette.window, 1.4); Layout.fillWidth: true }

        // --- Connection / instrument -----------------------------------------
        Label { text: qsTr("Instrument: %1").arg(currentInstrumentLabel()) }
        Label { text: qsTr("Connection status: %1").arg(appController.connectionStatus) }

        // --- Risk profile (optional – shown only if riskModel is available) ---
        Rectangle { height: 1; color: Qt.darker(root.palette.window, 1.4); Layout.fillWidth: true }

        Label {
            visible: typeof riskModel !== "undefined"
            text: qsTr("Profil ryzyka")
            font.pixelSize: 18
            font.bold: true
        }

        Label {
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Profil: %1").arg(riskModel.profileLabel)
                    : qsTr("Profil: —")
        }

        Label {
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Wartość portfela: %1")
                        .arg(Number(riskModel.portfolioValue).toLocaleString(Qt.locale(), "f", 0))
                    : qsTr("Wartość portfela: —")
        }

        Label {
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Drawdown: %1 %").arg(Number(riskModel.currentDrawdown * 100).toFixed(2))
                    : qsTr("Drawdown: —")
        }

        Label {
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Dźwignia: %1x").arg(Number(riskModel.usedLeverage).toFixed(2))
                    : qsTr("Dźwignia: —")
        }

        Label {
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Aktualizacja: %1").arg(riskModel.generatedAt.toString(Qt.ISODate))
                    : qsTr("Aktualizacja: —")
        }

        ListView {
            Layout.fillWidth: true
            visible: typeof riskModel !== "undefined" && riskModel.hasData && riskModel.count > 0
            implicitHeight: contentHeight
            model: (typeof riskModel !== "undefined") ? riskModel : null

            delegate: Rectangle {
                width: parent ? parent.width : 0
                height: 36
                color: model.breached ? Qt.rgba(0.6, 0.0, 0.0, 0.3) : "transparent"
                radius: 4
                border.color: Qt.rgba(1, 1, 1, 0.1)

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8

                    Label {
                        text: model.code
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }

                    Label {
                        text: qsTr("%1 / %2")
                              .arg(Number(model.currentValue).toLocaleString(Qt.locale(), "f", 0))
                              .arg(Number(model.maxValue).toLocaleString(Qt.locale(), "f", 0))
                    }
                }
            }
        }

        // --- Actions ----------------------------------------------------------
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
