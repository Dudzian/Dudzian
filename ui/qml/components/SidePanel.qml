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
            objectName: "performanceGuardTitleLabel"
            text: qsTr("Performance guard")
            font.pixelSize: 18
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 8

            Label {
                objectName: "fpsTargetLabel"
                text: qsTr("FPS target")
            }
            Label {
                objectName: "fpsTargetValueLabel"
                text: performanceGuard ? performanceGuard.fpsTarget.toString() : qsTr("—")
            }

            Label {
                objectName: "reduceMotionLabel"
                text: qsTr("Reduce motion after")
            }
            Label {
                objectName: "reduceMotionValueLabel"
                text: performanceGuard
                      ? qsTr("%1 s").arg(Number(performanceGuard.reduceMotionAfterSeconds).toFixed(2))
                      : qsTr("—")
            }

            Label {
                objectName: "jankBudgetLabel"
                text: qsTr("Jank budget")
            }
            Label {
                objectName: "jankBudgetValueLabel"
                text: performanceGuard
                      ? qsTr("%1 ms").arg(Number(performanceGuard.jankThresholdMs).toFixed(1))
                      : qsTr("—")
            }

            Label {
                objectName: "overlayLimitLabel"
                text: qsTr("Overlay limit")
            }
            Label {
                objectName: "overlayLimitValueLabel"
                text: performanceGuard ? performanceGuard.maxOverlayCount.toString() : qsTr("—")
            }

            Label {
                objectName: "disableSecondaryLabel"
                text: qsTr("Disable overlays <FPS")
            }
            Label {
                objectName: "disableSecondaryValueLabel"
                text: performanceGuard && performanceGuard.disableSecondaryWhenFpsBelow > 0
                      ? performanceGuard.disableSecondaryWhenFpsBelow.toString()
                      : qsTr("—")
            }
        }

        Label {
            objectName: "latestCloseLabel"
            text: qsTr("Latest close: %1")
                  .arg(ohlcvModel && ohlcvModel.latestClose() !== undefined
                           ? Number(ohlcvModel.latestClose()).toFixed(2)
                           : qsTr("--"))
            font.pixelSize: 16
        }

        Rectangle { height: 1; color: Qt.darker(root.palette.window, 1.4); Layout.fillWidth: true }

        // --- Connection / instrument -----------------------------------------
        Label {
            objectName: "instrumentLabel"
            text: qsTr("Instrument: %1").arg(currentInstrumentLabel())
        }
        Label {
            objectName: "connectionStatusLabel"
            text: qsTr("Connection status: %1").arg(appController.connectionStatus)
        }

        // --- Risk profile (optional – shown only if riskModel is available) ---
        Rectangle { height: 1; color: Qt.darker(root.palette.window, 1.4); Layout.fillWidth: true }

        Label {
            objectName: "riskProfileHeaderLabel"
            visible: typeof riskModel !== "undefined"
            text: qsTr("Profil ryzyka")
            font.pixelSize: 18
            font.bold: true
        }

        Label {
            objectName: "riskProfileLabel"
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Profil: %1").arg(riskModel.profileLabel)
                    : qsTr("Profil: —")
        }

        Label {
            objectName: "riskPortfolioLabel"
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Wartość portfela: %1")
                        .arg(Number(riskModel.portfolioValue).toLocaleString(Qt.locale(), "f", 0))
                    : qsTr("Wartość portfela: —")
        }

        Label {
            objectName: "riskDrawdownLabel"
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Drawdown: %1 %").arg(Number(riskModel.currentDrawdown * 100).toFixed(2))
                    : qsTr("Drawdown: —")
        }

        Label {
            objectName: "riskLeverageLabel"
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Dźwignia: %1x").arg(Number(riskModel.usedLeverage).toFixed(2))
                    : qsTr("Dźwignia: —")
        }

        Label {
            objectName: "riskGeneratedAtLabel"
            visible: typeof riskModel !== "undefined"
            text: (typeof riskModel !== "undefined" && riskModel.hasData)
                    ? qsTr("Aktualizacja: %1").arg(riskModel.generatedAt.toString(Qt.ISODate))
                    : qsTr("Aktualizacja: —")
        }

        ListView {
            objectName: "riskExposureList"
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
