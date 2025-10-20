import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property alias model: exposuresView.model
    property alias historyModel: historyView.model
    property var riskModel: root.model

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        Label {
            text: qsTr("Monitor ryzyka")
            font.pixelSize: 20
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 16
            rowSpacing: 12
            Layout.fillWidth: true

            Label { text: qsTr("Profil ryzyka") }
            Label {
                text: riskModel && riskModel.hasData ? riskModel.profileLabel : qsTr("—")
                font.bold: true
            }

            Label { text: qsTr("Wartość portfela") }
            Label {
                text: riskModel && riskModel.hasData
                      ? Qt.formatLocaleNumber(Qt.locale(), riskModel.portfolioValue, "f", 0)
                      : qsTr("—")
            }

            Label { text: qsTr("Bieżący drawdown") }
            ProgressBar {
                value: riskModel && riskModel.hasData ? Math.min(1, Math.max(0, riskModel.currentDrawdown / 0.1)) : 0
                Layout.preferredWidth: 220
                from: 0
                to: 1
                contentItem: Rectangle {
                    radius: 6
                    border.width: 0
                    color: Qt.lighter(root.palette.window, 1.12)
                }
                background: Rectangle {
                    radius: 6
                    border.color: Qt.rgba(0.3, 0.6, 0.9, 0.4)
                    color: Qt.rgba(0.2, 0.22, 0.26, 0.6)
                }
                ToolTip.visible: hovered
                ToolTip.text: riskModel && riskModel.hasData
                                ? qsTr("%1 %")
                                      .arg((riskModel.currentDrawdown * 100).toFixed(2))
                                : qsTr("brak danych")
            }

            Label { text: qsTr("Używana dźwignia") }
            ProgressBar {
                value: riskModel && riskModel.hasData ? Math.min(1, riskModel.usedLeverage / 10.0) : 0
                Layout.preferredWidth: 220
                from: 0
                to: 1
                ToolTip.visible: hovered
                ToolTip.text: riskModel && riskModel.hasData
                                ? qsTr("%1x").arg(riskModel.usedLeverage.toFixed(2))
                                : qsTr("brak danych")
            }

            Label { text: qsTr("Maks. dzienny loss") }
            Label {
                text: riskModel && riskModel.hasData
                      ? qsTr("%1 %").arg((riskModel.maxDailyLoss * 100).toFixed(2))
                      : qsTr("—")
            }

            Label { text: qsTr("Ostatnia aktualizacja") }
            Label {
                text: riskModel && riskModel.hasData && riskModel.generatedAt
                      ? riskModel.generatedAt.toString(Qt.ISODate)
                      : qsTr("—")
            }
        }

        Label {
            text: qsTr("Limity ekspozycji")
            font.bold: true
        }

        ListView {
            id: exposuresView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 6
            delegate: Frame {
                required property string code
                required property double currentValue
                required property double maxValue
                required property double thresholdValue
                required property bool breached

                Layout.fillWidth: true
                padding: 10
                background: Rectangle {
                    radius: 6
                    color: breached ? Qt.rgba(0.6, 0.07, 0.07, 0.5) : Qt.rgba(0.22, 0.25, 0.3, 0.45)
                    border.color: Qt.rgba(0.4, 0.7, 0.95, 0.3)
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Label {
                            text: code
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: breached ? qsTr("PRZEKROCZONY") : qsTr("Stabilny")
                            color: breached ? Qt.rgba(1, 0.86, 0.86, 1) : Qt.rgba(0.85, 0.92, 0.98, 0.8)
                        }
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        from: 0
                        to: thresholdValue > 0 ? thresholdValue : 1
                        value: Math.min(thresholdValue > 0 ? thresholdValue : 1, currentValue)
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0.16, 0.18, 0.22, 0.7)
                        }
                        contentItem: Rectangle {
                            radius: 6
                            color: breached ? Qt.rgba(0.9, 0.3, 0.3, 0.9) : Qt.rgba(0.26, 0.65, 0.85, 0.9)
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("%1 / %2 (próg %3)")
                              .arg(Qt.formatLocaleNumber(Qt.locale(), currentValue, "f", 0))
                              .arg(Qt.formatLocaleNumber(Qt.locale(), maxValue, "f", 0))
                              .arg(Qt.formatLocaleNumber(Qt.locale(), thresholdValue, "f", 0))
                        color: Qt.rgba(0.85, 0.88, 0.92, 1)
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            visible: historyModel && historyModel.hasSamples
            padding: 12
            background: Rectangle {
                radius: 8
                border.color: Qt.rgba(0.3, 0.55, 0.85, 0.35)
                color: Qt.rgba(0.18, 0.2, 0.26, 0.5)
            }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                Label {
                    text: qsTr("Podsumowanie historyczne")
                    font.bold: true
                }

                GridLayout {
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 6
                    Layout.fillWidth: true

                    Label { text: qsTr("Maksymalny drawdown") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1 %").arg((historyModel.maxDrawdown * 100).toFixed(2))
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Minimalny drawdown") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1 %").arg((historyModel.minDrawdown * 100).toFixed(2))
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Średni drawdown") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1 %").arg((historyModel.averageDrawdown * 100).toFixed(2))
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Maksymalna dźwignia") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1x").arg(historyModel.maxLeverage.toFixed(2))
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Średnia dźwignia") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1x").arg(historyModel.averageLeverage.toFixed(2))
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Minimum portfela") }
                    Label {
                        text: historyModel
                                  ? Qt.formatLocaleNumber(Qt.locale(), historyModel.minPortfolioValue, "f", 0)
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Maksimum portfela") }
                    Label {
                        text: historyModel
                                  ? Qt.formatLocaleNumber(Qt.locale(), historyModel.maxPortfolioValue, "f", 0)
                                  : qsTr("—")
                    }

                    Label { text: qsTr("Łączne naruszenia limitów") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1").arg(historyModel.totalBreachCount)
                                  : qsTr("0")
                        color: historyModel && historyModel.anyExposureBreached
                                   ? Qt.rgba(1, 0.75, 0.75, 1)
                                   : Qt.rgba(0.85, 0.92, 0.98, 0.8)
                        font.bold: historyModel && historyModel.anyExposureBreached
                    }

                    Label { text: qsTr("Szczyt wykorzystania limitu") }
                    Label {
                        text: historyModel
                                  ? qsTr("%1 %").arg((historyModel.maxExposureUtilization * 100).toFixed(1))
                                  : qsTr("—")
                        color: historyModel && historyModel.maxExposureUtilization > 1.0
                                   ? Qt.rgba(1, 0.75, 0.75, 1)
                                   : Qt.rgba(0.85, 0.92, 0.98, 0.8)
                        font.bold: historyModel && historyModel.maxExposureUtilization > 1.0
                    }
                }
            }
        }

        Label {
            text: qsTr("Historia próbek ryzyka")
            font.bold: true
            visible: historyView.count > 0
        }

        ListView {
            id: historyView
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(contentHeight + 12, 220)
            visible: count > 0
            clip: true
            spacing: 6
            ScrollBar.vertical: ScrollBar {}

            delegate: Frame {
                required property var timestamp
                required property double drawdown
                required property double leverage
                required property double portfolioValue
                required property string profileLabel
                required property bool hasBreach
                required property int breachCount
                required property double maxExposureUtilization
                required property var exposures

                Layout.fillWidth: true
                padding: 10
                background: Rectangle {
                    radius: 6
                    border.color: hasBreach ? Qt.rgba(0.9, 0.4, 0.4, 0.6) : Qt.rgba(0.35, 0.55, 0.8, 0.4)
                    color: hasBreach ? Qt.rgba(0.45, 0.12, 0.12, 0.55) : Qt.rgba(0.2, 0.22, 0.27, 0.45)
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: timestamp ? Qt.formatDateTime(timestamp, Qt.DefaultLocaleShortDate) : qsTr("—")
                            font.bold: true
                        }

                        Item { Layout.fillWidth: true }

                        Label {
                            text: profileLabel && profileLabel.length > 0 ? profileLabel : qsTr("profil nieznany")
                            color: Qt.rgba(0.8, 0.88, 0.98, 0.8)
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Label {
                            text: qsTr("Drawdown: %1 %").arg((drawdown * 100).toFixed(2))
                        }

                        Label {
                            text: qsTr("Dźwignia: %1x").arg(leverage.toFixed(2))
                        }

                        Label {
                            text: qsTr("Portfel: %1")
                                      .arg(Qt.formatLocaleNumber(Qt.locale(), portfolioValue, "f", 0))
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        visible: breachCount > 0 || maxExposureUtilization > 0

                        Label {
                            visible: breachCount > 0
                            text: qsTr("Naruszenia limitów: %1").arg(breachCount)
                            color: Qt.rgba(1, 0.8, 0.8, 1)
                            font.bold: true
                        }

                        Label {
                            visible: maxExposureUtilization > 0
                            text: qsTr("Maks. wykorzystanie: %1 %")
                                      .arg((maxExposureUtilization * 100).toFixed(1))
                            color: maxExposureUtilization > 1.0
                                       ? Qt.rgba(1, 0.75, 0.75, 1)
                                       : Qt.rgba(0.85, 0.92, 0.98, 0.8)
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: exposures && exposures.length > 0
                        spacing: 2

                        Repeater {
                            model: exposures ? exposures : []
                            delegate: RowLayout {
                                spacing: 8
                                Layout.fillWidth: true

                                Label {
                                    text: modelData.code
                                    font.bold: modelData.breached
                                    color: modelData.breached
                                               ? Qt.rgba(1, 0.75, 0.75, 1)
                                               : Qt.rgba(0.85, 0.92, 0.98, 0.9)
                                }

                                Label {
                                    text: qsTr("%1 / %2")
                                              .arg(Qt.formatLocaleNumber(Qt.locale(), modelData.currentValue, "f", 0))
                                              .arg(Qt.formatLocaleNumber(Qt.locale(), modelData.thresholdValue > 0
                                                                                       ? modelData.thresholdValue
                                                                                       : modelData.maxValue,
                                                                         "f",
                                                                         0))
                                    color: Qt.rgba(0.8, 0.85, 0.95, 0.8)
                                }

                                Item { Layout.fillWidth: true }

                                Label {
                                    text: qsTr("%1 %").arg((modelData.utilization * 100).toFixed(1))
                                    color: modelData.breached
                                               ? Qt.rgba(1, 0.75, 0.75, 1)
                                               : Qt.rgba(0.85, 0.92, 0.98, 0.8)
                                }
                            }
                        }
                    }
                }
            }
        }

        Label {
            visible: historyView.count === 0
            text: qsTr("Brak próbek historycznych — odśwież monitor ryzyka, aby je zebrać.")
            color: Qt.rgba(0.8, 0.82, 0.88, 0.7)
        }
    }
}
