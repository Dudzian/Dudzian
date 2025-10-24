import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Item {
    id: root

    property var priceModel: null
    property var indicatorModel: null
    property var signalModel: null
    property var regimeModel: null
    property PerformanceGuard performanceGuard
    property bool reduceMotion: false

    function latestRegimeSnapshot() {
        if (!regimeModel || regimeModel.count === undefined || regimeModel.count === 0)
            return null
        return regimeModel.get(regimeModel.count - 1)
    }

    function confidenceToPercent(value) {
        if (value === undefined || value === null)
            return 0
        var normalized = value / 1.5
        return Math.max(0, Math.min(1, normalized))
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        CandlestickChartView {
            id: priceView
            objectName: "priceView"
            Layout.fillWidth: true
            Layout.preferredHeight: parent.height > 0 ? parent.height * 0.55 : 360
            model: root.priceModel
            indicatorModel: root.indicatorModel
            performanceGuard: root.performanceGuard
            reduceMotion: root.reduceMotion
        }

        Frame {
            Layout.fillWidth: true
            background: Rectangle {
                color: Qt.darker(priceView.palette.window, 1.3)
                radius: 8
            }
            padding: 12

            ColumnLayout {
                anchors.fill: parent
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: qsTr("Reżim rynku")
                        font.bold: true
                        Layout.alignment: Qt.AlignVCenter
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        id: regimeLabel
                        text: {
                            const snapshot = latestRegimeSnapshot()
                            if (!snapshot)
                                return qsTr("brak danych")
                            return snapshot.regime || qsTr("nieznany")
                        }
                        font.pixelSize: 16
                        color: Qt.rgba(0.9, 0.92, 0.96, 1)
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Trend") }
                        ProgressBar {
                            value: confidenceToPercent(latestRegimeSnapshot() ? latestRegimeSnapshot().trendConfidence : 0)
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Mean reversion") }
                        ProgressBar {
                            value: confidenceToPercent(latestRegimeSnapshot() ? latestRegimeSnapshot().meanReversionConfidence : 0)
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Dzienny") }
                        ProgressBar {
                            value: confidenceToPercent(latestRegimeSnapshot() ? latestRegimeSnapshot().dailyConfidence : 0)
                        }
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Rectangle {
                color: Qt.darker(priceView.palette.window, 1.25)
                radius: 8
            }
            padding: 12

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: qsTr("Sygnały handlowe")
                        font.bold: true
                        font.pixelSize: 16
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: signalModel && signalModel.count !== undefined
                              ? qsTr("Widoczne: %1").arg(signalModel.count)
                              : qsTr("Brak danych")
                        color: Qt.rgba(0.78, 0.8, 0.86, 1)
                    }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 6
                    model: signalModel
                    objectName: "signalList"
                    delegate: Frame {
                        required property string code
                        required property string description
                        required property string timestampDisplay
                        required property real confidence
                        required property string regime

                        Layout.fillWidth: true
                        padding: 10
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0.18, 0.22, 0.28, 0.7)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4
                            Label {
                                text: description
                                font.bold: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: timestampDisplay
                                    color: Qt.rgba(0.75, 0.77, 0.82, 1)
                                }
                                Label {
                                    text: qsTr("Kod: %1").arg(code)
                                    color: Qt.rgba(0.75, 0.77, 0.82, 1)
                                }
                                Label {
                                    text: qsTr("Reżim: %1").arg(regime)
                                    color: Qt.rgba(0.75, 0.77, 0.82, 1)
                                }
                                Item { Layout.fillWidth: true }
                                Label {
                                    text: qsTr("Pewność: %1%2")
                                              .arg(Math.round(confidence * 100))
                                              .arg("%")
                                    color: Qt.rgba(0.92, 0.85, 0.45, 1)
                                }
                            }
                        }
                    }
                    PlaceholderMessage {
                        anchors.centerIn: parent
                        text: qsTr("Brak aktywnych sygnałów")
                        visible: !signalModel || signalModel.count === undefined || signalModel.count === 0
                    }
                }
            }
        }
    }
}
