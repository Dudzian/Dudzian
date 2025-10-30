import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var riskModel: appController ? appController.riskModel : (typeof riskModel !== "undefined" ? riskModel : null)
    property var riskHistoryModel: appController ? appController.riskHistoryModel : (typeof riskHistoryModel !== "undefined" ? riskHistoryModel : null)
    property var alertsModel: (typeof alertsModel !== "undefined" ? alertsModel : null)

    property var exchangeExposureItems: []
    property var strategyExposureItems: []
    property var historyPoints: []

    function rebuildHistoryPoints() {
        if (!riskHistoryModel || riskHistoryModel.count === undefined)
            return
        var points = []
        for (var i = 0; i < riskHistoryModel.count; ++i) {
            var entry = riskHistoryModel.get(i)
            if (!entry)
                continue
            points.push({ timestamp: entry.timestamp, value: entry.portfolioValue })
        }
        historyPoints = points
    }

    function rebuildExposureTables() {
        if (!riskModel || riskModel.count === undefined)
            return
        var exchanges = []
        var strategies = []
        for (var i = 0; i < riskModel.count; ++i) {
            var row = riskModel.get(i)
            if (!row || !row.code)
                continue
            var code = row.code
            if (code.indexOf("exchange:") === 0) {
                exchanges.push({
                    name: code.split(":")[1],
                    current: row.currentValue,
                    maximum: row.maxValue,
                    threshold: row.thresholdValue
                })
            } else if (code.indexOf("strategy:") === 0) {
                strategies.push({
                    name: code.split(":")[1],
                    current: row.currentValue,
                    maximum: row.maxValue,
                    threshold: row.thresholdValue
                })
            }
        }
        exchangeExposureItems = exchanges
        strategyExposureItems = strategies
    }

    Component.onCompleted: {
        rebuildHistoryPoints()
        rebuildExposureTables()
    }

    Connections {
        target: riskHistoryModel
        ignoreUnknownSignals: true
        function onHistoryChanged() { rebuildHistoryPoints() }
        function onSnapshotRecorded() { rebuildHistoryPoints() }
    }

    Connections {
        target: riskModel
        ignoreUnknownSignals: true
        function onRiskStateChanged() { rebuildExposureTables() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 18
        padding: 18

        Label {
            text: qsTr("Dashboard portfela")
            font.pixelSize: 24
            font.bold: true
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 320
            spacing: 18

            Components.EquityCurveDashboard {
                id: equityView
                Layout.fillWidth: true
                Layout.fillHeight: true
                points: historyPoints
                title: qsTr("Historia wartości portfela")
                accentColor: palette.highlight
            }

            Frame {
                Layout.preferredWidth: parent.width * 0.35
                Layout.fillHeight: true
                background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 10 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10
                    padding: 14

                    Label {
                        text: qsTr("Ekspozycja per giełda")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        objectName: "exchangeExposureList"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: exchangeExposureItems
                        delegate: Frame {
                            width: ListView.view.width
                            padding: 8
                            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.02) }
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 4
                                Label { text: model.name ? model.name.toUpperCase() : qsTr("Giełda") ; font.bold: true }
                                Label {
                                    Layout.fillWidth: true
                                    color: palette.mid
                                    text: qsTr("Ekspozycja: %1").arg(Number(model.current || 0).toLocaleString(Qt.locale(), "f", 2))
                                }
                                Label {
                                    Layout.fillWidth: true
                                    color: palette.mid
                                    text: model.threshold ? qsTr("Próg ostrzegawczy: %1").arg(Number(model.threshold).toLocaleString(Qt.locale(), "f", 2)) : qsTr("Brak progu")
                                }
                            }
                        }
                        placeholderText: qsTr("Brak danych ekspozycji")
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 18

            Frame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 10 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10
                    padding: 14

                    Label {
                        text: qsTr("Ekspozycja per strategia")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        objectName: "strategyExposureList"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: strategyExposureItems
                        delegate: Frame {
                            width: ListView.view.width
                            padding: 8
                            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.02) }
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 4
                                Label { text: model.name ? model.name : qsTr("Strategia"); font.bold: true }
                                Label {
                                    Layout.fillWidth: true
                                    color: palette.mid
                                    text: qsTr("Notional: %1").arg(Number(model.current || 0).toLocaleString(Qt.locale(), "f", 2))
                                }
                                Label {
                                    Layout.fillWidth: true
                                    color: palette.mid
                                    text: model.maximum ? qsTr("Limit: %1").arg(Number(model.maximum).toLocaleString(Qt.locale(), "f", 2)) : qsTr("Limit dynamiczny")
                                }
                            }
                        }
                        placeholderText: qsTr("Brak danych o strategiach")
                    }
                }
            }

            Frame {
                Layout.preferredWidth: parent.width * 0.35
                Layout.fillHeight: true
                background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 10 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10
                    padding: 14

                    Label {
                        text: qsTr("Aktywne alerty")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        objectName: "alertsListView"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: alertsModel ? alertsModel : []
                        delegate: Frame {
                            width: ListView.view.width
                            padding: 10
                            background: Rectangle {
                                radius: 8
                                color: model.severity === 2 ? Qt.rgba(0.58, 0.16, 0.2, 0.45)
                                                           : (model.severity === 1 ? Qt.rgba(0.96, 0.67, 0.17, 0.35)
                                                                                : Qt.darker(palette.base, 1.02))
                            }
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 4
                                Label { text: model.title || qsTr("Alert"); font.bold: true }
                                Label { Layout.fillWidth: true; wrapMode: Text.WordWrap; text: model.description || "" }
                                Label { text: Qt.formatDateTime(model.timestamp || new Date(), Qt.DefaultLocaleShortDate); color: palette.mid; font.pixelSize: 12 }
                            }
                        }
                        placeholderText: qsTr("Brak alertów")
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Button {
                            text: qsTr("Potwierdź wszystkie")
                            enabled: alertsModel && alertsModel.unacknowledgedCount > 0
                            onClicked: { if (alertsModel) alertsModel.acknowledgeAll() }
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            color: palette.mid
                            text: alertsModel ? qsTr("Łącznie: %1").arg(alertsModel.count || alertsModel.rowCount()) : ""
                        }
                    }
                }
            }
        }
    }
}
