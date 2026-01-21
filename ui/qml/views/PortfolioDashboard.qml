import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    // Nie shadowuj context property "appController" – inaczej robisz self-binding i QML potrafi nie wstać.
    property var controllerOverride: null

    readonly property var resolvedController: (
        root.controllerOverride
        ? root.controllerOverride
        : (typeof appController !== "undefined" && appController ? appController : null)
    )
    property var riskModelOverride: null
    property var riskHistoryModelOverride: null
    property var alertsModelOverride: null

    readonly property var resolvedRiskModel: (
        root.riskModelOverride
        ? root.riskModelOverride
        : (resolvedController && resolvedController.riskModel ? resolvedController.riskModel
           : (typeof riskModel !== "undefined" && riskModel ? riskModel : null))
    )
    readonly property var resolvedRiskHistoryModel: (
        root.riskHistoryModelOverride
        ? root.riskHistoryModelOverride
        : (resolvedController && resolvedController.riskHistoryModel ? resolvedController.riskHistoryModel
           : (typeof riskHistoryModel !== "undefined" && riskHistoryModel ? riskHistoryModel : null))
    )
    readonly property var resolvedAlertsModel: (
        root.alertsModelOverride
        ? root.alertsModelOverride
        : (resolvedController && resolvedController.alertsModel ? resolvedController.alertsModel
           : (typeof alertsModel !== "undefined" && alertsModel ? alertsModel : null))
    )

    property var exchangeExposureItems: []
    property var strategyExposureItems: []
    property var historyPoints: []
    property bool alertsLayoutRefreshScheduled: false

    Timer {
        id: alertsLayoutRefreshTimer
        interval: 0
        repeat: false
        running: false
        onTriggered: {
            alertsLayoutRefreshScheduled = false
            if (!root || !alertsListView)
                return
            refreshAlertsLayout()
        }
    }

    function rebuildHistoryPoints() {
        if (!resolvedRiskHistoryModel || resolvedRiskHistoryModel.count === undefined)
            return
        var points = []
        for (var i = 0; i < resolvedRiskHistoryModel.count; ++i) {
            var entry = resolvedRiskHistoryModel.get(i)
            if (!entry)
                continue
            points.push({ timestamp: entry.timestamp, value: entry.portfolioValue })
        }
        historyPoints = points
    }

    function rebuildExposureTables() {
        if (!resolvedRiskModel || resolvedRiskModel.count === undefined)
            return
        var exchanges = []
        var strategies = []
        for (var i = 0; i < resolvedRiskModel.count; ++i) {
            var row = resolvedRiskModel.get(i)
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
        target: resolvedRiskHistoryModel
        ignoreUnknownSignals: true
        function onHistoryChanged() { rebuildHistoryPoints() }
        function onSnapshotRecorded() { rebuildHistoryPoints() }
    }

    Connections {
        target: resolvedRiskModel
        ignoreUnknownSignals: true
        function onRiskStateChanged() { rebuildExposureTables() }
    }

    function refreshAlertsLayout() {
        if (!alertsListView)
            return
        if (alertsListView.forceLayout)
            alertsListView.forceLayout()
        else if (alertsListView.requestLayout)
            alertsListView.requestLayout()
    }

    function scheduleAlertsLayoutRefresh() {
        if (alertsLayoutRefreshScheduled)
            return
        alertsLayoutRefreshScheduled = true
        alertsLayoutRefreshTimer.start()
    }

    onResolvedAlertsModelChanged: scheduleAlertsLayoutRefresh()

    Connections {
        target: resolvedAlertsModel ? resolvedAlertsModel : null
        ignoreUnknownSignals: true
        function onAlertsChanged() { scheduleAlertsLayoutRefresh() }
        function onCountChanged() { scheduleAlertsLayoutRefresh() }
        function onModelReset() { scheduleAlertsLayoutRefresh() }
        function onRowsInserted() { scheduleAlertsLayoutRefresh() }
        function onRowsRemoved() { scheduleAlertsLayoutRefresh() }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 18

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
                objectName: "equityCurveDashboard"
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
                    anchors.margins: 14
                    spacing: 10

                    Label {
                        text: qsTr("Ekspozycja per giełda")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        id: exchangeExposureList
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
                        Label {
                            anchors.centerIn: parent
                            color: palette.mid
                            text: qsTr("Brak danych ekspozycji")
                            visible: exchangeExposureList.count === 0
                        }
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
                    anchors.margins: 14
                    spacing: 10

                    Label {
                        text: qsTr("Ekspozycja per strategia")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        id: strategyExposureList
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
                        Label {
                            anchors.centerIn: parent
                            color: palette.mid
                            text: qsTr("Brak danych o strategiach")
                            visible: strategyExposureList.count === 0
                        }
                    }
                }
            }

            Frame {
                Layout.preferredWidth: parent.width * 0.35
                Layout.fillHeight: true
                background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 10 }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    Label {
                        text: qsTr("Aktywne alerty")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    ListView {
                        id: alertsListView
                        objectName: "alertsListView"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: resolvedAlertsModel ? resolvedAlertsModel : []
                        delegate: Frame {
                            width: ListView.view.width
                            padding: 10
                            background: Rectangle {
                                radius: 8
                                color: severity === 2 ? Qt.rgba(0.58, 0.16, 0.2, 0.45)
                                                     : (severity === 1 ? Qt.rgba(0.96, 0.67, 0.17, 0.35)
                                                                                : Qt.darker(palette.base, 1.02))
                            }
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 4
                                Label { text: title || qsTr("Alert"); font.bold: true }
                                Label { Layout.fillWidth: true; wrapMode: Text.WordWrap; text: description || "" }
                                Label { text: Qt.formatDateTime(timestamp || new Date(), Qt.DefaultLocaleShortDate); color: palette.mid; font.pixelSize: 12 }
                            }
                        }
                        Label {
                            anchors.centerIn: parent
                            color: palette.mid
                            text: qsTr("Brak alertów")
                            visible: alertsListView.count === 0
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Button {
                            text: qsTr("Potwierdź wszystkie")
                            enabled: resolvedAlertsModel && (
                                (resolvedAlertsModel.unacknowledgedCount !== undefined && resolvedAlertsModel.unacknowledgedCount > 0)
                                || (resolvedAlertsModel.count !== undefined && resolvedAlertsModel.count > 0)
                            )
                            onClicked: { if (resolvedAlertsModel) resolvedAlertsModel.acknowledgeAll() }
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            color: palette.mid
                            text: resolvedAlertsModel ? qsTr("Łącznie: %1").arg(resolvedAlertsModel.count || resolvedAlertsModel.rowCount()) : ""
                        }
                    }
                }
            }
        }
    }
}
