import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCore

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property var monitor: (typeof decisionMonitorController !== "undefined" ? decisionMonitorController : null)
    property var decisionModel: (typeof decisionLogModel !== "undefined" ? decisionLogModel : null)
    property var outcomeSummary: monitor ? monitor.outcomeSummary : ({})
    property var flaggedDecisions: monitor ? monitor.flaggedDecisions : []
    property var recentDecisions: monitor ? monitor.recentDecisions : []
    property var scheduleSummary: monitor ? monitor.scheduleSummary : []
    property var lastUpdated: monitor ? monitor.lastUpdated : null
    readonly property var palette: Qt.application.palette

    function syncFromController() {
        if (!monitor)
            return
        outcomeSummary = monitor.outcomeSummary || ({})
        flaggedDecisions = monitor.flaggedDecisions || []
        recentDecisions = monitor.recentDecisions || []
        scheduleSummary = monitor.scheduleSummary || []
        lastUpdated = monitor.lastUpdated || null
    }

    Component.onCompleted: syncFromController()

    Connections {
        target: monitor
        ignoreUnknownSignals: true
        function onOutcomeSummaryChanged() { syncFromController() }
        function onFlaggedDecisionsChanged() { syncFromController() }
        function onRecentDecisionsChanged() { syncFromController() }
        function onScheduleSummaryChanged() { syncFromController() }
        function onLastUpdatedChanged() { syncFromController() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        padding: 16

        RowLayout {
            Layout.fillWidth: true
            Label {
                text: qsTr("Monitoring decyzji tradingowych")
                font.bold: true
                font.pointSize: 16
            }
            Item { Layout.fillWidth: true }
            SpinBox {
                id: limitSpin
                from: 10
                to: 250
                value: monitor ? monitor.recentLimit : 50
                enabled: monitor !== null
                onValueModified: {
                    if (monitor)
                        monitor.recentLimit = value
                }
            }
            Button {
                text: qsTr("Odśwież")
                enabled: monitor !== null
                onClicked: monitor ? monitor.refresh() : null
            }
        }

        Flow {
            Layout.fillWidth: true
            Layout.preferredHeight: implicitHeight
            spacing: 12
            Repeater {
                model: [
                    { key: "total", label: qsTr("Wpisy"), highlight: true },
                    { key: "executed", label: qsTr("Wykonane"), highlight: false },
                    { key: "rejected", label: qsTr("Odrzucone"), highlight: false },
                    { key: "approved", label: qsTr("Zatwierdzone"), highlight: false },
                    { key: "declined", label: qsTr("Anulowane"), highlight: false },
                    { key: "manual", label: qsTr("Manualne"), highlight: false },
                    { key: "automated", label: qsTr("Automatyczne"), highlight: false }
                ]
                delegate: Frame {
                    width: 140
                    height: 72
                    radius: 8
                    padding: 8
                    background: Rectangle {
                        radius: 8
                        color: modelData.highlight ? Qt.darker(palette.highlight, 0.8) : Qt.darker(palette.window, 1.1)
                        border.color: Qt.darker(color, 1.2)
                    }
                    Column {
                        anchors.fill: parent
                        spacing: 4
                        Label {
                            text: modelData.label
                            color: palette.brightText
                            font.bold: true
                        }
                        Label {
                            text: outcomeSummary[modelData.key] !== undefined ? outcomeSummary[modelData.key] : "—"
                            color: palette.text
                            font.pointSize: 14
                        }
                    }
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 16
            rowSpacing: 16
            Layout.preferredHeight: 220

            GroupBox {
                title: qsTr("Ostatnie decyzje")
                Layout.fillWidth: true
                Layout.fillHeight: true

                ListView {
                    anchors.fill: parent
                    model: recentDecisions
                    clip: true
                    delegate: Column {
                        width: ListView.view.width
                        spacing: 2
                        Label {
                            text: (modelData.timestampDisplay || modelData.timestamp || "") + " · " + (modelData.strategy || "—")
                            font.bold: true
                        }
                        Label {
                            text: qsTr("Tryb: %1, decyzja: %2 (%3)")
                                      .arg(modelData.decisionMode || "—")
                                      .arg(modelData.decisionState || "—")
                                      .arg(modelData.approved ? qsTr("zatwierdzona") : qsTr("odrzucona"))
                            color: palette.mid
                        }
                        Label {
                            text: (modelData.symbol || "") + " " + (modelData.side || "") + " " + (modelData.quantity || "")
                            color: palette.text
                        }
                        Rectangle {
                            width: parent.width
                            height: 1
                            color: Qt.darker(palette.window, 1.4)
                            visible: index < ListView.view.count - 1
                        }
                    }
                }
            }

            GroupBox {
                title: qsTr("Alerty / decyzje wymagające uwagi")
                Layout.fillWidth: true
                Layout.fillHeight: true

                ListView {
                    anchors.fill: parent
                    model: flaggedDecisions
                    clip: true
                    delegate: Column {
                        width: ListView.view.width
                        spacing: 2
                        Label {
                            text: (modelData.timestampDisplay || modelData.timestamp || "") + " – " + (modelData.strategy || "—")
                            color: palette.brightText
                        }
                        Label {
                            text: modelData.decisionReason && modelData.decisionReason.length > 0 ? modelData.decisionReason : qsTr("Brak uzasadnienia")
                            color: "#d95468"
                            wrapMode: Text.Wrap
                        }
                        Rectangle {
                            width: parent.width
                            height: 1
                            color: Qt.darker(palette.window, 1.4)
                            visible: index < ListView.view.count - 1
                        }
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Skuteczność per harmonogram")
            Layout.fillWidth: true
            Layout.fillHeight: true

            TableView {
                anchors.fill: parent
                columnSpacing: 8
                rowSpacing: 4
                clip: true
                model: scheduleSummary

                TableViewColumn { role: "schedule"; title: qsTr("Harmonogram"); width: 200 }
                TableViewColumn { role: "total"; title: qsTr("Wpisy"); width: 80 }
                TableViewColumn { role: "executed"; title: qsTr("Wykonane"); width: 90 }
                TableViewColumn { role: "rejected"; title: qsTr("Odrzucone"); width: 90 }
                TableViewColumn {
                    title: qsTr("Ostatnia decyzja")
                    width: 280
                    delegate: Label {
                        text: modelData.lastDecision && modelData.lastDecision.timestampDisplay ?
                                  modelData.lastDecision.timestampDisplay + " · " + (modelData.lastDecision.decisionState || "") :
                                  qsTr("—")
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Label {
                text: lastUpdated && lastUpdated.isValid ?
                          qsTr("Ostatnia aktualizacja: %1").arg(Qt.formatDateTime(lastUpdated, Qt.ISODateWithMs)) :
                          qsTr("Brak danych")
            }
            Item { Layout.fillWidth: true }
            Label {
                visible: decisionModel && decisionModel.logPath
                text: decisionModel && decisionModel.logPath ? qsTr("Ścieżka logu: %1").arg(decisionModel.logPath) : ""
                font.italic: true
            }
        }
    }
}
