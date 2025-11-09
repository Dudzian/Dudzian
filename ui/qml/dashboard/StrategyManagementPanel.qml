import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property var strategyController: null
    property var appController: null
    property bool busy: strategyController ? strategyController.busy : false
    property var schedulerItems: []
    property var decisionConfig: ({})
    property string lastError: ""

    padding: 16
    background: Rectangle {
        color: Qt.darker(palette.base, 1.05)
        radius: 10
    }

    function refreshSnapshots() {
        if (!strategyController) {
            schedulerItems = []
            decisionConfig = {}
            lastError = ""
            return
        }
        var config = {}
        try {
            config = strategyController.decisionConfigSnapshot() || {}
        } catch (err) {
            console.warn("StrategyManagementPanel: decisionConfigSnapshot failed", err)
        }
        decisionConfig = config || {}
        var list = []
        try {
            list = strategyController.schedulerList() || []
        } catch (err) {
            console.warn("StrategyManagementPanel: schedulerList failed", err)
        }
        schedulerItems = list
        lastError = strategyController.lastError || ""
        busy = strategyController.busy || false
    }

    onStrategyControllerChanged: refreshSnapshots()

    Component.onCompleted: refreshSnapshots()

    Connections {
        target: strategyController
        ignoreUnknownSignals: true

        function onSchedulerListChanged() { root.refreshSnapshots() }
        function onDecisionConfigChanged() { root.refreshSnapshots() }
        function onBusyChanged() { root.busy = strategyController ? strategyController.busy : false }
        function onLastErrorChanged() { root.lastError = strategyController ? strategyController.lastError : "" }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Zarządzanie strategiami")
                font.pixelSize: 18
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            BusyIndicator {
                running: root.busy
                visible: running
            }

            Button {
                text: qsTr("Odśwież")
                enabled: strategyController && !root.busy
                icon.name: "view-refresh"
                onClicked: {
                    if (strategyController && strategyController.refresh)
                        strategyController.refresh()
                }
            }
        }

        Label {
            Layout.fillWidth: true
            text: qsTr("Aktywna strategia: %1").arg(
                decisionConfig.default_strategy || decisionConfig.strategy || qsTr("(nie ustawiono)")
            )
            color: Qt.rgba(0.85, 0.86, 0.9, 1)
        }

        ListView {
            id: schedulerList
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: schedulerItems
            clip: true
            spacing: 8
            interactive: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

            delegate: Frame {
                required property var modelData
                width: schedulerList.width
                padding: 12
                background: Rectangle {
                    radius: 8
                    color: Qt.darker(palette.base, 1.02)
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            text: (modelData && modelData.name) ? modelData.name : qsTr("harmonogram")
                            font.bold: true
                        }

                        Rectangle {
                            width: 8
                            height: 8
                            radius: 4
                            color: (modelData && modelData.enabled === false) ? Qt.rgba(0.8, 0.3, 0.3, 1)
                                  : Qt.rgba(0.3, 0.7, 0.5, 1)
                        }

                        Item { Layout.fillWidth: true }

                        Label {
                            text: modelData && (modelData.next_run || modelData.nextRun)
                                  ? String(modelData.next_run || modelData.nextRun)
                                  : ""
                            color: palette.mid
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        color: palette.mid
                        text: modelData && (modelData.cron || modelData.expression)
                              ? String(modelData.cron || modelData.expression)
                              : qsTr("Brak zdefiniowanej reguły.")
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: modelData && (modelData.last_run || modelData.lastRun)
                                  ? qsTr("Ostatnie uruchomienie: %1").arg(String(modelData.last_run || modelData.lastRun))
                                  : ""
                            color: palette.mid
                        }

                        Item { Layout.fillWidth: true }

                        Button {
                            text: qsTr("Uruchom teraz")
                            enabled: strategyController && !root.busy
                            onClicked: {
                                if (strategyController && strategyController.runSchedulerNow)
                                    strategyController.runSchedulerNow(modelData && modelData.name ? modelData.name : "")
                            }
                        }

                        Button {
                            text: qsTr("Usuń")
                            enabled: strategyController && !root.busy
                            onClicked: {
                                if (strategyController && strategyController.removeSchedulerConfig)
                                    strategyController.removeSchedulerConfig(modelData && modelData.name ? modelData.name : "")
                            }
                        }
                    }
                }
            }

            footer: Item {
                width: schedulerList.width
                height: schedulerList.count > 0 ? 0 : 32
                Label {
                    anchors.centerIn: parent
                    text: qsTr("Brak skonfigurowanych harmonogramów")
                    color: palette.mid
                }
            }
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: lastError.length > 0
            text: lastError
            color: Qt.rgba(0.9, 0.55, 0.45, 1)
        }
    }
}
