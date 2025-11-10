import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCore

Item {
    id: root
    implicitWidth: 920
    implicitHeight: 600

    property var controller: (typeof strategyController !== "undefined" ? strategyController : null)
    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var riskPolicies: []
    property int selectedIndex: -1
    property string editorText: ""
    property string statusMessage: ""
    property color statusColor: highlightColor
    property var scheduleSnapshot: ({})
    readonly property color highlightColor: Qt.application.palette ? Qt.application.palette.highlight : "#3daee9"

    function syncPolicies() {
        if (!controller)
            return
        riskPolicies = controller.riskPolicies() || []
        if (riskPolicies.length === 0) {
            selectedIndex = -1
            editorText = ""
            return
        }
        if (selectedIndex < 0 || selectedIndex >= riskPolicies.length)
            selectedIndex = 0
        loadPolicy(selectedIndex)
    }

    function refreshSchedule() {
        if (!appController || !appController.riskRefreshSnapshot)
            return
        scheduleSnapshot = appController.riskRefreshSnapshot() || ({})
    }

    function loadPolicy(index) {
        if (!riskPolicies || index < 0 || index >= riskPolicies.length) {
            selectedIndex = -1
            editorText = ""
            return
        }
        selectedIndex = index
        var entry = riskPolicies[index] || ({})
        var pretty = ""
        try {
            pretty = JSON.stringify(entry, null, 2)
        } catch (err) {
            pretty = ""
        }
        editorText = pretty
        statusMessage = ""
    }

    function applyPolicy() {
        if (!controller || selectedIndex < 0 || selectedIndex >= riskPolicies.length)
            return
        var entry = riskPolicies[selectedIndex] || ({})
        var name = (entry.name || entry.id || "").toString().trim()
        if (!name) {
            statusMessage = qsTr("Brak identyfikatora polityki w zaznaczonym wpisie.")
            statusColor = "#d95468"
            return
        }
        var parsed = null
        try {
            parsed = JSON.parse(editorText)
        } catch (err) {
            statusMessage = qsTr("Niepoprawny JSON: %1").arg(err)
            statusColor = "#d95468"
            return
        }
        var ok = controller.saveRiskPolicy(name, parsed)
        if (ok) {
            statusMessage = qsTr("Zapisano politykę %1.").arg(name)
            statusColor = highlightColor
            controller.refresh()
        } else {
            statusMessage = controller.lastError || qsTr("Mostek konfiguracji zgłosił błąd zapisu.")
            statusColor = "#d95468"
        }
    }

    function removePolicy() {
        if (!controller || selectedIndex < 0 || selectedIndex >= riskPolicies.length)
            return
        var entry = riskPolicies[selectedIndex] || ({})
        var name = (entry.name || entry.id || "").toString().trim()
        if (!name)
            return
        if (!controller.removeRiskPolicy(name)) {
            statusMessage = controller.lastError || qsTr("Nie udało się usunąć polityki %1.").arg(name)
            statusColor = "#d95468"
            return
        }
        statusMessage = qsTr("Usunięto politykę %1.").arg(name)
        statusColor = highlightColor
        controller.refresh()
    }

    Component.onCompleted: {
        syncPolicies()
        refreshSchedule()
    }

    Connections {
        target: controller
        ignoreUnknownSignals: true
        function onRiskPoliciesChanged() { syncPolicies() }
        function onDecisionConfigChanged() { refreshSchedule() }
        function onSchedulerListChanged() { refreshSchedule() }
    }

    Connections {
        target: appController
        ignoreUnknownSignals: true
        function onRiskRefreshScheduleChanged() { refreshSchedule() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        padding: 16

        RowLayout {
            Layout.fillWidth: true
            Label {
                text: qsTr("Polityki zarządzania ryzykiem")
                font.bold: true
                font.pointSize: 16
            }
            Item { Layout.fillWidth: true }
            Button {
                text: qsTr("Odśwież")
                enabled: controller && !controller.busy
                onClicked: controller ? controller.refresh() : null
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Frame {
                Layout.fillHeight: true
                Layout.preferredWidth: parent.width * 0.35
                padding: 0

                ListView {
                    id: policyList
                    anchors.fill: parent
                    model: riskPolicies
                    clip: true
                    delegate: ItemDelegate {
                        width: ListView.view.width
                        text: (modelData && (modelData.name || modelData.id)) ? (modelData.name || modelData.id) : qsTr("Polityka %1").arg(index + 1)
                        onClicked: loadPolicy(index)
                        highlighted: index === selectedIndex
                    }
                }
            }

            Frame {
                Layout.fillHeight: true
                Layout.fillWidth: true
                padding: 12

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    TextArea {
                        id: editor
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        text: editorText
                        wrapMode: TextEdit.NoWrap
                        font.family: "Source Code Pro"
                        selectByMouse: true
                        onTextChanged: editorText = text
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Button {
                            text: qsTr("Zapisz zmiany")
                            enabled: controller && selectedIndex >= 0
                            onClicked: applyPolicy()
                        }

                        Button {
                            text: qsTr("Usuń politykę")
                            enabled: controller && selectedIndex >= 0
                            onClicked: removePolicy()
                        }

                        Item { Layout.fillWidth: true }
                    }

                    Label {
                        text: statusMessage
                        color: statusColor
                        visible: statusMessage.length > 0
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Harmonogram odświeżania ryzyka")
            Layout.fillWidth: true

            GridLayout {
                columns: 2
                columnSpacing: 12
                rowSpacing: 6
                Layout.fillWidth: true

                Label { text: qsTr("Aktywny") }
                Label { text: scheduleSnapshot.enabled === false ? qsTr("Nie") : qsTr("Tak") }

                Label { text: qsTr("Interwał (s)") }
                Label { text: Number(scheduleSnapshot.intervalSeconds || scheduleSnapshot.interval || 0).toFixed(1) }

                Label { text: qsTr("Następne odświeżenie") }
                Label {
                    text: scheduleSnapshot.nextRefreshUtc ? scheduleSnapshot.nextRefreshUtc.toLocaleString() : qsTr("—")
                }
            }
        }
    }
}
