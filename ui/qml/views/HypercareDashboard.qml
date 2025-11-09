import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 620

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var hypercareController: appController ? appController.hypercareController : (typeof hypercareController !== "undefined" ? hypercareController : null)
    property string summaryPath: hypercareController ? hypercareController.stage6SummaryPath : ""
    property var stage6Summary: ({})
    property var componentItems: []
    property var warningItems: []
    property var issueItems: []
    property string generatedAt: stage6Summary.generated_at ? stage6Summary.generated_at : ""
    property string overallStatus: stage6Summary.overall_status ? String(stage6Summary.overall_status) : "unknown"

    function statusColor(status) {
        var normalized = (status || "unknown").toLowerCase()
        if (normalized === "ok")
            return Qt.rgba(0.20, 0.63, 0.36, 1)
        if (normalized === "warn")
            return Qt.rgba(0.96, 0.68, 0.16, 1)
        if (normalized === "fail" || normalized === "critical")
            return Qt.rgba(0.82, 0.20, 0.27, 1)
        if (normalized === "skipped")
            return Qt.rgba(0.45, 0.45, 0.45, 1)
        return Qt.rgba(0.30, 0.36, 0.43, 1)
    }

    function refreshSummary() {
        if (!hypercareController)
            return
        var payload = hypercareController.loadSummary(summaryPath) || {}
        stage6Summary = payload
        generatedAt = payload.generated_at ? String(payload.generated_at) : ""
        overallStatus = payload.overall_status ? String(payload.overall_status) : "unknown"
        warningItems = Array.isArray(payload.warnings) ? payload.warnings : []
        issueItems = Array.isArray(payload.issues) ? payload.issues : []

        var items = []
        if (payload.components) {
            for (var name in payload.components) {
                if (!payload.components.hasOwnProperty(name))
                    continue
                var component = payload.components[name] || {}
                items.push({
                    name: name,
                    status: component.status ? String(component.status) : "unknown",
                    details: component
                })
            }
        }
        componentItems = items
    }

    onHypercareControllerChanged: {
        summaryPath = hypercareController ? hypercareController.stage6SummaryPath : ""
        refreshSummary()
    }

    Component.onCompleted: refreshSummary()

    Connections {
        target: hypercareController
        ignoreUnknownSignals: true
        function onStage6SummaryPathChanged() {
            if (!hypercareController)
                return
            summaryPath = hypercareController.stage6SummaryPath
            refreshSummary()
        }
        function onSelfHealingCompleted(exitCode) {
            refreshSummary()
        }
        function onRunningChanged() {
            if (hypercareController && !hypercareController.running)
                refreshSummary()
        }
    }

    ScrollView {
        anchors.fill: parent
        contentItem: ColumnLayout {
            spacing: 18
            padding: 18

            Label {
                text: qsTr("Panel hypercare Stage6")
                font.pixelSize: 26
                font.bold: true
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle { color: Qt.darker(palette.base, 1.04); radius: 10 }

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: qsTr("Status zbiorczy: %1").arg(overallStatus.toUpperCase())
                            font.pixelSize: 18
                            font.bold: true
                            color: statusColor(overallStatus)
                        }

                        Label {
                            text: generatedAt ? qsTr("Raport wygenerowany: %1").arg(generatedAt) : qsTr("Raport Stage6 nie został jeszcze utworzony")
                            color: palette.mid
                        }

                        TextField {
                            id: summaryPathField
                            Layout.fillWidth: true
                            text: summaryPath
                            placeholderText: qsTr("Ścieżka raportu Stage6")
                            selectByMouse: true
                            onEditingFinished: summaryPath = text
                        }

                        RowLayout {
                            spacing: 10

                            Button {
                                text: qsTr("Odśwież")
                                enabled: hypercareController && !hypercareController.running
                                onClicked: refreshSummary()
                            }

                            Button {
                                text: hypercareController && hypercareController.running ? qsTr("Self-healing w toku") : qsTr("Uruchom self-healing")
                                enabled: hypercareController && !hypercareController.running
                                onClicked: {
                                    if (!hypercareController)
                                        return
                                    hypercareController.triggerSelfHealing()
                                }
                            }

                            Button {
                                text: qsTr("Zapisz ścieżkę")
                                enabled: hypercareController !== null
                                onClicked: {
                                    if (!hypercareController)
                                        return
                                    hypercareController.stage6SummaryPath = summaryPath
                                    refreshSummary()
                                }
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.preferredWidth: 160
                        spacing: 8

                        BusyIndicator {
                            Layout.alignment: Qt.AlignHCenter
                            running: hypercareController && hypercareController.running
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            text: hypercareController && hypercareController.running
                                  ? qsTr("Automatyzacja hypercare trwa")
                                  : qsTr("Self-healing bezczynny")
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                            width: parent.width
                        }
                    }
                }
            }

            Frame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                background: Rectangle { color: Qt.darker(palette.base, 1.04); radius: 10 }

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 10

                        Label {
                            text: qsTr("Komponenty Stage6")
                            font.pixelSize: 18
                            font.bold: true
                        }

                        ListView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            spacing: 8
                            model: componentItems
                            delegate: Frame {
                                width: ListView.view.width
                                background: Rectangle { color: Qt.darker(palette.base, 1.02); radius: 8 }
                                padding: 12

                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: 6

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 10

                                        Label {
                                            text: model.name ? model.name.toUpperCase() : qsTr("Komponent")
                                            font.pixelSize: 16
                                            font.bold: true
                                        }

                                        Rectangle {
                                            width: 12
                                            height: 12
                                            radius: 6
                                            color: statusColor(model.status)
                                        }

                                        Label {
                                            text: model.status ? model.status.toUpperCase() : "UNKNOWN"
                                            color: palette.mid
                                        }
                                    }

                                    Label {
                                        visible: model.details && model.details.slo2 && model.details.slo2.breached && model.details.slo2.breached.length > 0
                                        text: visible ? qsTr("SLO2 naruszone: %1").arg(model.details.slo2.breached.join(", ")) : ""
                                        color: palette.highlight
                                        wrapMode: Text.WordWrap
                                    }

                                    Label {
                                        visible: model.details && model.details.slo2 && model.details.slo2.expected && model.details.slo2.expected.composites
                                        text: visible ? qsTr("Oczekiwane metryki: %1").arg(model.details.slo2.expected.composites.join(", ")) : ""
                                        color: palette.mid
                                        wrapMode: Text.WordWrap
                                    }

                                    Label {
                                        visible: model.details && model.details.summary && model.details.summary.overall_status
                                        text: visible ? qsTr("Stan szczegółowy: %1").arg(model.details.summary.overall_status) : ""
                                        color: palette.mid
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }
                            ScrollBar.vertical.policy: ScrollBar.AsNeeded
                            ScrollBar.vertical.interactive: true
                            placeholderText: qsTr("Brak danych hypercare")
                        }
                    }

                    ColumnLayout {
                        Layout.preferredWidth: parent.width * 0.32
                        Layout.fillHeight: true
                        spacing: 12

                        Label {
                            text: qsTr("Ostrzeżenia")
                            font.pixelSize: 18
                            font.bold: true
                        }

                        ListView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            spacing: 6
                            model: warningItems
                            delegate: Frame {
                                width: ListView.view.width
                                background: Rectangle { color: Qt.darker(palette.base, 1.01); radius: 6 }
                                padding: 10
                                Label {
                                    anchors.fill: parent
                                    text: modelData
                                    wrapMode: Text.WordWrap
                                    color: palette.mid
                                }
                            }
                            ScrollBar.vertical.policy: ScrollBar.AsNeeded
                            placeholderText: qsTr("Brak ostrzeżeń")
                        }

                        Label {
                            text: qsTr("Problemy krytyczne")
                            font.pixelSize: 18
                            font.bold: true
                        }

                        ListView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            spacing: 6
                            model: issueItems
                            delegate: Frame {
                                width: ListView.view.width
                                background: Rectangle { color: Qt.darker(palette.base, 1.01); radius: 6 }
                                padding: 10
                                Label {
                                    anchors.fill: parent
                                    text: modelData
                                    wrapMode: Text.WordWrap
                                    color: palette.highlight
                                }
                            }
                            ScrollBar.vertical.policy: ScrollBar.AsNeeded
                            placeholderText: qsTr("Brak zgłoszonych problemów")
                        }
                    }
                }
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle { color: Qt.darker(palette.base, 1.04); radius: 10 }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 10

                    Label {
                        text: qsTr("Logi self-healing")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    TextArea {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 100
                        readOnly: true
                        wrapMode: TextArea.Wrap
                        text: hypercareController && hypercareController.lastMessage ? hypercareController.lastMessage : qsTr("Brak komunikatów z ostatniego uruchomienia")
                    }

                    TextArea {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 100
                        readOnly: true
                        wrapMode: TextArea.Wrap
                        visible: hypercareController && hypercareController.lastError && hypercareController.lastError.length > 0
                        text: hypercareController ? hypercareController.lastError : ""
                        color: palette.highlight
                    }
                }
            }
        }
    }
}
