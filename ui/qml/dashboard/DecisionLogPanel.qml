import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property var decisionModel: null
    property var decisionFilterModel: null
    property var runtimeService: null
    property var appController: null
    property int pageSize: 100

    padding: 16
    background: Rectangle {
        color: Qt.darker(palette.base, 1.02)
        radius: 10
    }

    function effectiveModel() {
        return decisionFilterModel ? decisionFilterModel : decisionModel
    }

    function reloadLog() {
        if (appController && appController.reloadDecisionLog) {
            appController.reloadDecisionLog()
        } else if (runtimeService && runtimeService.loadRecentDecisions) {
            runtimeService.loadRecentDecisions(pageSize)
        }
    }

    Component.onCompleted: reloadLog()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Dziennik decyzji AI")
                font.pixelSize: 18
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Odśwież")
                icon.name: "view-refresh"
                onClicked: reloadLog()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: decisionSearch
                Layout.fillWidth: true
                placeholderText: qsTr("Szukaj w decyzjach…")
                onTextChanged: {
                    if (decisionFilterModel && decisionFilterModel.searchText !== text)
                        decisionFilterModel.searchText = text
                }
                Component.onCompleted: {
                    if (decisionFilterModel && decisionFilterModel.searchText)
                        text = decisionFilterModel.searchText
                }
            }

            ComboBox {
                id: approvalCombo
                Layout.preferredWidth: 200
                model: [
                    { text: qsTr("Wszystkie"), value: 0 },
                    { text: qsTr("Zatwierdzone"), value: 1 },
                    { text: qsTr("Oczekujące"), value: 2 }
                ]
                textRole: "text"
                valueRole: "value"
                enabled: decisionFilterModel
                onActivated: {
                    if (!decisionFilterModel)
                        return
                    const entry = model[index]
                    if (entry)
                        decisionFilterModel.approvalFilter = entry.value
                }
                Component.onCompleted: {
                    if (!decisionFilterModel)
                        return
                    const current = typeof decisionFilterModel.approvalFilter === "number"
                        ? decisionFilterModel.approvalFilter : 0
                    for (let i = 0; i < model.length; ++i) {
                        if (model[i].value === current) {
                            currentIndex = i
                            break
                        }
                    }
                }
            }
        }

        ListView {
            id: decisionList
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: root.effectiveModel()
            clip: true
            spacing: 8
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

            delegate: Frame {
                width: decisionList.width
                padding: 12
                background: Rectangle {
                    radius: 8
                    color: Qt.darker(palette.base, 1.04)
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: timestampDisplay || timestamp || ""
                            color: palette.mid
                        }

                        Label {
                            text: strategy || qsTr("(strategia)")
                            font.bold: true
                        }

                        Label {
                            text: symbol || ""
                            color: palette.mid
                        }

                        Item { Layout.fillWidth: true }

                        Label {
                            text: decisionState || ""
                            color: approved ? Qt.rgba(0.35, 0.75, 0.55, 1) : Qt.rgba(0.88, 0.65, 0.32, 1)
                            font.pixelSize: 12
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        text: decisionReason || event || ""
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: !!schedule
                        color: palette.mid
                        text: schedule ? qsTr("Harmonogram: %1").arg(schedule) : ""
                    }
                }
            }

            footer: Item {
                width: decisionList.width
                height: decisionList.count > 0 ? 0 : 40
                Label {
                    anchors.centerIn: parent
                    text: qsTr("Brak zarejestrowanych decyzji")
                    color: palette.mid
                }
            }
        }
    }
}
