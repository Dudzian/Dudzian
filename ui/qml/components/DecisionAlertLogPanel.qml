import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "."

Item {
    id: root

    property var alertsModel: null
    property var alertsFilterModel: null
    property var decisionModel: null
    property var decisionFilterModel: null
    property bool _syncingControls: false

    signal exportCompleted(url file)

    function approvalOptions() {
        return [
            { text: qsTr("Wszystkie"), value: 0 },
            { text: qsTr("Tylko zatwierdzone"), value: 1 },
            { text: qsTr("Tylko oczekujące"), value: 2 }
        ]
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Eksportuj dziennik decyzji")
        nameFilters: [qsTr("Pliki CSV (*.csv)"), qsTr("Wszystkie pliki (*)")]
        fileMode: FileDialog.SaveFile
        onAccepted: {
            if (decisionFilterModel && typeof decisionFilterModel.exportFilteredToCsv === "function") {
                const ok = decisionFilterModel.exportFilteredToCsv(exportDialog.currentFile)
                if (ok)
                    root.exportCompleted(exportDialog.currentFile)
            }
        }
    }

    function updateDecisionFilterControls() {
        if (root._syncingControls)
            return

        root._syncingControls = true

        if (!decisionFilterModel) {
            searchField.text = ""
            strategyField.text = ""
            regimeField.text = ""
            approvalCombo.currentIndex = 0
            root._syncingControls = false
            return
        }

        const search = decisionFilterModel.searchText || ""
        if (searchField.text !== search)
            searchField.text = search

        const strategy = decisionFilterModel.strategyFilter || ""
        if (strategyField.text !== strategy)
            strategyField.text = strategy

        const regime = decisionFilterModel.regimeFilter || ""
        if (regimeField.text !== regime)
            regimeField.text = regime

        const approval = typeof decisionFilterModel.approvalFilter === "number"
                ? decisionFilterModel.approvalFilter : 0
        let comboIndex = 0
        for (let i = 0; i < approvalCombo.model.length; ++i) {
            if (approvalCombo.model[i].value === approval) {
                comboIndex = i
                break
            }
        }
        if (approvalCombo.currentIndex !== comboIndex)
            approvalCombo.currentIndex = comboIndex

        root._syncingControls = false
    }

    onDecisionFilterModelChanged: updateDecisionFilterControls()

    Connections {
        target: decisionFilterModel
        function onFilterChanged() {
            updateDecisionFilterControls()
        }
    }

    Component.onCompleted: updateDecisionFilterControls()

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Frame {
            SplitView.preferredWidth: 360
            Layout.fillHeight: true
            padding: 12
            background: Rectangle {
                color: Qt.darker(palette.window, 1.2)
                radius: 8
            }

            AlertCenterPanel {
                anchors.fill: parent
                summaryModel: alertsModel
                listModel: alertsFilterModel
            }
        }

        Frame {
            SplitView.fillWidth: true
            Layout.fillHeight: true
            padding: 12
            background: Rectangle {
                color: Qt.darker(palette.window, 1.15)
                radius: 8
            }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: qsTr("Dziennik decyzji")
                        font.bold: true
                        font.pixelSize: 18
                    }
                    Item { Layout.fillWidth: true }
                    Button {
                        text: qsTr("Eksportuj CSV")
                        icon.name: "document-save"
                        onClicked: exportDialog.open()
                        enabled: decisionFilterModel && decisionFilterModel.rowCount !== undefined && decisionFilterModel.rowCount() > 0
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: searchField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Szukaj decyzji…")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel && decisionFilterModel.searchText !== text)
                                decisionFilterModel.searchText = text
                        }
                    }

                    ComboBox {
                        id: approvalCombo
                        model: approvalOptions()
                        textRole: "text"
                        valueRole: "value"
                        onActivated: {
                            if (root._syncingControls)
                                return
                            const opt = model[index]
                            if (decisionFilterModel && opt)
                                decisionFilterModel.approvalFilter = opt.value
                        }
                    }

                    TextField {
                        id: strategyField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Strategia")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.strategyFilter = text
                        }
                    }

                    TextField {
                        id: regimeField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Reżim")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.regimeFilter = text
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: Qt.rgba(1, 1, 1, 0.1)
                }

                ListView {
                    id: decisionView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 6
                    model: decisionFilterModel
                    delegate: Frame {
                        required property string timestampDisplay
                        required property string strategy
                        required property string symbol
                        required property string side
                        required property string decisionState
                        required property string decisionReason
                        required property bool approved
                        required property string decisionMode

                        Layout.fillWidth: true
                        padding: 10
                        background: Rectangle {
                            radius: 6
                            color: approved ? Qt.rgba(0.16, 0.32, 0.22, 0.7) : Qt.rgba(0.32, 0.18, 0.18, 0.7)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: timestampDisplay
                                    color: Qt.rgba(0.82, 0.84, 0.88, 1)
                                }
                                Label {
                                    text: strategy
                                    font.bold: true
                                }
                                Label {
                                    text: symbol + " (" + side + ")"
                                    color: Qt.rgba(0.86, 0.88, 0.92, 1)
                                }
                                Item { Layout.fillWidth: true }
                                Label {
                                    text: decisionMode
                                    color: Qt.rgba(0.76, 0.78, 0.82, 1)
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: qsTr("Stan: %1").arg(decisionState)
                                    color: Qt.rgba(0.86, 0.88, 0.92, 1)
                                }
                                Label {
                                    text: decisionReason
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                    PlaceholderMessage {
                        anchors.centerIn: parent
                        text: qsTr("Brak wpisów pasujących do filtra")
                        visible: !decisionFilterModel || decisionFilterModel.rowCount === undefined || decisionFilterModel.rowCount() === 0
                    }
                }
            }
        }
    }
}
