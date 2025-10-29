import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Item {
    id: root
    width: parent ? parent.width : 800
    height: parent ? parent.height : 600
    property var selectedPreset: null
    property string activationError: ""

    function openActivationDialog(preset) {
        selectedPreset = preset
        activationInput.text = "{\n  \"fingerprint\": \"\",\n  \"expires_at\": \"\"\n}"
        activationError = ""
        activationDialog.open()
    }

    Component.onCompleted: {
        if (marketplaceController)
            marketplaceController.refreshPresets()
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Marketplace presetów strategii")
                font.pixelSize: 22
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            ToolButton {
                icon.name: "view-refresh"
                text: qsTr("Odśwież")
                display: AbstractButton.TextBesideIcon
                onClicked: marketplaceController.refreshPresets()
            }
        }

        Label {
            id: statusLabel
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            text: marketplaceController.lastError.length > 0 ? marketplaceController.lastError
                                                           : qsTr("Załadowano %1 presetów").arg(marketplaceController.presets.length)
            color: marketplaceController.lastError.length > 0 ? "firebrick" : palette.windowText
        }

        BusyIndicator {
            Layout.alignment: Qt.AlignLeft
            visible: marketplaceController.busy
            running: marketplaceController.busy
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Flow {
                width: parent.width
                spacing: 16

                Repeater {
                    model: marketplaceController.presets
                    delegate: Frame {
                        id: card
                        property var preset: modelData
                        width: Math.min(parent.width, 420)
                        Layout.preferredWidth: 360
                        padding: 16

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                text: preset.name
                                font.pixelSize: 18
                                font.bold: true
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: qsTr("Profil: %1").arg(preset.profile)
                                color: card.palette.mid
                            }

                            Label {
                                text: preset.summary || preset.metadata?.summary || ""
                                wrapMode: Text.WordWrap
                                visible: text.length > 0
                            }

                            Label {
                                text: qsTr("Status licencji: %1").arg(preset.license.status)
                                color: preset.license.status === "active" ? "#0a8f08"
                                       : preset.license.status === "expired" ? "#c8500a"
                                       : card.palette.windowText
                                font.bold: preset.license.status === "active"
                            }

                            Label {
                                visible: preset.license.expires_at !== null
                                text: qsTr("Wygasa: %1").arg(preset.license.expires_at || qsTr("brak"))
                                color: card.palette.mid
                            }

                            Flow {
                                width: parent.width
                                spacing: 6
                                Repeater {
                                    model: preset.tags || []
                                    delegate: Rectangle {
                                        radius: 4
                                        color: Qt.darker(card.palette.base, 1.1)
                                        border.color: card.palette.mid
                                        border.width: 1
                                        padding: 4

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData
                                            font.pixelSize: 12
                                        }
                                    }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2
                                visible: preset.required_parameters !== undefined

                                Repeater {
                                    model: Object.keys(preset.required_parameters || {})
                                    delegate: Label {
                                        Layout.fillWidth: true
                                        text: qsTr("Parametry %1: %2").arg(modelData).arg((preset.required_parameters[modelData] || []).join(", "))
                                        wrapMode: Text.WordWrap
                                        font.pixelSize: 12
                                        color: card.palette.mid
                                    }
                                }
                            }

                            RowLayout {
                                Layout.topMargin: 12
                                spacing: 8

                                Button {
                                    text: qsTr("Aktywuj")
                                    onClicked: root.openActivationDialog(preset)
                                }

                                Button {
                                    text: qsTr("Dezaktywuj")
                                    enabled: preset.license.status !== "unlicensed"
                                    onClicked: marketplaceController.deactivatePreset(preset.preset_id)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: activationDialog
        title: selectedPreset ? qsTr("Aktywacja: %1").arg(selectedPreset.name) : qsTr("Aktywacja presetu")
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        onAccepted: {
            try {
                const payload = JSON.parse(activationInput.text)
                marketplaceController.activatePreset(selectedPreset.preset_id, payload)
                activationError = ""
            } catch (error) {
                activationError = qsTr("Niepoprawny JSON: %1").arg(error)
                activationDialog.open()
            }
        }

        ColumnLayout {
            width: 420
            spacing: 8

            Label {
                text: qsTr("Podaj payload licencji w formacie JSON")
                wrapMode: Text.WordWrap
            }

            TextArea {
                id: activationInput
                Layout.preferredWidth: 400
                Layout.preferredHeight: 180
                textFormat: TextEdit.PlainText
            }

            Label {
                visible: activationError.length > 0
                text: activationError
                color: "firebrick"
                wrapMode: Text.WordWrap
            }
        }
    }
}
