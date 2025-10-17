import QtQuick
import QtQml
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: adminDialog
    objectName: "adminDialog"
    title: qsTr("Administracja bezpieczeństwem")
    modal: true
    standardButtons: Dialog.Close
    width: 640
    height: 420
    property string selectedProfileId: ""

    onOpened: {
        if (securityController)
            securityController.refresh()
    }

    Connections {
        target: securityController
        function onUserProfilesChanged() {
            if (!securityController)
                return
            const profiles = securityController.userProfiles || []
            var stillExists = false
            for (var index = 0; index < profiles.length; ++index) {
                const profile = profiles[index]
                if (profile && profile.user_id === adminDialog.selectedProfileId) {
                    stillExists = true
                    break
                }
            }
            if (!stillExists)
                adminDialog.selectedProfileId = ""
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        GroupBox {
            title: qsTr("Licencja OEM")
            Layout.fillWidth: true

            GridLayout {
                columns: 2
                rowSpacing: 6
                columnSpacing: 12
                Layout.fillWidth: true

                Label { text: qsTr("Status:") }
                Label {
                    objectName: "licenseStatusValue"
                    text: securityController && securityController.licenseInfo.status || qsTr("brak")
                }

                Label { text: qsTr("Fingerprint:") }
                Label {
                    objectName: "licenseFingerprintValue"
                    text: securityController && securityController.licenseInfo.fingerprint || qsTr("n/d")
                    wrapMode: Text.WrapAnywhere
                }

                Label { text: qsTr("Ważna od:") }
                Label { text: securityController && securityController.licenseInfo.valid_from || qsTr("n/d") }

                Label { text: qsTr("Ważna do:") }
                Label { text: securityController && securityController.licenseInfo.valid_to || qsTr("n/d") }
            }
        }

        GroupBox {
            title: qsTr("Profile użytkowników")
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                ListView {
                    id: profilesView
                    objectName: "profilesView"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    currentIndex: -1
                    model: securityController ? securityController.userProfiles : []

                    delegate: Frame {
                        required property var modelData
                        required property int index
                        property bool selected: adminDialog.selectedProfileId === (modelData.user_id || "")
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            color: selected ? Qt.rgba(0.2, 0.4, 0.8, 0.2) : Qt.rgba(0.1, 0.1, 0.1, 0.12)
                            radius: 6
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            Label {
                                text: (modelData.display_name || modelData.user_id) + " (" + modelData.user_id + ")"
                                font.bold: true
                            }

                            Label {
                                text: qsTr("Role: %1").arg((modelData.roles || []).join(", "))
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: qsTr("Aktualizacja: %1").arg(modelData.updated_at || "-")
                                color: palette.mid
                                font.pointSize: font.pointSize - 1
                            }
                        }

                        TapHandler {
                            acceptedButtons: Qt.LeftButton
                            onTapped: {
                                adminDialog.selectedProfileId = modelData.user_id || ""
                                profilesView.currentIndex = index
                                userIdField.text = modelData.user_id || ""
                                displayNameField.text = modelData.display_name || ""
                                rolesField.text = (modelData.roles || []).join(", ")
                            }
                        }
                    }
                }

                Rectangle { height: 1; Layout.fillWidth: true; color: palette.mid }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: userIdField
                        objectName: "userIdField"
                        Layout.fillWidth: true
                        placeholderText: qsTr("Identyfikator użytkownika")
                    }

                    TextField {
                        id: displayNameField
                        objectName: "displayNameField"
                        Layout.fillWidth: true
                        placeholderText: qsTr("Wyświetlana nazwa (opcjonalnie)")
                    }
                }

                TextField {
                    id: rolesField
                    objectName: "rolesField"
                    Layout.fillWidth: true
                    placeholderText: qsTr("Role (oddzielone przecinkami)")
                }

                RowLayout {
                    Layout.fillWidth: true

                    Button {
                        text: qsTr("Zapisz profil")
                        objectName: "saveProfileButton"
                        enabled: securityController && !securityController.busy
                        onClicked: {
                            const id = userIdField.text.trim()
                            const roles = rolesField.text.split(",").map(r => r.trim()).filter(r => r.length > 0)
                            if (!id) {
                                return
                            }
                            if (securityController.assignProfile(id, roles, displayNameField.text)) {
                                rolesField.text = ""
                                displayNameField.text = ""
                                userIdField.text = ""
                                adminDialog.selectedProfileId = ""
                            }
                        }
                    }

                    Button {
                        text: qsTr("Usuń profil")
                        objectName: "removeProfileButton"
                        enabled: securityController && !securityController.busy && adminDialog.selectedProfileId.length > 0
                        onClicked: {
                            if (!securityController)
                                return
                            const id = adminDialog.selectedProfileId
                            if (!id)
                                return
                            if (securityController.removeProfile(id)) {
                                adminDialog.selectedProfileId = ""
                                userIdField.text = ""
                                displayNameField.text = ""
                                rolesField.text = ""
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: qsTr("Odśwież")
                        objectName: "refreshButton"
                        enabled: securityController && !securityController.busy
                        onClicked: securityController && securityController.refresh()
                    }
                }
            }
        }
    }
}

