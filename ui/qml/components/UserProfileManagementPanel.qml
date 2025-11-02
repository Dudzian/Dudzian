import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 640
    implicitHeight: 320

    property var userProfiles: (typeof appController !== "undefined" && appController) ? appController.userProfiles : null
    property var wizardController: (typeof configurationWizard !== "undefined") ? configurationWizard : null

    objectName: "profileManagementPanel"

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        Label {
            text: qsTr("Zarządzanie profilami użytkownika")
            font.pixelSize: Styles.AppTheme.fontSizeTitle
            font.bold: true
            font.family: Styles.AppTheme.fontFamily
            color: Styles.AppTheme.textPrimary
        }

        Label {
            text: qsTr("Profile sterują motywem, ulubionymi strategiami oraz rekomendacjami katalogu.")
            color: Styles.AppTheme.textSecondary
            font.pixelSize: Styles.AppTheme.fontSizeBody
            wrapMode: Text.WordWrap
        }

        ListView {
            id: profilesView
            objectName: "profilesListView"
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: userProfiles ? userProfiles.profiles : []
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            spacing: Styles.AppTheme.spacingSm

            delegate: Frame {
                required property var modelData
                Layout.fillWidth: true
                implicitHeight: 88
                background: Rectangle {
                    radius: Styles.AppTheme.radiusMedium
                    color: Styles.AppTheme.cardBackground(0.85)
                    border.color: Qt.rgba(1, 1, 1, 0.06)
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingSm
                    spacing: Styles.AppTheme.spacingXs

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: modelData && modelData.id ? modelData.id : qsTr("(brak id)")
                            font.pixelSize: Styles.AppTheme.fontSizeCaption
                            color: Styles.AppTheme.textTertiary
                        }

                        Rectangle {
                            visible: userProfiles && modelData && modelData.id === userProfiles.activeProfileId
                            implicitWidth: 8
                            implicitHeight: 8
                            radius: 4
                            color: Styles.AppTheme.accent
                        }

                        Item { Layout.fillWidth: true }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Styles.AppTheme.spacingSm

                        TextField {
                            id: nameField
                            objectName: "profileNameField_" + (modelData && modelData.id ? modelData.id : index)
                            text: modelData && modelData.displayName ? modelData.displayName : (modelData && modelData.id ? modelData.id : "")
                            Layout.fillWidth: true
                            placeholderText: qsTr("Nazwa profilu")
                        }

                        Button {
                            objectName: "activateProfileButton_" + (modelData && modelData.id ? modelData.id : index)
                            text: qsTr("Aktywuj")
                            enabled: !!userProfiles && modelData && modelData.id && modelData.id !== userProfiles.activeProfileId
                            onClicked: {
                                if (userProfiles && modelData && modelData.id)
                                    userProfiles.setActiveProfile(modelData.id)
                            }
                        }

                        Button {
                            objectName: "saveProfileButton_" + (modelData && modelData.id ? modelData.id : index)
                            text: qsTr("Zapisz")
                            enabled: !!userProfiles && modelData && modelData.id && nameField.text.trim().length > 0
                            onClicked: {
                                if (!userProfiles || !modelData || !modelData.id)
                                    return
                                const trimmed = nameField.text.trim()
                                if (trimmed.length === 0)
                                    return
                                userProfiles.renameProfile(modelData.id, trimmed)
                            }
                        }

                        Button {
                            objectName: "duplicateProfileButton_" + (modelData && modelData.id ? modelData.id : index)
                            text: qsTr("Duplikuj")
                            enabled: !!userProfiles && modelData && modelData.id
                            onClicked: {
                                if (!userProfiles || !modelData || !modelData.id)
                                    return
                                const trimmed = nameField.text.trim()
                                if (trimmed.length > 0)
                                    userProfiles.duplicateProfile(modelData.id, trimmed + " (kopia)")
                                else
                                    userProfiles.duplicateProfile(modelData.id)
                            }
                        }

                        Button {
                            objectName: "resetProfileButton_" + (modelData && modelData.id ? modelData.id : index)
                            text: qsTr("Resetuj")
                            enabled: !!userProfiles && modelData && modelData.id
                            onClicked: {
                                if (userProfiles && modelData && modelData.id)
                                    userProfiles.resetProfile(modelData.id)
                            }
                        }

                        Button {
                            objectName: "removeProfileButton_" + (modelData && modelData.id ? modelData.id : index)
                            text: qsTr("Usuń")
                            enabled: !!userProfiles && userProfiles.profiles && userProfiles.profiles.length > 1
                            onClicked: {
                                if (userProfiles && modelData && modelData.id)
                                    userProfiles.removeProfile(modelData.id)
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: Styles.AppTheme.spacingXs

                        readonly property var progressData: (modelData && modelData.setupProgress) ? modelData.setupProgress : ({ completedSteps: [], completed: false })
                        readonly property int totalSteps: wizardController && wizardController.steps ? wizardController.steps.length : 0
                        readonly property int completedCount: progressData.completedSteps ? progressData.completedSteps.length : 0

                        ProgressBar {
                            objectName: "wizardProgressBar_" + (modelData && modelData.id ? modelData.id : index)
                            Layout.fillWidth: true
                            from: 0
                            to: totalSteps > 0 ? totalSteps : 1
                            value: totalSteps > 0 ? Math.min(parent.completedCount, totalSteps) : parent.completedCount
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm

                            Label {
                                text: totalSteps > 0
                                      ? qsTr("Postęp: %1/%2 kroków").arg(Math.min(parent.completedCount, totalSteps)).arg(totalSteps)
                                      : qsTr("Zakończono %1 kroków").arg(parent.completedCount)
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textSecondary
                            }

                            Label {
                                objectName: "wizardStatusLabel_" + (modelData && modelData.id ? modelData.id : index)
                                text: progressData.completed ? qsTr("Ukończony") : qsTr("W toku")
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: progressData.completed ? Styles.AppTheme.positive : Styles.AppTheme.textSecondary
                            }

                            Item { Layout.fillWidth: true }

                            Button {
                                text: qsTr("Oznacz ukończenie")
                                visible: !!userProfiles
                                enabled: !!userProfiles && !progressData.completed
                                onClicked: {
                                    if (userProfiles && modelData && modelData.id)
                                        userProfiles.markWizardCompleted(modelData.id, true)
                                }
                            }

                            Button {
                                text: qsTr("Reset postępu")
                                visible: !!userProfiles
                                enabled: !!userProfiles && (parent.completedCount > 0 || progressData.completed)
                                onClicked: {
                                    if (userProfiles && modelData && modelData.id)
                                        userProfiles.resetWizardProgress(modelData.id)
                                }
                            }
                        }

                        Flow {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingXs
                            Repeater {
                                model: progressData.completedSteps ? progressData.completedSteps : []
                                delegate: Rectangle {
                                    radius: Styles.AppTheme.radiusSmall
                                    color: Styles.AppTheme.cardBackground(0.75)
                                    border.color: Qt.rgba(1, 1, 1, 0.08)
                                    implicitHeight: 24
                                    implicitWidth: label.implicitWidth + Styles.AppTheme.spacingSm * 2

                                    Label {
                                        id: label
                                        anchors.centerIn: parent
                                        text: modelData
                                        font.pixelSize: Styles.AppTheme.fontSizeCaption
                                        color: Styles.AppTheme.textPrimary
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: !model || model.length === 0
                text: qsTr("Brak zdefiniowanych profili użytkowników")
                color: Styles.AppTheme.textSecondary
                font.pixelSize: Styles.AppTheme.fontSizeBody
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            TextField {
                id: newProfileName
                objectName: "newProfileNameField"
                Layout.fillWidth: true
                placeholderText: qsTr("Nowy profil — np. Scalper, Hedging")
            }

            Button {
                id: createProfileButton
                objectName: "createProfileButton"
                text: qsTr("Dodaj profil")
                enabled: !!userProfiles && newProfileName.text.trim().length > 0
                onClicked: {
                    if (!userProfiles)
                        return
                    const trimmed = newProfileName.text.trim()
                    if (trimmed.length === 0)
                        return
                    const createdId = userProfiles.createProfile(trimmed)
                    if (createdId && createdId.length > 0)
                        newProfileName.text = ""
                }
            }
        }
    }
}
