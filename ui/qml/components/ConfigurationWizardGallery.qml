import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 640
    implicitHeight: 360

    property var wizardController: (typeof configurationWizard !== "undefined") ? configurationWizard : null
    property var userProfiles: (typeof appController !== "undefined" && appController) ? appController.userProfiles : null
    property string profileId: userProfiles ? userProfiles.activeProfileId : "default"
    property var activeProgress: ({})

    objectName: "configurationWizardGallery"

    function refreshProgress() {
        activeProgress = userProfiles ? userProfiles.activeWizardProgress : {}
    }

    onProfileIdChanged: refreshProgress()

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            Label {
                text: qsTr("Kreator konfiguracji profilu %1").arg(profileId)
                font.pixelSize: Styles.AppTheme.fontSizeTitle
                font.bold: true
                font.family: Styles.AppTheme.fontFamily
                color: Styles.AppTheme.textPrimary
                Layout.fillWidth: true
            }

            Button {
                text: qsTr("Uruchom kreator")
                icon.name: "media-playback-start"
                enabled: !!wizardController && !!userProfiles
                onClicked: {
                    if (wizardController)
                        wizardController.start(profileId)
                }
            }
        }

        Connections {
            target: userProfiles
            ignoreUnknownSignals: true
            function onWizardProgressChanged() { refreshProgress() }
            function onActiveProfileChanged() { refreshProgress() }
            function onProfilesChanged() { refreshProgress() }
        }

        Component.onCompleted: refreshProgress()

        Frame {
            Layout.fillWidth: true
            background: Rectangle {
                radius: Styles.AppTheme.radiusMedium
                color: Styles.AppTheme.cardBackground(0.88)
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingSm
                spacing: Styles.AppTheme.spacingXs

                readonly property int totalSteps: wizardController && wizardController.steps ? wizardController.steps.length : 0
                readonly property int completedCount: activeProgress && activeProgress.completedSteps ? activeProgress.completedSteps.length : 0
                readonly property bool completed: activeProgress && activeProgress.completed === true

                ProgressBar {
                    objectName: "wizardSummaryProgress"
                    Layout.fillWidth: true
                    from: 0
                    to: parent.totalSteps > 0 ? parent.totalSteps : 1
                    value: parent.totalSteps > 0 ? Math.min(parent.completedCount, parent.totalSteps) : parent.completedCount
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Styles.AppTheme.spacingSm

                    Label {
                        text: parent.totalSteps > 0
                              ? qsTr("Postęp profilu %1: %2/%3 kroków").arg(profileId).arg(Math.min(parent.completedCount, parent.totalSteps)).arg(parent.totalSteps)
                              : qsTr("Zakończono %1 kroków dla profilu %2").arg(parent.completedCount).arg(profileId)
                        font.pixelSize: Styles.AppTheme.fontSizeCaption
                        color: Styles.AppTheme.textSecondary
                    }

                    Label {
                        objectName: "wizardSummaryStatus"
                        text: parent.completed ? qsTr("Status: ukończony") : qsTr("Status: w toku")
                        font.pixelSize: Styles.AppTheme.fontSizeCaption
                        color: parent.completed ? Styles.AppTheme.positive : Styles.AppTheme.textSecondary
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: qsTr("Oznacz ukończony")
                        enabled: !!userProfiles && !parent.completed
                        onClicked: {
                            if (userProfiles)
                                userProfiles.markWizardCompleted(profileId, true)
                        }
                    }

                    Button {
                        text: qsTr("Resetuj")
                        enabled: !!userProfiles && (parent.completedCount > 0 || parent.completed)
                        onClicked: {
                            if (userProfiles)
                                userProfiles.resetWizardProgress(profileId)
                        }
                    }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: Styles.AppTheme.spacingXs
                    Repeater {
                        model: activeProgress && activeProgress.completedSteps ? activeProgress.completedSteps : []
                        delegate: Rectangle {
                            radius: Styles.AppTheme.radiusSmall
                            color: Styles.AppTheme.cardBackground(0.76)
                            border.color: Qt.rgba(1, 1, 1, 0.08)
                            implicitHeight: 22
                            implicitWidth: stepLabel.implicitWidth + Styles.AppTheme.spacingSm * 2

                            Label {
                                id: stepLabel
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

        ListView {
            id: stepsView
            objectName: "wizardStepsView"
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: wizardController ? wizardController.steps : []
            boundsBehavior: Flickable.StopAtBounds
            spacing: Styles.AppTheme.spacingSm
            clip: true

            delegate: Frame {
                required property var modelData
                Layout.fillWidth: true
                implicitHeight: 96
                background: Rectangle {
                    color: Styles.AppTheme.cardBackground(0.86)
                    radius: Styles.AppTheme.radiusMedium
                    border.color: Qt.rgba(1, 1, 1, 0.05)
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingMd
                    spacing: Styles.AppTheme.spacingXs

                    Label {
                        text: modelData.title || qsTr("Nieznany krok")
                        font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                        font.bold: true
                        color: Styles.AppTheme.textPrimary
                    }

                    Label {
                        text: modelData.description || qsTr("Brak opisu")
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        wrapMode: Text.WordWrap
                        color: Styles.AppTheme.textSecondary
                    }

                    RowLayout {
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: qsTr("Kategoria: %1").arg(modelData.category || qsTr("—"))
                            font.pixelSize: Styles.AppTheme.fontSizeCaption
                            color: Styles.AppTheme.textTertiary
                            Layout.fillWidth: true
                        }

                        Label {
                            text: modelData.metadata && modelData.metadata.requiresLicense
                                  ? qsTr("Wymaga licencji")
                                  : modelData.metadata && modelData.metadata.requiresSignature
                                        ? qsTr("Wymaga podpisu")
                                        : ""
                            font.pixelSize: Styles.AppTheme.fontSizeCaption
                            color: Styles.AppTheme.warning
                            visible: text.length > 0
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: !model || model.length === 0
                text: qsTr("Brak zdefiniowanych kroków kreatora")
                color: Styles.AppTheme.textSecondary
                font.pixelSize: Styles.AppTheme.fontSizeBody
            }
        }
    }
}
