import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../dashboard" as Dashboard
import "../settings" as Settings
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 640

    property var appController: (typeof appController !== "undefined") ? appController : null
    property var configurationWizard: (typeof configurationWizard !== "undefined") ? configurationWizard : null
    property var workbenchController: (typeof workbenchController !== "undefined") ? workbenchController : null

    objectName: "strategyExperienceView"

    Flickable {
        anchors.fill: parent
        contentWidth: parent.width
        contentHeight: contentColumn.implicitHeight + Styles.AppTheme.spacingXl
        clip: true
        ScrollBar.vertical: ScrollBar {}

        ColumnLayout {
            id: contentColumn
            width: parent.width
            spacing: Styles.AppTheme.spacingLg
            padding: Styles.AppTheme.spacingLg

            Components.UserProfileManagementPanel {
                Layout.fillWidth: true
                Layout.preferredHeight: implicitHeight
                userProfiles: appController ? appController.userProfiles : null
                wizardController: configurationWizard
            }

            Dashboard.StrategyOverviewPanel {
                Layout.fillWidth: true
                Layout.preferredHeight: implicitHeight
                userProfiles: appController ? appController.userProfiles : null
            }

            Components.ConfigurationWizardGallery {
                Layout.fillWidth: true
                Layout.preferredHeight: implicitHeight
                wizardController: configurationWizard
                userProfiles: appController ? appController.userProfiles : null
            }

            Settings.ThemePersonalization {
                Layout.fillWidth: true
                Layout.preferredHeight: implicitHeight
                userProfiles: appController ? appController.userProfiles : null
            }
        }
    }
}
