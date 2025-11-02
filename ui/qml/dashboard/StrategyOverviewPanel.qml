import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 720
    implicitHeight: 480

    property var userProfiles: (typeof appController !== "undefined" && appController) ? appController.userProfiles : null
    property string profileId: userProfiles ? userProfiles.activeProfileId : "default"
    property var favorites: []
    property var recommendations: []

    objectName: "strategyOverviewPanel"

    function refreshData() {
        if (!userProfiles) {
            favorites = []
            recommendations = []
            return
        }
        favorites = userProfiles.favoriteStrategies(profileId) || []
        recommendations = userProfiles.recommendedStrategies(profileId) || []
    }

    onUserProfilesChanged: refreshData()

    Component.onCompleted: refreshData()

    Connections {
        target: userProfiles
        ignoreUnknownSignals: true

        function onProfilesChanged() { refreshData() }
        function onActiveProfileChanged() { refreshData() }
        function onCatalogIntegrationChanged() { refreshData() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingLg

        GroupBox {
            title: qsTr("Ulubione strategie")
            Layout.fillWidth: true
            Layout.preferredHeight: Math.max(220, favoritesView.contentHeight + Styles.AppTheme.spacingLg * 2)
            background: Rectangle {
                color: Styles.AppTheme.cardBackground(0.92)
                radius: Styles.AppTheme.radiusLarge
            }

            ListView {
                id: favoritesView
                objectName: "favoritesView"
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingLg
                model: favorites
                clip: true
                delegate: Rectangle {
                    width: parent ? parent.width : 0
                    height: 52
                    radius: Styles.AppTheme.radiusMedium
                    color: Styles.AppTheme.cardBackground(0.82)
                    border.color: Qt.rgba(1, 1, 1, 0.06)

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: Styles.AppTheme.spacingSm
                        spacing: Styles.AppTheme.spacingSm

                        Label {
                            text: modelData
                            font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                            font.family: Styles.AppTheme.fontFamily
                            color: Styles.AppTheme.textPrimary
                            Layout.fillWidth: true
                        }

                        Button {
                            text: qsTr("Usuń")
                            icon.name: "list-remove"
                            visible: !!userProfiles
                            onClicked: {
                                if (userProfiles)
                                    userProfiles.toggleFavoriteStrategy(profileId, modelData)
                            }
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: favoritesView
                visible: favorites.length === 0
                text: qsTr("Dodaj strategie do ulubionych w katalogu")
                color: Styles.AppTheme.textSecondary
                font.pixelSize: Styles.AppTheme.fontSizeBody
                font.family: Styles.AppTheme.fontFamily
            }
        }

        GroupBox {
            title: qsTr("Rekomendacje katalogu")
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Rectangle {
                color: Styles.AppTheme.cardBackground(0.9)
                radius: Styles.AppTheme.radiusLarge
            }

            ListView {
                id: recommendationsView
                objectName: "recommendationsView"
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingLg
                model: recommendations
                clip: true
                delegate: Rectangle {
                    width: parent ? parent.width : 0
                    height: 68
                    radius: Styles.AppTheme.radiusMedium
                    color: Styles.AppTheme.cardBackground(0.78)
                    border.color: Qt.rgba(1, 1, 1, 0.04)

                    property var metadata: (modelData && modelData.metadata) ? modelData.metadata : {}

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: Styles.AppTheme.spacingSm
                        spacing: Styles.AppTheme.spacingSm

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingXs

                            Label {
                                text: modelData && modelData.name ? modelData.name : qsTr("(brak nazwy)")
                                font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                                font.bold: true
                                color: Styles.AppTheme.textPrimary
                            }

                            Label {
                                text: modelData && modelData.engine
                                      ? qsTr("Silnik: %1").arg(modelData.engine)
                                      : qsTr("Silnik nieznany")
                                font.pixelSize: Styles.AppTheme.fontSizeBody
                                color: Styles.AppTheme.textSecondary
                            }
                        }

                        ColumnLayout {
                            Layout.alignment: Qt.AlignVCenter
                            spacing: Styles.AppTheme.spacingXs

                            Label {
                                text: metadata && metadata.risk_profile
                                      ? qsTr("Profil: %1").arg(metadata.risk_profile)
                                      : qsTr("Profil: —")
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textTertiary
                            }

                            Label {
                                text: metadata && metadata.tags && metadata.tags.length > 0
                                      ? metadata.tags.join(", ")
                                      : qsTr("Brak tagów")
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textTertiary
                            }
                        }

                        Button {
                            text: qsTr("Dodaj do ulubionych")
                            icon.name: "list-add"
                            visible: !!userProfiles && modelData && modelData.name
                            onClicked: {
                                if (userProfiles && modelData && modelData.name)
                                    userProfiles.toggleFavoriteStrategy(profileId, modelData.name)
                            }
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: recommendationsView
                visible: recommendations.length === 0
                text: qsTr("Brak rekomendacji z katalogu strategii")
                color: Styles.AppTheme.textSecondary
                font.pixelSize: Styles.AppTheme.fontSizeBody
                font.family: Styles.AppTheme.fontFamily
            }
        }
    }
}
