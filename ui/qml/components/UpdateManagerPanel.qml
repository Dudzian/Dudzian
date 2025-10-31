import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    width: parent ? parent.width : 800
    height: panel.implicitHeight

    property var manager: (typeof updateManager !== "undefined" ? updateManager : null)

    ColumnLayout {
        id: panel
        objectName: "updateManagerPanel"
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        Label {
            text: qsTr("Aktualizacje offline")
            font.pixelSize: Styles.AppTheme.fontSizeHeadline
            font.family: Styles.AppTheme.fontFamily
            font.bold: true
            color: Styles.AppTheme.textPrimary
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: qsTr("Instaluj podpisane pakiety aktualizacji offline. Pakiety muszą pochodzić z zaufanego źródła i być zgodne z fingerprintem urządzenia.")
            font.pixelSize: Styles.AppTheme.fontSizeBody
            font.family: Styles.AppTheme.fontFamily
            color: Styles.AppTheme.textSecondary
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            Button {
                text: qsTr("Odśwież")
                enabled: manager && !manager.busy
                onClicked: manager.refresh()
            }
            Label {
                text: manager && manager.busy ? qsTr("Wyszukiwanie aktualizacji...") : ""
                color: Styles.AppTheme.textTertiary
                font.family: Styles.AppTheme.fontFamily
                font.pixelSize: Styles.AppTheme.fontSizeBody
            }
            Label {
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                text: manager && manager.lastError ? manager.lastError : ""
                color: Styles.AppTheme.negative
                font.family: Styles.AppTheme.fontFamily
                font.pixelSize: Styles.AppTheme.fontSizeBody
            }
        }

        GroupBox {
            objectName: "availableUpdatesGroup"
            Layout.fillWidth: true
            title: qsTr("Dostępne pakiety")
            background: Rectangle {
                radius: Styles.AppTheme.radiusLarge
                color: Styles.AppTheme.cardBackground(0.82)
            }

            ListView {
                anchors.fill: parent
                model: manager ? manager.availableUpdates : []
                delegate: Frame {
                    width: parent ? parent.width - 24 : 760
                    background: Rectangle {
                        radius: Styles.AppTheme.radiusMedium
                        color: Styles.AppTheme.cardBackground(0.92)
                    }
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: Styles.AppTheme.spacingSm
                        spacing: Styles.AppTheme.spacingXs
                        Label {
                            text: (modelData.id || "?") + " • " + (modelData.version || "-")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: modelData.description || qsTr("Brak opisu pakietu.")
                            font.family: Styles.AppTheme.fontFamily
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textSecondary
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Styles.AppTheme.spacingSm
                            Label {
                                text: qsTr("Fingerprint: %1").arg(modelData.fingerprint || qsTr("dowolny"))
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textTertiary
                            }
                            Label {
                                text: modelData.differential ? qsTr("Łatka różnicowa (%1)").arg(modelData.baseId || "?") : qsTr("Pełny pakiet")
                                font.family: Styles.AppTheme.fontFamily
                                font.pixelSize: Styles.AppTheme.fontSizeCaption
                                color: Styles.AppTheme.textTertiary
                            }
                            Item { Layout.fillWidth: true }
                            Button {
                                text: qsTr("Zainstaluj")
                                enabled: manager && !manager.busy
                                onClicked: manager.applyUpdate(modelData.id)
                            }
                        }
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }

        GroupBox {
            objectName: "installedUpdatesGroup"
            Layout.fillWidth: true
            title: qsTr("Zainstalowane pakiety")
            background: Rectangle {
                radius: Styles.AppTheme.radiusLarge
                color: Styles.AppTheme.cardBackground(0.82)
            }

            ListView {
                anchors.fill: parent
                model: manager ? manager.installedUpdates : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: Styles.AppTheme.spacingSm
                    Label {
                        Layout.preferredWidth: 220
                        text: modelData.id || "?"
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textPrimary
                    }
                    Label {
                        Layout.preferredWidth: 140
                        text: modelData.version || ""
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Label {
                        Layout.preferredWidth: 180
                        text: modelData.installedAt || modelData.patchedAt || ""
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Item { Layout.fillWidth: true }
                    Button {
                        text: qsTr("Rollback")
                        visible: modelData.id
                        enabled: manager && !manager.busy
                        onClicked: manager.rollbackUpdate(modelData.id)
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }
}
