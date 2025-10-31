import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    width: parent ? parent.width : 800
    height: panel.implicitHeight

    property var manager: (typeof updateManager !== "undefined" ? updateManager : null)

    ColumnLayout {
        id: panel
        anchors.fill: parent
        spacing: 12

        Label {
            text: qsTr("Aktualizacje offline")
            font.pixelSize: 22
            font.bold: true
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: qsTr("Instaluj podpisane pakiety aktualizacji offline. Pakiety muszą pochodzić z zaufanego źródła i być zgodne z fingerprintem urządzenia.")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Button {
                text: qsTr("Odśwież")
                enabled: manager && !manager.busy
                onClicked: manager.refresh()
            }
            Label {
                text: manager && manager.busy ? qsTr("Wyszukiwanie aktualizacji...") : ""
                color: palette.mid
            }
            Label {
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                text: manager && manager.lastError ? manager.lastError : ""
                color: palette.negative
            }
        }

        GroupBox {
            Layout.fillWidth: true
            title: qsTr("Dostępne pakiety")
            background: Rectangle { radius: 12; color: Qt.rgba(1,1,1,0.04) }

            ListView {
                anchors.fill: parent
                model: manager ? manager.availableUpdates : []
                delegate: Frame {
                    width: parent ? parent.width - 24 : 760
                    background: Rectangle { radius: 10; color: Qt.rgba(1,1,1,0.05) }
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        Label { text: (modelData.id || "?") + " • " + (modelData.version || "-"); font.bold: true }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: modelData.description || qsTr("Brak opisu pakietu.")
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Label { text: qsTr("Fingerprint: %1").arg(modelData.fingerprint || qsTr("dowolny")) }
                            Label { text: modelData.differential ? qsTr("Łatka różnicowa (%1)").arg(modelData.baseId || "?") : qsTr("Pełny pakiet") }
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
            Layout.fillWidth: true
            title: qsTr("Zainstalowane pakiety")
            background: Rectangle { radius: 12; color: Qt.rgba(1,1,1,0.04) }

            ListView {
                anchors.fill: parent
                model: manager ? manager.installedUpdates : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: 12
                    Label { Layout.preferredWidth: 220; text: modelData.id || "?" }
                    Label { Layout.preferredWidth: 140; text: modelData.version || "" }
                    Label { Layout.preferredWidth: 180; text: modelData.installedAt || modelData.patchedAt || "" }
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
