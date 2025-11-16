import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".." as Design

Control {
    id: root
    implicitWidth: 320
    implicitHeight: 220
    padding: 20

    property string title: ""
    property string subtitle: ""
    property real cornerRadius: 18
    property alias headerContent: headerExtras.data
    default property alias contentData: bodySlot.data

    background: Design.FrostedGlass {
        anchors.fill: parent
        radius: root.cornerRadius
    }

    contentItem: ColumnLayout {
        anchors.fill: parent
        anchors.margins: root.padding
        spacing: 12

        ColumnLayout {
            id: headerBlock
            spacing: 4
            visible: root.title.length > 0 || root.subtitle.length > 0 || headerExtras.data.length > 0

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Label {
                    text: root.title
                    visible: root.title.length > 0
                    color: Design.Palette.textPrimary
                    font.pixelSize: Design.Typography.headlineSmall
                    font.bold: true
                }

                Item { Layout.fillWidth: true }

                Item {
                    id: headerExtras
                    Layout.alignment: Qt.AlignRight
                }
            }

            Label {
                visible: root.subtitle.length > 0
                text: root.subtitle
                color: Design.Palette.textSecondary
                font.pixelSize: Design.Typography.body
                wrapMode: Text.WordWrap
            }
        }

        ColumnLayout {
            id: bodySlot
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
        }
    }
}
