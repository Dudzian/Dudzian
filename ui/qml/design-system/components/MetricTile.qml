import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import ".." as Design

Card {
    id: tile
    property string label: ""
    property string value: "--"
    property string trend: ""
    implicitWidth: 180
    implicitHeight: 120

    ColumnLayout {
        anchors.fill: parent
        spacing: 6

        Label {
            text: tile.label
            color: Design.Palette.textSecondary
            font.pixelSize: Design.Typography.caption
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        Label {
            text: tile.value
            color: Design.Palette.textPrimary
            font.pixelSize: Design.Typography.headlineMedium
            font.bold: true
            Layout.fillWidth: true
        }

        Label {
            text: tile.trend
            visible: tile.trend.length > 0
            color: tile.trend.startsWith("-") ? Design.Palette.danger : Design.Palette.success
            font.pixelSize: Design.Typography.body
        }
    }
}
