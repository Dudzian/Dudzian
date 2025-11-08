import QtQuick
import QtQuick.Controls
import ".." as Design

Frame {
    id: root
    implicitWidth: 320
    implicitHeight: 200
    padding: 16
    background: Rectangle {
        radius: 12
        color: Design.Palette.surface
        border.color: Design.Palette.border
        border.width: 1
    }
}
