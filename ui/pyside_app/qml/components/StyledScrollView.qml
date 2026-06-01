import QtQuick
import QtQuick.Controls

ScrollView {
    id: control
    property var designSystem
    clip: true
    contentWidth: availableWidth
    ScrollBar.vertical: ScrollBar {
        id: verticalBar
        policy: ScrollBar.AsNeeded
        interactive: true
        minimumSize: 0.08
        width: 10
        padding: 1
        background: Rectangle {
            radius: 5
            color: control.designSystem ? Qt.rgba(1, 1, 1, 0.04) : Qt.rgba(1, 1, 1, 0.06)
        }
        contentItem: Rectangle {
            implicitWidth: 8
            radius: 4
            color: control.designSystem
                   ? (verticalBar.pressed ? control.designSystem.color("accent") : control.designSystem.color("surfaceElevated"))
                   : (verticalBar.pressed ? "#55c7ff" : "#596172")
            border.color: control.designSystem ? control.designSystem.color("border") : "#3a4358"
            border.width: 1
        }
    }
    ScrollBar.horizontal: ScrollBar {
        policy: ScrollBar.AsNeeded
        interactive: true
        height: 10
        padding: 1
        background: Rectangle {
            radius: 5
            color: control.designSystem ? Qt.rgba(1, 1, 1, 0.04) : Qt.rgba(1, 1, 1, 0.06)
        }
        contentItem: Rectangle {
            implicitHeight: 8
            radius: 4
            color: control.designSystem ? control.designSystem.color("surfaceElevated") : "#596172"
            border.color: control.designSystem ? control.designSystem.color("border") : "#3a4358"
            border.width: 1
        }
    }
}
