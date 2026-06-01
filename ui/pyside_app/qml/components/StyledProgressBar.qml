import QtQuick
import QtQuick.Controls

ProgressBar {
    id: control
    property var designSystem
    property string label: ""
    from: 0
    to: 100
    implicitHeight: 30
    background: Rectangle {
        radius: 15
        color: control.designSystem ? control.designSystem.color("surfaceMuted") : "#252c3d"
        border.color: control.designSystem ? control.designSystem.color("border") : "#3a4358"
        border.width: 1
    }
    contentItem: Item {
        implicitHeight: 30
        Rectangle {
            width: Math.max(28, parent.width * control.visualPosition)
            height: parent.height
            radius: 15
            color: control.designSystem ? control.designSystem.color("accent") : "#55c7ff"
            opacity: 0.9
            Behavior on width { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
        }
        Text {
            anchors.centerIn: parent
            text: control.label.length > 0 ? control.label : Math.round(control.value) + "%"
            color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
            font.bold: true
            font.pixelSize: 12
        }
    }
}
