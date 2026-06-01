import QtQuick
import QtQuick.Controls

Switch {
    id: control
    property var designSystem
    implicitWidth: 58
    implicitHeight: 34
    indicator: Rectangle {
        implicitWidth: 52
        implicitHeight: 28
        x: control.leftPadding
        y: parent.height / 2 - height / 2
        radius: 14
        color: control.checked ? (control.designSystem ? control.designSystem.color("accent") : "#55c7ff") : (control.designSystem ? control.designSystem.color("surfaceMuted") : "#30384a")
        border.color: control.designSystem ? control.designSystem.color("border") : "#46506a"
        Rectangle {
            x: control.checked ? parent.width - width - 4 : 4
            anchors.verticalCenter: parent.verticalCenter
            width: 20
            height: 20
            radius: 10
            color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
            Behavior on x { NumberAnimation { duration: 120 } }
        }
    }
    contentItem: Text {
        text: control.text
        color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
        verticalAlignment: Text.AlignVCenter
        leftPadding: control.indicator.width + control.spacing
    }
}
