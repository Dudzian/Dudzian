import QtQuick
import QtQuick.Controls

TextField {
    id: control
    property var designSystem
    color: designSystem ? designSystem.color("textPrimary") : "#f5f7ff"
    placeholderTextColor: designSystem ? designSystem.color("textSecondary") : "#8d96ad"
    selectedTextColor: designSystem ? designSystem.color("surface") : "#101522"
    selectionColor: designSystem ? designSystem.color("accent") : "#55c7ff"
    implicitHeight: 40
    leftPadding: 12
    rightPadding: 12
    background: Rectangle {
        radius: 10
        color: control.designSystem ? control.designSystem.color("surfaceMuted") : "#252c3d"
        border.color: control.activeFocus && control.designSystem ? control.designSystem.color("accent") : (control.designSystem ? control.designSystem.color("border") : "#3a4358")
        border.width: control.activeFocus ? 2 : 1
    }
}
