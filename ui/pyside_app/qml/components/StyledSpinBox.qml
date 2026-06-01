import QtQuick
import QtQuick.Controls

SpinBox {
    id: control
    property var designSystem
    implicitHeight: 40
    editable: true
    contentItem: TextInput {
        text: control.textFromValue(control.value, control.locale)
        font: control.font
        color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
        selectionColor: control.designSystem ? control.designSystem.color("accent") : "#55c7ff"
        selectedTextColor: control.designSystem ? control.designSystem.color("surface") : "#101522"
        horizontalAlignment: Qt.AlignHCenter
        verticalAlignment: Qt.AlignVCenter
        readOnly: !control.editable
        validator: control.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
    }
    up.indicator: Rectangle {
        x: control.mirrored ? 0 : parent.width - width
        height: parent.height
        width: 34
        radius: 10
        color: control.designSystem ? control.designSystem.color("surfaceElevated") : "#30384a"
        border.color: control.designSystem ? control.designSystem.color("border") : "#46506a"
        Text {
            anchors.centerIn: parent
            text: "+"
            color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
            font.bold: true
        }
    }
    down.indicator: Rectangle {
        x: control.mirrored ? parent.width - width : 0
        height: parent.height
        width: 34
        radius: 10
        color: control.designSystem ? control.designSystem.color("surfaceElevated") : "#30384a"
        border.color: control.designSystem ? control.designSystem.color("border") : "#46506a"
        Text {
            anchors.centerIn: parent
            text: "−"
            color: control.designSystem ? control.designSystem.color("textPrimary") : "#f5f7ff"
            font.bold: true
        }
    }
    background: Rectangle {
        radius: 10
        color: control.designSystem ? control.designSystem.color("surfaceMuted") : "#252c3d"
        border.color: control.activeFocus && control.designSystem ? control.designSystem.color("accent") : (control.designSystem ? control.designSystem.color("border") : "#3a4358")
        border.width: control.activeFocus ? 2 : 1
    }
}
