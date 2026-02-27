import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Button {
    id: control
    property var designSystem
    property string iconName: ""
    property color backgroundColor: designSystem ? designSystem.color("surfaceMuted") : "#2a303c"
    property color foregroundColor: designSystem ? designSystem.color("textPrimary") : "#f5f7ff"
    property bool subtle: false
    implicitHeight: 40
    implicitWidth: Math.max(40, contentItem.implicitWidth + 16)
    readonly property string glyphText: designSystem && iconName.length > 0 && designSystem.iconGlyph
                                        ? designSystem.iconGlyph(iconName)
                                        : ""

    background: Rectangle {
        radius: 10
        color: control.subtle ? Qt.rgba(0, 0, 0, 0) : control.backgroundColor
        border.color: control.subtle
                        ? Qt.rgba(1, 1, 1, 0.12)
                        : (designSystem ? designSystem.color("border") : Qt.rgba(1, 1, 1, 0.2))
        border.width: control.subtle ? 1 : 0
        opacity: control.enabled ? 1 : 0.4
    }

    contentItem: RowLayout {
        spacing: 8
        anchors.centerIn: parent
        Text {
            visible: control.glyphText.length > 0
            text: control.glyphText
            font.family: control.designSystem ? control.designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
            font.pixelSize: 18
            color: control.foregroundColor
        }
        IconGlyph {
            visible: control.glyphText.length === 0 && control.iconName.length > 0
            source: control.iconName.length > 0 && designSystem
                    ? designSystem.iconSource(control.iconName)
                    : ""
            width: 18
            height: 18
            fillMode: Image.PreserveAspectFit
            color: control.foregroundColor
        }
        Label {
            visible: control.text.length > 0
            text: control.text
            font.bold: true
            color: control.foregroundColor
        }
    }
}
