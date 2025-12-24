import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../design-system" as DesignSystem

Button {
    id: control
    property alias icon: iconProps
    property color backgroundColor: DesignSystem.Palette.surface
    property color foregroundColor: DesignSystem.Palette.textPrimary
    property bool subtle: false

    QtObject {
        id: iconProps
        property string name: ""
        property url source: ""
    }

    implicitHeight: 40
    implicitWidth: Math.max(40, contentItem.implicitWidth + 16)

    background: Rectangle {
        radius: 10
        color: control.subtle ? Qt.rgba(0, 0, 0, 0) : control.backgroundColor
        border.color: control.subtle ? Qt.rgba(1, 1, 1, 0.12) : DesignSystem.Palette.border
        border.width: control.subtle ? 1 : 0
        opacity: control.enabled ? 1 : 0.4
    }

    contentItem: RowLayout {
        spacing: 8
        anchors.centerIn: parent

        DesignSystem.Icon {
            visible: iconProps.name.length > 0
            name: iconProps.name
            size: 18
            color: control.foregroundColor
        }

        Image {
            visible: iconProps.name.length === 0 && iconProps.source.toString().length > 0
            source: iconProps.source
            width: 18
            height: 18
            fillMode: Image.PreserveAspectFit
        }

        Label {
            visible: control.text.length > 0
            text: control.text
            font.bold: true
            color: control.foregroundColor
        }
    }
}
