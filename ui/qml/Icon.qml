import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    // Public API used by other components (e.g. StrategyAiPanel):
    // - iconName: label / symbol text to show when there is no image
    // - iconSource: URL of the image to show as icon
    // - iconColor: color of the text-based icon
    // - iconSize: icon size in pixels
    property string iconName: ""
    property url iconSource: ""
    property color iconColor: "white"
    property int iconSize: 16

    // Make layout predictable when this icon is placed in other components
    implicitWidth: iconSize
    implicitHeight: iconSize

    // Image-based icon path
    Image {
        id: imageIcon
        anchors.centerIn: parent
        width: root.iconSize
        height: root.iconSize
        visible: root.iconSource !== ""
        source: root.iconSource
        fillMode: Image.PreserveAspectFit
        sourceSize.width: root.iconSize
        sourceSize.height: root.iconSize
    }

    // Text fallback path when no image source is provided
    Text {
        id: textIcon
        anchors.centerIn: parent
        visible: root.iconSource === "" && root.iconName !== ""
        text: root.iconName
        color: root.iconColor
        font.pixelSize: root.iconSize
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
