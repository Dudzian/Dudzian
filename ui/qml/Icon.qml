import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    // Core icon properties
    property string name: ""
    property alias iconName: name

    property url source: ""
    property alias iconSource: source

    property color color: "white"
    property alias iconColor: color

    property int pixelSize: 16
    property alias iconSize: pixelSize

    // Keep layout sane in containers
    implicitWidth: pixelSize
    implicitHeight: pixelSize

    // Image-based icon
    Image {
        id: imageIcon
        anchors.centerIn: parent
        width: root.pixelSize
        height: root.pixelSize
        visible: root.source !== ""
        source: root.source
        fillMode: Image.PreserveAspectFit
        sourceSize.width: root.pixelSize
        sourceSize.height: root.pixelSize
    }

    // Text fallback when no image source is provided.
    Text {
        id: textIcon
        anchors.centerIn: parent
        visible: root.source === "" && root.name !== ""
        text: root.name
        color: root.color
        font.pixelSize: root.pixelSize
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
