import QtQuick
import QtQuick.Effects

Item {
    id: root
    property url source: ""
    property color color: "transparent"
    property int fillMode: Image.PreserveAspectFit
    readonly property bool hasTint: root.color.a > 0
    readonly property bool hasSource: String(root.source).length > 0
    implicitWidth: Math.max(18, iconImage.implicitWidth)
    implicitHeight: Math.max(18, iconImage.implicitHeight)

    Image {
        id: iconImage
        anchors.fill: parent
        source: root.source
        fillMode: root.fillMode
        cache: true
        asynchronous: true
        opacity: root.hasSource ? (root.hasTint ? 0 : 1) : 0
        smooth: true
        mipmap: true
    }

    MultiEffect {
        anchors.fill: iconImage
        source: iconImage
        visible: root.hasSource && root.hasTint
        colorization: 1.0
        colorizationColor: root.color
    }
}
