import QtQuick
import QtQuick.Controls
import QtQuick.Effects
import Qt5Compat.GraphicalEffects

Item {
    id: root
    implicitWidth: 320
    implicitHeight: 200

    property Item sourceItem: null
    property real radius: 18
    property real blurRadius: 64
    property color tintColor: Qt.rgba(16 / 255, 19 / 255, 38 / 255, 0.78)
    property color borderColor: Qt.rgba(1, 1, 1, 0.08)
    property real borderWidth: 1
    default property alias contentData: contentLayer.data

    Rectangle {
        id: baseTint
        anchors.fill: parent
        radius: root.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(1, 1, 1, 0.06) }
            GradientStop { position: 1.0; color: root.tintColor }
        }
    }

    ShaderEffectSource {
        id: blurSource
        anchors.fill: parent
        sourceItem: root.sourceItem ? root.sourceItem : baseTint
        hideSource: false
        live: root.sourceItem !== null
        recursive: true
    }

    MultiEffect {
        anchors.fill: parent
        source: blurSource
        blurEnabled: true
        blur: 0.45
        blurMax: root.blurRadius
        brightness: 0.05
        saturation: 1.05
        // colorizationStrength is not available in QtQuick.Effects MultiEffect on some Qt versions.
        // Approximate the intended subtle tint by baking the strength into the alpha channel.
        colorization: Qt.rgba(1, 1, 1, 0.028)
    }

    DropShadow {
        anchors.fill: baseTint
        source: baseTint
        horizontalOffset: 0
        verticalOffset: 6
        radius: 28
        samples: 32
        color: Qt.rgba(0, 0, 0, 0.35)
    }

    Rectangle {
        anchors.fill: parent
        radius: root.radius
        color: Qt.rgba(1, 1, 1, 0.02)
        border.color: root.borderColor
        border.width: root.borderWidth
    }

    Item {
        id: contentLayer
        anchors.fill: parent
        anchors.margins: 18
    }
}
