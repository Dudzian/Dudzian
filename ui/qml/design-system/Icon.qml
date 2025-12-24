import QtQuick
import QtQuick.Controls
import Qt5Compat.GraphicalEffects
import ".." as Design
import "fonts/FontAwesomeData.js" as FontAwesomeData

Item {
    id: root
    implicitWidth: size
    implicitHeight: size

    property string name: ""
    property string glyph: ""
    property color color: Design.Palette.textPrimary
    property real size: 20
    property url source: ""

    readonly property bool showsImage: resolvedGlyph.length === 0 && source.toString().length > 0

    readonly property string resolvedGlyph: glyph.length > 0
                                         ? glyph
                                         : (glyphMap[name] ? glyphMap[name] : "")

    readonly property var glyphMap: ({
        "cloud": "\uf0c2",
        "bolt": "\uf0e7",
        "shield": "\uf3ed",
        "refresh": "\uf021",
        "brain": "\uf5dc",
        "chart": "\uf080",
        "layer": "\uf5fd",
        "robot": "\uf544",
        "wand": "\uf72b",
        "globe": "\uf0ac"
    })

    FontLoader {
        id: fontLoader
        source: FontAwesomeData.solidDataUrl()
    }

    Text {
        id: glyphText
        anchors.centerIn: parent
        text: root.resolvedGlyph
        visible: root.resolvedGlyph.length > 0
        color: root.color
        font.pixelSize: root.size
        font.family: fontLoader.name.length > 0 ? fontLoader.name : "Font Awesome 6 Free"
        font.styleName: "Solid"
        renderType: Text.NativeRendering
    }

    Image {
        id: iconImage
        anchors.centerIn: parent
        visible: false
        source: root.showsImage ? root.source : ""
        width: root.size
        height: root.size
        fillMode: Image.PreserveAspectFit
    }

    ColorOverlay {
        anchors.centerIn: parent
        width: root.size
        height: root.size
        source: iconImage
        visible: root.showsImage
        color: root.color
    }
}
