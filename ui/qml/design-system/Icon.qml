import QtQuick
import QtQuick.Controls
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
        anchors.centerIn: parent
        visible: root.resolvedGlyph.length === 0 && root.source.length > 0
        source: root.source
        width: root.size
        height: root.size
        fillMode: Image.PreserveAspectFit
        color: root.color
    }
}
