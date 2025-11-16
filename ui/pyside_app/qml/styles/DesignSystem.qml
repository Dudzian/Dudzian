import QtQuick
import "../../../qml/design-system/fonts/FontAwesomeData.js" as FontAwesomeData

QtObject {
    id: designSystem
    property var themeBridge

    FontLoader {
        id: fontAwesomeLoader
        source: FontAwesomeData.solidDataUrl()
    }

    function color(token) {
        return themeBridge ? themeBridge.color(token) : "#FFFFFF";
    }

    function iconSource(token) {
        return themeBridge ? themeBridge.iconUrl(token) : "";
    }

    function iconGlyph(token) {
        return themeBridge && themeBridge.iconGlyph ? themeBridge.iconGlyph(token) : "";
    }

    function fontAwesomeFamily() {
        return fontAwesomeLoader.name && fontAwesomeLoader.name.length > 0 ? fontAwesomeLoader.name : "Font Awesome 6 Free";
    }

    function gradientColors(token) {
        return themeBridge ? themeBridge.gradient(token) : [];
    }
}
