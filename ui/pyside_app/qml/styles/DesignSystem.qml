import QtQuick

QtObject {
    id: designSystem
    property var themeBridge

    function color(token) {
        return themeBridge ? themeBridge.color(token) : "#FFFFFF";
    }

    function iconSource(token) {
        return themeBridge ? themeBridge.iconUrl(token) : "";
    }

    function gradientColors(token) {
        return themeBridge ? themeBridge.gradient(token) : [];
    }
}
