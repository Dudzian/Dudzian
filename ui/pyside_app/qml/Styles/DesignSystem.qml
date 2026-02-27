import QtQuick

QtObject {
    id: root

    // In MainWindow.qml this is set as: themeBridge: theme
    property var themeBridge

    // Local fallback when used without explicit injection
    function _bridge() {
        if (root.themeBridge)
            return root.themeBridge
        if (typeof theme !== "undefined")
            return theme
        return null
    }

    function color(token) {
        var b = _bridge()
        // Touch palette so bindings depend on paletteChanged notifications.
        if (b && b.palette !== undefined) {
            var _p = b.palette
        }
        return (b && b.color) ? b.color(token) : "#ffffff"
    }

    function iconSource(token) {
        var b = _bridge()
        if (b && b.palette !== undefined) {
            var _p = b.palette
        }
        return (b && b.iconUrl) ? b.iconUrl(token) : ""
    }

    function gradientColors(token) {
        var b = _bridge()
        if (b && b.palette !== undefined) {
            var _p = b.palette
        }
        return (b && b.gradient) ? b.gradient(token) : []
    }

    function iconGlyph(token) {
        var b = _bridge()
        if (b && b.palette !== undefined) {
            var _p = b.palette
        }
        return (b && b.iconGlyph) ? b.iconGlyph(token) : ""
    }

    function fontAwesomeFamily() {
        return "Font Awesome 6 Free"
    }
}
