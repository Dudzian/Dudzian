import QtQuick
import Styles 1.0 as StylesModule

// Wrapper to allow path-based imports ("import \"styles\" as Styles") while
// delegating all behaviour to the module-backed implementation. This keeps the
// interface identical, including the themeBridge property and helper methods.
QtObject {
    id: designSystem

    property var themeBridge
    // Keep a single underlying implementation that receives the same bridge.
    property QtObject _impl: StylesModule.DesignSystem {
        themeBridge: designSystem.themeBridge
    }

    function color(token) {
        return _impl.color(token)
    }

    function iconSource(token) {
        return _impl.iconSource(token)
    }

    function iconGlyph(token) {
        return _impl.iconGlyph(token)
    }

    function fontAwesomeFamily() {
        return _impl.fontAwesomeFamily()
    }

    function gradientColors(token) {
        return _impl.gradientColors(token)
    }
}
