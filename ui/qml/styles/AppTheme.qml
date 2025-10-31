pragma Singleton
import QtQuick

QtObject {
    readonly property color backgroundPrimary: "#0E1320"
    readonly property color backgroundOverlay: "#161C2A"
    readonly property color surfaceStrong: "#1F2536"
    readonly property color surfaceMuted: "#242B3D"
    readonly property color surfaceSubtle: "#2C3448"

    readonly property color textPrimary: "#F5F7FA"
    readonly property color textSecondary: "#A4ACC4"
    readonly property color textTertiary: "#7C86A4"

    readonly property color accent: "#4FA3FF"
    readonly property color accentMuted: "#3577D4"
    readonly property color positive: "#3FD0A4"
    readonly property color negative: "#FF6B6B"
    readonly property color warning: "#F8C572"

    readonly property int radiusSmall: 6
    readonly property int radiusMedium: 10
    readonly property int radiusLarge: 14

    readonly property int spacingXs: 4
    readonly property int spacingSm: 8
    readonly property int spacingMd: 12
    readonly property int spacingLg: 18
    readonly property int spacingXl: 24

    readonly property int fontSizeCaption: 11
    readonly property int fontSizeBody: 14
    readonly property int fontSizeSubtitle: 16
    readonly property int fontSizeTitle: 20
    readonly property int fontSizeHeadline: 24

    readonly property string fontFamily: "Inter"
    readonly property string monoFontFamily: "Fira Code"

    function cardBackground(opacity) {
        var value = opacity === undefined ? 1.0 : opacity
        return Qt.rgba(surfaceStrong.r, surfaceStrong.g, surfaceStrong.b, value)
    }

    function overlayBackground(opacity) {
        var value = opacity === undefined ? 0.86 : opacity
        return Qt.rgba(backgroundPrimary.r, backgroundPrimary.g, backgroundPrimary.b, value)
    }

    function iconSource(name) {
        return "qrc:/icons/%1.svg".arg(name)
    }
}
