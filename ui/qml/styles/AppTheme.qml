pragma Singleton
import QtQuick

QtObject {
    id: root

    readonly property var defaultPalette: ({
        backgroundPrimary: "#0E1320",
        backgroundOverlay: "#161C2A",
        surfaceStrong: "#1F2536",
        surfaceMuted: "#242B3D",
        surfaceSubtle: "#2C3448",
        textPrimary: "#F5F7FA",
        textSecondary: "#A4ACC4",
        textTertiary: "#7C86A4",
        accent: "#4FA3FF",
        accentMuted: "#3577D4",
        positive: "#3FD0A4",
        negative: "#FF6B6B",
        warning: "#F8C572"
    })

    property var palette: defaultPalette

    readonly property color backgroundPrimary: palette.backgroundPrimary
    readonly property color backgroundOverlay: palette.backgroundOverlay
    readonly property color surfaceStrong: palette.surfaceStrong
    readonly property color surfaceMuted: palette.surfaceMuted
    readonly property color surfaceSubtle: palette.surfaceSubtle

    readonly property color textPrimary: palette.textPrimary
    readonly property color textSecondary: palette.textSecondary
    readonly property color textTertiary: palette.textTertiary

    readonly property color accent: palette.accent
    readonly property color accentMuted: palette.accentMuted
    readonly property color positive: palette.positive
    readonly property color negative: palette.negative
    readonly property color warning: palette.warning

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

    function applyPalette(overrides) {
        var base = Object.assign({}, defaultPalette)
        if (overrides) {
            for (var key in overrides) {
                if (!overrides.hasOwnProperty(key))
                    continue
                if (base.hasOwnProperty(key) && overrides[key])
                    base[key] = overrides[key]
            }
        }
        palette = base
    }
}
