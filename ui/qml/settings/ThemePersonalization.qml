import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 480
    implicitHeight: 260

    property var userProfiles: (typeof appController !== "undefined" && appController) ? appController.userProfiles : null
    property string profileId: userProfiles ? userProfiles.activeProfileId : "default"
    property var availableThemes: userProfiles ? userProfiles.availableThemes : []
    property var paletteOverrides: ({})
    property var editablePaletteRoles: [
        ({ role: "accent", label: qsTr("Akcent") }),
        ({ role: "accentMuted", label: qsTr("Akcent przygaszony") }),
        ({ role: "positive", label: qsTr("Pozytywny") }),
        ({ role: "negative", label: qsTr("Negatywny") }),
        ({ role: "warning", label: qsTr("Ostrzeżenie") })
    ]
    property int paletteVersion: 0

    signal paletteVersionChanged(int version)

    objectName: "themePersonalization"

    function refreshThemes() {
        availableThemes = userProfiles ? userProfiles.availableThemes : []
        refreshPalette()
    }

    function refreshPalette() {
        paletteOverrides = userProfiles ? userProfiles.paletteOverrides(profileId) : {}
        paletteVersion = paletteVersion + 1
        paletteVersionChanged(paletteVersion)
    }

    onUserProfilesChanged: refreshThemes()
    onProfileIdChanged: refreshPalette()

    Component.onCompleted: refreshThemes()

    Connections {
        target: userProfiles
        ignoreUnknownSignals: true
        function onProfilesChanged() { refreshThemes() }
        function onActiveProfileChanged() { refreshThemes() }
        function onThemePaletteChanged() { refreshPalette() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        Label {
            text: qsTr("Personalizacja motywu")
            font.pixelSize: Styles.AppTheme.fontSizeTitle
            font.bold: true
            font.family: Styles.AppTheme.fontFamily
            color: Styles.AppTheme.textPrimary
        }

        Flow {
            id: themeFlow
            objectName: "themeFlow"
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: Styles.AppTheme.spacingSm

            Repeater {
                id: themeRepeater
                objectName: "themeRepeater"
                model: availableThemes

                delegate: Frame {
                    required property string modelData
                    width: 140
                    height: 120
                    background: Rectangle {
                        radius: Styles.AppTheme.radiusMedium
                        color: Styles.AppTheme.cardBackground(0.82)
                        border.color: Qt.rgba(1, 1, 1, 0.06)
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: Styles.AppTheme.spacingSm
                        spacing: Styles.AppTheme.spacingXs
                        objectName: "themeCard"

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 56
                            radius: Styles.AppTheme.radiusSmall
                            color: {
                                if (!userProfiles)
                                    return Styles.AppTheme.accent
                                var palette = userProfiles.themePalette(modelData)
                                return palette && palette.accent ? palette.accent : Styles.AppTheme.accent
                            }
                        }

                        Label {
                            text: displayName(modelData)
                            font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                            color: Styles.AppTheme.textPrimary
                            horizontalAlignment: Text.AlignHCenter
                            Layout.fillWidth: true
                        }

                        Button {
                            objectName: "themeButton_" + modelData
                            text: isActiveTheme(modelData) ? qsTr("Aktywny") : qsTr("Ustaw")
                            enabled: !!userProfiles && !isActiveTheme(modelData)
                            onClicked: {
                                if (userProfiles)
                                    userProfiles.applyTheme(profileId, modelData)
                            }
                        }
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Niestandardowe kolory")
            Layout.fillWidth: true
            Layout.preferredHeight: Math.max(160, overridesColumn.implicitHeight + Styles.AppTheme.spacingLg * 2)
            background: Rectangle {
                color: Styles.AppTheme.cardBackground(0.9)
                radius: Styles.AppTheme.radiusLarge
            }

            ColumnLayout {
                id: overridesColumn
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingLg
                spacing: Styles.AppTheme.spacingSm

                Repeater {
                    model: editablePaletteRoles
                    delegate: RowLayout {
                        required property var modelData
                        property string role: modelData.role
                        property string currentValue: root.overrideValue(role)

                        spacing: Styles.AppTheme.spacingSm
                        Layout.fillWidth: true

                        Connections {
                            target: root
                            function onPaletteVersionChanged() {
                                currentValue = root.overrideValue(role)
                                if (colorInput.text !== currentValue)
                                    colorInput.text = currentValue
                            }
                        }

                        Label {
                            text: modelData.label
                            Layout.fillWidth: true
                            font.pixelSize: Styles.AppTheme.fontSizeBody
                            color: Styles.AppTheme.textPrimary
                        }

                        Rectangle {
                            width: 36
                            height: 24
                            radius: Styles.AppTheme.radiusSmall
                            border.color: Qt.rgba(1, 1, 1, 0.1)
                            color: root.paletteColor(role)
                        }

                        TextField {
                            id: colorInput
                            objectName: "overrideField_" + role
                            text: currentValue
                            Layout.preferredWidth: 120
                            placeholderText: root.defaultColor(role)
                            inputMethodHints: Qt.ImhPreferUppercase
                            validator: RegularExpressionValidator {
                                regularExpression: /#?([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})/
                            }
                            onEditingFinished: {
                                if (!userProfiles)
                                    return
                                const trimmed = text.trim()
                                root.applyOverride(role, trimmed)
                            }
                        }

                        Button {
                            objectName: "resetOverrideButton_" + role
                            text: qsTr("Reset")
                            enabled: root.overrideExists(role)
                            onClicked: root.applyOverride(role, "")
                        }
                    }
                }

                Button {
                    objectName: "clearOverridesButton"
                    text: qsTr("Wyczyść wszystkie niestandardowe kolory")
                    enabled: root.hasOverrides()
                    onClicked: {
                        if (!userProfiles)
                            return
                        userProfiles.clearPaletteOverrides(profileId)
                        root.refreshPalette()
                    }
                }
            }
        }
    }

    function displayName(themeId) {
        switch (themeId) {
        case "midnight": return qsTr("Midnight")
        case "aurora": return qsTr("Aurora")
        case "solarized": return qsTr("Solarized")
        default: return themeId
        }
    }

    function isActiveTheme(themeId) {
        if (!userProfiles)
            return false
        var profile = userProfiles.activeProfile || {}
        return profile.theme === themeId
    }

    function overrideValue(role) {
        if (!paletteOverrides)
            return ""
        const value = paletteOverrides[role]
        return value ? value : ""
    }

    function paletteColor(role) {
        if (!userProfiles)
            return Styles.AppTheme.accent
        const palette = userProfiles.activeThemePalette || {}
        return palette[role] ? palette[role] : Styles.AppTheme.accent
    }

    function defaultColor(role) {
        if (!userProfiles)
            return Styles.AppTheme.accent
        const profile = userProfiles.activeProfile || {}
        const theme = profile.theme || "midnight"
        const base = userProfiles.themePalette(theme) || {}
        return base[role] ? base[role] : Styles.AppTheme.accent
    }

    function hasOverrides() {
        return paletteOverrides && Object.keys(paletteOverrides).length > 0
    }

    function overrideExists(role) {
        return paletteOverrides && Object.prototype.hasOwnProperty.call(paletteOverrides, role)
    }

    function applyOverride(role, rawValue) {
        if (!userProfiles)
            return
        var trimmed = (rawValue || "").trim()
        if (trimmed.length > 0 && trimmed.charAt(0) !== "#")
            trimmed = "#" + trimmed
        userProfiles.setPaletteOverride(profileId, role, trimmed)
        refreshPalette()
    }
}
