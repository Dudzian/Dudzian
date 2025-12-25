import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "../components" as Components

Item {
    id: root
    property var designSystem
    property var strategyManagementController
    property var layoutController

    function presetsModel() {
        return strategyManagementController ? strategyManagementController.presets : []
    }

    function assignedPortfolios(entry) {
        if (!entry || !entry.assignedPortfolios)
            return []
        return entry.assignedPortfolios
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Label {
                text: qsTr("Manager strategii & Marketplace")
                font.pixelSize: 20
                font.bold: true
                color: designSystem ? designSystem.color("textPrimary") : "#fff"
            }
            Item { Layout.fillWidth: true }
            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Odśwież")
                iconName: "refresh"
                enabled: strategyManagementController && !strategyManagementController.busy
                onClicked: strategyManagementController.refreshMarketplace()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 18
            color: designSystem ? designSystem.color("surfaceElevated") : "#202433"
            opacity: 0.95
            border.color: designSystem ? designSystem.color("border") : "#30384a"
            border.width: 1
            Layout.margins: 4
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8

                Label {
                    text: strategyManagementController ? strategyManagementController.statusMessage : qsTr("Brak statusu")
                    color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                    wrapMode: Text.WordWrap
                }

                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true
                    TextField {
                        id: portfolioInput
                        Layout.fillWidth: true
                        placeholderText: qsTr("ID portfela docelowego")
                    }
                    Components.IconButton {
                        designSystem: designSystem
                        subtle: true
                        text: qsTr("Otwórz w układzie")
                        iconName: "package"
                        onClicked: {
                            if (layoutController)
                                layoutController.setPanelVisibility("strategyManagerPanel", true)
                        }
                    }
                }
            }
        }

        Loader {
            active: strategyManagementController && strategyManagementController.busy
            Layout.preferredHeight: active ? 4 : 0
            sourceComponent: Rectangle {
                Layout.fillWidth: true
                height: 4
                gradient: Gradient {
                    GradientStop { position: 0; color: designSystem ? designSystem.color("gradientHeroStart") : "#3f51b5" }
                    GradientStop { position: 1; color: designSystem ? designSystem.color("gradientHeroEnd") : "#00bcd4" }
                }
                SequentialAnimation on opacity {
                    loops: Animation.Infinite
                    NumberAnimation { from: 0.3; to: 1.0; duration: 800 }
                    NumberAnimation { from: 1.0; to: 0.3; duration: 800 }
                }
            }
        }

        ListView {
            id: presetList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            model: presetsModel()
            delegate: Rectangle {
                width: ListView.view.width
                color: designSystem ? designSystem.color("surface") : "#1c2030"
                radius: 16
                border.color: designSystem ? designSystem.color("border") : "#2c3348"
                border.width: 1
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 6
                    Label {
                        text: modelData.name + " • v" + (modelData.version || "-")
                        font.bold: true
                        color: designSystem ? designSystem.color("textPrimary") : "#fff"
                    }
                    Label {
                        text: modelData.summary || qsTr("Brak podsumowania")
                        color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                        wrapMode: Text.WordWrap
                    }
                    Item {
                        Layout.fillWidth: true
                        visible: modelData.userPreferences && modelData.userPreferences.length > 0
                        implicitHeight: personaSection.implicitHeight + 12
                        Rectangle {
                            id: personaFrame
                            anchors.fill: parent
                            radius: 14
                            color: designSystem ? designSystem.color("surfaceElevated") : "#23283b"
                            opacity: 0.92
                            border.color: designSystem ? designSystem.color("border") : "#30384a"
                            border.width: 1
                        }
                        MultiEffect {
                            anchors.fill: personaFrame
                            source: personaFrame
                            blurEnabled: true
                            blur: 1.0
                            blurMax: 18
                            saturation: 0.9
                            brightness: 0.04
                        }
                        ColumnLayout {
                            id: personaSection
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 4
                            Repeater {
                                model: modelData.userPreferences || []
                                delegate: ColumnLayout {
                                    spacing: 2
                                    Label {
                                        text: qsTr("Persona: %1").arg(modelData.persona || qsTr("profil"))
                                        font.bold: true
                                        color: designSystem ? designSystem.color("textPrimary") : "#fff"
                                    }
                                    RowLayout {
                                        spacing: 6
                                        Text {
                                            text: designSystem ? designSystem.iconGlyph("shield") : "\uf3ed"
                                            font.family: designSystem ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                                            font.pixelSize: 14
                                            color: designSystem ? designSystem.color("accent") : "#4fc3f7"
                                        }
                                        Label {
                                            text: qsTr("Ryzyko: %1").arg(modelData.risk_target || qsTr("brak"))
                                            color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                                        }
                                        Text {
                                            text: designSystem ? designSystem.iconGlyph("package") : "\uf49e"
                                            font.family: designSystem ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                                            font.pixelSize: 14
                                            color: designSystem ? designSystem.color("accent") : "#4fc3f7"
                                        }
                                        Label {
                                            text: modelData.recommended_budget
                                                  ? qsTr("Budżet: %1 USD").arg(Number(modelData.recommended_budget).toLocaleString(Qt.locale(), 'f', 0))
                                                  : qsTr("Budżet: brak")
                                            color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                                        }
                                    }
                                    RowLayout {
                                        spacing: 6
                                        Text {
                                            property string fallbackGlyph: "\uf017"
                                            text: designSystem && designSystem.iconGlyph("clock") && designSystem.iconGlyph("clock").length > 0
                                                  ? designSystem.iconGlyph("clock")
                                                  : fallbackGlyph
                                            font.family: designSystem ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                                            font.pixelSize: 14
                                            color: designSystem ? designSystem.color("accent") : "#4fc3f7"
                                        }
                                        Label {
                                            text: qsTr("Horyzont: %1").arg(modelData.holding_period || qsTr("brak"))
                                            color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Label {
                        text: modelData.signatureVerified ? qsTr("Podpis zweryfikowany") : qsTr("Podpis NIEZWERYFIKOWANY")
                        color: modelData.signatureVerified ? designSystem.color("textSecondary") : "#ff8a80"
                    }
                    Label {
                        text: modelData.license && modelData.license.status
                              ? qsTr("Licencja: %1").arg(modelData.license.status)
                              : qsTr("Brak licencji w pakiecie")
                        color: designSystem ? designSystem.color("textSecondary") : "#b5bfd8"
                    }
                    Flow {
                        width: parent.width
                        spacing: 4
                        Repeater {
                            model: assignedPortfolios(modelData)
                            delegate: Rectangle {
                                radius: 12
                                color: designSystem ? designSystem.color("surfaceElevated") : "#2b3045"
                                border.color: designSystem ? designSystem.color("border") : "#3d4560"
                                border.width: 1
                                Text {
                                    id: portfolioLabel
                                    anchors.centerIn: parent
                                    text: modelData
                                    color: designSystem ? designSystem.color("textPrimary") : "#fff"
                                }
                                implicitWidth: Math.max(portfolioLabel.implicitWidth + 16, 48)
                                implicitHeight: 24
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Zainstaluj i przypisz")
                            iconName: "package"
                            enabled: strategyManagementController && !strategyManagementController.busy
                            onClicked: {
                                var result = strategyManagementController.activateAndAssign(modelData.presetId, portfolioInput.text)
                                if (result && result.assignmentError)
                                    console.warn("Assignment error", result.assignmentError)
                            }
                        }
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Tylko przypisz")
                            iconName: "fingerprint"
                            subtle: true
                            enabled: strategyManagementController && !strategyManagementController.busy
                            onClicked: strategyManagementController.assignPresetToPortfolio(modelData.presetId, portfolioInput.text)
                        }
                        Item { Layout.fillWidth: true }
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Odśwież dane")
                            iconName: "refresh"
                            subtle: true
                            onClicked: strategyManagementController.refreshMarketplace()
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        if (strategyManagementController && strategyManagementController.refreshMarketplace)
            strategyManagementController.refreshMarketplace()
    }
}
