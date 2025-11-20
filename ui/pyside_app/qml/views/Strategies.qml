import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles
import "../components" as Components

ColumnLayout {
    id: root
    property var runtimeService
    property var designSystem

    spacing: 12

    Label {
        text: qsTr("Strategie i parametry")
        font.bold: true
        color: designSystem.color("textPrimary")
    }

    ListView {
        id: strategyList
        Layout.fillWidth: true
        Layout.fillHeight: true
        spacing: 12
        clip: true
        model: runtimeService ? runtimeService.strategyConfigs : []

        delegate: Rectangle {
            width: ListView.view.width
            radius: 14
            color: index % 2 === 0 ? designSystem.color("surface") : designSystem.color("surfaceMuted")
            border.color: designSystem.color("border")
            border.width: 1
            opacity: 0.95
            height: container.implicitHeight + 20

            property var workingCopy: JSON.parse(JSON.stringify(modelData))

            ColumnLayout {
                id: container
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    spacing: 8
                    Label {
                        text: workingCopy.name || workingCopy.id
                        font.bold: true
                        color: designSystem.color("textPrimary")
                        Layout.fillWidth: true
                    }
                    Label {
                        text: qsTr("Tryb: %1").arg(workingCopy.mode || "?")
                        color: designSystem.color("textSecondary")
                    }
                    Label {
                        text: qsTr("Profil: %1").arg(workingCopy.profile || "balanced")
                        color: designSystem.color("textSecondary")
                    }
                }

                Repeater {
                    model: Object.keys(workingCopy.params || {})
                    delegate: RowLayout {
                        spacing: 6
                        Label {
                            text: modelData
                            color: designSystem.color("textSecondary")
                            Layout.preferredWidth: 160
                        }
                        TextField {
                            id: valueField
                            text: String((workingCopy.params || {})[modelData])
                            Layout.fillWidth: true
                            color: designSystem.color("textPrimary")
                            background: Rectangle {
                                implicitHeight: 36
                                radius: 8
                                color: designSystem.color("surface")
                                border.color: designSystem.color("border")
                            }
                            onEditingFinished: {
                                var asNumber = Number(text)
                                workingCopy.params[modelData] = isNaN(asNumber) ? text : asNumber
                            }
                        }
                    }
                }

                RowLayout {
                    spacing: 8
                    Components.IconButton {
                        designSystem: designSystem
                        text: qsTr("Zapisz zmiany")
                        iconName: "save"
                        onClicked: {
                            if (runtimeService && runtimeService.saveStrategyConfig) {
                                const result = runtimeService.saveStrategyConfig(workingCopy.id, workingCopy)
                                statusLabel.text = result && result.message ? result.message : qsTr("Zapisano")
                            }
                        }
                    }
                    Label {
                        id: statusLabel
                        color: designSystem.color("textSecondary")
                    }
                }
            }
        }
    }

    Rectangle {
        Layout.fillWidth: true
        height: 1
        color: designSystem.color("border")
        opacity: 0.3
    }

    Components.IconButton {
        designSystem: designSystem
        text: qsTr("Odśwież strategie")
        iconName: "refresh"
        subtle: true
        onClicked: runtimeService && runtimeService.loadStrategyConfigs()
    }
}
