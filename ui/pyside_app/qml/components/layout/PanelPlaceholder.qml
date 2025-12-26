import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import QtQuick.Controls.impl

Rectangle {
    id: root
    required property string panelId
    required property string title
    required property Component contentComponent
    property string iconName: ""
    property var designSystem
    property Item dockManager
    property int columnIndex: 0
    property int panelOrder: 0
    radius: 20
    color: designSystem ? designSystem.color("surface") : Qt.rgba(0.12, 0.12, 0.12, 1)
    border.color: designSystem ? designSystem.color("border") : Qt.rgba(1, 1, 1, 0.05)
    border.width: 1
    opacity: dragHandler.active ? 0.85 : 1

    layer.enabled: true
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowBlur: dragHandler.active ? 0.6 : 0.4
        shadowColor: Qt.rgba(0, 0, 0, dragHandler.active ? 0.6 : 0.35)
        shadowHorizontalOffset: 0
        shadowVerticalOffset: dragHandler.active ? 12 : 6
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            spacing: 8
            IconImage {
                source: designSystem ? designSystem.iconSource(iconName) : ""
                width: 20
                height: 20
                fillMode: Image.PreserveAspectFit
                visible: source.length > 0
                color: designSystem ? designSystem.color("accent") : "#7dd3fc"
            }
            Label {
                text: title
                font.bold: true
                color: designSystem ? designSystem.color("textPrimary") : "white"
                Layout.fillWidth: true
            }
            ToolButton {
                text: qsTr("Ukryj")
                icon.name: "close"
                visible: dockManager
                onClicked: dockManager ? dockManager.hidePanel(root.panelId) : undefined
            }
        }

        Loader {
            id: contentLoader
            Layout.fillWidth: true
            sourceComponent: contentComponent
            active: contentComponent !== null
        }
    }

    DragHandler {
        id: dragHandler
        target: null
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
    }

    Drag.active: dragHandler.active
    Drag.hotSpot.x: width / 2
    Drag.hotSpot.y: 32
    Drag.mimeData: ({ "panelId": panelId })
    Drag.source: root
}
