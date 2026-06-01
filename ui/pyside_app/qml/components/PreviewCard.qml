import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: card
    default property alias content: extraContent.data
    property var designSystem
    property alias cardBody: extraContent
    property int cardPadding: 16
    property string title: ""
    property string description: ""
    property color cardColor: designSystem ? designSystem.color("surface") : "#1b2130"

    Layout.fillWidth: true
    radius: 18
    color: card.cardColor
    border.color: designSystem ? designSystem.color("border") : "#30384a"
    border.width: 1
    implicitHeight: cardContent.implicitHeight + cardPadding * 2

    ColumnLayout {
        id: cardContent
        anchors.fill: parent
        anchors.margins: card.cardPadding
        spacing: 8

        RowLayout {
            visible: card.title.length > 0
            Layout.fillWidth: true
            spacing: 8
            Rectangle {
                objectName: "previewCardTitleAccentBar"
                width: 3
                height: Math.max(18, cardTitle.implicitHeight)
                radius: 2
                color: card.designSystem ? card.designSystem.color("accent") : "#55c7ff"
            }
            Label {
                id: cardTitle
                text: card.title
                font.bold: true
                font.pixelSize: 16
                color: card.designSystem ? card.designSystem.color("textPrimary") : "#f5f7ff"
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }

        Label {
            visible: card.description.length > 0
            text: card.description
            color: card.designSystem ? card.designSystem.color("textSecondary") : "#b5bfd8"
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        ColumnLayout {
            id: extraContent
            objectName: "previewCardExtraContent"
            property var designSystem: card.designSystem
            Layout.fillWidth: true
            spacing: 8
        }
    }
}
