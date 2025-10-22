import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "newsFeedPanel"
    property var newsHeadlines: []

    Layout.fillWidth: true
    Layout.fillHeight: true

    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            text: qsTr("Strumień newsów")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: newsList
            objectName: "newsHeadlinesList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}
            model: root.newsHeadlines || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label {
                        text: modelData.title || qsTr("Nagłówek")
                        font.bold: true
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Label {
                        text: modelData.sentiment ? qsTr("%1").arg(modelData.sentiment) : ""
                        color: palette.highlight
                        font.bold: true
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Label {
                        text: modelData.source ? qsTr("Źródło: %1").arg(modelData.source) : ""
                        color: palette.mid
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: Utils.formatTimestamp(modelData.publishedAt)
                        color: palette.mid
                    }
                }

                Label {
                    text: Utils.formatText(modelData.summary, qsTr("Brak podsumowania"))
                    wrapMode: Text.WordWrap
                }

                Label {
                    visible: modelData.url && modelData.url.length > 0
                    text: modelData.url
                    color: palette.highlight
                    font.underline: true
                    elide: Text.ElideRight
                }

                Rectangle {
                    visible: index < (ListView.view.count - 1)
                    Layout.topMargin: 8
                    Layout.fillWidth: true
                    height: 1
                    color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.15)
                }
            }

            footer: Item {
                visible: (root.newsHeadlines || []).length === 0
                implicitHeight: 24
                Label {
                    anchors.centerIn: parent
                    text: qsTr("Brak wiadomości do wyświetlenia")
                    color: palette.mid
                }
            }
        }
    }
}
