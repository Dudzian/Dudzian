import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "marketSentimentPanel"
    property var marketSentiment: ({})

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
            text: qsTr("Sentyment rynku")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            id: summaryGrid
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 16
            rowSpacing: 6

            Label { text: qsTr("Trend") }
            Label {
                text: Utils.formatText(root.marketSentiment.trend, qsTr("n/d"))
                font.bold: true
            }

            Label { text: qsTr("Wynik globalny") }
            Label {
                text: Utils.formatPercent(root.marketSentiment.globalScore || 0, 1)
                font.bold: true
            }

            Label { text: qsTr("Pewność") }
            Label { text: Utils.formatPercent(root.marketSentiment.confidence || 0, 1) }

            Label { text: qsTr("Volatility Index") }
            Label { text: Utils.formatNumber(root.marketSentiment.volatilityIndex, 1) }

            Label { text: qsTr("Przepływy on-chain") }
            Label { text: Utils.formatNumber(root.marketSentiment.onChainFlow, 2) }

            Label { text: qsTr("Sentyment newsów") }
            Label { text: Utils.formatPercent(root.marketSentiment.newsScore || 0, 1) }

            Label { text: qsTr("Aktualizacja") }
            Label {
                text: Utils.formatTimestamp(root.marketSentiment.derivedAt)
                color: palette.mid
            }
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.2)
        }

        Label {
            text: qsTr("Kontrybucje źródeł")
            font.bold: true
        }

        ListView {
            id: sourcesList
            objectName: "marketSentimentSourcesList"
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(180, contentHeight)
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}
            model: root.marketSentiment.sources || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 4

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Label {
                        text: modelData.name || qsTr("Źródło")
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: Utils.formatPercent(modelData.score || 0, 1)
                        font.bold: true
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label {
                        text: qsTr("Waga: %1").arg(Utils.formatPercent(modelData.weight || 0, 1))
                        color: palette.mid
                    }

                    Label {
                        text: modelData.sentiment ? qsTr("Bias: %1").arg(modelData.sentiment) : ""
                        color: palette.mid
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: Utils.formatTimestamp(modelData.updatedAt)
                        color: palette.mid
                    }
                }

                Label {
                    visible: modelData.notes && modelData.notes.length > 0
                    text: modelData.notes
                    color: palette.mid
                    wrapMode: Text.WordWrap
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
                visible: (root.marketSentiment.sources || []).length === 0
                implicitHeight: 24
                Column {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 4

                    Label {
                        text: qsTr("Brak danych o źródłach")
                        color: palette.mid
                    }
                }
            }
        }
    }
}
