import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "exchangePanel"
    property var exchangeConnections: []

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
            text: qsTr("Zarządzanie giełdami")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: exchangeListView
            objectName: "exchangeListView"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.exchangeConnections || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 4

                Label {
                    text: (modelData.exchange || qsTr("Giełda")) + (modelData.symbol ? qsTr(" • %1").arg(modelData.symbol) : "")
                    font.bold: true
                }

                Label {
                    text: modelData.venueSymbol ? qsTr("Instrument: %1").arg(modelData.venueSymbol) : ""
                    visible: !!modelData.venueSymbol
                    color: palette.mid
                }

                Label {
                    text: modelData.status || ""
                    wrapMode: Text.WordWrap
                }

                Label {
                    text: qsTr("Offline: %1 • Automatyzacja: %2 • FPS: %3")
                        .arg(Utils.formatBoolean(modelData.offline))
                        .arg(Utils.formatBoolean(modelData.automationRunning))
                        .arg(modelData.fpsTarget || 0)
                    color: palette.mid
                }

                Rectangle {
                    visible: index < (ListView.view.count - 1)
                    height: 1
                    color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.3)
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                }
            }
        }

        Label {
            visible: (root.exchangeConnections || []).length === 0
            text: qsTr("Brak aktywnych połączeń giełdowych.")
            color: palette.mid
        }
    }
}
