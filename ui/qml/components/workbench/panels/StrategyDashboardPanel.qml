import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    objectName: "strategyPanel"
    property var schedulerEntries: []

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
            text: qsTr("Dashboard strategii")
            font.pointSize: 15
            font.bold: true
        }

        ListView {
            id: strategyListView
            objectName: "strategyListView"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 12
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { }
            model: root.schedulerEntries || []

            delegate: ColumnLayout {
                width: ListView.view ? ListView.view.width : parent.width
                spacing: 4

                Label {
                    text: (modelData.name || qsTr("Strategia")) + (modelData.enabled === false ? qsTr(" (wyłączona)") : "")
                    font.bold: true
                }

                Label {
                    text: qsTr("Harmonogramy: %1 • Strefa: %2")
                        .arg(modelData.scheduleCount || 0)
                        .arg(modelData.timezone || qsTr("brak"))
                    color: palette.mid
                }

                Label {
                    visible: !!modelData.nextRun
                    text: modelData.nextRun ? qsTr("Następne uruchomienie: %1").arg(modelData.nextRun) : ""
                    color: palette.mid
                }

                Label {
                    visible: !!modelData.notes
                    text: modelData.notes
                    wrapMode: Text.WordWrap
                    color: palette.buttonText
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
            visible: (root.schedulerEntries || []).length === 0
            text: qsTr("Brak zarejestrowanych strategii w schedulerze.")
            color: palette.mid
        }
    }
}
