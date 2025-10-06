import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Pane {
    id: footer
    implicitHeight: 44
    padding: 12
    background: Rectangle {
        color: Qt.darker(footer.palette.window, 1.3)
    }

    RowLayout {
        anchors.fill: parent
        spacing: 16

        Label {
            text: qsTr("Status: %1").arg(appController.connectionStatus)
        }

        Label {
            text: qsTr("Samples: %1").arg(ohlcvModel.count)
        }

        Item { Layout.fillWidth: true }

        Label {
            text: Qt.formatDateTime(new Date(), "HH:mm:ss")
            Timer {
                interval: 1000
                running: true
                repeat: true
                onTriggered: parent.text = Qt.formatDateTime(new Date(), "HH:mm:ss")
            }
        }
    }
}
