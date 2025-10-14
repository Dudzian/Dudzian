import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

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
            text: qsTr("Instrument: %1").arg(appController.instrumentLabel)
        }

        Label {
            text: qsTr("Status: %1").arg(appController.connectionStatus)
        }

        Label {
            text: qsTr("Samples: %1")
                  .arg(ohlcvModel && ohlcvModel.count !== undefined ? ohlcvModel.count : 0)
        }

        Label {
            text: qsTr("Okna: %1")
                  .arg(((Window.window && Window.window.extraWindowCount !== undefined)
                        ? Window.window.extraWindowCount : 0) + 1)
        }

        Label {
            text: licenseController.licenseActive
                    ? qsTr("Licencja: %1 • ważna do %2 • FP %3")
                          .arg(licenseController.licenseProfile)
                          .arg(licenseController.licenseExpiresAt.length > 0
                                   ? Qt.formatDateTime(new Date(licenseController.licenseExpiresAt), "yyyy-MM-dd")
                                   : qsTr("-")
                          )
                          .arg(licenseController.licenseFingerprint)
                    : qsTr("Licencja: nieaktywna")
            color: licenseController.licenseActive
                    ? palette.windowText
                    : Qt.rgba(0.94, 0.36, 0.32, 1)
        }

        Label {
            text: appController.reduceMotionActive
                    ? qsTr("Animacje: ograniczone")
                    : qsTr("Animacje: pełne")
            color: appController.reduceMotionActive
                    ? Qt.rgba(0.96, 0.74, 0.23, 1)
                    : palette.windowText
        }

        Item { Layout.fillWidth: true }

        Label {
            id: clockLabel
            text: Qt.formatDateTime(new Date(), "HH:mm:ss")

            Timer {
                interval: 1000
                running: true
                repeat: true
                onTriggered: clockLabel.text = Qt.formatDateTime(new Date(), "HH:mm:ss")
            }
        }
    }
}
