import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "licensePanel"
    property var licenseStatus: ({})

    Layout.fillWidth: true
    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 8

        Label {
            text: qsTr("Status licencji")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Aktywna"); font.bold: true }
            Label { text: Utils.formatBoolean(root.licenseStatus.active) }

            Label { text: qsTr("Edycja"); font.bold: true }
            Label { text: root.licenseStatus.edition || qsTr("n/d") }

            Label { text: qsTr("ID licencji"); font.bold: true }
            Label { text: root.licenseStatus.licenseId || qsTr("n/d") }

            Label { text: qsTr("Właściciel"); font.bold: true }
            Label { text: root.licenseStatus.holderName || qsTr("n/d") }

            Label { text: qsTr("Siedziby"); font.bold: true }
            Label {
                text: root.licenseStatus.environments && root.licenseStatus.environments.length
                    ? root.licenseStatus.environments.join(", ")
                    : qsTr("brak")
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Moduły"); font.bold: true }
            Label {
                text: root.licenseStatus.modules && root.licenseStatus.modules.length
                    ? root.licenseStatus.modules.join(", ")
                    : qsTr("brak")
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Runtime"); font.bold: true }
            Label {
                text: root.licenseStatus.runtime && root.licenseStatus.runtime.length
                    ? root.licenseStatus.runtime.join(", ")
                    : qsTr("brak")
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Wsparcie"); font.bold: true }
            Label {
                text: root.licenseStatus.maintenanceActive
                    ? qsTr("Aktywne do %1").arg(root.licenseStatus.maintenanceUntil || qsTr("n/d"))
                    : qsTr("Nieaktywne")
            }

            Label { text: qsTr("Trial"); font.bold: true }
            Label {
                text: root.licenseStatus.trialActive
                    ? qsTr("Do %1").arg(root.licenseStatus.trialExpiresAt || qsTr("n/d"))
                    : qsTr("Brak")
            }
        }
    }
}
