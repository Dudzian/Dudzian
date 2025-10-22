import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    objectName: "aiPanel"
    property var configuration: ({})

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
            text: qsTr("Konfiguracja AI")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 8
            Layout.fillWidth: true

            Label { text: qsTr("Polityka"); font.bold: true }
            Label { text: root.configuration.policy || qsTr("brak") }

            Label { text: qsTr("Okno decyzyjne"); font.bold: true }
            Label { text: root.configuration.decision_window || qsTr("n/d") }

            Label { text: qsTr("Profil ryzyka"); font.bold: true }
            Label { text: root.configuration.risk_profile || qsTr("n/d") }

            Label { text: qsTr("Rewizja modelu"); font.bold: true }
            Label { text: root.configuration.modelRevision || qsTr("n/d") }
        }

        ColumnLayout {
            spacing: 4
            Layout.fillWidth: true

            Label {
                text: qsTr("Nadpisania profili")
                font.bold: true
            }

            Repeater {
                model: root.configuration.overrides || []
                delegate: Label {
                    text: qsTr("• %1 – maxDD %2 • TP %3")
                        .arg(modelData.profile || qsTr("profil"))
                        .arg(modelData.maxDrawdown !== undefined ? Number(modelData.maxDrawdown * 100).toFixed(1) + "%" : "–")
                        .arg(modelData.takeProfit !== undefined ? Number(modelData.takeProfit * 100).toFixed(1) + "%" : "–")
                }
            }

            Label {
                visible: !(root.configuration.overrides && root.configuration.overrides.length)
                text: qsTr("Brak nadpisań.")
                color: palette.mid
            }
        }

        ColumnLayout {
            spacing: 4
            Layout.fillWidth: true

            Label {
                text: qsTr("Wykorzystywane cechy")
                font.bold: true
            }

            Label {
                text: root.configuration.features && root.configuration.features.length
                    ? root.configuration.features.join(", ")
                    : qsTr("Brak danych")
                wrapMode: Text.WordWrap
            }
        }
    }
}
