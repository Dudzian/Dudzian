import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var controller: (typeof privacySettingsController !== "undefined" ? privacySettingsController : null)

    Component.onCompleted: {
        if (controller && controller.refresh)
            controller.refresh()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: qsTr("Anonimowa telemetria pozwala na ulepszanie produktu na podstawie zanonimizowanych danych użycia. Dane są przetwarzane lokalnie i mogą zostać wyeksportowane ręcznie.")
        }

        GroupBox {
            title: qsTr("Zgoda użytkownika")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Switch {
                        id: optInSwitch
                        text: qsTr("Włącz anonimową telemetrię")
                        checked: controller ? controller.optInEnabled : false
                        onToggled: {
                            if (!controller)
                                return
                            controller.setOptIn(checked, fingerprintField.text)
                        }
                    }
                    TextField {
                        id: fingerprintField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Fingerprint sprzętowy (opcjonalnie)")
                    }
                    Button {
                        text: qsTr("Odśwież identyfikator")
                        enabled: controller && controller.optInEnabled
                        onClicked: {
                            if (controller)
                                controller.refreshPseudonym(fingerprintField.text)
                        }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    text: controller && controller.pseudonym.length > 0
                          ? qsTr("Pseudonim: %1").arg(controller.pseudonym)
                          : qsTr("Pseudonim zostanie wygenerowany po wyrażeniu zgody.")
                }

                Label {
                    Layout.fillWidth: true
                    text: qsTr("Identyfikator instalacji: %1").arg(controller ? controller.installationId : "")
                    color: palette.mid
                }

                Label {
                    Layout.fillWidth: true
                    text: controller && controller.lastExportAt.length > 0
                          ? qsTr("Ostatni eksport: %1").arg(controller.lastExportAt)
                          : qsTr("Brak wyeksportowanych raportów.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Kolejka danych")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                Label {
                    Layout.fillWidth: true
                    text: qsTr("Liczba zdarzeń w kolejce: %1").arg(controller ? controller.queuedEvents : 0)
                }

                Label {
                    Layout.fillWidth: true
                    text: qsTr("Lokalizacja kolejki: %1").arg(controller ? controller.queuePath : "")
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }

                TextArea {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    readOnly: true
                    text: controller ? controller.previewJson : "[]"
                    wrapMode: Text.WordWrap
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Button {
                        text: qsTr("Eksportuj do pliku")
                        enabled: controller && controller.queuedEvents > 0
                        onClicked: {
                            if (!controller)
                                return
                            const path = controller.exportTelemetry()
                            if (path && path.length > 0)
                                exportInfo.text = qsTr("Wyeksportowano do: %1").arg(path)
                        }
                    }
                    Button {
                        text: qsTr("Wyczyść kolejkę")
                        enabled: controller && controller.queuedEvents > 0
                        onClicked: {
                            if (controller)
                                controller.clearQueue()
                        }
                    }
                    Item { Layout.fillWidth: true }
                }

                Label {
                    id: exportInfo
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    color: palette.mid
                }
            }
        }

        Item { Layout.fillHeight: true }
    }
}
