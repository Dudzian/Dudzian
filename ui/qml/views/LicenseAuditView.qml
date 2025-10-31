import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var controller: typeof licenseAuditController !== "undefined" ? licenseAuditController : null
    property string exportStatus: ""

    implicitWidth: 720
    implicitHeight: 540

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            Label {
                text: controller && controller.lastUpdated
                      ? qsTr("Ostatnia aktualizacja: %1").arg(controller.lastUpdated)
                      : qsTr("Raport nie został jeszcze pobrany")
            }

            Item { Layout.fillWidth: true }

            Button {
                id: refreshButton
                text: qsTr("Odśwież")
                enabled: controller && !controller.busy
                icon.name: "view-refresh"
                onClicked: controller && controller.refreshReport()
            }
        }

        GroupBox {
            title: qsTr("Podsumowanie")
            Layout.fillWidth: true

            GridLayout {
                columns: 2
                columnSpacing: 12
                rowSpacing: 6
                anchors.fill: parent
                anchors.margins: 8

                Label { text: qsTr("Aktywacje") }
                Label {
                    text: controller && controller.summary.total_activations !== undefined
                          ? controller.summary.total_activations
                          : 0
                }

                Label { text: qsTr("Urządzenia") }
                Label {
                    text: controller && controller.summary.unique_devices !== undefined
                          ? controller.summary.unique_devices
                          : 0
                }

                Label { text: qsTr("ID licencji") }
                Label {
                    text: controller && controller.summary.license_id ? controller.summary.license_id : qsTr("brak")
                    font.family: "monospace"
                }

                Label { text: qsTr("Edycja") }
                Label {
                    text: controller && controller.summary.edition ? controller.summary.edition : qsTr("brak")
                }

                Label { text: qsTr("Ostatnia aktywacja") }
                Label {
                    text: controller && controller.summary.latest_activation
                          ? controller.summary.latest_activation
                          : qsTr("brak")
                    font.family: "monospace"
                }
            }
        }

        GroupBox {
            title: qsTr("Historia aktywacji")
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: activationList
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                model: controller ? controller.activations : []

                delegate: Frame {
                    required property var modelData
                    Layout.fillWidth: true
                    padding: 8

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 4

                        Label {
                            text: qsTr("Data: %1").arg(modelData.timestamp || qsTr("brak"))
                            font.family: "monospace"
                        }

                        Label {
                            text: qsTr("Licencja: %1").arg(modelData.license_id || qsTr("brak"))
                        }

                        Label {
                            text: qsTr("Edycja: %1").arg(modelData.edition || qsTr("brak"))
                        }

                        Label {
                            text: qsTr("Urządzenie: %1").arg(modelData.local_hwid_hash || qsTr("brak"))
                            font.family: "monospace"
                        }

                        Label {
                            text: qsTr("Powtórna aktywacja: %1").arg(modelData.repeat_activation ? qsTr("tak") : qsTr("nie"))
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar {}
            }
        }

        GroupBox {
            title: qsTr("Ostrzeżenia")
            Layout.fillWidth: true
            visible: controller && controller.warnings && controller.warnings.length > 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 6

                Repeater {
                    model: controller ? controller.warnings : []
                    delegate: Label {
                        required property string modelData
                        text: modelData
                        color: Qt.rgba(0.85, 0.45, 0.18, 1)
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("Eksport raportu")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: exportDirectoryField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Katalog docelowy, np. reports/security")
                        text: "reports/security"
                    }

                    Button {
                        text: qsTr("Eksportuj")
                        enabled: controller && !controller.busy
                        onClicked: {
                            exportStatus = ""
                            exportStatusLabel.color = Qt.rgba(0.35, 0.65, 0.35, 1)
                            if (!controller)
                                return
                            const ok = controller.exportReport(exportDirectoryField.text, exportBasenameField.text)
                            if (!ok && controller.errorMessage)
                                exportStatus = controller.errorMessage
                        }
                    }
                }

                TextField {
                    id: exportBasenameField
                    Layout.fillWidth: true
                    placeholderText: qsTr("Nazwa pliku, np. license_audit")
                    text: "license_audit"
                }

                Label {
                    id: exportStatusLabel
                    Layout.fillWidth: true
                    text: exportStatus
                    color: Qt.rgba(0.35, 0.65, 0.35, 1)
                    wrapMode: Text.WordWrap
                }
            }
        }
    }

    Connections {
        target: controller
        enabled: !!controller

        function onExportCompleted(jsonPath, markdownPath) {
            exportStatus = qsTr("Zapisano: %1, %2").arg(jsonPath).arg(markdownPath)
            exportStatusLabel.color = Qt.rgba(0.35, 0.65, 0.35, 1)
        }

        function onExportFailed(message) {
            exportStatus = message
            exportStatusLabel.color = Qt.rgba(0.85, 0.3, 0.3, 1)
        }
    }
}
