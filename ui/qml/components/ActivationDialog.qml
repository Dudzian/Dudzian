import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: activationDialog
    property var controller
    modal: true
    width: 640
    height: 520
    title: qsTr("Aktywacja licencji OEM")
    standardButtons: Dialog.Close

    onOpened: {
        if (controller)
            controller.refresh()
    }

    contentItem: ScrollView {
        ColumnLayout {
            width: parent ? parent.width : implicitWidth
            spacing: 12

            Label {
                text: qsTr("Fingerprint (SHA-256): %1")
                        .arg(controller && controller.fingerprint.payload && controller.fingerprint.payload.fingerprint
                             ? controller.fingerprint.payload.fingerprint.value : qsTr("(brak)"))
                wrapMode: Text.Wrap
                Layout.fillWidth: true
            }

            Label {
                visible: controller && controller.lastError.length > 0
                text: controller ? controller.lastError : ""
                color: "#d9534f"
                wrapMode: Text.Wrap
                Layout.fillWidth: true
            }

            GroupBox {
                title: qsTr("Komponenty sprzętowe")
                Layout.fillWidth: true
                contentItem: ColumnLayout {
                    spacing: 6
                    Repeater {
                        model: controller && controller.fingerprint.payload ? (controller.fingerprint.payload.component_list || []) : []
                        delegate: ColumnLayout {
                            Layout.fillWidth: true
                            Label {
                                text: qsTr("%1").arg(modelData.name || qsTr("(nieznany)"))
                                font.bold: true
                            }
                            Label {
                                visible: !!modelData.raw
                                text: qsTr("RAW: %1").arg(modelData.raw || qsTr("(brak)"))
                                wrapMode: Text.Wrap
                            }
                            Label {
                                visible: !!modelData.normalized
                                text: qsTr("Normalized: %1").arg(modelData.normalized || qsTr("(brak)"))
                                wrapMode: Text.Wrap
                                color: "#6c757d"
                            }
                            Label {
                                visible: !!modelData.digest
                                text: qsTr("Digest: %1").arg(modelData.digest || qsTr("(brak)"))
                                font.family: "monospace"
                                wrapMode: Text.Wrap
                            }
                            Rectangle {
                                height: 1
                                color: Qt.rgba(1,1,1,0.1)
                                Layout.fillWidth: true
                                visible: index < (Repeater.view.count - 1)
                            }
                        }
                    }
                }
            }

            GroupBox {
                title: qsTr("Zarejestrowane licencje")
                Layout.fillWidth: true
                contentItem: ColumnLayout {
                    spacing: 4
                    Repeater {
                        model: controller ? controller.licenses : []
                        delegate: ColumnLayout {
                            Layout.fillWidth: true
                            Label {
                                text: qsTr("ID licencji: %1").arg(modelData.licenseId || qsTr("(brak)"))
                                font.bold: true
                            }
                            Label {
                                text: qsTr("Tryb: %1, klucz: %2").arg(modelData.mode || "-").arg(modelData.signatureKey || "-")
                                color: "#6c757d"
                            }
                            Label {
                                text: qsTr("Wydano: %1").arg(modelData.issuedAt || "-")
                            }
                            Rectangle {
                                height: 1
                                color: Qt.rgba(1,1,1,0.1)
                                Layout.fillWidth: true
                                visible: index < (Repeater.view.count - 1)
                            }
                        }
                    }
                    Label {
                        visible: !controller || controller.licenses.length === 0
                        text: qsTr("Brak licencji w rejestrze.")
                        color: "#6c757d"
                    }
                }
            }

            RowLayout {
                Layout.alignment: Qt.AlignRight
                spacing: 8
                Button {
                    text: qsTr("Odśwież")
                    icon.name: "view-refresh"
                    onClicked: controller.refresh()
                }
                Button {
                    text: qsTr("Załaduj rejestr")
                    onClicked: controller.reloadRegistry()
                }
            }
        }
    }
}

