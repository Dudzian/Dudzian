import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var activationControllerRef: typeof activationController !== "undefined" ? activationController : null
    property var appControllerRef: typeof appController !== "undefined" ? appController : null
    property var securityControllerRef: typeof securityController !== "undefined" ? securityController : null

    implicitWidth: parent ? parent.width : 640

    function licenseSummary() {
        if (securityControllerRef && securityControllerRef.licenseInfo)
            return securityControllerRef.licenseInfo
        if (appControllerRef && appControllerRef.securityCache && appControllerRef.securityCache.oemLicense)
            return appControllerRef.securityCache.oemLicense
        return {}
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        GroupBox {
            title: qsTr("Podsumowanie licencji OEM")
            Layout.fillWidth: true

            GridLayout {
                columns: 2
                columnSpacing: 12
                rowSpacing: 6
                anchors.fill: parent
                anchors.margins: 8

                property var summary: licenseSummary()

                Label { text: qsTr("Status") }
                Label {
                    text: summary.status || qsTr("nieznany")
                    font.bold: true
                }

                Label { text: qsTr("Fingerprint (payload)") }
                Label {
                    text: summary.fingerprint || qsTr("brak")
                    font.family: "monospace"
                    wrapMode: Text.WordWrap
                }

                Label { text: qsTr("Fingerprint lokalny") }
                Label {
                    text: summary.local_fingerprint || qsTr("brak")
                    font.family: "monospace"
                    wrapMode: Text.WordWrap
                }

                Label { text: qsTr("Edycja") }
                Label { text: summary.edition || summary.profile || qsTr("brak") }

                Label { text: qsTr("ID licencji") }
                Label { text: summary.license_id || qsTr("brak") }

                Label { text: qsTr("Utrzymanie do") }
                Label { text: summary.maintenance_until || qsTr("brak") }

                Label { text: qsTr("Moduły") }
                Label {
                    text: summary.modules ? summary.modules.join(", ") : qsTr("brak")
                    wrapMode: Text.WordWrap
                }

                Label { text: qsTr("Środowiska") }
                Label {
                    text: summary.environments ? summary.environments.join(", ") : qsTr("brak")
                    wrapMode: Text.WordWrap
                }

                Label { text: qsTr("Ostatnie ostrzeżenia") }
                Label {
                    text: summary.warnings && summary.warnings.length > 0 ? summary.warnings.join("\n") : qsTr("brak")
                    wrapMode: Text.WordWrap
                }

                Label { text: qsTr("Ostatnie błędy") }
                Label {
                    text: summary.errors && summary.errors.length > 0 ? summary.errors.join("\n") : qsTr("brak")
                    wrapMode: Text.WordWrap
                    color: summary.errors && summary.errors.length > 0 ? Qt.rgba(0.9, 0.35, 0.35, 1) : palette.text
                }
            }
        }

        GroupBox {
            title: qsTr("Rejestr aktywacji (registry.jsonl)")
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label {
                        text: activationControllerRef && activationControllerRef.licenses
                              ? qsTr("Wpisów: %1").arg(activationControllerRef.licenses.length)
                              : qsTr("Wpisów: 0")
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: qsTr("Odśwież rejestr")
                        enabled: !!activationControllerRef
                        onClicked: activationControllerRef.reloadRegistry()
                    }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: activationControllerRef ? activationControllerRef.licenses : []

                    delegate: Frame {
                        required property var modelData
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            color: Qt.rgba(0.14, 0.22, 0.32, 0.35)
                            radius: 6
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 2

                            Label {
                                text: qsTr("Fingerprint: %1").arg(modelData.fingerprint || qsTr("brak"))
                                font.family: "monospace"
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: qsTr("Tryb: %1").arg(modelData.mode || qsTr("brak"))
                            }

                            Label {
                                text: qsTr("ID licencji: %1").arg(modelData.licenseId || qsTr("brak"))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Wydana: %1").arg(modelData.issuedAt || qsTr("brak"))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Klucz podpisu: %1").arg(modelData.signatureKey || qsTr("brak"))
                                color: palette.mid
                            }
                        }
                    }
                }
            }
        }
    }
}
