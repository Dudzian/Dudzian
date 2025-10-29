import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property url catalogSource: Qt.resolvedUrl("../../../config/marketplace/catalog.json")
    property bool autoReload: true
    property int reloadInterval: 60000
    signal packageActivated(string packageId)

    ListModel { id: packageModel }

    function loadCatalog() {
        if (!catalogSource)
            return
        var request = new XMLHttpRequest()
        request.open("GET", catalogSource)
        request.onreadystatechange = function() {
            if (request.readyState !== XMLHttpRequest.DONE)
                return
            if (request.status !== 0 && request.status !== 200) {
                statusLabel.text = qsTr("Nie udało się pobrać katalogu (%1)").arg(request.status)
                return
            }
            try {
                var doc = JSON.parse(request.responseText)
                packageModel.clear()
                if (doc && doc.packages) {
                    for (var i = 0; i < doc.packages.length; ++i) {
                        var item = doc.packages[i]
                        var artifact = item.distribution && item.distribution.length ? item.distribution[0] : null
                        packageModel.append({
                            packageId: item.package_id,
                            name: item.display_name || item.package_id,
                            summary: item.summary || "",
                            version: item.version || "",
                            tagList: item.tags || [],
                            licenseName: item.license ? item.license.name : "",
                            artifactUri: artifact ? artifact.uri : "",
                            documentationUrl: item.documentation_url || "",
                            notes: item.release_notes || []
                        })
                    }
                    statusLabel.text = qsTr("Załadowano %1 paczek").arg(packageModel.count)
                } else {
                    statusLabel.text = qsTr("Katalog nie zawiera paczek")
                }
            } catch (error) {
                statusLabel.text = qsTr("Błąd podczas parsowania katalogu: %1").arg(error)
            }
        }
        request.send()
    }

    Timer {
        id: reloadTimer
        interval: root.reloadInterval
        repeat: true
        running: root.autoReload
        onTriggered: loadCatalog()
    }

    Component.onCompleted: loadCatalog()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Marketplace konfiguracji")
                font.pixelSize: 22
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            ToolButton {
                icon.name: "view-refresh"
                text: qsTr("Odśwież")
                display: AbstractButton.TextBesideIcon
                onClicked: loadCatalog()
            }
        }

        Label {
            id: statusLabel
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            text: qsTr("Trwa ładowanie katalogu...")
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Flow {
                id: cardFlow
                width: parent.width
                spacing: 16
                Repeater {
                    model: packageModel
                    delegate: Frame {
                        id: card
                        width: Math.min(cardFlow.width, 420)
                        Layout.preferredWidth: 360
                        padding: 16

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                text: model.name
                                font.pixelSize: 18
                                font.bold: true
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: qsTr("Wersja: %1").arg(model.version)
                                color: card.palette.mid
                            }

                            Label {
                                text: model.summary
                                wrapMode: Text.WordWrap
                            }

                            Flow {
                                width: parent.width
                                spacing: 6
                                Repeater {
                                    model: model.tagList
                                    delegate: Rectangle {
                                        radius: 4
                                        color: Qt.darker(card.palette.base, 1.1)
                                        border.color: card.palette.mid
                                        border.width: 1
                                        padding: 4

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData
                                            font.pixelSize: 12
                                        }
                                    }
                                }
                            }

                            Label {
                                visible: model.licenseName.length > 0
                                text: qsTr("Licencja: %1").arg(model.licenseName)
                                color: card.palette.mid
                            }

                            RowLayout {
                                Layout.topMargin: 8
                                spacing: 8

                                Button {
                                    text: qsTr("Pobierz")
                                    icon.name: "download"
                                    enabled: model.artifactUri.length > 0
                                    onClicked: root.packageActivated(model.packageId)
                                }

                                Button {
                                    text: qsTr("Dokumentacja")
                                    visible: model.documentationUrl.length > 0
                                    onClicked: Qt.openUrlExternally(model.documentationUrl)
                                }
                            }

                            Label {
                                visible: model.notes.length > 0
                                text: qsTr("Zmiany: %1").arg(model.notes.join(", "))
                                wrapMode: Text.WordWrap
                                font.pixelSize: 12
                                color: card.palette.mid
                            }
                        }
                    }
                }
            }
        }
    }
}
