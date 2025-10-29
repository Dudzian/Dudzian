import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    width: parent ? parent.width : 800
    height: parent ? parent.height : 600

    property var selectedPortfolio: null
    property string errorMessage: ""

    function refresh() {
        if (portfolioController)
            portfolioController.refreshPortfolios()
    }

    Component.onCompleted: refresh()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Zarządzanie portfelami multi-account")
                font.pixelSize: 22
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            ToolButton {
                icon.name: "view-refresh"
                text: qsTr("Odśwież")
                display: AbstractButton.TextBesideIcon
                onClicked: root.refresh()
            }
        }

        Label {
            Layout.fillWidth: true
            visible: errorMessage.length > 0 || (portfolioController && portfolioController.lastError.length > 0)
            wrapMode: Text.WordWrap
            color: "firebrick"
            text: errorMessage.length > 0 ? errorMessage : (portfolioController ? portfolioController.lastError : "")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Button {
                text: qsTr("Dodaj / aktualizuj")
                onClicked: editDialog.openForCreate()
            }

            Button {
                text: qsTr("Usuń zaznaczony")
                enabled: selectedPortfolio !== null
                onClicked: {
                    if (!portfolioController || !selectedPortfolio)
                        return
                    portfolioController.removePortfolio(selectedPortfolio.portfolio_id || selectedPortfolio.id)
                    root.refresh()
                }
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: portfolioList
                width: parent.width
                clip: true
                spacing: 6
                model: portfolioController ? portfolioController.portfolios : []

                delegate: Frame {
                    width: Math.min(parent.width, 560)
                    padding: 12
                    property var itemData: modelData
                    background: Rectangle { radius: 6; color: palette.base }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: {
                            root.selectedPortfolio = itemData
                        }
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 4

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Label {
                                text: itemData.portfolio_id || itemData.id
                                font.pixelSize: 18
                                font.bold: true
                            }
                            Label {
                                text: qsTr("Preset: %1").arg(itemData.primary_preset || itemData.preset)
                                color: palette.mid
                                Layout.fillWidth: true
                            }
                        }

                        Label {
                            visible: (itemData.fallback_presets || []).length > 0
                            text: qsTr("Fallback: %1").arg((itemData.fallback_presets || []).join(", "))
                            color: palette.mid
                        }

                        Label {
                            text: qsTr("Followerzy: %1").arg((itemData.followers || []).length)
                            color: palette.mid
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            Repeater {
                                model: itemData.followers || []
                                delegate: Label {
                                    Layout.fillWidth: true
                                    text: qsTr("• %1 (skala %2)").arg(modelData.portfolio_id || modelData.id).arg(modelData.scaling || 1.0)
                                    font.pixelSize: 12
                                    color: palette.windowText
                                }
                            }
                        }

                        Button {
                            text: qsTr("Edytuj")
                            Layout.alignment: Qt.AlignRight
                            onClicked: editDialog.openForEdit(itemData)
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: editDialog
        modal: true
        title: qsTr("Konfiguracja portfela")
        standardButtons: Dialog.Ok | Dialog.Cancel
        property bool isEdit: false
        property var payload: ({})

        function openForCreate() {
            isEdit = false
            payload = { portfolio_id: "", primary_preset: "", fallback_presets: [], followers: [] }
            followerModel.clear()
            editDialog.open()
        }

        function openForEdit(data) {
            isEdit = true
            payload = JSON.parse(JSON.stringify(data || {}))
            followerModel.clear()
            const followers = payload.followers || []
            for (let i = 0; i < followers.length; ++i)
                followerModel.append(followers[i])
            editDialog.open()
        }

        onAccepted: {
            payload.portfolio_id = portfolioIdField.text.trim()
            payload.primary_preset = presetField.text.trim()
            payload.fallback_presets = fallbackField.text.trim().length > 0 ? fallbackField.text.split(/\s*,\s*/) : []
            payload.followers = []
            for (let i = 0; i < followerModel.count; ++i)
                payload.followers.push(followerModel.get(i))
            if (!payload.portfolio_id || !payload.primary_preset) {
                root.errorMessage = qsTr("Id portfela i preset są wymagane")
                return
            }
            if (!portfolioController) {
                root.errorMessage = qsTr("Kontroler portfeli jest niedostępny")
                return
            }
            const success = portfolioController.applyPortfolio(payload)
            if (success)
                root.refresh()
        }

        contentItem: ColumnLayout {
            spacing: 8
            width: 420

            TextField {
                id: portfolioIdField
                Layout.fillWidth: true
                placeholderText: qsTr("Id portfela")
                text: editDialog.payload.portfolio_id || ""
            }

            TextField {
                id: presetField
                Layout.fillWidth: true
                placeholderText: qsTr("Preset główny")
                text: editDialog.payload.primary_preset || editDialog.payload.preset || ""
            }

            TextField {
                id: fallbackField
                Layout.fillWidth: true
                placeholderText: qsTr("Fallback (oddzielone przecinkami)")
                text: (editDialog.payload.fallback_presets || []).join(", ")
            }

            Label {
                text: qsTr("Followerzy")
                font.bold: true
            }

            ListView {
                id: followerList
                Layout.fillWidth: true
                Layout.preferredHeight: 160
                clip: true
                spacing: 4
                model: ListModel { id: followerModel }

                delegate: Frame {
                    width: parent.width
                    padding: 6
                    RowLayout {
                        anchors.fill: parent
                        spacing: 6
                        TextField {
                            Layout.fillWidth: true
                            text: model.portfolio_id || ""
                            placeholderText: qsTr("Id followera")
                            onEditingFinished: followerModel.setProperty(index, "portfolio_id", text.trim())
                        }
                        SpinBox {
                            value: model.scaling || 1.0
                            from: 0.1
                            to: 5.0
                            stepSize: 0.1
                            onValueChanged: followerModel.setProperty(index, "scaling", value)
                        }
                        Button {
                            text: qsTr("Usuń")
                            onClicked: followerModel.remove(index)
                        }
                    }
                }
            }

            Button {
                text: qsTr("Dodaj followera")
                onClicked: followerModel.append({ portfolio_id: "", scaling: 1.0 })
            }
        }
    }
}
