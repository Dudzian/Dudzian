import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../design-system" as DesignSystem
import "../design-system/components" as DesignComponents

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

    Rectangle {
        anchors.fill: parent
        color: DesignSystem.Palette.background
        radius: 12

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 16

            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Label {
                    text: qsTr("Zarządzanie portfelami multi-account")
                    color: DesignSystem.Palette.textPrimary
                    font.pixelSize: DesignSystem.Typography.headlineMedium
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
                color: DesignSystem.Palette.danger
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
                    spacing: 12
                    model: portfolioController ? portfolioController.portfolios : []

                    delegate: DesignComponents.Card {
                        width: Math.min(parent.width, 620)
                        property var itemData: modelData
                        background.radius: 12
                        background.color: root.selectedPortfolio === itemData
                                           ? Qt.rgba(0.2, 0.38, 0.6, 0.6)
                                           : DesignSystem.Palette.surface

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: {
                                root.selectedPortfolio = itemData
                            }
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 6

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: itemData.portfolio_id || itemData.id
                                    color: DesignSystem.Palette.textPrimary
                                    font.pixelSize: DesignSystem.Typography.title
                                    font.bold: true
                                }
                                Label {
                                    text: qsTr("Preset: %1").arg(itemData.primary_preset || itemData.preset)
                                    color: DesignSystem.Palette.textSecondary
                                    Layout.fillWidth: true
                                }
                            }

                            Label {
                                visible: (itemData.fallback_presets || []).length > 0
                                text: qsTr("Fallback: %1").arg((itemData.fallback_presets || []).join(", "))
                                color: DesignSystem.Palette.textSecondary
                            }

                            Label {
                                text: qsTr("Followerzy: %1").arg((itemData.followers || []).length)
                                color: DesignSystem.Palette.textSecondary
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                Repeater {
                                    model: itemData.followers || []
                                    delegate: Label {
                                        Layout.fillWidth: true
                                        text: qsTr("• %1 (skala %2)")
                                                .arg(modelData.portfolio_id || modelData.id)
                                                .arg(modelData.scaling || 1.0)
                                        font.pixelSize: DesignSystem.Typography.caption
                                        color: DesignSystem.Palette.textPrimary
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

        ListModel { id: followerModel }

        contentItem: ColumnLayout {
            spacing: 8
            width: 420

            Label {
                text: qsTr("Id portfela")
                color: DesignSystem.Palette.textSecondary
            }
            TextField {
                id: portfolioIdField
                text: editDialog.payload.portfolio_id || ""
                placeholderText: qsTr("np. core-alpha")
            }

            Label {
                text: qsTr("Główny preset")
                color: DesignSystem.Palette.textSecondary
            }
            TextField {
                id: presetField
                text: editDialog.payload.primary_preset || ""
                placeholderText: qsTr("np. mean-reversion-prod")
            }

            Label {
                text: qsTr("Fallback preset")
                color: DesignSystem.Palette.textSecondary
            }
            TextField {
                id: fallbackField
                text: (editDialog.payload.fallback_presets || []).join(", ")
                placeholderText: qsTr("Preset1, Preset2")
            }

            Label {
                text: qsTr("Followerzy")
                color: DesignSystem.Palette.textSecondary
            }

            ListView {
                height: 140
                clip: true
                model: followerModel
                delegate: RowLayout {
                    width: parent.width
                    spacing: 8

                    TextField {
                        Layout.fillWidth: true
                        text: model.portfolio_id || model.id
                        placeholderText: qsTr("Id portfela following")
                        onTextChanged: followerModel.setProperty(index, "portfolio_id", text)
                    }

                    SpinBox {
                        value: model.scaling || 1.0
                        from: 0.1
                        to: 10
                        stepSize: 0.1
                        onValueChanged: followerModel.setProperty(index, "scaling", value)
                    }

                    ToolButton {
                        icon.name: "list-remove"
                        onClicked: followerModel.remove(index)
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
