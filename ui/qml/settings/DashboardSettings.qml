import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    implicitWidth: 720
    implicitHeight: 540

    property var settingsController: (typeof settingsController !== "undefined" ? settingsController : null)
    readonly property var themeOptions: [
        { id: "system", label: qsTr("Systemowy") },
        { id: "light", label: qsTr("Jasny") },
        { id: "dark", label: qsTr("Ciemny") }
    ]

    function cardTitle(cardId) {
        switch (cardId) {
        case "io_queue":
            return qsTr("Kolejki I/O")
        case "guardrails":
            return qsTr("Guardrail'e")
        case "retraining":
            return qsTr("Retraining")
        default:
            return cardId
        }
    }

    function isCardHidden(cardId) {
        if (!root.settingsController)
            return false
        return root.settingsController.hiddenCards.indexOf(cardId) !== -1
    }

    function themeIndex(themeId) {
        for (var i = 0; i < root.themeOptions.length; i++) {
            if (root.themeOptions[i].id === themeId)
                return i
        }
        return 0
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 18

        Label {
            text: qsTr("Ustawienia dashboardu runtime")
            font.pixelSize: 20
            font.bold: true
            color: Styles.AppTheme.textPrimary
        }

        Rectangle {
            color: Styles.AppTheme.surfaceStrong
            radius: 6
            Layout.fillWidth: true
            Layout.fillHeight: false

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Label {
                    text: qsTr("Kolejność i widoczność kart")
                    font.bold: true
                    color: Styles.AppTheme.textPrimary
                }

                ColumnLayout {
                    id: cardsList
                    spacing: 8

                    Repeater {
                        model: root.settingsController ? root.settingsController.cardOrder : []
                        delegate: Frame {
                            Layout.fillWidth: true
                            background: Rectangle {
                                radius: 6
                                color: Qt.rgba(0.14, 0.16, 0.22, 0.9)
                            }

                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 12

                                Text {
                                    Layout.fillWidth: true
                                    text: root.cardTitle(modelData)
                                    color: Styles.AppTheme.textPrimary
                                }

                                Switch {
                                    id: visibilitySwitch
                                    checked: !root.isCardHidden(modelData)
                                    text: checked ? qsTr("Widoczna") : qsTr("Ukryta")
                                    onToggled: {
                                        if (!root.settingsController)
                                            return
                                        var currentlyHidden = root.isCardHidden(modelData)
                                        if (checked === !currentlyHidden)
                                            return
                                        root.settingsController.setCardVisibility(modelData, checked)
                                    }
                                }

                                Button {
                                    text: "▲"
                                    enabled: index > 0
                                    onClicked: {
                                        if (root.settingsController)
                                            root.settingsController.moveCard(modelData, -1)
                                    }
                                }

                                Button {
                                    text: "▼"
                                    enabled: index < (cardsList.cardsListRepeaterCount - 1)
                                    onClicked: {
                                        if (root.settingsController)
                                            root.settingsController.moveCard(modelData, 1)
                                    }
                                }
                            }
                        }
                    }

                    property int cardsListRepeaterCount: root.settingsController ? root.settingsController.cardOrder.length : 0
                }

                Label {
                    visible: root.settingsController && root.settingsController.cardOrder.length === 0
                    text: qsTr("Brak kart do skonfigurowania")
                    color: Styles.AppTheme.textSecondary
                }
            }
        }

        Rectangle {
            color: Styles.AppTheme.surfaceStrong
            radius: 6
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Label {
                    text: qsTr("Parametry ogólne")
                    font.bold: true
                    color: Styles.AppTheme.textPrimary
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12

                    Text {
                        text: qsTr("Interwał odświeżania (ms)")
                        color: Styles.AppTheme.textSecondary
                    }

                    SpinBox {
                        id: refreshSpin
                        Layout.preferredWidth: 160
                        from: 1000
                        to: 60000
                        stepSize: 500
                        value: root.settingsController ? root.settingsController.refreshIntervalMs : 4000
                        onValueModified: {
                            if (root.settingsController)
                                root.settingsController.setRefreshIntervalMs(value)
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12

                    Text {
                        text: qsTr("Motyw interfejsu")
                        color: Styles.AppTheme.textSecondary
                    }

                    ComboBox {
                        id: themeCombo
                        Layout.preferredWidth: 200
                        textRole: "label"
                        valueRole: "id"
                        model: root.themeOptions
                        currentIndex: root.themeIndex(root.settingsController ? root.settingsController.theme : "system")
                        onActivated: {
                            if (!root.settingsController)
                                return
                            var value = model[currentIndex]
                            if (value && value.id)
                                root.settingsController.setTheme(value.id)
                        }
                    }
                }

                Label {
                    text: root.settingsController ? qsTr("Plik ustawień: %1").arg(root.settingsController.settingsPath) : ""
                    color: Styles.AppTheme.textSecondary
                    wrapMode: Text.WrapAnywhere
                    Layout.fillWidth: true
                }

                Button {
                    text: qsTr("Przywróć domyślne")
                    Layout.alignment: Qt.AlignLeft
                    onClicked: {
                        if (root.settingsController)
                            root.settingsController.resetDefaults()
                    }
                }
            }
        }

        Item { Layout.fillHeight: true }
    }

    Connections {
        target: root.settingsController
        enabled: !!root.settingsController
        function onRefreshIntervalMsChanged() {
            refreshSpin.value = root.settingsController.refreshIntervalMs
        }
        function onThemeChanged() {
            themeCombo.currentIndex = root.themeIndex(root.settingsController.theme)
        }
    }
}
