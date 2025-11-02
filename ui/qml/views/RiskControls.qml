import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    implicitWidth: 720
    implicitHeight: 520

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property var limitsModel: appController && appController.riskLimitsModel ? appController.riskLimitsModel
                                     : (typeof limitsModel !== "undefined" ? limitsModel : null)
    property var costModel: appController && appController.riskCostModel ? appController.riskCostModel
                                   : (typeof costModel !== "undefined" ? costModel : null)
    property bool killSwitchFallback: false
    property bool killSwitchState: false

    function refreshKillSwitchState() {
        if (appController && appController.hasOwnProperty("riskKillSwitchEngaged"))
            killSwitchState = !!appController.riskKillSwitchEngaged
        else
            killSwitchState = killSwitchFallback
    }

    function commitLimitValue(key, value) {
        if (!limitsModel)
            return
        if (typeof limitsModel.setLimitValue === "function") {
            limitsModel.setLimitValue(key, value)
        } else if (typeof limitsModel.setLimitValueAt === "function") {
            var idx = limitsModel.indexForKey ? limitsModel.indexForKey(key) : -1
            if (idx >= 0)
                limitsModel.setLimitValueAt(idx, value)
        }
    }

    function formatNumber(value, decimals) {
        if (value === undefined || value === null)
            return "—"
        var digits = decimals !== undefined ? decimals : 2
        return Number(value).toLocaleString(Qt.locale(), "f", digits)
    }

    function formatLimitValue(value, isPercent) {
        if (value === undefined || value === null)
            return "—"
        var scaled = isPercent ? value * 100.0 : value
        return Number(scaled).toLocaleString(Qt.locale(), "f", isPercent ? 2 : 2)
    }

    function applyKillSwitch(value) {
        if (appController && typeof appController.setRiskKillSwitchEngaged === "function") {
            appController.setRiskKillSwitchEngaged(value)
        } else {
            killSwitchFallback = value
        }
        killSwitchState = value
    }

    onAppControllerChanged: refreshKillSwitchState()
    Component.onCompleted: refreshKillSwitchState()

    Connections {
        target: appController
        ignoreUnknownSignals: true
        function onRiskKillSwitchChanged() {
            root.refreshKillSwitchState()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 14
        padding: 14

        Label {
            text: qsTr("Zarządzanie ryzykiem")
            font.pixelSize: 24
            font.bold: true
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height * 0.55
            padding: 12
            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.05) }

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                Label {
                    text: qsTr("Limity profilu")
                    font.pixelSize: 18
                    font.bold: true
                }

                ListView {
                    id: limitsView
                    objectName: "limitsListView"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    model: root.limitsModel
                    delegate: Frame {
                        width: ListView.view.width
                        padding: 8
                        background: Rectangle { radius: 6; color: Qt.darker(palette.base, 1.02) }

                        RowLayout {
                            anchors.fill: parent
                            spacing: 10

                            Label {
                                Layout.fillWidth: true
                                text: model.label || model.key
                            }

                            TextField {
                                id: valueField
                                Layout.preferredWidth: 120
                                text: root.formatLimitValue(model.value, model.isPercent)
                                inputMethodHints: Qt.ImhFormattedNumbersOnly
                                validator: DoubleValidator {
                                    bottom: model.isPercent ? model.minimum * 100.0 : model.minimum
                                    top: model.isPercent ? model.maximum * 100.0 : model.maximum
                                    decimals: 4
                                }

                                onActiveFocusChanged: {
                                    if (!activeFocus)
                                        text = root.formatLimitValue(model.value, model.isPercent)
                                }

                                onEditingFinished: {
                                    var locale = Qt.locale()
                                    var parsed = Number.fromLocaleString(locale, text)
                                    if (isNaN(parsed)) {
                                        text = root.formatLimitValue(model.value, model.isPercent)
                                        return
                                    }
                                    var normalized = model.isPercent ? parsed / 100.0 : parsed
                                    root.commitLimitValue(model.key, normalized)
                                    text = root.formatLimitValue(model.value, model.isPercent)
                                }
                            }
                        }
                    }
                    ScrollBar.vertical: ScrollBar {}
                    placeholderText: qsTr("Brak zdefiniowanych limitów ryzyka")
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            padding: 12
            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.05) }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                Label {
                    text: qsTr("Metryki kosztów i ekspozycji")
                    font.pixelSize: 18
                    font.bold: true
                }

                ListView {
                    id: costView
                    objectName: "costListView"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 160
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    model: root.costModel
                    delegate: RowLayout {
                        width: ListView.view.width
                        spacing: 10

                        Label {
                            Layout.fillWidth: true
                            text: model.label || model.key
                        }

                        Label {
                            text: model.formatted !== undefined ? model.formatted : root.formatNumber(model.value, 2)
                            font.bold: true
                        }
                    }
                    ScrollBar.vertical: ScrollBar {}
                    placeholderText: qsTr("Brak zagregowanych metryk kosztów")
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            padding: 12
            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.08) }

            RowLayout {
                anchors.fill: parent
                spacing: 12

                Switch {
                    id: killSwitchToggle
                    objectName: "killSwitchToggle"
                    checked: root.killSwitchState
                    text: checked ? qsTr("Kill-switch aktywny") : qsTr("Kill-switch nieaktywny")
                    onToggled: root.applyKillSwitch(checked)
                }

                Item { Layout.fillWidth: true }

                Button {
                    text: qsTr("Wyłącz")
                    enabled: killSwitchToggle.checked
                    onClicked: killSwitchToggle.checked = false
                }

                Button {
                    text: qsTr("Aktywuj")
                    enabled: !killSwitchToggle.checked
                    onClicked: killSwitchToggle.checked = true
                }
            }
        }
    }
}

