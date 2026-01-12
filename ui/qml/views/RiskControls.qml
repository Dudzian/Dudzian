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
    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var guardrails: runtimeService && runtimeService.guardrails ? runtimeService.guardrails : ({})

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

    function syncGuardrails() {
        if (!runtimeService)
            return
        guardrails = runtimeService.guardrails || {}
    }

    onAppControllerChanged: refreshKillSwitchState()
    Component.onCompleted: {
        refreshKillSwitchState()
        syncGuardrails()
    }

    Connections {
        target: appController
        ignoreUnknownSignals: true
        function onRiskKillSwitchChanged() {
            root.refreshKillSwitchState()
        }
    }

    Connections {
        target: runtimeService
        ignoreUnknownSignals: true
        function onGuardrailsChanged() { root.syncGuardrails() }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 14

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

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ListView {
                        id: limitsView
                        objectName: "limitsListView"
                        anchors.fill: parent
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
                    }

                    Label {
                        anchors.centerIn: parent
                        text: qsTr("Brak zdefiniowanych limitów ryzyka")
                        color: palette.mid
                        mouseTransparent: true
                        visible: !limitsView.model || limitsView.count === 0
                    }
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

                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 160

                    ListView {
                        id: costView
                        objectName: "costListView"
                        anchors.fill: parent
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
                    }

                    Label {
                        anchors.centerIn: parent
                        text: qsTr("Brak zagregowanych metryk kosztów")
                        color: palette.mid
                        mouseTransparent: true
                        visible: !costView.model || costView.count === 0
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            padding: 12
            background: Rectangle { radius: 8; color: Qt.darker(palette.base, 1.08) }

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: qsTr("Guardraile automatyzacji")
                        font.pixelSize: 18
                        font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: runtimeService && runtimeService.executionMode === "auto"
                              ? qsTr("Tryb: automatyczny") : qsTr("Tryb: manualny")
                        color: palette.mid
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label {
                        text: qsTr("Maks. ekspozycja (%)")
                        Layout.fillWidth: true
                    }
                    TextField {
                        id: maxExposureField
                        Layout.preferredWidth: 120
                        text: guardrails && guardrails.maxExposure !== undefined ? (guardrails.maxExposure * 100).toFixed(2) : "35.00"
                        validator: DoubleValidator { bottom: 0.0; top: 100.0; decimals: 3 }
                        onEditingFinished: {
                            var parsed = Number.fromLocaleString(Qt.locale(), text)
                            if (isNaN(parsed) || !runtimeService) {
                                text = guardrails && guardrails.maxExposure !== undefined ? (guardrails.maxExposure * 100).toFixed(2) : "35.00"
                                return
                            }
                            runtimeService.setMaxExposureLimit(parsed / 100.0)
                            text = guardrails && guardrails.maxExposure !== undefined ? (guardrails.maxExposure * 100).toFixed(2) : text
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label {
                        text: qsTr("Dzienny limit strat (%)")
                        Layout.fillWidth: true
                    }
                    TextField {
                        id: dailyLossField
                        Layout.preferredWidth: 120
                        text: guardrails && guardrails.dailyLossLimitPct !== undefined ? (guardrails.dailyLossLimitPct * 100).toFixed(2) : "3.00"
                        validator: DoubleValidator { bottom: 0.0; top: 50.0; decimals: 3 }
                        onEditingFinished: {
                            var parsedLoss = Number.fromLocaleString(Qt.locale(), text)
                            if (isNaN(parsedLoss) || !runtimeService) {
                                text = guardrails && guardrails.dailyLossLimitPct !== undefined ? (guardrails.dailyLossLimitPct * 100).toFixed(2) : "3.00"
                                return
                            }
                            runtimeService.setDailyLossLimitPct(parsedLoss / 100.0)
                            text = guardrails && guardrails.dailyLossLimitPct !== undefined ? (guardrails.dailyLossLimitPct * 100).toFixed(2) : text
                        }
                    }
                }

                CheckBox {
                    id: blockOnSlaToggle
                    checked: guardrails && guardrails.blockOnSlaAlerts
                    text: qsTr("Blokuj otwieranie pozycji przy alertach SLA")
                    onToggled: runtimeService ? runtimeService.setBlockOnSlaAlerts(checked) : null
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
