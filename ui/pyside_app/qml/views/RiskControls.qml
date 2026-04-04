import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

ColumnLayout {
    id: root
    property var runtimeService
    property var designSystem
    property var opportunitySettings: runtimeService && runtimeService.opportunityRuntimeSettings
                                      ? runtimeService.opportunityRuntimeSettings : ({})

    spacing: 12

    Label {
        text: qsTr("Limity ryzyka i ochrona")
        font.bold: true
        color: designSystem.color("textPrimary")
    }

    ColumnLayout {
        spacing: 8
        RowLayout {
            spacing: 10
            Label {
                text: qsTr("Take profit [%]")
                color: designSystem.color("textSecondary")
                Layout.preferredWidth: 200
            }
            SpinBox {
                id: tpSpin
                property real percentValue: value / 10.0
                from: 0
                to: 200
                value: runtimeService ? Math.round((runtimeService.riskControls.takeProfitPct || 0) * 10) : 0
                stepSize: 1
                textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) }
                valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 10) }
            }
        }
        RowLayout {
            spacing: 10
            Label { text: qsTr("Stop loss [%]"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
            SpinBox {
                id: slSpin
                property real percentValue: value / 10.0
                from: 0
                to: 200
                value: runtimeService ? Math.round((runtimeService.riskControls.stopLossPct || 0) * 10) : 0
                stepSize: 1
                textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) }
                valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 10) }
            }
        }
        RowLayout {
            spacing: 10
            Label { text: qsTr("Maks. liczba pozycji"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
            SpinBox {
                id: positionsSpin
                from: 0
                to: 50
                value: runtimeService ? runtimeService.riskControls.maxOpenPositions : 0
                stepSize: 1
            }
        }
        RowLayout {
            spacing: 10
            Label { text: qsTr("Limit pozycji [USD]"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
            TextField {
                id: positionLimit
                text: runtimeService ? String(runtimeService.riskControls.maxPositionUsd || 0) : "0"
                Layout.fillWidth: true
                color: designSystem.color("textPrimary")
                background: Rectangle {
                    implicitHeight: 36
                    radius: 8
                    color: designSystem.color("surface")
                    border.color: designSystem.color("border")
                }
            }
        }
        RowLayout {
            spacing: 10
            Label { text: qsTr("Maks. slippage [%]"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
            SpinBox {
                id: slippageSpin
                property real percentValue: value / 10.0
                from: 0
                to: 50
                value: runtimeService ? Math.round((runtimeService.riskControls.maxSlippagePct || 0) * 10) : 0
                stepSize: 1
                textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) }
                valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 10) }
            }
        }
        RowLayout {
            spacing: 10
            Label { text: qsTr("Risk kill-switch (portfel/ryzyko)"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
            Switch {
                id: killSwitchToggle
                checked: runtimeService ? runtimeService.riskControls.killSwitch : false
            }
        }
    }

    Connections {
        target: runtimeService
        function onRiskControlsChanged() {
            if (!runtimeService)
                return
            const rc = runtimeService.riskControls || {}
            tpSpin.value = Math.round((rc.takeProfitPct || 0) * 10)
            slSpin.value = Math.round((rc.stopLossPct || 0) * 10)
            positionsSpin.value = rc.maxOpenPositions || 0
            positionLimit.text = String(rc.maxPositionUsd || 0)
            slippageSpin.value = Math.round((rc.maxSlippagePct || 0) * 10)
            killSwitchToggle.checked = rc.killSwitch || false
        }
        function onOpportunityRuntimeSettingsChanged() {
            root.opportunitySettings = runtimeService ? runtimeService.opportunityRuntimeSettings || ({}) : ({})
        }
    }

    GroupBox {
        title: qsTr("Opportunity AI / Policy Runtime")
        Layout.fillWidth: true

        ColumnLayout {
            anchors.fill: parent
            spacing: 8

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity AI enabled"); Layout.fillWidth: true; color: designSystem.color("textSecondary") }
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: opportunitySettings && opportunitySettings.opportunityAiEnabled ? "#2ecc71" : "#7f8c8d"
                }
                Switch {
                    checked: opportunitySettings && opportunitySettings.opportunityAiEnabled
                    onToggled: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({
                        opportunityAiEnabled: checked
                    }) : null
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity manual kill-switch"); Layout.fillWidth: true; color: designSystem.color("textSecondary") }
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: opportunitySettings && opportunitySettings.manualKillSwitch ? "#e74c3c" : "#7f8c8d"
                }
                Switch {
                    checked: opportunitySettings && opportunitySettings.manualKillSwitch
                    onToggled: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({
                        manualKillSwitch: checked
                    }) : null
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Env override: AI enabled"); Layout.fillWidth: true; color: designSystem.color("textSecondary") }
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: opportunitySettings && opportunitySettings.envOverrideEnabledActive ? "#e74c3c" : "#7f8c8d"
                }
                Label {
                    text: opportunitySettings && opportunitySettings.envOverrideEnabledActive
                          ? qsTr("Aktywny (%1)").arg(String(opportunitySettings.envOverrideEnabledValue))
                          : qsTr("Brak")
                    color: designSystem.color("textSecondary")
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Env override: kill-switch"); Layout.fillWidth: true; color: designSystem.color("textSecondary") }
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: opportunitySettings && opportunitySettings.envOverrideKillSwitchActive ? "#e74c3c" : "#7f8c8d"
                }
                Label {
                    text: opportunitySettings && opportunitySettings.envOverrideKillSwitchActive
                          ? qsTr("Aktywny (%1)").arg(String(opportunitySettings.envOverrideKillSwitchValue))
                          : qsTr("Brak")
                    color: designSystem.color("textSecondary")
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity policy mode"); Layout.fillWidth: true; color: designSystem.color("textSecondary") }
                ComboBox {
                    model: ["shadow", "assist", "live"]
                    currentIndex: {
                        var mode = opportunitySettings && opportunitySettings.policyMode
                                   ? String(opportunitySettings.policyMode) : "shadow"
                        var idx = model.indexOf(mode)
                        return idx >= 0 ? idx : 0
                    }
                    onActivated: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({
                        policyMode: currentText
                    }) : null
                }
                Label { text: qsTr("Effective"); color: designSystem.color("textSecondary") }
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: opportunitySettings && opportunitySettings.effectiveAiEnabled ? "#2ecc71" : "#e74c3c"
                }
            }

            Label {
                Layout.fillWidth: true
                color: designSystem.color("textSecondary")
                text: qsTr("Source: %1").arg(opportunitySettings && opportunitySettings.sourceOfTruth
                                            ? String(opportunitySettings.sourceOfTruth)
                                            : "runtime_control_plane")
            }
            Label {
                Layout.fillWidth: true
                color: designSystem.color("textSecondary")
                text: qsTr("Uwaga: risk kill-switch ≠ opportunity AI manual kill-switch.")
            }
        }
    }

    RowLayout {
        spacing: 8
        Components.IconButton {
            designSystem: designSystem
            text: qsTr("Zapisz limity")
            iconName: "save"
            onClicked: {
                if (!runtimeService || !runtimeService.saveRiskControls)
                    return
                const result = runtimeService.saveRiskControls({
                    takeProfitPct: tpSpin.percentValue,
                    stopLossPct: slSpin.percentValue,
                    maxOpenPositions: positionsSpin.value,
                    maxPositionUsd: Number(positionLimit.text),
                    maxSlippagePct: slippageSpin.percentValue,
                    killSwitch: killSwitchToggle.checked
                })
                statusLabel.text = result && result.message ? result.message : qsTr("Zapisano")
            }
        }
        Components.IconButton {
            designSystem: designSystem
            text: qsTr("Odśwież")
            iconName: "refresh"
            subtle: true
            onClicked: runtimeService && runtimeService.loadRiskControls()
        }
        Label {
            id: statusLabel
            color: designSystem.color("textSecondary")
        }
    }
}
