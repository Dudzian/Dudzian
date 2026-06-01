import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

ScrollView {
    id: root
    objectName: "riskControlsPreviewPanel"
    property var runtimeService
    property var designSystem
    property var riskControls: runtimeService && runtimeService.riskControls ? runtimeService.riskControls : ({})
    property var opportunitySettings: runtimeService && runtimeService.opportunityRuntimeSettings
                                      ? runtimeService.opportunityRuntimeSettings : ({})
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            Label {
                objectName: "riskControlsPreviewTitle"
                text: qsTr("Kontrola ryzyka")
                font.bold: true
                font.pixelSize: 22
                color: designSystem.color("textPrimary")
            }
            Label {
                text: qsTr("Preview limitów, runtime policy i kill-switch. Live trading disabled/blocked, exchange/order disabled, API keys not required.")
                color: designSystem.color("textSecondary")
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Runtime loop")
                description: qsTr("Not started — panel nie uruchamia workerów ani adapterów giełdowych.")
                Layout.fillWidth: true
            }
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Order policy")
                description: qsTr("Order submission disabled. Zmiany są bezpiecznym preview ustawień UI.")
                Layout.fillWidth: true
            }
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("API keys")
                description: qsTr("Not required — panel nie czyta sekretów ani konfiguracji środowiska.")
                Layout.fillWidth: true
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Limity pozycji")
            description: qsTr("Jawnie ostylowane SpinBox/TextField utrzymują kontrast na ciemnym tle.")

            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 10
                columnSpacing: 14

                Label { text: qsTr("Take profit [%]"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox {
                    id: tpSpin
                    designSystem: root.designSystem
                    property real percentValue: value / 10.0
                    from: 0; to: 200; value: runtimeService ? Math.round((root.riskControls.takeProfitPct || 0) * 10) : 18; stepSize: 1
                    textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) }
                    valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 10) }
                    Layout.fillWidth: true
                }

                Label { text: qsTr("Stop loss [%]"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox {
                    id: slSpin
                    designSystem: root.designSystem
                    property real percentValue: value / 10.0
                    from: 0; to: 200; value: runtimeService ? Math.round((root.riskControls.stopLossPct || 0) * 10) : 11; stepSize: 1
                    textFromValue: function(value, locale) { return Number(value / 10).toLocaleString(locale, 'f', 1) }
                    valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 10) }
                    Layout.fillWidth: true
                }

                Label { text: qsTr("Maks. otwarte pozycje"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox {
                    id: positionsSpin
                    designSystem: root.designSystem
                    from: 0; to: 25; value: runtimeService ? (root.riskControls.maxOpenPositions || 0) : 3
                    Layout.fillWidth: true
                }

                Label { text: qsTr("Maks. pozycja [USD]"); color: root.designSystem.color("textSecondary") }
                Components.StyledTextField {
                    id: positionLimit
                    designSystem: root.designSystem
                    text: String(runtimeService ? (root.riskControls.maxPositionUsd || 0) : 2500)
                    Layout.fillWidth: true
                }

                Label { text: qsTr("Maks. slippage [%]"); color: root.designSystem.color("textSecondary") }
                Components.StyledSpinBox {
                    id: slippageSpin
                    designSystem: root.designSystem
                    property real percentValue: value / 100.0
                    from: 0; to: 1000; value: runtimeService ? Math.round((root.riskControls.maxSlippagePct || 0) * 100) : 35; stepSize: 5
                    textFromValue: function(value, locale) { return Number(value / 100).toLocaleString(locale, 'f', 2) }
                    valueFromText: function(text, locale) { return Math.round(Number.fromLocaleString(locale, text) * 100) }
                    Layout.fillWidth: true
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Kill-switch preview")
            description: qsTr("Przełączniki są jawnie ostylowane. Risk kill-switch pozostaje oddzielony od opportunity governor manual kill-switch.")

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Risk kill-switch"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                Rectangle { width: 12; height: 12; radius: 6; color: killSwitchToggle.checked ? "#e74c3c" : "#2ecc71" }
                Components.StyledSwitch {
                    id: killSwitchToggle
                    designSystem: root.designSystem
                    checked: runtimeService ? root.riskControls.killSwitch : false
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity governor enabled"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                Rectangle { width: 12; height: 12; radius: 6; color: opportunityEnabled.checked ? "#2ecc71" : "#7f8c8d" }
                Components.StyledSwitch {
                    id: opportunityEnabled
                    designSystem: root.designSystem
                    checked: root.opportunitySettings && root.opportunitySettings.opportunityAiEnabled ? true : false
                    onToggled: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({ opportunityAiEnabled: checked }) : null
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity manual kill-switch"); color: root.designSystem.color("textPrimary"); Layout.fillWidth: true }
                Rectangle { width: 12; height: 12; radius: 6; color: opportunityKill.checked ? "#e74c3c" : "#7f8c8d" }
                Components.StyledSwitch {
                    id: opportunityKill
                    designSystem: root.designSystem
                    checked: root.opportunitySettings && root.opportunitySettings.manualKillSwitch ? true : false
                    onToggled: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({ manualKillSwitch: checked }) : null
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Runtime policy mode")
            description: qsTr("Tryb policy jest lokalnym preview. Efektywne AI pozostaje ograniczone przez blokady runtime.")

            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("Opportunity policy mode"); color: root.designSystem.color("textSecondary"); Layout.fillWidth: true }
                ComboBox {
                    id: policyMode
                    model: ["shadow", "assist", "live"]
                    currentIndex: {
                        var mode = root.opportunitySettings && root.opportunitySettings.policyMode ? String(root.opportunitySettings.policyMode) : "shadow"
                        var idx = model.indexOf(mode)
                        return idx >= 0 ? idx : 0
                    }
                    onActivated: runtimeService ? runtimeService.applyOpportunityRuntimeSettings({ policyMode: currentText }) : null
                    contentItem: Text {
                        text: policyMode.displayText
                        color: root.designSystem.color("textPrimary")
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: 12
                    }
                    background: Rectangle {
                        radius: 10
                        color: root.designSystem.color("surfaceMuted")
                        border.color: root.designSystem.color("border")
                        border.width: 1
                    }
                    popup.background: Rectangle { color: root.designSystem.color("surfaceElevated"); border.color: root.designSystem.color("border"); radius: 10 }
                }
                Label {
                    text: root.opportunitySettings && root.opportunitySettings.effectiveAiEnabled ? qsTr("Effective: enabled") : qsTr("Effective: blocked")
                    color: root.designSystem.color("textSecondary")
                }
            }
            Label {
                Layout.fillWidth: true
                color: root.designSystem.color("textSecondary")
                wrapMode: Text.WordWrap
                text: qsTr("Source: %1").arg(root.opportunitySettings && root.opportunitySettings.sourceOfTruth ? String(root.opportunitySettings.sourceOfTruth) : "runtime_control_plane")
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Components.IconButton {
                designSystem: root.designSystem
                text: qsTr("Zapisz limity")
                iconName: "save"
                onClicked: {
                    if (!runtimeService || !runtimeService.saveRiskControls) {
                        statusLabel.text = qsTr("Demo/offline — brak zapisu do runtime")
                        return
                    }
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
                designSystem: root.designSystem
                text: qsTr("Odśwież")
                iconName: "refresh"
                subtle: true
                onClicked: {
                    if (runtimeService && runtimeService.refreshRiskControls)
                        runtimeService.refreshRiskControls()
                    statusLabel.text = qsTr("Odświeżono preview")
                }
            }
            Label {
                id: statusLabel
                text: qsTr("Safe preview gotowy")
                color: root.designSystem.color("textSecondary")
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }
        }
    }
}
