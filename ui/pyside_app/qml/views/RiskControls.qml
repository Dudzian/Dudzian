import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles
import "../components" as Components

ColumnLayout {
    id: root
    property var runtimeService
    property var designSystem

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
            Label { text: qsTr("Kill-switch aktywny"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 200 }
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
