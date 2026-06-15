import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "riskControlsPreviewPanel"
    property var runtimeService
    property var previewState
    property var profiles: ["Conservative", "Balanced", "Aggressive", "Custom", "AI Recommended"]
    contentWidth: availableWidth
    clip: true

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "riskControlsTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "riskControlsTitle"; text: qsTr("Ryzyko"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Kokpit ryzyka dla bezpiecznego Paper Preview. Profile Conservative/Balanced/Aggressive/Custom/AI Recommended aktualizują wyłącznie lokalne limity; live trading, exchange I/O i order submission pozostają wyłączone."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Risk profile segmented control")
            description: qsTr("Choose Conservative, Balanced, Aggressive, Custom or AI Recommended. Active segment updates local preview riskProfile and active limits only.")
            RowLayout {
                objectName: "riskProfileSegmentedControl"
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.profiles
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.riskProfile === modelData
                        Layout.fillWidth: true
                        implicitHeight: 44
                        radius: 14
                        color: active ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        ToolTip.delay: 800
                        ToolTip.visible: riskProfileMouseArea.containsMouse
                        ToolTip.text: previewState.tooltipText("Risk profile " + modelData)
                        Label { anchors.centerIn: parent; text: modelData; color: active ? designSystem.color("surface") : designSystem.color("textPrimary"); font.bold: true }
                        MouseArea { id: riskProfileMouseArea; anchors.fill: parent; hoverEnabled: true; onClicked: previewState.setRiskProfile(modelData) }
                    }
                }
            }
        }

        Components.PreviewCard {
            objectName: "riskCustomProfileCard"
            designSystem: root.designSystem
            title: qsTr("Własny / Custom — preview-only limits")
            description: qsTr("Edytowalne wartości są zapisywane tylko w lokalnym UI preview state. Nie zapisują real runtime config i nie uruchamiają live/exchange/order path.")
            GridLayout {
                objectName: "riskCustomFieldsGrid"
                Layout.fillWidth: true
                columns: width > 900 ? 3 : 2
                rowSpacing: 8
                columnSpacing: 8
                Repeater {
                    model: [
                        ({ label: "max position", field: "maxPosition", value: previewState.maxPosition, tip: "max position" }),
                        ({ label: "max open positions", field: "maxOpenPositions", value: String(previewState.maxOpenPositions), tip: "max position" }),
                        ({ label: "stop loss", field: "stopLoss", value: previewState.stopLoss, tip: "stop loss" }),
                        ({ label: "take profit", field: "takeProfit", value: previewState.takeProfit, tip: "take profit" }),
                        ({ label: "max slippage", field: "maxSlippage", value: previewState.maxSlippage, tip: "slippage" }),
                        ({ label: "max drawdown", field: "maxDrawdown", value: previewState.maxDrawdown, tip: "drawdown" }),
                        ({ label: "daily loss limit", field: "dailyLossLimit", value: previewState.dailyLossLimit, tip: "daily loss limit" }),
                        ({ label: "per-symbol exposure", field: "perSymbolExposure", value: previewState.perSymbolExposure, tip: "daily loss limit" }),
                        ({ label: "confidence floor", field: "confidenceFloor", value: previewState.confidenceFloor, tip: "confidence floor" }),
                        ({ label: "cooldown", field: "cooldown", value: previewState.cooldown, tip: "confidence floor" }),
                        ({ label: "max allocation", field: "maxAllocation", value: previewState.maxAllocation, tip: "daily loss limit" })
                    ]
                    delegate: ColumnLayout {
                        required property var modelData
                        Layout.fillWidth: true
                        Label { text: modelData.label; color: designSystem.color("textSecondary"); font.pixelSize: 12 }
                        Rectangle {
                            objectName: "riskCustomField_" + modelData.field
                            Layout.fillWidth: true
                            implicitHeight: 40
                            radius: 10
                            color: designSystem.color("surfaceMuted")
                            border.color: customInput.activeFocus ? designSystem.color("accent") : designSystem.color("border")
                            ToolTip.delay: 800
                            ToolTip.visible: customMouse.containsMouse
                            ToolTip.text: previewState.tooltipText(modelData.tip)
                            TextInput {
                                id: customInput
                                anchors.fill: parent
                                anchors.leftMargin: 12
                                anchors.rightMargin: 12
                                verticalAlignment: TextInput.AlignVCenter
                                color: designSystem.color("textPrimary")
                                selectedTextColor: designSystem.color("surface")
                                selectionColor: designSystem.color("accent")
                                text: modelData.value
                                onEditingFinished: previewState.setCustomRiskValue(modelData.field, text)
                            }
                            MouseArea { id: customMouse; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                    Components.StyledSwitch {
                        objectName: "riskAllowAiOverrideToggle"
                        designSystem: root.designSystem
                        text: qsTr("allow AI override toggle preview-only")
                        checked: previewState.allowAiOverride
                        ToolTip.delay: 800
                        ToolTip.visible: hovered
                        ToolTip.text: previewState.tooltipText("allow AI override")
                        onToggled: previewState.setCustomRiskValue("allowAiOverride", checked)
                    }
                }
            }
        }

        Components.PreviewCard {
            objectName: "riskExplanationCard"
            designSystem: root.designSystem
            title: qsTr("Dlaczego takie ustawienia?")
            description: previewState.riskExplanation
        }

        Components.PreviewCard {
            objectName: "riskActiveLimitsTable"
            designSystem: root.designSystem
            title: qsTr("Aktywne limity")
            description: qsTr("Parametr • aktualna wartość • źródło • komentarz")
            GridLayout {
                Layout.fillWidth: true
                columns: 4
                columnSpacing: 8
                rowSpacing: 6
                Label { text: qsTr("parametr"); color: designSystem.color("textPrimary"); font.bold: true }
                Label { text: qsTr("aktualna wartość"); color: designSystem.color("textPrimary"); font.bold: true }
                Label { text: qsTr("źródło"); color: designSystem.color("textPrimary"); font.bold: true }
                Label { text: qsTr("komentarz"); color: designSystem.color("textPrimary"); font.bold: true }
                Repeater {
                    model: previewState.riskActiveLimits
                    delegate: Repeater {
                        required property var modelData
                        model: [modelData.parameter, modelData.value, modelData.source, modelData.comment]
                        delegate: Label { required property string modelData; text: modelData; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 1000 ? 4 : 2
            rowSpacing: 10
            columnSpacing: 10
            Components.PreviewCard { objectName: "riskStateCard"; descriptionObjectName: "riskStateCardDescription"; designSystem: root.designSystem; title: qsTr("Risk state"); description: previewState.riskState; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "riskMaxPositionCard"; descriptionObjectName: "riskMaxPositionCardDescription"; designSystem: root.designSystem; title: qsTr("Max position"); description: previewState.maxPosition; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max open positions"); description: String(previewState.maxOpenPositions); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Stop loss"); description: previewState.stopLoss; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Take profit"); description: previewState.takeProfit; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Max slippage"); description: previewState.maxSlippage; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "riskMaxDrawdownCard"; descriptionObjectName: "riskMaxDrawdownCardDescription"; designSystem: root.designSystem; title: qsTr("Max drawdown"); description: previewState.maxDrawdown; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "riskDailyLossLimitCard"; descriptionObjectName: "riskDailyLossLimitCardDescription"; designSystem: root.designSystem; title: qsTr("Daily loss limit"); description: previewState.dailyLossLimit; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "riskPerSymbolExposureCard"; descriptionObjectName: "riskPerSymbolExposureCardDescription"; designSystem: root.designSystem; title: qsTr("Per-symbol exposure"); description: previewState.perSymbolExposure; Layout.fillWidth: true }
            Components.PreviewCard { objectName: "riskConfidenceFloorCard"; descriptionObjectName: "riskConfidenceFloorCardDescription"; designSystem: root.designSystem; title: qsTr("Confidence floor"); description: previewState.confidenceFloor; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Cooldown"); description: previewState.cooldown; Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Allow AI override preview-only"); description: previewState.allowAiOverride ? qsTr("enabled locally") : qsTr("disabled locally"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            objectName: "riskSafetyBoundaryCard"
            descriptionObjectName: "riskSafetyBoundaryDescription"
            designSystem: root.designSystem
            title: qsTr("Safety boundary")
            description: qsTr("LIVE DISABLED • EXCHANGE I/O DISABLED • ORDER SUBMISSION DISABLED • API KEYS NOT REQUIRED IN PREVIEW • SECRETS NOT READ • RUNTIME LOOP NOT STARTED • PREVIEW LOCAL ONLY • NO LIVE SIDE EFFECTS • LIVE MODE BLOCKED • RISK GATE / SAFETY LOCK ACTIVE • Safety kill-switch armed • live disabled • exchange route disabled • order submission disabled • paper bridge not connected/planned • Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • No real orders • Runtime loop not started / production runtime loop not started • Risk settings are local preview only. Zablokowane przez ryzyko: brak zmiany PnL/equity, brak pozycji, brak order fill. Blocked reasons: confidence floor, scanner risk score, max position/exposure, daily loss/drawdown, kill-switch/risk lock. Operator can explain blocked state local only. Blocked events update audit/logs only.")
        }
    }
}
