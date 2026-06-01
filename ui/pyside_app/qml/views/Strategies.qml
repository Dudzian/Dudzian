import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "strategiesPreviewPanel"
    property var runtimeService

    property var demoStrategies: [
        ({ id: "btc-shadow-momentum", name: qsTr("BTC Shadow Momentum"), mode: "demo/offline", profile: "balanced", params: ({ max_position_usd: 2500, confidence_floor: 0.72, cooldown_sec: 90 }) }),
        ({ id: "btc-range-guard", name: qsTr("BTC Range Guard"), mode: "preview", profile: "defensive", params: ({ max_drawdown_pct: 2.5, stop_loss_pct: 1.1, take_profit_pct: 1.8 }) })
    ]
    property var strategyModel: runtimeService && runtimeService.strategyConfigs && runtimeService.strategyConfigs.length > 0
                                ? runtimeService.strategyConfigs
                                : demoStrategies

    ColumnLayout {
        width: root.availableWidth
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            Label {
                objectName: "strategiesPreviewTitle"
                text: qsTr("Strategie i parametry")
                font.bold: true
                font.pixelSize: 22
                color: designSystem.color("textPrimary")
                Layout.fillWidth: true
            }
            Label {
                text: qsTr("Panel preview używa bezpiecznych parametrów demo/offline. Live trading disabled, exchange/order disabled, API keys not required.")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
                Layout.fillWidth: true
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Status konfiguracji")
                description: runtimeService && runtimeService.strategyConfigs && runtimeService.strategyConfigs.length > 0
                             ? qsTr("Załadowano lokalne konfiguracje strategii z preview bridge.")
                             : qsTr("Brak konfiguracji runtime — pokazuję statyczny demo/offline empty state.")
                Layout.fillWidth: true
            }
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Runtime policy")
                description: qsTr("Runtime loop not started. Zapis w preview aktualizuje tylko lokalny kontroler UI, bez zleceń.")
                Layout.fillWidth: true
            }
        }

        Repeater {
            model: root.strategyModel
            delegate: Components.PreviewCard {
                designSystem: root.designSystem
                title: (modelData.name || modelData.id) + qsTr(" • profil %1").arg(modelData.profile || "balanced")
                description: qsTr("Tryb: %1. Formularz ma jawne kolory i nie używa surowych natywnych pól Windows.").arg(modelData.mode || "preview")
                property var workingCopy: JSON.parse(JSON.stringify(modelData))

                Repeater {
                    model: Object.keys(workingCopy.params || {})
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        Label {
                            text: modelData
                            color: root.designSystem.color("textSecondary")
                            Layout.preferredWidth: 180
                            elide: Text.ElideRight
                        }
                        Components.StyledTextField {
                            designSystem: root.designSystem
                            text: String((workingCopy.params || {})[modelData])
                            Layout.fillWidth: true
                            onEditingFinished: {
                                var asNumber = Number(text)
                                workingCopy.params[modelData] = isNaN(asNumber) ? text : asNumber
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Components.IconButton {
                        designSystem: root.designSystem
                        text: qsTr("Zapisz preview")
                        iconName: "copy"
                        onClicked: {
                            if (runtimeService && runtimeService.saveStrategyConfig) {
                                const result = runtimeService.saveStrategyConfig(workingCopy.id, workingCopy)
                                statusLabel.text = result && result.message ? result.message : qsTr("Zapisano w preview")
                            } else {
                                statusLabel.text = qsTr("Demo/offline — brak zapisu do runtime")
                            }
                        }
                    }
                    Components.IconButton {
                        designSystem: root.designSystem
                        text: qsTr("Reset widoku")
                        iconName: "refresh"
                        subtle: true
                        onClicked: statusLabel.text = qsTr("Parametry demo pozostają lokalne")
                    }
                    Label {
                        id: statusLabel
                        text: qsTr("Gotowe do podglądu")
                        color: root.designSystem.color("textSecondary")
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Components.IconButton {
                designSystem: root.designSystem
                text: qsTr("Odśwież strategie")
                iconName: "refresh"
                subtle: true
                onClicked: runtimeService && runtimeService.loadStrategyConfigs()
            }
            Label {
                text: qsTr("Safe demo/offline: brak live adapters, brak order execution.")
                color: root.designSystem.color("textSecondary")
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }
        }
    }
}
