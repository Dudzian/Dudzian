import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../design-system" as DesignSystem

Item {
    id: root
    objectName: "strategyManagerView"
    property var marketplaceController: (typeof marketplaceController !== "undefined" ? marketplaceController : null)
    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property string targetPortfolioId: ""
    property string statusMessage: ""
    property string statusError: ""
    property var presetsCache: []
    property var activationSummary: ({})
    property var guardrailTransitions: []

    function controller() {
        return marketplaceController ? marketplaceController : (typeof appController !== "undefined" ? appController : null)
    }

    function refreshMarketplace() {
        const ctrl = controller()
        if (!ctrl || !ctrl.refreshPresets)
            return
        if (ctrl.busy !== undefined)
            ctrl.busy = true
        try {
            ctrl.refreshPresets()
        } finally {
            presetsCache = ctrl.presets || []
            statusError = ctrl.lastError || ""
            if (!statusError || statusError.length === 0)
                statusMessage = qsTr("Załadowano %1 presetów").arg(presetsCache.length)
        }
    }

    function refreshActivationSummary() {
        if (!runtimeService || runtimeService.regimeActivationSummary === undefined)
            return
        const raw = runtimeService.regimeActivationSummary
        if (!raw || raw.length === 0) {
            activationSummary = {}
            guardrailTransitions = []
            return
        }
        try {
            const parsed = JSON.parse(raw)
            activationSummary = parsed.activePreset || {}
            guardrailTransitions = parsed.guardrailTrace || []
        } catch (err) {
            console.warn("StrategyManager activation summary parse failed", err)
        }
    }

    function assignPreset(presetId, portfolioId) {
        const ctrl = controller()
        if (!ctrl || !ctrl.assignPresetToPortfolio)
            return false
        const trimmedPreset = (presetId || "").trim()
        const trimmedPortfolio = (portfolioId || "").trim()
        if (!trimmedPreset || !trimmedPortfolio)
            return false
        const ok = ctrl.assignPresetToPortfolio(trimmedPreset, trimmedPortfolio)
        if (ok) {
            statusMessage = qsTr("Przypisano preset %1 do %2").arg(trimmedPreset).arg(trimmedPortfolio)
            statusError = ""
            refreshMarketplace()
        } else {
            statusError = ctrl.lastError || qsTr("Nie udało się przypisać portfela")
        }
        return ok
    }

    function quickInstall(presetId) {
        const ctrl = controller()
        if (!ctrl || !ctrl.activateAndAssignPreset)
            return
        const trimmedPortfolio = (targetPortfolioId || "").trim()
        const result = ctrl.activateAndAssignPreset(presetId, trimmedPortfolio)
        if (result && result.success) {
            statusMessage = qsTr("Zainstalowano preset %1").arg(presetId)
            statusError = ""
            refreshMarketplace()
        } else {
            const message = result && result.error ? result.error : (ctrl.lastError || "")
            statusError = message || qsTr("Instalacja presetu nie powiodła się")
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Label {
                text: qsTr("Strategy Manager")
                font.bold: true
                font.pointSize: 18
            }
            Item { Layout.fillWidth: true }
            Components.IconButton {
                text: qsTr("Odśwież")
                icon.name: "reload"
                onClicked: refreshMarketplace()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            TextField {
                id: portfolioField
                Layout.fillWidth: true
                placeholderText: qsTr("Identyfikator portfela docelowego")
                text: root.targetPortfolioId
                onEditingFinished: root.targetPortfolioId = text
            }
            Components.IconButton {
                text: qsTr("Zapisz ID portfela")
                icon.name: "bookmark"
                onClicked: root.targetPortfolioId = portfolioField.text
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: Qt.darker(palette.window, 1.1)
            border.width: 1
            border.color: Qt.darker(palette.window, 1.3)
            padding: 12
            implicitHeight: statusText.implicitHeight + 24
            Label {
                id: statusText
                anchors.fill: parent
                anchors.margins: 8
                text: root.statusError && root.statusError.length > 0 ? root.statusError : root.statusMessage
                color: root.statusError ? "#ff8a80" : palette.windowText
                wrapMode: Text.WordWrap
            }
        }

        Frame {
            Layout.fillWidth: true
            padding: 12
            background: Rectangle {
                color: Qt.darker(palette.base, 1.05)
                radius: 8
            }
            ColumnLayout {
                anchors.fill: parent
                spacing: 4
                Label {
                    text: qsTr("Aktywny preset w runtime")
                    font.bold: true
                }
                Label {
                    text: activationSummary && activationSummary.preset_name
                          ? activationSummary.preset_name
                          : qsTr("Brak aktywnego presetu")
                    font.pointSize: 14
                    font.bold: true
                }
                Label {
                    text: activationSummary && activationSummary.regime
                          ? qsTr("Reżim: %1").arg(activationSummary.regime)
                          : qsTr("Reżim: —")
                    color: palette.mid
                }
                Label {
                    text: activationSummary && activationSummary.preset_hash
                          ? qsTr("Hash: %1").arg(activationSummary.preset_hash)
                          : qsTr("Hash: —")
                    color: palette.mid
                }
                Label {
                    text: activationSummary && activationSummary.used_fallback
                          ? qsTr("Wykorzystano fallback")
                          : qsTr("Tryb standardowy")
                    color: activationSummary && activationSummary.used_fallback ? "#f39c12" : palette.mid
                }
                ColumnLayout {
                    visible: guardrailTransitions && guardrailTransitions.length > 0
                    spacing: 2
                    Label {
                        text: qsTr("Ostatnie blokady guardrail")
                        font.bold: true
                    }
                    Repeater {
                        model: guardrailTransitions
                        delegate: Label {
                            Layout.fillWidth: true
                            text: {
                                var ts = modelData.timestamp ? new Date(modelData.timestamp).toLocaleString() : ""
                                var reason = modelData.reasons && modelData.reasons.length > 0
                                        ? modelData.reasons.join(", ")
                                        : qsTr("brak danych")
                                return ts ? qsTr("%1 • %2").arg(ts).arg(reason) : reason
                            }
                            color: "#f1c40f"
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }

        ListView {
            id: presetsList
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            clip: true
            model: root.presetsCache
            delegate: Rectangle {
                width: ListView.view.width
                radius: 12
                color: Qt.darker(palette.base, 1.05)
                border.width: 1
                border.color: Qt.darker(palette.base, 1.2)
                property var presetEntry: modelData || ({})
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 4
                    Label {
                        text: (presetEntry.name || presetEntry.presetId || "") + " • v" + (presetEntry.version || "-")
                        font.bold: true
                    }
                    Label {
                        text: presetEntry.summary || qsTr("Brak opisu")
                        color: palette.mid
                        wrapMode: Text.WordWrap
                    }
                    DesignSystem.FrostedGlass {
                        Layout.fillWidth: true
                        visible: presetEntry.userPreferences && presetEntry.userPreferences.length > 0
                        sourceItem: root
                        radius: 14
                        blurRadius: 36
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 4
                            Repeater {
                                model: presetEntry.userPreferences || []
                                delegate: ColumnLayout {
                                    spacing: 2
                                    Label {
                                        text: qsTr("Persona: %1").arg(modelData.persona || qsTr("profil"))
                                        font.bold: true
                                    }
                                    RowLayout {
                                        spacing: 6
                                        DesignSystem.Icon {
                                            glyph: "\uf3ed"
                                            size: 14
                                            color: DesignSystem.Palette.accent
                                        }
                                        Label {
                                            text: qsTr("Ryzyko: %1").arg(modelData.risk_target || qsTr("brak"))
                                            color: DesignSystem.Palette.textSecondary
                                        }
                                        DesignSystem.Icon {
                                            glyph: "\uf49e"
                                            size: 14
                                            color: DesignSystem.Palette.accent
                                        }
                                        Label {
                                            text: modelData.recommended_budget
                                                  ? qsTr("Budżet: %1 USD").arg(Number(modelData.recommended_budget).toLocaleString(Qt.locale(), 'f', 0))
                                                  : qsTr("Budżet: brak")
                                            color: DesignSystem.Palette.textSecondary
                                        }
                                    }
                                    RowLayout {
                                        spacing: 6
                                        DesignSystem.Icon {
                                            glyph: "\uf017"
                                            size: 14
                                            color: DesignSystem.Palette.accent
                                        }
                                        Label {
                                            text: qsTr("Horyzont: %1").arg(modelData.holding_period || qsTr("brak"))
                                            color: DesignSystem.Palette.textSecondary
                                        }
                                        DesignSystem.Icon {
                                            glyph: "\uf544"
                                            size: 14
                                            color: DesignSystem.Palette.accent
                                        }
                                        Label {
                                            text: modelData.automation
                                                  ? qsTr("Automatyzacja: %1").arg(modelData.automation)
                                                  : qsTr("Automatyzacja: brak")
                                            color: DesignSystem.Palette.textSecondary
                                        }
                                    }
                                    Label {
                                        text: modelData.notes || ""
                                        visible: text.length > 0
                                        wrapMode: Text.WordWrap
                                        color: DesignSystem.Palette.textSecondary
                                    }
                                }
                            }
                        }
                    }
                    Label {
                        text: presetEntry.assignedPortfolios && presetEntry.assignedPortfolios.length > 0
                              ? qsTr("Portfele: %1").arg(presetEntry.assignedPortfolios.join(", "))
                              : qsTr("Brak przypisań")
                        color: palette.mid
                    }
                    RowLayout {
                        spacing: 8
                        Components.IconButton {
                            text: qsTr("Zainstaluj i przypisz")
                            icon.name: "download"
                            onClicked: root.quickInstall(presetEntry.presetId)
                        }
                        Components.IconButton {
                            text: qsTr("Tylko przypisz")
                            icon.name: "task"
                            subtle: true
                            enabled: root.targetPortfolioId.length > 0
                            onClicked: root.assignPreset(presetEntry.presetId, root.targetPortfolioId)
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        presetsCache = controller() && controller().presets ? controller().presets : []
        refreshActivationSummary()
    }

    onRuntimeServiceChanged: refreshActivationSummary()

    Connections {
        target: runtimeService ? runtimeService : null
        function onRegimeActivationSummaryChanged() {
            refreshActivationSummary()
        }
    }
}
