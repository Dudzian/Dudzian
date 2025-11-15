import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Item {
    id: root
    objectName: "strategyManagerView"
    property var marketplaceController: (typeof marketplaceController !== "undefined" ? marketplaceController : null)
    property string targetPortfolioId: ""
    property string statusMessage: ""
    property string statusError: ""
    property var presetsCache: []

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
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 4
                    Label {
                        text: (modelData.name || modelData.presetId || "") + " • v" + (modelData.version || "-")
                        font.bold: true
                    }
                    Label {
                        text: modelData.summary || qsTr("Brak opisu")
                        color: palette.mid
                        wrapMode: Text.WordWrap
                    }
                    Label {
                        text: modelData.assignedPortfolios && modelData.assignedPortfolios.length > 0
                              ? qsTr("Portfele: %1").arg(modelData.assignedPortfolios.join(", "))
                              : qsTr("Brak przypisań")
                        color: palette.mid
                    }
                    RowLayout {
                        spacing: 8
                        Components.IconButton {
                            text: qsTr("Zainstaluj i przypisz")
                            icon.name: "download"
                            onClicked: root.quickInstall(modelData.presetId)
                        }
                        Components.IconButton {
                            text: qsTr("Tylko przypisz")
                            icon.name: "task"
                            subtle: true
                            enabled: root.targetPortfolioId.length > 0
                            onClicked: root.assignPreset(modelData.presetId, root.targetPortfolioId)
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        presetsCache = controller() && controller().presets ? controller().presets : []
    }
}
