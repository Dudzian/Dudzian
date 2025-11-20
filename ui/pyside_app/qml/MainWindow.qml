import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "components" as Components
import "components/layout" as LayoutComponents
import "styles" as Styles
import "views" as Views

ApplicationWindow {
    id: root
    width: 1280
    height: 720
    visible: true
    title: qsTr("Stage6 PySide UI")
    color: designSystem.color("background")

    property var panelMetadata: [
        ({ panelId: "sidePanel", title: qsTr("Panel statusu"), icon: "fingerprint", defaultColumn: 0, defaultOrder: 0 }),
        ({ panelId: "telemetryPanel", title: qsTr("Telemetria feedu"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 1 }),
        ({ panelId: "aiDecisionsPanel", title: qsTr("Decyzje AI"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 2 }),
        ({ panelId: "diagnosticsPanel", title: qsTr("Diagnostyka"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 3 }),
        ({ panelId: "chartView", title: qsTr("Chart & Decision Stream"), icon: "cloud", defaultColumn: 1, defaultOrder: 0 }),
        ({ panelId: "strategyWorkbench", title: qsTr("Strategy Workbench"), icon: "package", defaultColumn: 1, defaultOrder: 1 }),
        ({ panelId: "modeWizardPanel", title: qsTr("Tryby pracy"), icon: "mode_wizard", defaultColumn: 1, defaultOrder: 2 }),
        ({ panelId: "strategyManagerPanel", title: qsTr("Strategy Manager"), icon: "strategy_manager", defaultColumn: 1, defaultOrder: 3 })
    ]

    property var panelRegistry: ({
        "sidePanel": { title: qsTr("Panel statusu"), icon: "fingerprint", component: sidePanelComponent },
        "telemetryPanel": { title: qsTr("Telemetria feedu"), icon: "diagnostics", component: telemetryPanelComponent },
        "chartView": { title: qsTr("Chart & Decision Stream"), icon: "cloud", component: chartViewComponent },
        "strategyWorkbench": { title: qsTr("Strategy Workbench"), icon: "package", component: strategyWorkbenchComponent },
        "modeWizardPanel": { title: qsTr("Tryby pracy"), icon: "mode_wizard", component: modeWizardPanelComponent },
        "strategyManagerPanel": { title: qsTr("Strategy Manager"), icon: "strategy_manager", component: strategyManagerPanelComponent },
        "diagnosticsPanel": { title: qsTr("Diagnostyka"), icon: "diagnostics", component: diagnosticsPanelComponent },
        "aiDecisionsPanel": { title: qsTr("Decyzje AI"), icon: "mode_wizard", component: aiDecisionsPanelComponent }
    })

    Styles.DesignSystem {
        id: designSystem
        themeBridge: theme
    }

    Dialog {
        id: startupDialog
        modal: true
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        title: qsTr("Stan backendu")
        property string body: ""
        onAccepted: visible = false

        contentItem: ColumnLayout {
            spacing: 12
            padding: 16

            Label {
                id: statusBody
                text: startupDialog.body
                wrapMode: Text.WordWrap
                color: designSystem.color("textPrimary")
                Layout.preferredWidth: 420
            }

            Label {
                text: qsTr("Jeśli problem dotyczy konfiguracji, sprawdź plik runtime.yaml lub flagę cloud.")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
                visible: startupDialog.title.indexOf(qsTr("Błąd")) !== -1
            }
        }
    }

    Connections {
        target: runtimeService
        function onErrorMessageChanged() {
            if (!runtimeService)
                return
            if (runtimeService.errorMessage && runtimeService.errorMessage.length > 0) {
                startupDialog.title = qsTr("Błąd uruchomienia runtime")
                startupDialog.body = runtimeService.errorMessage
                startupDialog.open()
            }
        }
        function onCloudRuntimeStatusChanged() {
            if (!runtimeService)
                return
            const status = runtimeService.cloudRuntimeStatus || {}
            if (status.status === "ready") {
                const targetLabel = status.target || (cloudRuntimeEnabled ? qsTr("profil cloud") : qsTr("tryb lokalny"))
                startupDialog.title = qsTr("Runtime gotowy")
                startupDialog.body = qsTr("Połączenie z backendem %1 aktywne.").arg(targetLabel)
                startupDialog.open()
            }
        }
    }

    Menu {
        id: panelMenu
        Repeater {
            model: layoutController ? layoutController.availablePanels : []
            delegate: MenuItem {
                readonly property var entry: modelData
                text: entry && entry.title ? entry.title : entry.panelId
                checkable: true
                checked: entry && entry.visible !== false
                onTriggered: {
                    if (layoutController && entry) {
                        var currentVisible = layoutController.isPanelVisible(entry.panelId)
                        layoutController.setPanelVisibility(entry.panelId, !currentVisible)
                    }
                }
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
            GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
        }
        z: -2
    }

    header: ToolBar {
        id: toolbar
        implicitHeight: 64
        background: Item {
            anchors.fill: parent
            Rectangle {
                id: toolbarGradient
                anchors.fill: parent
                gradient: Gradient {
                    GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
                    GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
                }
                opacity: 0.9
            }
            MultiEffect {
                anchors.fill: parent
                source: toolbarGradient
                blurEnabled: true
                blurRadius: 24
                saturation: 0.95
                brightness: 0.05
            }
        }
        RowLayout {
            anchors.fill: parent
            spacing: 12

            Label {
                text: qsTr("Endpoint: %1").arg(uiConfig && uiConfig.endpoint ? uiConfig.endpoint : "local-demo")
                font.bold: true
                color: designSystem.color("textPrimary")
                Layout.alignment: Qt.AlignVCenter
            }

            Rectangle { width: 1; height: parent.height * 0.6; color: designSystem.color("border"); opacity: 0.4 }

            Label {
                readonly property var cloudStatus: runtimeService ? runtimeService.cloudRuntimeStatus : ({})
                readonly property var handshake: cloudStatus.handshake || ({})
                text: cloudRuntimeEnabled
                      ? qsTr("Cloud: %1 • handshake: %2")
                            .arg(cloudStatus.target || "client.yaml")
                            .arg(handshake.status || qsTr("oczekuje"))
                      : qsTr("Cloud runtime: wyłączony")
                color: designSystem.color("textSecondary")
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            Components.IconButton {
                id: layoutButton
                designSystem: designSystem
                text: qsTr("Panele")
                iconName: "package"
                subtle: true
                onClicked: panelMenu.popup(layoutButton)
            }

            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Strategie")
                iconName: "strategy_manager"
                subtle: true
                onClicked: {
                    if (layoutController)
                        layoutController.setPanelVisibility("strategyManagerPanel", true)
                }
            }

            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Tryby pracy")
                iconName: "mode_wizard"
                subtle: true
                onClicked: modeWizardDialog.open()
            }

            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Odśwież dane")
                iconName: "refresh"
                backgroundColor: designSystem.color("accent")
                foregroundColor: designSystem.color("surface")
                onClicked: runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
            }
        }
    }

    LayoutComponents.DockManager {
        id: dockManager
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: toolbar.bottom
        anchors.margins: 16
        layoutController: layoutController
        panelRegistry: panelRegistry
        designSystem: designSystem
    }

    Component {
        id: sidePanelComponent
        ColumnLayout {
            spacing: 8
            Label {
                text: licensingController.licenseAccepted
                      ? qsTr("Licencja aktywna: %1").arg(licensingController.licenseId || "-")
                      : qsTr("Licencja nieaktywna")
                font.bold: true
                wrapMode: Text.WordWrap
                color: designSystem.color("textPrimary")
            }
            Label {
                text: licensingController.statusDetails.length > 0
                      ? licensingController.statusDetails
                      : qsTr("Użyj kreatora, aby aktywować licencję OEM.")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
            }
            RowLayout {
                spacing: 8
                Components.IconButton {
                    designSystem: designSystem
                    iconName: "fingerprint"
                    text: qsTr("Fingerprint")
                    subtle: true
                    onClicked: licensingController.refreshFingerprint()
                }
                Components.IconButton {
                    designSystem: designSystem
                    iconName: "shield"
                    text: qsTr("Reset")
                    subtle: true
                    onClicked: licensingController.resetStatus()
                }
            }
            Rectangle { height: 1; width: parent.width; color: designSystem.color("border"); opacity: 0.3 }
            ColumnLayout {
                visible: cloudRuntimeEnabled
                spacing: 4
                readonly property var cloudStatus: runtimeService ? runtimeService.cloudRuntimeStatus : ({})
                readonly property var handshake: cloudStatus.handshake || ({})
                Label {
                    text: qsTr("Cloud endpoint: %1").arg(cloudStatus.target || "client.yaml")
                    color: designSystem.color("textSecondary")
                    wrapMode: Text.WordWrap
                }
                Label {
                    text: qsTr("Licencja: %1 • HWID: %2")
                          .arg(handshake.licenseId || "?")
                          .arg(handshake.fingerprint || "?")
                    color: designSystem.color("textSecondary")
                    wrapMode: Text.WordWrap
                }
                Label {
                    text: qsTr("Status handshake: %1").arg(handshake.status || qsTr("oczekuje"))
                    color: designSystem.color(handshake.status === "ok" ? "success" : "warning")
                }
                Components.IconButton {
                    designSystem: designSystem
                    iconName: "refresh"
                    text: qsTr("Odnów handshake")
                    subtle: true
                    onClicked: runtimeService && runtimeService.refreshCloudHandshake()
                }
            }
            Label {
                text: qsTr("Profil UI: %1").arg(uiConfig && uiConfig.profile ? uiConfig.profile : "default")
                color: designSystem.color("textSecondary")
            }
        }
    }

    Component {
        id: telemetryPanelComponent
        ColumnLayout {
            spacing: 8
            Label {
                text: qsTr("Status feedu: %1").arg(runtimeService.feedHealth.status || qsTr("inicjalizacja"))
                font.bold: true
                color: designSystem.color("textPrimary")
            }
            Label {
                text: qsTr("Ostatni błąd: %1").arg(runtimeService.feedHealth.lastError || "-")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
            }
            Label {
                text: qsTr("Reconnects: %1  •  Downtime: %2 ms")
                      .arg(runtimeService.feedHealth.reconnects || 0)
                      .arg(Math.round(runtimeService.feedHealth.downtimeMs || 0))
                color: designSystem.color("textSecondary")
            }
            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Ping feed")
                iconName: "refresh"
                subtle: true
                onClicked: runtimeService.loadRecentDecisions(0)
            }
        }
    }

    Component {
        id: aiDecisionsPanelComponent
        Views.AiDecisionsView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: designSystem
        }
    }

    Component {
        id: chartViewComponent
        ColumnLayout {
            spacing: 12
            Label {
                text: qsTr("Confidence chart & dziennik decyzji")
                font.bold: true
                color: designSystem.color("textPrimary")
            }
            Canvas {
                id: chartCanvas
                Layout.fillWidth: true
                Layout.preferredHeight: 160
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.fillStyle = designSystem.color("surfaceMuted")
                    ctx.fillRect(0, 0, width, height)
                    var data = runtimeService.decisions || []
                    if (data.length === 0)
                        return
                    var windowSize = Math.min(40, data.length)
                    var step = width / Math.max(windowSize - 1, 1)
                    ctx.strokeStyle = designSystem.color("accent")
                    ctx.lineWidth = 2
                    ctx.beginPath()
                    for (var i = 0; i < windowSize; ++i) {
                        var entry = data[data.length - windowSize + i]
                        var confidence = entry.decision && entry.decision.confidence !== undefined
                                ? Number(entry.decision.confidence)
                                : 0.35
                        confidence = Math.max(0.05, Math.min(confidence, 1.0))
                        var x = i * step
                        var y = height - (confidence * height)
                        if (i === 0)
                            ctx.moveTo(x, y)
                        else
                            ctx.lineTo(x, y)
                    }
                    ctx.stroke()
                }
            }
            Connections {
                target: runtimeService
                function onDecisionsChanged() {
                    chartCanvas.requestPaint()
                }
            }
            ListView {
                id: decisionList
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: runtimeService.decisions
                clip: true
                delegate: Rectangle {
                    width: ListView.view.width
                    color: index % 2 === 0 ? designSystem.color("surfaceMuted") : Qt.rgba(0, 0, 0, 0)
                    opacity: 0.85
                    height: column.implicitHeight + 18
                    radius: 12
                    border.color: designSystem.color("border")
                    border.width: 1
                    Column {
                        id: column
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 4
                        Label {
                            text: qsTr("%1 • %2 • %3").arg(modelData.timestamp || "-")
                                                        .arg(modelData.portfolio || "-")
                                                        .arg(modelData.marketRegime && modelData.marketRegime.label
                                                                 ? modelData.marketRegime.label : "")
                            font.bold: true
                            color: designSystem.color("textPrimary")
                            wrapMode: Text.Wrap
                        }
                        Label {
                            text: modelData.decision && modelData.decision.shouldTrade
                                  ? qsTr("Decyzja: %1 %2 @ %3").arg(modelData.symbol || "-")
                                        .arg(modelData.side || "")
                                        .arg(modelData.price || "")
                                  : qsTr("Decyzja: brak transakcji")
                            wrapMode: Text.Wrap
                            color: designSystem.color("textPrimary")
                        }
                        Label {
                            text: modelData.ai && modelData.ai.strategy ? qsTr("AI: %1").arg(modelData.ai.strategy) : ""
                            color: designSystem.color("textSecondary")
                            visible: text.length > 0
                        }
                    }
                }
            }
        }
    }

    Component {
        id: strategyWorkbenchComponent
        ColumnLayout {
            spacing: 12
            property var strategies: []
            property var marketplacePresets: strategyManagementController ? strategyManagementController.presets : []
            function rebuild() {
                var data = runtimeService.decisions || []
                var stats = {}
                for (var i = 0; i < data.length; ++i) {
                    var entry = data[i]
                    var strategy = entry.ai && entry.ai.strategy ? entry.ai.strategy : qsTr("Nieznana strategia")
                    if (!stats[strategy])
                        stats[strategy] = { count: 0, lastSymbol: entry.symbol || "-" }
                    stats[strategy].count += 1
                    stats[strategy].lastSymbol = entry.symbol || stats[strategy].lastSymbol
                }
                var collection = []
                for (var key in stats) {
                    collection.push({ name: key, count: stats[key].count, symbol: stats[key].lastSymbol })
                }
                collection.sort(function(a, b) { return b.count - a.count })
                strategies = collection
            }
            Component.onCompleted: rebuild()
            Connections {
                target: runtimeService
                function onDecisionsChanged() {
                    rebuild()
                }
            }
            Label {
                text: qsTr("Strategie AI oraz marketplace")
                font.bold: true
                color: designSystem.color("textPrimary")
            }
            ListView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: strategies
                delegate: RowLayout {
                    width: ListView.view.width
                    spacing: 8
                    Label {
                        text: modelData.name
                        font.bold: true
                        color: designSystem.color("textPrimary")
                        Layout.fillWidth: true
                    }
                    Label {
                        text: qsTr("%1 zdarzeń").arg(modelData.count)
                        color: designSystem.color("textSecondary")
                    }
                    Label {
                        text: modelData.symbol
                        color: designSystem.color("textSecondary")
                    }
                }
            }
            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Odśwież strategie")
                iconName: "refresh"
                onClicked: runtimeService.loadRecentDecisions(0)
            }
            Rectangle {
                Layout.fillWidth: true
                radius: 14
                color: designSystem.color("surface")
                opacity: 0.95
                border.color: designSystem.color("border")
                border.width: 1
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 6
                    Label {
                        text: qsTr("Presety Marketplace")
                        font.bold: true
                        color: designSystem.color("textPrimary")
                    }
                    Repeater {
                        model: marketplacePresets
                        delegate: ColumnLayout {
                            visible: index < 3
                            spacing: 2
                            Label {
                                text: modelData.name + " • v" + (modelData.version || "-")
                                font.bold: true
                                color: designSystem.color("textSecondary")
                            }
                            Label {
                                text: modelData.license && modelData.license.status ? qsTr("Licencja: %1").arg(modelData.license.status) : qsTr("Brak licencji")
                                color: designSystem.color("textSecondary")
                            }
                            Label {
                                text: modelData.assignedPortfolios && modelData.assignedPortfolios.length > 0
                                      ? qsTr("Portfele: %1").arg(modelData.assignedPortfolios.join(", "))
                                      : qsTr("Nieprzypisany")
                                color: designSystem.color("textSecondary")
                            }
                            RowLayout {
                                spacing: 6
                                Components.IconButton {
                                    designSystem: designSystem
                                    text: qsTr("Zastosuj")
                                    iconName: "strategy_manager"
                                    subtle: true
                                    onClicked: {
                                        if (strategyManagementController)
                                            strategyManagementController.activateAndAssign(modelData.presetId, "")
                                    }
                                }
                                Components.IconButton {
                                    designSystem: designSystem
                                    text: qsTr("Otwórz w managerze")
                                    iconName: "package"
                                    subtle: true
                                    onClicked: {
                                        if (layoutController)
                                            layoutController.setPanelVisibility("strategyManagerPanel", true)
                                    }
                                }
                            }
                            Rectangle { height: 1; color: designSystem.color("border"); opacity: 0.4; Layout.fillWidth: true }
                        }
                    }
                    RowLayout {
                        spacing: 8
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Odśwież marketplace")
                            iconName: "refresh"
                            subtle: true
                            onClicked: {
                                if (strategyManagementController)
                                    strategyManagementController.refreshMarketplace()
                            }
                        }
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Strategy Manager")
                            iconName: "strategy_manager"
                            onClicked: {
                                if (layoutController)
                                    layoutController.setPanelVisibility("strategyManagerPanel", true)
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: modeWizardPanelComponent
        Views.ModeWizard {
            designSystem: designSystem
            modeWizardController: modeWizardController
            compact: true
            onLaunchWizardRequested: modeWizardDialog.open()
            layoutController: layoutController
            strategyManagementController: strategyManagementController
        }
    }

    Component {
        id: strategyManagerPanelComponent
        Views.StrategyManager {
            designSystem: designSystem
            strategyManagementController: strategyManagementController
            layoutController: layoutController
        }
    }

    Component {
        id: diagnosticsPanelComponent
        ColumnLayout {
            spacing: 12
            Label {
                text: diagnosticsController.statusMessageId
                color: designSystem.color("textPrimary")
            }
            Components.IconButton {
                designSystem: designSystem
                iconName: "diagnostics"
                text: diagnosticsController.busy ? qsTr("Generuję…") : qsTr("Generuj paczkę")
                enabled: !diagnosticsController.busy
                backgroundColor: designSystem.color("accent")
                foregroundColor: designSystem.color("surface")
                onClicked: diagnosticsController.generateDiagnostics()
            }
            Label {
                text: diagnosticsController.lastArchivePath.length > 0
                      ? qsTr("Ostatnia paczka: %1").arg(diagnosticsController.lastArchivePath)
                      : ""
                visible: text.length > 0
                color: designSystem.color("textSecondary")
            }
        }
    }

    Component.onCompleted: {
        if (runtimeService && runtimeService.loadRecentDecisions)
            runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
        if (layoutController && layoutController.registerPanels)
            layoutController.registerPanels(panelMetadata)
    }

    Timer {
        interval: 15000
        repeat: true
        running: true
        onTriggered: runtimeService.loadRecentDecisions(0)
    }

    Dialog {
        id: modeWizardDialog
        modal: true
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.8, 1100)
        height: Math.min(parent.height * 0.85, 780)
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        background: Rectangle {
            anchors.fill: parent
            radius: 24
            color: Qt.rgba(0, 0, 0, 0.75)
            border.color: designSystem.color("border")
            border.width: 1
        }
        contentItem: Views.ModeWizard {
            anchors.fill: parent
            anchors.margins: 16
            designSystem: designSystem
            modeWizardController: modeWizardController
            compact: false
            onLaunchWizardRequested: modeWizardDialog.close()
        }
    }
}
