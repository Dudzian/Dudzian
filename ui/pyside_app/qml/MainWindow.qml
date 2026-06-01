import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "components" as Components
import "components/layout" as LayoutComponents
import Styles 1.0 as StylesModule
import "views" as Views

ApplicationWindow {
    id: root
    width: 1280
    height: 720
    visible: true
    title: qsTr("Stage6 PySide UI")
    color: designSystem.color("background")
    property var contextGrpcBridge: (typeof grpcBridge !== "undefined" ? grpcBridge : null)
    property var runtimeService: contextGrpcBridge && contextGrpcBridge.runtimeService ? contextGrpcBridge.runtimeService : null
    property var contextRuntimeState: (typeof runtimeState !== "undefined" ? runtimeState : null)
    property string defaultPanelId: "sidePanel"
    property string currentPanelId: defaultPanelId
    readonly property var rootDesignSystem: designSystem

    function showPanel(panelId) {
        if (!panelId)
            return
        currentPanelId = panelId
        if (layoutController)
            layoutController.setPanelVisibility(panelId, true)
    }

    function showOperatorDashboard() {
        showPanel(defaultPanelId)
    }

    function selectedPanelComponent() {
        if (panelRegistry && panelRegistry[currentPanelId] && panelRegistry[currentPanelId].component)
            return panelRegistry[currentPanelId].component
        return sidePanelComponent
    }

    property var panelMetadata: [
        ({ panelId: "sidePanel", title: qsTr("Dashboard operatora"), icon: "fingerprint", defaultColumn: 0, defaultOrder: 0 }),
        ({ panelId: "telemetryPanel", title: qsTr("Telemetria feedu"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 1 }),
        ({ panelId: "aiDecisionsPanel", title: qsTr("Decyzje governor"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 2 }),
        ({ panelId: "diagnosticsPanel", title: qsTr("Diagnostyka"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 3 }),
        ({ panelId: "chartView", title: qsTr("Strumień decyzji"), icon: "cloud", defaultColumn: 1, defaultOrder: 0 }),
        ({ panelId: "strategyWorkbench", title: qsTr("Warsztat strategii"), icon: "package", defaultColumn: 1, defaultOrder: 1 }),
        ({ panelId: "strategiesPanel", title: qsTr("Strategie"), icon: "strategy_manager", defaultColumn: 1, defaultOrder: 2 }),
        ({ panelId: "riskControlsPanel", title: qsTr("Kontrola ryzyka"), icon: "shield", defaultColumn: 1, defaultOrder: 3 }),
        ({ panelId: "modeWizardPanel", title: qsTr("Tryby pracy"), icon: "mode_wizard", defaultColumn: 1, defaultOrder: 4 }),
        ({ panelId: "strategyManagerPanel", title: qsTr("Menedżer strategii"), icon: "strategy_manager", defaultColumn: 1, defaultOrder: 5 })
    ]

    property var panelRegistry: ({
        "sidePanel": { title: qsTr("Dashboard operatora"), icon: "fingerprint", component: sidePanelComponent },
        "telemetryPanel": { title: qsTr("Telemetria feedu"), icon: "diagnostics", component: telemetryPanelComponent },
        "chartView": { title: qsTr("Strumień decyzji"), icon: "cloud", component: chartViewComponent },
        "strategyWorkbench": { title: qsTr("Warsztat strategii"), icon: "package", component: strategyWorkbenchComponent },
        "strategiesPanel": { title: qsTr("Strategie"), icon: "strategy_manager", component: strategiesPanelComponent },
        "riskControlsPanel": { title: qsTr("Kontrola ryzyka"), icon: "shield", component: riskControlsPanelComponent },
        "modeWizardPanel": { title: qsTr("Tryby pracy"), icon: "mode_wizard", component: modeWizardPanelComponent },
        "strategyManagerPanel": { title: qsTr("Menedżer strategii"), icon: "strategy_manager", component: strategyManagerPanelComponent },
        "diagnosticsPanel": { title: qsTr("Diagnostyka"), icon: "diagnostics", component: diagnosticsPanelComponent },
        "aiDecisionsPanel": { title: qsTr("Decyzje governor"), icon: "mode_wizard", component: aiDecisionsPanelComponent }
    })

    StylesModule.DesignSystem {
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
            anchors.fill: parent
            anchors.margins: 16
            spacing: 12

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
                checked: entry && entry.panelId === root.defaultPanelId ? true : entry && entry.visible !== false
                onTriggered: {
                    if (layoutController && entry) {
                        if (entry.panelId === root.defaultPanelId) {
                            root.showOperatorDashboard()
                            return
                        }
                        var currentVisible = layoutController.isPanelVisible(entry.panelId)
                        root.currentPanelId = entry.panelId
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
                blur: 1.0
                blurMax: 24
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
                text: contextRuntimeState ? contextRuntimeState.cloudStatusLabel : qsTr("Cloud runtime: wyłączony")
                color: designSystem.color("textSecondary")
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            Components.IconButton {
                id: layoutButton
                designSystem: rootDesignSystem
                text: qsTr("Panele")
                iconName: "package"
                subtle: true
                onClicked: panelMenu.popup(layoutButton)
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: qsTr("Strategie")
                iconName: "strategy_manager"
                subtle: true
                onClicked: {
                    root.showPanel("strategyManagerPanel")
                }
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: qsTr("Tryby pracy")
                iconName: "mode_wizard"
                subtle: true
                onClicked: root.showPanel("modeWizardPanel")
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: qsTr("Odśwież dane")
                iconName: ""
                backgroundColor: designSystem.color("accent")
                foregroundColor: designSystem.color("surface")
                onClicked: runtimeService && runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
            }
        }
    }

    Rectangle {
        id: centralContentRoot
        objectName: "centralContentRoot"
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: parent.top
        anchors.margins: 16
        radius: 28
        color: Qt.rgba(0, 0, 0, 0.08)
        border.color: designSystem.color("border")
        border.width: 1

        Loader {
            id: centralContentLoader
            objectName: "centralContentLoader"
            anchors.fill: parent
            anchors.margins: 0
            active: true
            sourceComponent: root.selectedPanelComponent()
        }
    }

    LayoutComponents.DockManager {
        id: dockManager
        anchors.fill: centralContentRoot
        layoutController: layoutController
        panelRegistry: panelRegistry
        designSystem: rootDesignSystem
        visible: false
    }

    Component {
        id: sidePanelComponent
        Views.OperatorDashboard {
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: telemetryPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "telemetryFeedPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    Label {
                        objectName: "telemetryFeedPreviewTitle"
                        text: qsTr("Telemetria feedu")
                        font.bold: true
                        font.pixelSize: 22
                        color: designSystem.color("textPrimary")
                    }
                    Label {
                        text: qsTr("Bezpieczny podgląd health-checków feedu. Demo/offline: runtime loop not started, exchange/order disabled.")
                        wrapMode: Text.WordWrap
                        color: designSystem.color("textSecondary")
                        Layout.fillWidth: true
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Status feedu")
                        description: qsTr("Stan: %1").arg(contextRuntimeState ? contextRuntimeState.feedHealth.status || qsTr("inicjalizacja") : qsTr("inicjalizacja"))
                        Layout.fillWidth: true
                    }
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Reconnects")
                        description: qsTr("Reconnects: %1 • Downtime: %2 ms")
                              .arg(contextRuntimeState ? contextRuntimeState.feedHealth.reconnects || 0 : 0)
                              .arg(Math.round(contextRuntimeState ? contextRuntimeState.feedHealth.downtimeMs || 0 : 0))
                        Layout.fillWidth: true
                    }
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Ostatni błąd")
                        description: qsTr("%1").arg(contextRuntimeState ? contextRuntimeState.feedHealth.lastError || qsTr("brak") : qsTr("brak"))
                        Layout.fillWidth: true
                    }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Demo/offline heartbeat")
                    description: qsTr("BTC/USDT heartbeat OK • ETH/USDT stale guard OK • loading/empty state gotowy do prezentacji.")
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Components.IconButton {
                            designSystem: rootDesignSystem
                            text: qsTr("Ping feed")
                            iconName: "refresh"
                            subtle: true
                            onClicked: runtimeService && runtimeService.loadRecentDecisions(0)
                        }
                        Label {
                            text: qsTr("Brak realnych requestów sieciowych w panelu preview.")
                            color: designSystem.color("textSecondary")
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }

    Component {
        id: aiDecisionsPanelComponent
        Views.AiDecisionsView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: chartViewComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "decisionStreamPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label {
                    objectName: "decisionStreamPreviewTitle"
                    text: qsTr("Strumień decyzji i dziennik governora")
                    font.bold: true
                    font.pixelSize: 22
                    color: designSystem.color("textPrimary")
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Wykres confidence oraz dziennik zdarzeń w trybie demo/offline. Order execution disabled.")
                    wrapMode: Text.WordWrap
                    color: designSystem.color("textSecondary")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Confidence preview")
                    description: qsTr("Canvas zachowany, z poprawionym paddingiem i pustym stanem.")
                    Canvas {
                        id: chartCanvas
                        Layout.fillWidth: true
                        Layout.preferredHeight: 180
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            ctx.fillStyle = designSystem.color("surfaceMuted")
                            ctx.fillRect(0, 0, width, height)
                            var data = runtimeService ? runtimeService.decisions || [] : []
                            if (data.length === 0) {
                                ctx.strokeStyle = designSystem.color("border")
                                ctx.lineWidth = 1
                                for (var g = 1; g < 4; ++g) {
                                    ctx.beginPath(); ctx.moveTo(0, height * g / 4); ctx.lineTo(width, height * g / 4); ctx.stroke()
                                }
                                return
                            }
                            var windowSize = Math.min(40, data.length)
                            var step = width / Math.max(windowSize - 1, 1)
                            ctx.strokeStyle = designSystem.color("accent")
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            for (var i = 0; i < windowSize; ++i) {
                                var entry = data[data.length - windowSize + i]
                                var confidence = entry.decision && entry.decision.confidence !== undefined ? Number(entry.decision.confidence) : 0.35
                                confidence = Math.max(0.05, Math.min(confidence, 1.0))
                                var x = i * step
                                var y = height - (confidence * height)
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                            }
                            ctx.stroke()
                        }
                    }
                }
                Connections {
                    target: runtimeService
                    function onDecisionsChanged() { chartCanvas.requestPaint() }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Zdarzenia governora")
                    description: (!runtimeService || !runtimeService.decisions || runtimeService.decisions.length === 0)
                                 ? qsTr("Brak danych live — pokazuję pusty stan demo/offline.")
                                 : qsTr("Ostatnie decyzje z lokalnego preview bridge.")
                    ListView {
                        id: decisionList
                        Layout.fillWidth: true
                        Layout.preferredHeight: 260
                        model: runtimeService ? runtimeService.decisions : []
                        clip: true
                        spacing: 8
                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                            width: 10
                            background: Rectangle { radius: 5; color: Qt.rgba(1, 1, 1, 0.04) }
                            contentItem: Rectangle { radius: 4; color: designSystem.color("surfaceElevated"); border.color: designSystem.color("border"); border.width: 1 }
                        }
                        delegate: Rectangle {
                            width: ListView.view.width
                            color: designSystem.color("surfaceMuted")
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
                                    text: qsTr("%1 • %2 • %3").arg(modelData.timestamp || "-").arg(modelData.portfolio || "-").arg(modelData.marketRegime && modelData.marketRegime.label ? modelData.marketRegime.label : "")
                                    font.bold: true
                                    color: designSystem.color("textPrimary")
                                    wrapMode: Text.Wrap
                                }
                                Label {
                                    text: modelData.decision && modelData.decision.shouldTrade ? qsTr("Decyzja: %1 %2 @ %3").arg(modelData.symbol || "-").arg(modelData.side || "").arg(modelData.price || "") : qsTr("Decyzja: brak transakcji")
                                    wrapMode: Text.Wrap
                                    color: designSystem.color("textPrimary")
                                }
                                Label {
                                    text: modelData.ai && modelData.ai.strategy ? qsTr("Governor: %1").arg(modelData.ai.strategy) : ""
                                    color: designSystem.color("textSecondary")
                                    visible: text.length > 0
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: strategyWorkbenchComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "strategyWorkbenchPreviewPanel"
            contentWidth: availableWidth
            clip: true
            property var strategies: []
            property var marketplacePresets: strategyManagementController ? strategyManagementController.presets : []
            function rebuild() {
                var data = runtimeService ? runtimeService.decisions || [] : []
                var stats = {}
                for (var i = 0; i < data.length; ++i) {
                    var entry = data[i]
                    var strategy = entry.ai && entry.ai.strategy ? entry.ai.strategy : qsTr("Nieznana strategia")
                    if (!stats[strategy]) stats[strategy] = { count: 0, lastSymbol: entry.symbol || "-" }
                    stats[strategy].count += 1
                    stats[strategy].lastSymbol = entry.symbol || stats[strategy].lastSymbol
                }
                var collection = []
                for (var key in stats) collection.push({ name: key, count: stats[key].count, symbol: stats[key].lastSymbol })
                collection.sort(function(a, b) { return b.count - a.count })
                strategies = collection
            }
            Component.onCompleted: rebuild()
            Connections { target: runtimeService; function onDecisionsChanged() { rebuild() } }
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label {
                    objectName: "strategyWorkbenchPreviewTitle"
                    text: qsTr("Warsztat strategii")
                    font.bold: true
                    font.pixelSize: 22
                    color: designSystem.color("textPrimary")
                }
                Label {
                    text: qsTr("Demo/offline workspace do analizy strategii bez uruchamiania live tradingu ani order execution.")
                    wrapMode: Text.WordWrap
                    color: designSystem.color("textSecondary")
                    Layout.fillWidth: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Sygnały"); description: strategies.length > 0 ? qsTr("%1 strategii z decyzji preview").arg(strategies.length) : qsTr("Brak live danych — statyczny empty state demo/offline"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Marketplace"); description: marketplacePresets.length > 0 ? qsTr("Presety dostępne lokalnie") : qsTr("Marketplace unavailable w tym preview"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety"); description: qsTr("Runtime loop not started, API keys not required"); Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Kandydaci strategii")
                    description: strategies.length === 0 ? qsTr("Demo/offline: BTC/USDT momentum, BTC/USDT range guard i BTC/USDT risk hedge czekają na dane preview.") : qsTr("Agregacja ostatnich zdarzeń governora.")
                    Repeater {
                        model: strategies.length > 0 ? strategies : [
                            ({ name: "BTC/USDT Momentum Preview", count: 4, symbol: "BTC/USDT" }),
                            ({ name: "BTC/USDT Range Guard", count: 4, symbol: "BTC/USDT" }),
                            ({ name: "BTC/USDT Risk Hedge", count: 4, symbol: "BTC/USDT" })
                        ]
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            Label { text: modelData.name; font.bold: true; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                            Label { text: qsTr("%1 zdarzeń").arg(modelData.count); color: designSystem.color("textSecondary") }
                            Label { text: modelData.symbol; color: designSystem.color("textSecondary") }
                        }
                    }
                    RowLayout {
                        spacing: 8
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Odśwież strategie"); iconName: "refresh"; subtle: true; onClicked: runtimeService && runtimeService.loadRecentDecisions(0) }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Otwórz menedżer"); iconName: "strategy_manager"; onClicked: root.showPanel("strategyManagerPanel") }
                    }
                }
            }
        }
    }

    Component {
        id: strategiesPanelComponent
        Views.Strategies {
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: riskControlsPanelComponent
        Views.RiskControls {
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: modeWizardPanelComponent
        Views.ModeWizard {
            width: parent ? parent.width : 900
            height: parent ? parent.height : 620
            designSystem: rootDesignSystem
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
            width: parent ? parent.width : 900
            height: parent ? parent.height : 620
            designSystem: rootDesignSystem
            strategyManagementController: strategyManagementController
            layoutController: layoutController
        }
    }

    Component {
        id: diagnosticsPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "diagnosticsPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label {
                    objectName: "diagnosticsPreviewTitle"
                    text: qsTr("Diagnostyka")
                    font.bold: true
                    font.pixelSize: 22
                    color: designSystem.color("textPrimary")
                }
                Label {
                    text: qsTr("Panel paczki diagnostycznej w trybie safe preview. Nie czyta sekretów, nie uruchamia runtime loop ani połączeń giełdowych.")
                    color: designSystem.color("textSecondary")
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Status"); description: diagnosticsController.busy ? qsTr("Generuję paczkę diagnostyczną…") : qsTr("Gotowe — paczka może zostać wygenerowana lokalnie"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Zakres") ; description: qsTr("UI logs, konfiguracja preview i metadane bez sekretów"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety") ; description: qsTr("API keys not required, exchange/order disabled"); Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Akcje diagnostyczne")
                    description: qsTr("Surowy identyfikator statusu został zastąpiony czytelnym komunikatem dla demo.")
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton {
                            designSystem: rootDesignSystem
                            iconName: "diagnostics"
                            text: diagnosticsController.busy ? qsTr("Generuję…") : qsTr("Generuj paczkę")
                            enabled: !diagnosticsController.busy
                            backgroundColor: designSystem.color("accent")
                            foregroundColor: designSystem.color("surface")
                            onClicked: diagnosticsController.generateDiagnostics()
                        }
                        Label {
                            text: diagnosticsController.lastArchivePath.length > 0 ? qsTr("Ostatnia paczka: %1").arg(diagnosticsController.lastArchivePath) : qsTr("Brak wygenerowanej paczki w tej sesji")
                            color: designSystem.color("textSecondary")
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        if (runtimeService && runtimeService.loadRecentDecisions)
            runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
        if (layoutController && layoutController.registerPanels) {
            layoutController.registerPanels(panelMetadata)
            showOperatorDashboard()
        }
    }

    Timer {
        interval: 15000
        repeat: true
        running: true
        onTriggered: runtimeService && runtimeService.loadRecentDecisions(0)
    }

    Dialog {
        id: modeWizardDialog
        modal: true
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.8, 1100)
        height: Math.min(parent.height * 0.85, 780)
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        Overlay.modal: Rectangle {
            color: Qt.rgba(0, 0, 0, 0.62)
        }
        background: Rectangle {
            anchors.fill: parent
            radius: 24
            color: designSystem.color("surface")
            border.color: designSystem.color("border")
            border.width: 1
        }
        contentItem: Views.ModeWizard {
            anchors.fill: parent
            anchors.margins: 16
            designSystem: rootDesignSystem
            modeWizardController: modeWizardController
            compact: false
            onLaunchWizardRequested: modeWizardDialog.close()
        }
    }
}
