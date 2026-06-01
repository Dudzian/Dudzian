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
    title: qsTr("Dudzian Product Preview — safe dry-run")
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
        ({ panelId: "sidePanel", title: qsTr("Dashboard"), icon: "fingerprint", defaultColumn: 0, defaultOrder: 0 }),
        ({ panelId: "aiCenterPanel", title: qsTr("AI Center"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 1 }),
        ({ panelId: "tradingUniversePanel", title: qsTr("Trading Universe"), icon: "cloud", defaultColumn: 0, defaultOrder: 2 }),
        ({ panelId: "strategiesPanel", title: qsTr("Strategie"), icon: "strategy_manager", defaultColumn: 0, defaultOrder: 3 }),
        ({ panelId: "riskControlsPanel", title: qsTr("Ryzyko"), icon: "shield", defaultColumn: 0, defaultOrder: 4 }),
        ({ panelId: "aiDecisionsPanel", title: qsTr("Decyzje"), icon: "mode_wizard", defaultColumn: 0, defaultOrder: 5 }),
        ({ panelId: "telemetryPanel", title: qsTr("Telemetria"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 6 }),
        ({ panelId: "diagnosticsPanel", title: qsTr("Diagnostyka"), icon: "diagnostics", defaultColumn: 0, defaultOrder: 7 })
    ]

    property var productTabs: panelMetadata

    property var panelRegistry: ({
        "sidePanel": { title: qsTr("Dashboard"), icon: "fingerprint", component: sidePanelComponent },
        "aiCenterPanel": { title: qsTr("AI Center"), icon: "mode_wizard", component: aiCenterPanelComponent },
        "tradingUniversePanel": { title: qsTr("Trading Universe"), icon: "cloud", component: tradingUniversePanelComponent },
        "strategiesPanel": { title: qsTr("Strategie"), icon: "strategy_manager", component: strategiesPanelComponent },
        "riskControlsPanel": { title: qsTr("Ryzyko"), icon: "shield", component: riskControlsPanelComponent },
        "aiDecisionsPanel": { title: qsTr("Decyzje"), icon: "mode_wizard", component: aiDecisionsPanelComponent },
        "telemetryPanel": { title: qsTr("Telemetria"), icon: "diagnostics", component: telemetryPanelComponent },
        "diagnosticsPanel": { title: qsTr("Diagnostyka"), icon: "diagnostics", component: diagnosticsPanelComponent },
        "chartView": { title: qsTr("Strumień decyzji"), icon: "cloud", component: chartViewComponent },
        "strategyWorkbench": { title: qsTr("Warsztat strategii"), icon: "package", component: strategyWorkbenchComponent },
        "modeWizardPanel": { title: qsTr("Tryby pracy"), icon: "mode_wizard", component: modeWizardPanelComponent },
        "strategyManagerPanel": { title: qsTr("Menedżer strategii"), icon: "strategy_manager", component: strategyManagerPanelComponent }
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
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12

            ColumnLayout {
                Layout.alignment: Qt.AlignVCenter
                spacing: 2
                Label {
                    text: qsTr("Dudzian Bot Preview")
                    font.bold: true
                    font.pixelSize: 16
                    color: designSystem.color("textPrimary")
                }
                Label {
                    text: qsTr("Live trading disabled • Exchange I/O disabled • Order submission disabled")
                    color: designSystem.color("textSecondary")
                    font.pixelSize: 11
                }
            }

            Rectangle { width: 1; height: parent.height * 0.6; color: designSystem.color("border"); opacity: 0.45 }

            Flickable {
                id: tabOverflow
                objectName: "productPreviewTabBar"
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                Layout.alignment: Qt.AlignVCenter
                clip: true
                contentWidth: browserTabBar.implicitWidth
                boundsBehavior: Flickable.StopAtBounds
                flickableDirection: Flickable.HorizontalFlick

                Row {
                    id: browserTabBar
                    height: parent.height
                    spacing: 6
                    Repeater {
                        model: root.productTabs
                        delegate: Rectangle {
                            required property var modelData
                            property bool hovered: false
                            readonly property bool active: root.currentPanelId === modelData.panelId
                            width: Math.max(120, tabLabel.implicitWidth + 34)
                            height: 42
                            radius: 14
                            color: active ? designSystem.color("surface") : (hovered ? Qt.rgba(0.33, 0.78, 1.0, 0.12) : Qt.rgba(0, 0, 0, 0.20))
                            border.color: active ? designSystem.color("accent") : (hovered ? designSystem.color("textSecondary") : designSystem.color("border"))
                            border.width: active ? 2 : 1
                            Rectangle {
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                height: active ? 3 : 0
                                radius: 2
                                color: designSystem.color("accent")
                            }
                            Label {
                                id: tabLabel
                                anchors.centerIn: parent
                                text: modelData.title
                                font.bold: active
                                color: active ? designSystem.color("textPrimary") : designSystem.color("textSecondary")
                            }
                            MouseArea {
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onEntered: parent.hovered = true
                                onExited: parent.hovered = false
                                onPressed: parent.opacity = 0.82
                                onReleased: parent.opacity = 1.0
                                onClicked: root.showPanel(modelData.panelId)
                            }
                        }
                    }
                }
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: qsTr("Odśwież preview")
                iconName: "refresh"
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
        id: aiCenterPanelComponent
        Views.AiControlCenter {
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: tradingUniversePanelComponent
        Views.TradingUniverse {
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
                        text: qsTr("Telemetria")
                        font.bold: true
                        font.pixelSize: 22
                        color: designSystem.color("textPrimary")
                    }
                    Label {
                        text: qsTr("Pełnoprawna Telemetria preview: feed status, reconnects, downtime, last heartbeat, data freshness. Runtime loop not started, Exchange I/O disabled, Order submission disabled.")
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
                        title: qsTr("Feed status")
                        description: qsTr("Stan: %1").arg(contextRuntimeState ? contextRuntimeState.feedHealth.status || qsTr("inicjalizacja") : qsTr("inicjalizacja"))
                        Layout.fillWidth: true
                    }
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Reconnects / Downtime")
                        description: qsTr("Reconnects: %1 • Downtime: %2 ms")
                              .arg(contextRuntimeState ? contextRuntimeState.feedHealth.reconnects || 0 : 0)
                              .arg(Math.round(contextRuntimeState ? contextRuntimeState.feedHealth.downtimeMs || 0 : 0))
                        Layout.fillWidth: true
                    }
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Last heartbeat / Data freshness")
                        description: qsTr("Last heartbeat: 12:04:18Z • Data freshness: mock/local preview • last error: %1").arg(contextRuntimeState ? contextRuntimeState.feedHealth.lastError || qsTr("brak") : qsTr("brak"))
                        Layout.fillWidth: true
                    }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Mock local preview rows")
                    description: qsTr("Last heartbeat 12:04:18Z • Data freshness: mock/local preview • BTC/USDT heartbeat OK • ETH/USDT stale guard OK • SOL/USDT telemetry preview")
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
                            text: qsTr("Mock local preview rows only. Runtime loop not started, exchange/order disabled.")
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
                    text: qsTr("Final diagnostics preview: status paczki diagnostycznej, zakres, safety, ostatnia paczka i generowanie lokalne. No secrets / no .env / no keychain. Local preview only.")
                    color: designSystem.color("textSecondary")
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Status"); description: diagnosticsController && diagnosticsController.busy ? qsTr("Generuję paczkę diagnostyczną…") : qsTr("Gotowe — paczka może zostać wygenerowana lokalnie"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Zakres") ; description: qsTr("UI logs, preview config, telemetry snapshot, governor state — no secrets / no .env / no keychain"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety") ; description: qsTr("API keys not required, Exchange I/O disabled, Order submission disabled, Runtime loop not started, Preview only"); Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Ostatnia paczka i akcje diagnostyczne")
                    description: qsTr("Generuj paczkę lokalnie w safe preview. Local preview only, bez sekretów i bez runtime loop.")
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton {
                            designSystem: rootDesignSystem
                            iconName: "diagnostics"
                            text: diagnosticsController && diagnosticsController.busy ? qsTr("Generuję…") : qsTr("Generuj paczkę")
                            enabled: !diagnosticsController || !diagnosticsController.busy
                            backgroundColor: designSystem.color("accent")
                            foregroundColor: designSystem.color("surface")
                            onClicked: diagnosticsController && diagnosticsController.generateDiagnostics()
                        }
                        Label {
                            text: diagnosticsController && diagnosticsController.lastArchivePath && diagnosticsController.lastArchivePath.length > 0 ? qsTr("Ostatnia paczka: %1").arg(diagnosticsController.lastArchivePath) : qsTr("Last diagnostic bundle path: mock/local preview path unavailable in this session")
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
