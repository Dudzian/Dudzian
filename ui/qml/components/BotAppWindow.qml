import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import Qt.labs.settings
import "."
import "workbench"
import "../views" as Views
import "../styles" as Styles

ApplicationWindow {
    id: window
    property alias performanceGuard: guardModel.guard
    property alias sidePanel: sidePanel
    property alias chartView: chartView
    property alias workbenchView: strategyWorkbench
    property var extraWindows: []
    property int extraWindowCount: 0
    property bool wizardCompleted: licenseController ? licenseController.licenseActive : false

    title: qsTr("Bot Trading Shell")
    minimumWidth: 960
    minimumHeight: 540
    visible: true

    color: Qt.darker(palette.window, 1.05)
    readonly property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)

    Component.onCompleted: {
        if (typeof appController !== "undefined" && appController && appController.userProfiles)
            Styles.AppTheme.applyPalette(appController.userProfiles.activeThemePalette)
        else
            Styles.AppTheme.applyPalette(null)
    }

    Connections {
        target: (typeof appController !== "undefined" && appController) ? appController.userProfiles : null
        ignoreUnknownSignals: true

        function onThemePaletteChanged() {
            if (appController && appController.userProfiles)
                Styles.AppTheme.applyPalette(appController.userProfiles.activeThemePalette)
        }

        function onActiveProfileChanged() {
            if (appController && appController.userProfiles)
                Styles.AppTheme.applyPalette(appController.userProfiles.activeThemePalette)
        }
    }

    header: ToolBar {
        enabled: licenseController.licenseActive
        contentHeight: 48
        RowLayout {
            anchors.fill: parent
            spacing: 12

            Label {
                text: qsTr("Connection: %1").arg(appController.connectionStatus)
                font.bold: true
                Layout.alignment: Qt.AlignVCenter
            }

            Rectangle {
                width: 1
                color: palette.mid
                height: parent.height * 0.6
                Layout.margins: 8
            }

            Label {
                text: qsTr("FPS target: %1").arg(performanceGuard.fpsTarget)
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Nowe okno")
                icon.name: "window-new"
                onClicked: window.openChartWindow()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Otwórz dodatkowe okno wykresu")
            }

            Button {
                text: qsTr("Aktywacja")
                icon.name: "emblem-verified"
                onClicked: activationDialog.open()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Wyświetl fingerprint i licencje OEM")
            }

            Button {
                text: qsTr("Panel administracyjny")
                icon.name: "preferences-system"
                onClicked: adminPanelDrawer.open()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Konfiguracja strategii, monitorowanie i licencja")
            }

            Button {
                text: qsTr("Raporty")
                icon.name: "view-list-details"
                onClicked: reportDialog.open()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Przeglądaj i usuwaj raporty z magazynu UI")
            }

            Button {
                text: qsTr("Ponów połączenie")
                onClicked: {
                    appController.stop()
                    appController.start()
                }
            }
        }
    }

    ActivationDialog {
        id: activationDialog
        controller: activationController
    }

    AdminPanel {
        id: adminPanelDrawer
        parent: window.contentItem
        height: window.height
    }

    Dialog {
        id: reportDialog
        objectName: "reportDialog"
        title: qsTr("Przegląd raportów")
        modal: true
        standardButtons: Dialog.Close
        width: 720
        height: 480

        onOpened: {
            if (reportController)
                reportController.refresh()
        }

        ReportBrowser {
            anchors.fill: parent
            controller: reportController
        }
    }

    Connections {
        target: securityController
        function onAdminEventLogged(message) {
            console.info("Security event:", message)
        }
    }

    background: Rectangle {
        color: Qt.darker(palette.window, 1.15)
    }

    Item {
        id: guardModel
        property PerformanceGuard guard: appController.performanceGuard

        Connections {
            target: appController
            function onPerformanceGuardChanged() {
                guard = appController.performanceGuard
            }
        }
    }

    Item {
        anchors.fill: parent
        anchors.topMargin: header.height
        anchors.bottomMargin: footer.implicitHeight

        RowLayout {
            anchors.fill: parent
            spacing: 0

            SidePanel {
                id: sidePanel
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                performanceGuard: guardModel.guard
                onOpenWindowRequested: window.openChartWindow()
            }

            Rectangle {
                width: 1
                color: Qt.darker(palette.window, 1.2)
                Layout.fillHeight: true
            }

            TabView {
                id: mainTabView
                Layout.fillWidth: true
                Layout.fillHeight: true

                Component.onCompleted: currentIndex = wizardCompleted ? 1 : 0

                Tab {
                    title: qsTr("Kreator")

                    Views.SetupWizard {
                        anchors.fill: parent
                        appController: appController
                        strategyController: strategyController
                        workbenchController: workbenchController
                        licenseController: licenseController
                        riskModel: riskModel
                        onWizardCompleted: {
                            window.wizardCompleted = true
                            if (licenseController && !licenseController.licenseActive)
                                licenseController.refreshLicenseStatus()
                            mainTabView.currentIndex = 1
                        }
                    }
                }

                Tab {
                    title: qsTr("Wykres")
                    enabled: licenseController.licenseActive

                    Item {
                        anchors.fill: parent

                        MarketMultiStreamView {
                            id: chartView
                            anchors.fill: parent
                            priceModel: ohlcvModel
                            indicatorModel: indicatorSeriesModel
                            signalModel: signalListModel
                            regimeModel: marketRegimeTimelineModel
                            performanceGuard: guardModel.guard
                            reduceMotion: appController.reduceMotionActive
                        }
                    }
                }

                Tab {
                    title: qsTr("Dashboard")
                    enabled: licenseController.licenseActive

                    Views.PortfolioDashboard {
                        anchors.fill: parent
                        appController: appController
                        riskModel: riskModel
                        riskHistoryModel: riskHistoryModel
                        alertsModel: alertsModel
                    }
                }

                Tab {
                    title: qsTr("Workbench")
                    enabled: licenseController.licenseActive

                    StrategyWorkbench {
                        id: strategyWorkbench
                        anchors.fill: parent
                        appController: appController
                        strategyController: strategyController
                        workbenchController: workbenchController
                        riskModel: riskModel
                        riskHistoryModel: riskHistoryModel
                        licenseController: licenseController
                    }
                }

                Tab {
                    title: qsTr("Konfigurator")
                    enabled: licenseController.licenseActive

                    Views.StrategyConfigurator {
                        anchors.fill: parent
                        appController: appController
                        strategyController: strategyController
                        workbenchController: workbenchController
                        riskModel: riskModel
                        licenseController: licenseController
                    }
                }

                Tab {
                    title: qsTr("Ryzyko live")
                    enabled: licenseController.licenseActive

                    Flickable {
                        anchors.fill: parent
                        contentWidth: Math.max(width, dashboard.implicitWidth)
                        contentHeight: Math.max(height, dashboard.implicitHeight)
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar {}

                        LiveRiskDashboard {
                            id: dashboard
                            width: Math.max(parent.width, implicitWidth)
                            riskModel: riskModel
                            riskHistoryModel: riskHistoryModel
                            capitalAllocationModel: appController && appController.capitalAllocationModel ? appController.capitalAllocationModel : null
                            capitalAllocation: appController && appController.capitalAllocationSnapshot ? appController.capitalAllocationSnapshot : []
                        }
                    }
                }

                Tab {
                    title: qsTr("Monitoring")
                    enabled: licenseController.licenseActive

                    Flickable {
                        anchors.fill: parent
                        contentWidth: Math.max(width, monitoringDashboard.implicitWidth)
                        contentHeight: Math.max(height, monitoringDashboard.implicitHeight)
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar {}

                        Views.MonitoringDashboard {
                            id: monitoringDashboard
                            width: Math.max(parent.width, implicitWidth)
                        }
                    }
                }

                Tab {
                    title: qsTr("Moduły")
                    enabled: licenseController.licenseActive

                    ModuleBrowser {
                        anchors.fill: parent
                        viewsModel: moduleViewsModel
                    }
                }

                Tab {
                    title: qsTr("Portfele")
                    enabled: licenseController.licenseActive

                    PortfolioManagerView {
                        anchors.fill: parent
                    }
                }

                Tab {
                    title: qsTr("Strategie 360°")
                    enabled: licenseController.licenseActive

                    Views.StrategyExperience {
                        anchors.fill: parent
                        appController: appController
                        configurationWizard: configurationWizard
                        workbenchController: workbenchController
                    }
                }

                Tab {
                    title: qsTr("Marketplace")
                    enabled: licenseController.licenseActive

                    Views.Marketplace {
                        anchors.fill: parent
                        appController: appController
                    }
                }
            }
        }
    }

    footer: StatusFooter {
        id: footer
        Layout.fillWidth: true
    }

    FirstRunWizard {
        anchors.fill: parent
        visible: !wizardCompleted
    }

    AlertToastOverlay {
        id: alertToastOverlay
        enabled: appController ? appController.alertToastsEnabled : false
    }

    Settings {
        id: windowSettings
        category: "bot_trading_shell/main_window"
        property real savedX: window.x
        property real savedY: window.y
        property real savedWidth: window.width
        property real savedHeight: window.height
        property int savedExtraWindows: 0
    }

    Component {
        id: chartWindowComponent
        ChartWindow {
            ohlcvModel: ohlcvModel
            performanceGuard: guardModel.guard
            instrumentLabel: appController.instrumentLabel
        }
    }

    Shortcut {
        sequences: [ StandardKey.New ]
        onActivated: window.openChartWindow()
    }

    Component.onCompleted: {
        if (!isNaN(windowSettings.savedX)) window.x = windowSettings.savedX
        if (!isNaN(windowSettings.savedY)) window.y = windowSettings.savedY
        if (!isNaN(windowSettings.savedWidth)) window.width = windowSettings.savedWidth
        if (!isNaN(windowSettings.savedHeight)) window.height = windowSettings.savedHeight

        for (let i = 0; i < windowSettings.savedExtraWindows; ++i) {
            openChartWindow({ restoreSession: true })
        }
        if (typeof appController !== 'undefined' && appController.notifyWindowCount)
            appController.notifyWindowCount(extraWindowCount + 1)
    }

    onXChanged: windowSettings.savedX = window.x
    onYChanged: windowSettings.savedY = window.y
    onWidthChanged: windowSettings.savedWidth = window.width
    onHeightChanged: windowSettings.savedHeight = window.height

    onClosing: {
        windowSettings.savedExtraWindows = extraWindows.length
        for (let i = extraWindows.length - 1; i >= 0; --i) {
            if (extraWindows[i])
                extraWindows[i].close()
        }
    }

    Connections {
        target: appController
        function onPerformanceGuardChanged() {
            guardModel.guard = appController.performanceGuard
            window.syncPerformanceGuard()
        }
        function onInstrumentChanged() {
            window.syncInstrumentLabel()
        }
        function onAlertToastsEnabledChanged() {
            if (!appController.alertToastsEnabled)
                alertToastOverlay.clear()
        }
    }

    Connections {
        target: licenseController
        ignoreUnknownSignals: true
        function onLicenseActiveChanged() {
            window.wizardCompleted = licenseController.licenseActive
            if (window.wizardCompleted && mainTabView.currentIndex === 0 && mainTabView.count > 1)
                mainTabView.currentIndex = 1
        }
    }

    Connections {
        target: appController ? appController.alertsModel : null
        ignoreUnknownSignals: true
        function onAlertRaised(id, severity, title, description) {
            if (!appController || !appController.alertToastsEnabled)
                return
            alertToastOverlay.showToast(id, severity, title, description)
        }
    }

    function openChartWindow(options) {
        if (chartWindowComponent.status !== Component.Ready)
            return
        const params = {
            ohlcvModel: ohlcvModel,
            performanceGuard: guardModel.guard,
            instrumentLabel: sidePanel.currentInstrumentLabel ? sidePanel.currentInstrumentLabel() : appController.instrumentLabel
        }
        const win = chartWindowComponent.createObject(window, params)
        if (!win)
            return
        extraWindows.push(win)
        win.windowClosed.connect(function() {
            removeExtraWindow(win)
        })
        win.show()
        syncInstrumentLabel()
        if (!options || !options.restoreSession) {
            windowSettings.savedExtraWindows = extraWindows.length
        }
        extraWindowCount = extraWindows.length
        if (typeof appController !== 'undefined' && appController.notifyWindowCount)
            appController.notifyWindowCount(extraWindowCount + 1)
    }

    function removeExtraWindow(win) {
        const index = extraWindows.indexOf(win)
        if (index >= 0)
            extraWindows.splice(index, 1)
        extraWindowCount = extraWindows.length
        windowSettings.savedExtraWindows = extraWindows.length
        if (typeof appController !== 'undefined' && appController.notifyWindowCount)
            appController.notifyWindowCount(extraWindowCount + 1)
    }

    function syncPerformanceGuard() {
        for (let i = 0; i < extraWindows.length; ++i) {
            extraWindows[i].performanceGuard = guardModel.guard
        }
    }

    function syncInstrumentLabel() {
        const label = sidePanel.currentInstrumentLabel ? sidePanel.currentInstrumentLabel() : appController.instrumentLabel
        for (let i = 0; i < extraWindows.length; ++i) {
            extraWindows[i].instrumentLabel = label
        }
    }
}
