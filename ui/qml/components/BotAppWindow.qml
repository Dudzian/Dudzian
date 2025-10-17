import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import Qt.labs.settings
import "."

ApplicationWindow {
    id: window
    property alias performanceGuard: guardModel.guard
    property alias sidePanel: sidePanel
    property alias chartView: chartView
    property var extraWindows: []
    property int extraWindowCount: 0

    title: qsTr("Bot Trading Shell")
    minimumWidth: 960
    minimumHeight: 540
    visible: true

    color: Qt.darker(palette.window, 1.05)
    readonly property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)

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
                text: qsTr("Administracja")
                icon.name: "preferences-system"
                onClicked: adminDialog.open()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Zarządzaj licencją i profilami użytkowników")
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

    AdminDialog {
        id: adminDialog
        visible: false
        x: Math.max(0, (window.width - width) / 2)
        y: Math.max(0, (window.height - height) / 2)
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
        enabled: licenseController.licenseActive

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

            CandlestickChartView {
                id: chartView
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: ohlcvModel
                performanceGuard: guardModel.guard
                reduceMotion: appController.reduceMotionActive
            }
        }
    }

    footer: StatusFooter {
        id: footer
        Layout.fillWidth: true
    }

    LicenseActivationOverlay {
        anchors.fill: parent
        visible: !licenseController.licenseActive
        enabled: !licenseController.licenseActive
        focus: visible
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
