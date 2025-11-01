import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "./" as Workbench
import "./panels" as Panels

Item {
    id: root
    property alias viewModel: viewModel

    // Allow overriding context objects, otherwise fall back to global context properties
    property var appController: null
    property var strategyController: null
    property var workbenchController: null
    property var riskModel: null
    property var riskHistoryModel: null
    property var licenseController: null

    implicitWidth: 960
    implicitHeight: 540

    Workbench.StrategyWorkbenchViewModel {
        id: viewModel
        objectName: "strategyWorkbenchViewModel"
        appController: root.appController ? root.appController : (typeof appController !== "undefined" ? appController : null)
        strategyController: root.strategyController ? root.strategyController : (typeof strategyController !== "undefined" ? strategyController : null)
        workbenchController: root.workbenchController ? root.workbenchController : (typeof workbenchController !== "undefined" ? workbenchController : null)
        riskModel: root.riskModel ? root.riskModel : (typeof riskModel !== "undefined" ? riskModel : null)
        riskHistoryModel: root.riskHistoryModel ? root.riskHistoryModel : (typeof riskHistoryModel !== "undefined" ? riskHistoryModel : null)
        licenseController: root.licenseController ? root.licenseController : (typeof licenseController !== "undefined" ? licenseController : null)
    }

    function formatNumber(value, digits) {
        if (value === null || value === undefined)
            return "–"
        var precision = digits !== undefined ? digits : 2
        return Number(value).toLocaleString(Qt.locale(), "f", precision)
    }

    function formatPercent(value) {
        if (value === null || value === undefined)
            return "–"
        return Number(value * 100).toLocaleString(Qt.locale(), "f", 2) + "%"
    }

    ScrollView {
        id: mainScroll
        anchors.fill: parent
        enabled: viewModel.catalogReady && !viewModel.workbenchBusy
        opacity: (viewModel.catalogReady && !viewModel.workbenchBusy) ? 1 : 0.4
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        Behavior on opacity {
            NumberAnimation { duration: 160 }
        }

        ColumnLayout {
            id: content
            width: parent.width
            spacing: 16
            padding: 16

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: Qt.darker(palette.base, 1.05)
                    radius: 8
                }

                ColumnLayout {
                    spacing: 12
                    width: parent.width
                    anchors.margins: 12

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Label {
                            text: viewModel.demoModeActive ? qsTr("Tryb demo: %1").arg(viewModel.demoModeTitle) : qsTr("Tryb live")
                            font.bold: true
                            Layout.alignment: Qt.AlignVCenter
                        }

                        Label {
                            text: viewModel.demoModeActive ? viewModel.demoModeDescription : qsTr("Dane odczytywane bezpośrednio z runtime")
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            maximumLineCount: 2
                            elide: Text.ElideRight
                        }

                        ComboBox {
                            id: demoPresetSelector
                            objectName: "demoPresetSelector"
                            model: viewModel.demoPresets
                            textRole: "title"
                            Layout.preferredWidth: 220
                            onActivated: function(index) {
                                if (index < 0 || index >= viewModel.demoPresets.length)
                                    return
                                viewModel.activateDemoMode(viewModel.demoPresets[index].id)
                            }
                        }

                        Button {
                            id: demoDisableButton
                            objectName: "demoDisableButton"
                            text: qsTr("Wyłącz demo")
                            enabled: viewModel.demoModeActive
                            onClicked: viewModel.disableDemoMode()
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                Frame {
                    id: strategyPanel
                    objectName: "strategyPanel"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 2
                    background: Rectangle {
                        color: Qt.darker(palette.window, 1.05)
                        radius: 8
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Label {
                            text: qsTr("Dashboard strategii")
                            font.pointSize: 15
                            font.bold: true
                        }

                        ListView {
                            id: strategyListView
                            objectName: "strategyListView"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            model: viewModel.schedulerEntries
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            delegate: ColumnLayout {
                                width: ListView.view.width
                                spacing: 4

                                Label {
                                    text: (modelData.name || qsTr("Strategia")) + (modelData.enabled === false ? qsTr(" (wyłączona)") : "")
                                    font.bold: true
                                }
                                Label {
                                    text: qsTr("Okna harmonogramu: %1 • Strefa: %2").arg(modelData.scheduleCount || 0).arg(modelData.timezone || qsTr("brak"))
                                    color: palette.mid
                                }
                                Label {
                                    visible: !!modelData.nextRun
                                    text: modelData.nextRun ? qsTr("Najbliższe uruchomienie: %1").arg(modelData.nextRun) : ""
                                    color: palette.mid
                                }
                                Label {
                                    visible: !!modelData.notes
                                    text: modelData.notes
                                    wrapMode: Text.WordWrap
                                    color: palette.windowText
                                }

                                Rectangle {
                                    width: parent.width
                                    height: 1
                                    color: palette.mid
                                    opacity: index === ListView.view.count - 1 ? 0 : 0.4
                                }
                            }
                        }
                    }
                }

                Frame {
                    id: exchangePanel
                    objectName: "exchangePanel"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 2
                    background: Rectangle {
                        color: Qt.darker(palette.window, 1.05)
                        radius: 8
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Label {
                            text: qsTr("Zarządzanie giełdami")
                            font.pointSize: 15
                            font.bold: true
                        }

                        Repeater {
                            model: viewModel.exchangeConnections

                            ColumnLayout {
                                width: parent.width
                                spacing: 4

                                Label {
                                    text: (modelData.exchange || "") + " • " + (modelData.symbol || "")
                                    font.bold: true
                                }
                                Label {
                                    text: qsTr("Venue: %1").arg(modelData.venueSymbol || qsTr("brak"))
                                    color: palette.mid
                                }
                                Label {
                                    text: qsTr("Status: %1").arg(modelData.status || qsTr("n/a"))
                                }
                                Label {
                                    text: qsTr("Automatyzacja: %1").arg(modelData.automationRunning ? qsTr("aktywna") : qsTr("wstrzymana"))
                                    color: palette.mid
                                }
                                Label {
                                    text: qsTr("Docelowe FPS: %1").arg(modelData.fpsTarget || 0)
                                    color: palette.mid
                                }

                                Rectangle {
                                    width: parent.width
                                    height: 1
                                    color: palette.mid
                                    opacity: index === (viewModel.exchangeConnections.length - 1) ? 0 : 0.4
                                }
                            }
                        }
                    }
                }
            }

            Panels.AutoModePanel {
                Layout.fillWidth: true
                runtimeService: root.appController && root.appController.runtimeService
                                 ? root.appController.runtimeService()
                                 : (typeof runtimeService !== "undefined" ? runtimeService : null)
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                Frame {
                    id: aiPanel
                    objectName: "aiPanel"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 2
                    background: Rectangle {
                        color: Qt.darker(palette.window, 1.05)
                        radius: 8
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Label {
                            text: qsTr("Konfiguracja AI")
                            font.pointSize: 15
                            font.bold: true
                        }

                        ColumnLayout {
                            spacing: 6

                            Repeater {
                                model: Object.keys(viewModel.aiConfiguration || {})
                                delegate: RowLayout {
                                    spacing: 8
                                    Layout.fillWidth: true

                                    Label {
                                        text: modelData
                                        font.bold: true
                                        Layout.preferredWidth: 140
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                        text: {
                                            var config = viewModel.aiConfiguration || {}
                                            var value = config[modelData]
                                            if (Array.isArray(value))
                                                return value.map(function(entry) {
                                                    return typeof entry === "object" ? JSON.stringify(entry) : entry
                                                }).join(", ")
                                            if (typeof value === "object")
                                                return JSON.stringify(value)
                                            return value
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Frame {
                    id: analysisPanel
                    objectName: "analysisPanel"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 2
                    background: Rectangle {
                        color: Qt.darker(palette.window, 1.05)
                        radius: 8
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Label {
                            text: qsTr("Analiza wyników")
                            font.pointSize: 15
                            font.bold: true
                        }

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 6

                            Label { text: qsTr("Profil") ; font.bold: true }
                            Label { text: viewModel.portfolioSummary.profileLabel || qsTr("brak") }

                            Label { text: qsTr("Ostatnia wartość") ; font.bold: true }
                            Label { text: formatNumber(viewModel.portfolioSummary.latestValue, 2) }

                            Label { text: qsTr("Maks. DD") ; font.bold: true }
                            Label { text: formatPercent(viewModel.portfolioSummary.maxDrawdown || 0) }

                            Label { text: qsTr("Średni DD") ; font.bold: true }
                            Label { text: formatPercent(viewModel.portfolioSummary.averageDrawdown || 0) }

                            Label { text: qsTr("Maks. lewar") ; font.bold: true }
                            Label { text: formatNumber(viewModel.portfolioSummary.maxLeverage, 2) }

                            Label { text: qsTr("Średni lewar") ; font.bold: true }
                            Label { text: formatNumber(viewModel.portfolioSummary.averageLeverage, 2) }

                            Label { text: qsTr("Ekspozycje naruszone") ; font.bold: true }
                            Label { text: viewModel.portfolioSummary.anyBreach ? qsTr("tak") : qsTr("nie") }
                        }

                        Label {
                            text: qsTr("Ekspozycje")
                            font.bold: true
                        }

                        ListView {
                            id: exposureList
                            objectName: "exposureList"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            model: viewModel.riskSnapshot.exposures || []
                            delegate: RowLayout {
                                width: ListView.view.width
                                spacing: 12

                                Label {
                                    text: modelData.code || ""
                                    font.bold: true
                                    Layout.preferredWidth: 120
                                }
                                Label {
                                    text: qsTr("Aktualna: %1").arg(formatNumber(modelData.current, 3))
                                    Layout.fillWidth: true
                                }
                                Label {
                                    text: qsTr("Limit: %1").arg(formatNumber(modelData.threshold, 3))
                                }
                                Label {
                                    text: modelData.breach ? qsTr("NARUSZENIE") : qsTr("OK")
                                    color: modelData.breach ? "#d1495b" : palette.mid
                                }
                            }
                        }
                    }
                }
            }

            Frame {
                id: licensePanel
                objectName: "licensePanel"
                Layout.fillWidth: true
                background: Rectangle {
                    color: Qt.darker(palette.window, 1.08)
                    radius: 8
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 8

                    Label {
                        text: qsTr("Status licencji")
                        font.pointSize: 15
                        font.bold: true
                    }

                    GridLayout {
                        columns: 2
                        columnSpacing: 12
                        rowSpacing: 4

                        Label { text: qsTr("Aktywna") ; font.bold: true }
                        Label { text: viewModel.licenseStatus.active ? qsTr("tak") : qsTr("nie") }

                        Label { text: qsTr("Edycja") ; font.bold: true }
                        Label { text: viewModel.licenseStatus.edition || qsTr("brak") }

                        Label { text: qsTr("ID") ; font.bold: true }
                        Label { text: viewModel.licenseStatus.licenseId || qsTr("brak") }

                        Label { text: qsTr("Użytkownik") ; font.bold: true }
                        Label { text: viewModel.licenseStatus.holderName || qsTr("brak") }

                        Label { text: qsTr("Seat'y") ; font.bold: true }
                        Label { text: viewModel.licenseStatus.seats || 0 }

                        Label { text: qsTr("Moduły") ; font.bold: true }
                        Label { text: (viewModel.licenseStatus.modules || []).join(", ") }

                        Label { text: qsTr("Runtime") ; font.bold: true }
                        Label { text: (viewModel.licenseStatus.runtime || []).join(", ") }
                    }
                }
            }
        }
    }

    Frame {
        id: workbenchErrorBanner
        visible: viewModel.catalogReady && viewModel.workbenchError.length > 0
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            margins: 16
        }
        z: 25
        padding: 12
        background: Rectangle {
            color: Qt.rgba(0.35, 0.07, 0.07, 0.9)
            radius: 8
            border.color: Qt.rgba(0.85, 0.35, 0.32, 0.9)
            border.width: 1
        }

        RowLayout {
            anchors.fill: parent
            spacing: 12

            Label {
                text: viewModel.workbenchError
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
                color: "#ffd7d7"
            }

            Button {
                text: qsTr("Ukryj")
                visible: viewModel.workbenchError.length > 0
                onClicked: viewModel.clearWorkbenchError()
            }
        }
    }

    Item {
        anchors.fill: parent
        visible: !viewModel.catalogReady && !viewModel.workbenchBusy
        z: 20

        Rectangle {
            anchors.fill: parent
            color: Qt.rgba(0, 0, 0, 0.55)
        }

        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.AllButtons
            hoverEnabled: true
        }

        Column {
            anchors.centerIn: parent
            spacing: 12
            width: Math.min(parent.width * 0.75, 460)

            Label {
                text: qsTr("Katalog strategii nie jest gotowy")
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                font.pointSize: 16
                font.bold: true
                color: "#ffffff"
            }

            Label {
                visible: viewModel.workbenchError.length > 0
                text: viewModel.workbenchError
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                color: "#ffd7d7"
            }

            Label {
                text: qsTr("Uzupełnij ścieżki do core.yaml, ui_config_bridge.py oraz interpretera Pythona, aby wczytać katalog dla Workbencha.")
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                color: "#f0f0f0"
            }
        }
    }

    Item {
        anchors.fill: parent
        visible: viewModel.workbenchBusy
        z: 30

        Rectangle {
            anchors.fill: parent
            color: Qt.rgba(0, 0, 0, 0.35)
        }

        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.AllButtons
            hoverEnabled: true
        }

        Column {
            anchors.centerIn: parent
            spacing: 12

            BusyIndicator {
                id: busyIndicator
                running: viewModel.workbenchBusy
                implicitWidth: 64
                implicitHeight: 64
            }

            Label {
                text: qsTr("Ładowanie katalogu strategii…")
                color: "#ffffff"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
            }
        }
    }
}
