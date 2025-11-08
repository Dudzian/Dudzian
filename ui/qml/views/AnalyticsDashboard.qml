import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml
import "../design-system" as DesignSystem
import "../design-system/components" as DesignComponents
import "../design-system/charts" as DesignCharts
import "AnalyticsUtils.js" as AnalyticsUtils

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property var resultsModel: (typeof resultsDashboard !== "undefined" ? resultsDashboard : null)
    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property var pnlSeries: resultsModel ? resultsModel.equityTimeline : []
    property var heatmapRows: []
    property var sandboxDecisions: []
    property int sandboxDecisionCount: 0
    property var sandboxLastTimestamp: null
    property string sandboxLastUpdateText: qsTr("Brak danych")
    property string sandboxError: runtimeService && runtimeService.errorMessage ? runtimeService.errorMessage : ""
    property var sandboxTopSymbols: []
    property var sandboxBottomSymbols: []
    property int sandboxDecisionWindow: 120
    property var sandboxWindowOptions: [
        { label: qsTr("Ostatnie %1 decyzji").arg(30), value: 30 },
        { label: qsTr("Ostatnie %1 decyzji").arg(60), value: 60 },
        { label: qsTr("Ostatnie %1 decyzji").arg(120), value: 120 },
        { label: qsTr("Ostatnie %1 decyzji").arg(240), value: 240 }
    ]

    Timer {
        id: refreshTimer
        interval: 4000
        running: root.runtimeService !== null && root.runtimeService !== undefined
        repeat: true
        triggeredOnStart: true
        onTriggered: root.refreshSandbox()
    }

    Connections {
        target: resultsModel
        function onTimelineChanged() {
            root.pnlSeries = resultsModel ? resultsModel.equityTimeline : []
        }
    }

    Connections {
        target: runtimeService
        ignoreUnknownSignals: true
        function onDecisionsChanged() {
            if (!runtimeService)
                return
            var records = runtimeService.decisions || []
            root.applySandboxRecords(records)
        }
        function onErrorMessageChanged() {
            root.sandboxError = runtimeService && runtimeService.errorMessage ? runtimeService.errorMessage : ""
        }
    }

    onResultsModelChanged: {
        pnlSeries = resultsModel ? resultsModel.equityTimeline : []
    }

    onRuntimeServiceChanged: {
        sandboxError = runtimeService && runtimeService.errorMessage ? runtimeService.errorMessage : ""
        if (runtimeService) {
            refreshSandbox()
        } else {
            sandboxDecisions = []
            heatmapRows = []
            sandboxDecisionCount = 0
            sandboxLastTimestamp = null
            sandboxLastUpdateText = qsTr("Brak danych")
            sandboxTopSymbols = []
            sandboxBottomSymbols = []
        }
    }

    function refreshSandbox() {
        if (!runtimeService || !runtimeService.loadRecentDecisions) {
            sandboxError = runtimeService && runtimeService.errorMessage ? runtimeService.errorMessage : ""
            sandboxDecisionCount = 0
            sandboxLastTimestamp = null
            sandboxLastUpdateText = qsTr("Brak danych")
            sandboxTopSymbols = []
            sandboxBottomSymbols = []
            return
        }
        var limit = sandboxDecisionWindow > 0 ? sandboxDecisionWindow : 120
        var records = runtimeService.loadRecentDecisions(limit) || []
        applySandboxRecords(records)
        sandboxError = runtimeService.errorMessage || ""
    }

    function applySandboxRecords(records) {
        sandboxDecisions = records
        heatmapRows = AnalyticsUtils.buildHeatmap(records, {
            noSymbol: qsTr("N/D"),
            noDate: qsTr("Brak daty")
        })
        sandboxDecisionCount = AnalyticsUtils.toDecisionCount(records)
        var latest = AnalyticsUtils.latestTimestamp(records)
        sandboxLastTimestamp = latest
        sandboxLastUpdateText = latest ? Qt.formatDateTime(latest, Qt.DefaultLocaleShortDate) : qsTr("Brak danych")
        var rankings = AnalyticsUtils.rankSymbols(records, {
            limit: 5,
            noSymbol: qsTr("N/D")
        })
        sandboxTopSymbols = rankings.top
        sandboxBottomSymbols = rankings.bottom
    }

    function sandboxWindowIndex() {
        for (var i = 0; i < sandboxWindowOptions.length; ++i) {
            if (sandboxWindowOptions[i].value === sandboxDecisionWindow)
                return i
        }
        return 0
    }

    function selectSandboxWindow(index) {
        if (index < 0 || index >= sandboxWindowOptions.length)
            return
        var option = sandboxWindowOptions[index]
        if (!option)
            return
        if (sandboxDecisionWindow === option.value)
            return
        sandboxDecisionWindow = option.value
        refreshSandbox()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Zwrot skumulowany")
                value: resultsModel ? Qt.formatLocaleNumber(resultsModel.cumulativeReturn * 100, "f", 2) + "%" : "--"
            }
            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Maks. obsunięcie")
                value: resultsModel ? Qt.formatLocaleNumber(resultsModel.maxDrawdown * 100, "f", 2) + "%" : "--"
            }
            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Sharpe")
                value: resultsModel ? Qt.formatLocaleNumber(resultsModel.sharpeRatio, "f", 2) : "--"
            }
            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Volatility")
                value: resultsModel ? Qt.formatLocaleNumber(resultsModel.annualizedVolatility * 100, "f", 2) + "%" : "--"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Decyzje sandbox")
                value: sandboxDecisionCount.toString()
            }

            DesignComponents.MetricTile {
                Layout.fillWidth: true
                label: qsTr("Ostatnia aktualizacja")
                value: sandboxLastUpdateText
            }

            Item { Layout.fillWidth: true }

            RowLayout {
                spacing: 8
                Layout.alignment: Qt.AlignVCenter

                Label {
                    text: qsTr("Zakres sandbox")
                    color: DesignSystem.Palette.textSecondary
                    font.pixelSize: DesignSystem.Typography.body
                }

                ComboBox {
                    id: sandboxWindowSelector
                    model: root.sandboxWindowOptions
                    textRole: "label"
                    valueRole: "value"
                    currentIndex: root.sandboxWindowIndex()
                    onActivated: root.selectSandboxWindow(index)
                    implicitWidth: 200
                }

                Button {
                    text: qsTr("Odśwież dane")
                    onClicked: root.refreshSandbox()
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            DesignComponents.Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                background.color: DesignSystem.Palette.elevated

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    Label {
                        text: qsTr("Najlepsze instrumenty")
                        color: DesignSystem.Palette.textPrimary
                        font.pixelSize: DesignSystem.Typography.title
                        font.bold: true
                    }

                    Repeater {
                        model: root.sandboxTopSymbols
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                Layout.fillWidth: true
                                text: modelData.label
                                color: DesignSystem.Palette.textPrimary
                                font.pixelSize: DesignSystem.Typography.body
                            }

                            Label {
                                text: Qt.formatLocaleNumber(modelData.value, "f", 2)
                                color: DesignSystem.Palette.success
                                font.pixelSize: DesignSystem.Typography.body
                                font.bold: true
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: root.sandboxTopSymbols.length === 0
                        text: qsTr("Brak wyników")
                        color: DesignSystem.Palette.textSecondary
                        font.pixelSize: DesignSystem.Typography.body
                    }
                }
            }

            DesignComponents.Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                background.color: DesignSystem.Palette.elevated

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    Label {
                        text: qsTr("Najsłabsze instrumenty")
                        color: DesignSystem.Palette.textPrimary
                        font.pixelSize: DesignSystem.Typography.title
                        font.bold: true
                    }

                    Repeater {
                        model: root.sandboxBottomSymbols
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                Layout.fillWidth: true
                                text: modelData.label
                                color: DesignSystem.Palette.textPrimary
                                font.pixelSize: DesignSystem.Typography.body
                            }

                            Label {
                                text: Qt.formatLocaleNumber(modelData.value, "f", 2)
                                color: DesignSystem.Palette.warning
                                font.pixelSize: DesignSystem.Typography.body
                                font.bold: true
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: root.sandboxBottomSymbols.length === 0
                        text: qsTr("Brak wyników")
                        color: DesignSystem.Palette.textSecondary
                        font.pixelSize: DesignSystem.Typography.body
                    }
                }
            }
        }

        DesignComponents.Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 260
            background.color: DesignSystem.Palette.elevated

            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                Label {
                    text: qsTr("Krzywa P&L")
                    color: DesignSystem.Palette.textPrimary
                    font.pixelSize: DesignSystem.Typography.title
                    font.bold: true
                }

                DesignCharts.PnlChart {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    data: root.pnlSeries
                    lineColor: DesignSystem.Palette.accent
                }
            }
        }

        DesignComponents.Card {
            Layout.fillWidth: true
            Layout.fillHeight: true
            background.color: DesignSystem.Palette.elevated

            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                Label {
                    text: qsTr("Heatmapa sandbox")
                    color: DesignSystem.Palette.textPrimary
                    font.pixelSize: DesignSystem.Typography.title
                    font.bold: true
                }

                Label {
                    Layout.fillWidth: true
                    visible: sandboxError.length > 0
                    text: sandboxError
                    wrapMode: Text.WordWrap
                    color: DesignSystem.Palette.warning
                    font.pixelSize: DesignSystem.Typography.body
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    DesignCharts.HeatmapChart {
                        anchors.fill: parent
                        visible: root.heatmapRows.length > 0
                        rows: root.heatmapRows
                    }

                    Label {
                        anchors.centerIn: parent
                        visible: root.heatmapRows.length === 0 && sandboxError.length === 0
                        text: qsTr("Brak danych sandbox")
                        color: DesignSystem.Palette.textSecondary
                        font.pixelSize: DesignSystem.Typography.body
                    }
                }
            }
        }
    }
}
