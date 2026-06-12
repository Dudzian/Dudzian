import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "portfolioPerformanceRoot"
    property var previewState
    contentWidth: availableWidth
    clip: true
    implicitWidth: 1040
    implicitHeight: 680

    function pnlColor(valueText) {
        if (String(valueText).indexOf("-") === 0) return designSystem.color("critical")
        if (String(valueText).indexOf("+") === 0) return designSystem.color("accent")
        return designSystem.color("textSecondary")
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            Rectangle { objectName: "portfolioPerformanceTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6
                Label { objectName: "portfolioPerformanceTitle"; text: qsTr("Portfel / Wyniki"); font.bold: true; font.pixelSize: 28; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Portfolio/Wyniki to preview/report state. Live trading disabled. Exchange I/O disabled. Order submission disabled. API keys not required / not read. Time filters do not alter active Paper session."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Wartość portfela")
                description: qsTr("%1 / %2 • raport %3").arg(previewState.formatMoney(previewState.portfolioTotalEquityUsd, "USD")).arg(previewState.formatMoney(previewState.portfolioTotalEquityPln, "PLN")).arg(previewState.portfolioSelectedRange)
                Layout.preferredWidth: 340
            }
        }

        Components.PreviewCard {
            objectName: "portfolioSafetyBoundaryCard"
            designSystem: root.designSystem
            title: qsTr("Granica bezpieczeństwa preview-only")
            description: qsTr("Raport portfolio jest lokalnym snapshotem UI. runtime loop not started, exchange I/O disabled, order submission disabled, API keys not required, no secrets/env/keychain reads. Filtry czasu aktualizują tylko Portfolio report / selected range, nigdy Paper session PnL / equity.")
        }

        Components.PreviewCard {
            objectName: "portfolioTimeFiltersCard"
            designSystem: root.designSystem
            title: qsTr("Filtry czasu")
            description: previewState.portfolioRangeLabel
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: previewState.portfolioTimeFilters
                    delegate: Components.IconButton {
                        required property string modelData
                        designSystem: root.designSystem
                        text: modelData
                        subtle: previewState.portfolioSelectedRange !== modelData
                        backgroundColor: previewState.portfolioSelectedRange === modelData ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                        foregroundColor: previewState.portfolioSelectedRange === modelData ? designSystem.color("surface") : designSystem.color("textPrimary")
                        onClicked: previewState.setPortfolioTimeRange(modelData)
                    }
                }
            }
        }

        Components.PreviewCard {
            objectName: "portfolioCustomRangeCard"
            designSystem: root.designSystem
            title: qsTr("Custom range")
            description: qsTr("Preview-only pola daty/czasu. Zastosuj zakres zmienia tylko portfolio/report snapshot i etykietę zakresu; Paper session state pozostaje bez zmian.")
            GridLayout {
                Layout.fillWidth: true
                columns: width > 760 ? 3 : 1
                rowSpacing: 8
                columnSpacing: 8
                Components.StyledTextField { id: customFromField; objectName: "portfolioCustomFromInput"; designSystem: root.designSystem; text: previewState.portfolioCustomFrom; placeholderText: qsTr("from / start, np. 2026-06-01 00:00"); Layout.fillWidth: true }
                Components.StyledTextField { id: customToField; objectName: "portfolioCustomToInput"; designSystem: root.designSystem; text: previewState.portfolioCustomTo; placeholderText: qsTr("to / end, np. 2026-06-02 23:59"); Layout.fillWidth: true }
                Components.IconButton { objectName: "portfolioApplyCustomRangeButton"; designSystem: root.designSystem; text: qsTr("Zastosuj zakres"); helpText: previewState.tooltipText("Zastosuj zakres"); backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.applyPortfolioCustomRange(customFromField.text, customToField.text) }
            }
        }

        GridLayout {
            objectName: "portfolioTopSectionsGrid"
            Layout.fillWidth: true
            columns: width > 980 ? 2 : 1
            rowSpacing: 10
            columnSpacing: 10

            Components.PreviewCard {
                objectName: "portfolioAccountStateCard"
                designSystem: root.designSystem
                title: qsTr("Stan konta")
                description: qsTr("Fiat balance/equity i trading balance/equity są raportowym preview; nie są pobierane z giełdy.")
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Fiat balance / equity"); description: previewState.portfolioFiatAccountLabel; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Trading balance / equity"); description: previewState.formatMoney(previewState.portfolioTradingEquityUsdt, "USDT"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Available balance"); description: previewState.formatMoney(previewState.portfolioAvailableBalanceUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("In positions"); description: previewState.formatMoney(previewState.portfolioInPositionsUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Reserved / margin preview"); description: previewState.formatMoney(previewState.portfolioReservedMarginUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Trading available"); description: previewState.formatMoney(previewState.portfolioTradingAvailableUsdt, "USDT"); Layout.fillWidth: true }
                }
            }

            Components.PreviewCard {
                objectName: "portfolioLastCycleCard"
                designSystem: root.designSystem
                title: qsTr("Ostatni cykl transakcyjny")
                description: qsTr("Cycle id / timestamp preview, PnL, trades, winners / losers, fees i net result — wszystko local-only.")
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Cycle id / timestamp"); description: previewState.portfolioLastCycleId + " • " + previewState.portfolioLastCycleTimestamp; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Cycle PnL"); description: previewState.formatUsd(previewState.portfolioLastCyclePnlUsd); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Cycle trades count"); description: String(previewState.portfolioLastCycleTradesCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Winners / losers"); description: String(previewState.portfolioLastCycleWinners) + " / " + String(previewState.portfolioLastCycleLosers); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Fees"); description: previewState.formatMoney(previewState.portfolioLastCycleFeesUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Net result"); description: previewState.formatUsd(previewState.portfolioLastCycleNetUsd); Layout.fillWidth: true }
                }
            }

            Components.PreviewCard {
                objectName: "portfolioPaperSessionCard"
                descriptionObjectName: "previewPortfolioSummaryLabel"
                designSystem: root.designSystem
                title: qsTr("Bieżąca sesja Paper")
                description: qsTr("Paper session equity: %1 • Paper session PnL: %2 • orders: %3 • simulated: %4 • blocked: %5").arg(previewState.formatMoney(previewState.paperEquity, "USD")).arg(previewState.formatUsd(previewState.paperPnl)).arg(previewState.paperOrderRows.length).arg(previewState.paperSimulatedCount).arg(previewState.paperBlockedCount)
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper session equity"); description: previewState.formatMoney(previewState.paperEquity, "USD"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper session PnL"); description: previewState.formatUsd(previewState.paperPnl); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Paper session ticks"); description: String(previewState.paperSessionTicks); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Simulated orders"); description: String(previewState.paperSimulatedCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Blocked / no-order counts"); description: String(previewState.paperBlockedCount) + " / " + String(previewState.paperNoOrderCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Session status"); description: previewState.paperSessionStatus; Layout.fillWidth: true }
                }
            }

            Components.PreviewCard {
                objectName: "portfolioAllTimeResultCard"
                designSystem: root.designSystem
                title: qsTr("Wynik całkowity")
                description: qsTr("All-time PnL, realized/unrealized PnL, fees total, net PnL i ROI % preview.")
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("All-time PnL"); description: previewState.formatUsd(previewState.portfolioAllTimePnlUsd); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Realized PnL"); description: previewState.formatUsd(previewState.portfolioRealizedPnlUsd); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Unrealized PnL"); description: previewState.formatUsd(previewState.portfolioUnrealizedPnlUsd); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Fees total"); description: previewState.formatMoney(previewState.portfolioFeesUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Net PnL"); description: previewState.formatUsd(previewState.portfolioNetPnlUsd); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: root.designSystem; title: qsTr("ROI % preview"); description: String(previewState.portfolioRoiPercent.toFixed(2)) + "%"; Layout.fillWidth: true }
                }
            }
        }

        GridLayout {
            objectName: "portfolioPerformanceTiles"
            Layout.fillWidth: true
            columns: width > 1120 ? 4 : (width > 760 ? 3 : 1)
            rowSpacing: 10
            columnSpacing: 10
            Repeater {
                model: previewState.portfolioPerformanceCards
                delegate: Components.PreviewCard {
                    required property var modelData
                    designSystem: root.designSystem
                    title: qsTr(modelData.title)
                    description: previewState.portfolioCardDescription(modelData.field)
                    Layout.fillWidth: true
                }
            }
        }

        Components.PreviewCard {
            objectName: "portfolioCycleTableCard"
            designSystem: root.designSystem
            title: qsTr("Tabela wyników / cykle")
            description: qsTr("Lista ostatnich cykli raportu (max 10–12 widocznych): Time, Pair/Cycle, Trades, Gross PnL, Fees, Net PnL, Result.")
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 38
                radius: 10
                color: designSystem.color("surfaceMuted")
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    Label { text: qsTr("Time"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 135 }
                    Label { text: qsTr("Pair/Cycle"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 125 }
                    Label { text: qsTr("Trades"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 70 }
                    Label { text: qsTr("Gross PnL"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 110 }
                    Label { text: qsTr("Fees"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 }
                    Label { text: qsTr("Net PnL"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 110 }
                    Label { text: qsTr("Result"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                }
            }
            ListView {
                objectName: "portfolioCycleList"
                Layout.fillWidth: true
                Layout.preferredHeight: 330
                clip: true
                spacing: 8
                model: previewState.portfolioCycleRows
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view ? ListView.view.width : 980
                    height: cycleRow.implicitHeight + 18
                    radius: 12
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    RowLayout {
                        id: cycleRow
                        anchors.fill: parent
                        anchors.margins: 9
                        Label { text: modelData.startTime; color: designSystem.color("textPrimary"); Layout.preferredWidth: 135 }
                        Label { text: modelData.pair + " • " + modelData.closeReason; color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 125 }
                        Label { text: String(previewState.portfolioLastCycleTradesCount); color: designSystem.color("textSecondary"); Layout.preferredWidth: 70 }
                        Label { text: modelData.result; color: root.pnlColor(modelData.result); font.bold: true; Layout.preferredWidth: 110 }
                        Label { text: modelData.fee; color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 }
                        Label { text: modelData.result; color: root.pnlColor(modelData.result); font.bold: true; Layout.preferredWidth: 110 }
                        Rectangle { Layout.preferredWidth: 150; implicitHeight: 26; radius: 13; color: Qt.rgba(1, 1, 1, 0.05); border.color: root.pnlColor(modelData.result); Label { anchors.centerIn: parent; text: modelData.result.indexOf("-") === 0 ? qsTr("LOSS preview") : qsTr("WIN preview"); color: root.pnlColor(modelData.result); font.bold: true; font.pixelSize: 11 } }
                    }
                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: previewState.portfolioRangeLabel = qsTr("Podgląd cyklu: %1 • %2 • local-only").arg(modelData.pair).arg(modelData.closeReason)
                    }
                }
            }
        }
    }
}
