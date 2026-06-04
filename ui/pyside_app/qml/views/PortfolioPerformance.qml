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
                Label { text: qsTr("Produktowy kokpit portfela i wyników dla safe preview. Wszystkie wartości są lokalne/mock: runtime loop not started, exchange I/O disabled, order submission disabled, API keys not required, no secrets/env/keychain reads."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Components.PreviewCard {
                designSystem: root.designSystem
                title: qsTr("Wartość portfela")
                description: qsTr("%1 / %2 • zakres %3").arg(previewState.formatMoney(previewState.portfolioTotalEquityUsd, "USD")).arg(previewState.formatMoney(previewState.portfolioTotalEquityPln, "PLN")).arg(previewState.portfolioSelectedRange)
                Layout.preferredWidth: 340
            }
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
            objectName: "portfolioBreakdownCard"
            designSystem: root.designSystem
            title: qsTr("Rozbicie wyniku")
            description: qsTr("Lokalne rozbicie PnL. Funding i inne koszty są statycznym mockiem; bez backendu i bez network/API calls.")
            GridLayout {
                Layout.fillWidth: true
                columns: width > 760 ? 5 : 2
                rowSpacing: 8
                columnSpacing: 8
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Zrealizowany PnL"); description: previewState.formatUsd(previewState.portfolioRealizedPnlUsd); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Niezrealizowany PnL"); description: previewState.formatUsd(previewState.portfolioUnrealizedPnlUsd); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Prowizje"); description: previewState.formatMoney(previewState.portfolioFeesUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Funding / inne koszty"); description: previewState.formatMoney(previewState.portfolioFundingOtherCostsUsd, previewState.portfolioBaseCurrency); Layout.fillWidth: true }
                Components.PreviewCard { designSystem: root.designSystem; title: qsTr("PnL netto"); description: previewState.formatUsd(previewState.portfolioNetPnlUsd); Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            objectName: "portfolioCycleTableCard"
            designSystem: root.designSystem
            title: qsTr("Cykle transakcyjne")
            description: qsTr("Tabela preview: czas startu, czas końca, para, strategia, wynik, fee, status, powód zamknięcia. Klikalność pozostaje local-only i nie wykonuje realnych akcji.")
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 38
                radius: 10
                color: designSystem.color("surfaceMuted")
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    Label { text: qsTr("czas startu"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 130 }
                    Label { text: qsTr("czas końca"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 130 }
                    Label { text: qsTr("para"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 }
                    Label { text: qsTr("strategia"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 190 }
                    Label { text: qsTr("wynik"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 110 }
                    Label { text: qsTr("fee"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 86 }
                    Label { text: qsTr("status"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 120 }
                    Label { text: qsTr("powód zamknięcia"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
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
                        Label { text: modelData.startTime; color: designSystem.color("textPrimary"); Layout.preferredWidth: 130 }
                        Label { text: modelData.endTime; color: designSystem.color("textPrimary"); Layout.preferredWidth: 130 }
                        Label { text: modelData.pair; color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 90 }
                        Label { text: modelData.strategy; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.preferredWidth: 190 }
                        Label { text: modelData.result; color: root.pnlColor(modelData.result); font.bold: true; Layout.preferredWidth: 110 }
                        Label { text: modelData.fee; color: designSystem.color("textSecondary"); Layout.preferredWidth: 86 }
                        Rectangle { Layout.preferredWidth: 120; implicitHeight: 26; radius: 13; color: Qt.rgba(1, 1, 1, 0.05); border.color: designSystem.color("accent"); Label { anchors.centerIn: parent; text: modelData.status; color: designSystem.color("textPrimary"); font.bold: true; font.pixelSize: 11 } }
                        Label { text: modelData.closeReason; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
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
