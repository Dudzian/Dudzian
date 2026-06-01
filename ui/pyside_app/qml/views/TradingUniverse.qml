import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "tradingUniverseRoot"
    contentWidth: availableWidth
    clip: true
    property var selectedExchanges: ["Demo Exchange"]
    property var selectedPairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    property var exchanges: ["Binance", "Bybit", "OKX", "KuCoin", "Coinbase", "Demo Exchange"]
    property var pairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"]

    function toggleValue(list, value) {
        var copy = list.slice()
        var idx = copy.indexOf(value)
        if (idx >= 0)
            copy.splice(idx, 1)
        else
            copy.push(value)
        return copy
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 16
        Label { objectName: "tradingUniverseTitle"; text: qsTr("Trading Universe Preview"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Real-product looking exchange and coin universe, but every toggle is UI-only dry-run. No API calls, no exchange I/O, no order execution."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        GridLayout {
            Layout.fillWidth: true
            columns: 4
            rowSpacing: 12
            columnSpacing: 12
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Selected preview summary"); description: qsTr("Exchanges: %1 • Pairs: %2").arg(root.selectedExchanges.join(", ")).arg(root.selectedPairs.join(", ")); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Market data status"); description: qsTr("mock/local preview • no live data • static rows only"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("API key status"); description: qsTr("API keys not required • no .env • no keychain"); Layout.fillWidth: true }
            Components.PreviewCard { designSystem: root.designSystem; title: qsTr("Live trading status / Order route"); description: qsTr("Live trading disabled • order route blocked • Order submission disabled"); Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Exchange cards / toggles")
            description: qsTr("Binance, Bybit, OKX, KuCoin, Coinbase and Demo Exchange are preview-only cards with no API connection.")
            Flow {
                Layout.fillWidth: true
                spacing: 10
                Repeater {
                    model: root.exchanges
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool selected: root.selectedExchanges.indexOf(modelData) >= 0
                        width: 230
                        height: 92
                        radius: 16
                        color: selected ? Qt.rgba(0.33, 0.78, 1.0, 0.18) : designSystem.color("surfaceMuted")
                        border.color: selected ? designSystem.color("accent") : designSystem.color("border")
                        border.width: selected ? 2 : 1
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 4
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                                Components.StyledSwitch { designSystem: root.designSystem; checked: selected }
                            }
                            Label { text: qsTr("preview only • no API connection"); color: designSystem.color("textSecondary"); font.pixelSize: 11 }
                            Label { text: qsTr("no live orders • Exchange I/O disabled"); color: designSystem.color("textSecondary"); font.pixelSize: 11 }
                        }
                        MouseArea { anchors.fill: parent; onClicked: root.selectedExchanges = root.toggleValue(root.selectedExchanges, modelData) }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Coin/pair cards / toggles")
            description: qsTr("Selection is local UI state only and is not written to runtime/live config.")
            Flow {
                Layout.fillWidth: true
                spacing: 10
                Repeater {
                    model: root.pairs
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool selected: root.selectedPairs.indexOf(modelData) >= 0
                        width: 150
                        height: 58
                        radius: 14
                        color: selected ? Qt.rgba(0.33, 0.78, 1.0, 0.18) : designSystem.color("surfaceMuted")
                        border.color: selected ? designSystem.color("accent") : designSystem.color("border")
                        border.width: selected ? 2 : 1
                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                            Components.StyledSwitch { designSystem: root.designSystem; checked: selected }
                        }
                        MouseArea { anchors.fill: parent; onClicked: root.selectedPairs = root.toggleValue(root.selectedPairs, modelData) }
                    }
                }
            }
        }
    }
}
