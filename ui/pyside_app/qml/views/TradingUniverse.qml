import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "tradingUniverseRoot"
    property var previewState
    contentWidth: availableWidth
    clip: true
    property var filters: ["All", "Selected", "Top volume", "AI candidates", "Excluded"]
    property var topVolumePairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "TON/USDT", "LINK/USDT"]
    property var aiCandidatePairs: ["BTC/USDT", "SOL/USDT", "INJ/USDT", "TAO/USDT", "RENDER/USDT", "FET/USDT", "ARB/USDT", "OP/USDT"]

    function filteredPairs() {
        var pairs = previewState.marketsImported ? previewState.previewMarketPairs.slice() : previewState.previewMarketPairs.slice(0, 8)
        var term = previewState.marketSearch.toLowerCase()
        var out = []
        for (var i = 0; i < pairs.length; ++i) {
            var pair = pairs[i]
            if (term.length > 0 && pair.toLowerCase().indexOf(term) < 0)
                continue
            if (previewState.marketFilter === "Selected" && !previewState.hasValue(previewState.selectedPairs, pair))
                continue
            if (previewState.marketFilter === "Top volume" && topVolumePairs.indexOf(pair) < 0)
                continue
            if (previewState.marketFilter === "AI candidates" && aiCandidatePairs.indexOf(pair) < 0)
                continue
            if (previewState.marketFilter === "Excluded" && !previewState.hasValue(previewState.blacklistPairs, pair))
                continue
            out.push(pair)
        }
        return out
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        Label { objectName: "tradingUniverseTitle"; text: qsTr("Trading Universe"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
        Label { text: qsTr("Market selector final-product preview: exchange cards, mock Import markets preview, Search pair, filters, whitelist/blacklist preview state. Żadna akcja nie woła API giełdy."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Exchange cards/toggles")
            description: qsTr("Status każdej giełdy: disconnected / preview-only / api keys not required / exchange I/O disabled")
            Flow {
                Layout.fillWidth: true
                spacing: 10
                Repeater {
                    model: previewState.previewExchanges
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool selected: previewState.hasValue(previewState.selectedExchanges, modelData)
                        width: 220
                        height: 96
                        radius: 16
                        color: selected ? Qt.rgba(0.33, 0.78, 1.0, 0.18) : designSystem.color("surfaceMuted")
                        border.color: selected ? designSystem.color("accent") : designSystem.color("border")
                        border.width: selected ? 2 : 1
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 3
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                                Components.StyledSwitch { designSystem: root.designSystem; checked: selected; onToggled: previewState.toggleExchange(modelData) }
                            }
                            Label { text: selected ? qsTr("preview-only • disconnected") : qsTr("disconnected • preview-only"); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true }
                            Label { text: qsTr("api keys not required • exchange I/O disabled"); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true }
                        }
                        MouseArea { anchors.fill: parent; onClicked: previewState.toggleExchange(modelData) }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Mock market import and filters • Market data status • API key status • Live trading status / Order route")
            description: qsTr("Import markets preview ładuje lokalną listę minimum 25 par preview — bez real exchange I/O.")
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Import markets preview"); iconName: "cloud"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.importMarketsPreview() }
                Components.StyledTextField { objectName: "tradingUniverseSearchPair"; designSystem: root.designSystem; placeholderText: qsTr("Search pair"); text: previewState.marketSearch; onTextChanged: previewState.marketSearch = text; Layout.preferredWidth: 260 }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("select all"); subtle: true; onClicked: previewState.selectAllPairs() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("clear selected"); subtle: true; onClicked: previewState.clearSelectedPairs() }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.filters
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool active: previewState.marketFilter === modelData
                        width: 130
                        height: 38
                        radius: 12
                        color: active ? Qt.rgba(0.33, 0.78, 1.0, 0.20) : designSystem.color("surfaceMuted")
                        border.color: active ? designSystem.color("accent") : designSystem.color("border")
                        Label { anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: active }
                        MouseArea { anchors.fill: parent; onClicked: previewState.marketFilter = modelData }
                    }
                }
            }
            Label { text: qsTr("Whitelist preview state: %1 • Blacklist preview state: %2 • selected pairs: %3").arg(previewState.whitelistPairs.join(", ")).arg(previewState.blacklistPairs.join(", ")).arg(previewState.selectedPairs.length); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Local preview markets list")
            description: qsTr("BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT ADA/USDT DOGE/USDT AVAX/USDT DOT/USDT LINK/USDT LTC/USDT BCH/USDT ATOM/USDT NEAR/USDT ARB/USDT OP/USDT INJ/USDT APT/USDT SUI/USDT TON/USDT PEPE/USDT WIF/USDT FET/USDT RENDER/USDT TAO/USDT UNI/USDT AAVE/USDT ETC/USDT FIL/USDT ICP/USDT")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.filteredPairs()
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool selected: previewState.hasValue(previewState.selectedPairs, modelData)
                        readonly property bool excluded: previewState.hasValue(previewState.blacklistPairs, modelData)
                        width: 154
                        height: 70
                        radius: 14
                        color: selected ? Qt.rgba(0.33, 0.78, 1.0, 0.18) : (excluded ? Qt.rgba(1, 0.44, 0.55, 0.14) : designSystem.color("surfaceMuted"))
                        border.color: selected ? designSystem.color("accent") : (excluded ? designSystem.color("critical") : designSystem.color("border"))
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 2
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                                Components.StyledSwitch { designSystem: root.designSystem; checked: selected; onToggled: previewState.togglePair(modelData) }
                            }
                            Label { text: selected ? qsTr("whitelist preview") : (excluded ? qsTr("excluded preview") : qsTr("AI candidate/top volume preview")); color: designSystem.color("textSecondary"); font.pixelSize: 10; Layout.fillWidth: true }
                            Label { text: qsTr("local only • no API"); color: designSystem.color("textSecondary"); font.pixelSize: 10; Layout.fillWidth: true }
                        }
                        MouseArea { anchors.fill: parent; acceptedButtons: Qt.LeftButton | Qt.RightButton; onClicked: function(mouse) { if (mouse.button === Qt.RightButton) previewState.toggleBlacklist(modelData); else previewState.togglePair(modelData) } }
                    }
                }
            }
        }
    }
}
