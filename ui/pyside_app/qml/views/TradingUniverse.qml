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
    property var quoteFilters: ["All", "USDT", "USDC", "BTC", "ETH"]
    property var categoryFilters: ["All", "Major", "AI", "Meme", "DeFi", "Layer1", "Layer2", "High volume"]

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "tradingUniverseTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "tradingUniverseTitle"; text: qsTr("Trading Universe"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Skalowalny selektor rynku dla Paper Preview. To local preview catalog z ponad 100 parami; bez real API calls, bez exchange route i bez order submission."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Exchange market import flow")
            description: qsTr("choose exchange -> sandbox/testnet/API planned/disabled -> import markets preview -> AI scans eligible pairs -> paper/testserver route planned/disabled • paper/testserver trading planned/disabled • preview-only local catalog")
            GridLayout {
                Layout.fillWidth: true
                columns: width > 1100 ? 5 : 1
                rowSpacing: 8
                columnSpacing: 8
                Repeater {
                    model: [
                        ({ step: "Step 1", title: "choose exchange", state: "UI-only" }),
                        ({ step: "Step 2", title: "sandbox/testnet/API planned/disabled", state: "API keys not required" }),
                        ({ step: "Step 3", title: "import markets preview", state: "local catalog, no API calls" }),
                        ({ step: "Step 4", title: "AI scans all eligible pairs", state: "preview-only" }),
                        ({ step: "Step 5", title: "paper/testserver route planned/disabled", state: "order submission disabled" })
                    ]
                    delegate: Rectangle {
                        required property var modelData
                        Layout.fillWidth: true
                        implicitHeight: 86
                        radius: 14
                        color: designSystem.color("surfaceMuted")
                        border.color: designSystem.color("border")
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            Label { text: modelData.step; color: designSystem.color("accent"); font.bold: true }
                            Label { text: modelData.title; color: designSystem.color("textPrimary"); font.bold: true; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                            Label { text: modelData.state; color: designSystem.color("textSecondary"); font.pixelSize: 11; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Exchange selector • Market data status • API key status • Live trading status / Order route")
            description: qsTr("Karty giełd są tylko lokalnym wyborem preview. Live trading disabled • Exchange route disabled • Order submission disabled • API keys not required.")
            Flow {
                Layout.fillWidth: true
                spacing: 10
                Repeater {
                    model: previewState.previewExchanges
                    delegate: Rectangle {
                        required property string modelData
                        readonly property bool selected: previewState.hasValue(previewState.selectedExchanges, modelData)
                        width: 255
                        height: 124
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
                            Label { text: selected ? qsTr("wybrana • preview-only") : qsTr("rozłączona • preview-only"); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                            Label { text: qsTr("sandbox/testnet/API planned/disabled"); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                            Label { text: qsTr("exchange route disabled • no real API calls"); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        }
                        MouseArea { anchors.fill: parent; onClicked: previewState.toggleExchange(modelData) }
                    }
                }
            }
        }

        Components.PreviewCard {
            designSystem: root.designSystem
            title: qsTr("Market universe selector")
            description: qsTr("Search pair, quote filter: USDT, USDC, BTC, ETH; category filter: Major, AI, Meme, DeFi, Layer1, Layer2, High volume; selected only; AI candidates; excluded/blacklist.")
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Import markets preview"); iconName: "cloud"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.importMarketsPreview() }
                Components.StyledTextField { objectName: "tradingUniverseSearchPair"; designSystem: root.designSystem; placeholderText: qsTr("Search pair"); text: previewState.marketSearch; onTextChanged: previewState.marketSearch = text; width: 240 }
                Repeater { model: root.quoteFilters; delegate: Components.IconButton { required property string modelData; designSystem: root.designSystem; text: qsTr(modelData); subtle: previewState.marketQuoteFilter !== modelData; onClicked: previewState.marketQuoteFilter = modelData } }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater { model: root.categoryFilters; delegate: Components.IconButton { required property string modelData; designSystem: root.designSystem; text: qsTr(modelData); subtle: previewState.marketCategoryFilter !== modelData; onClicked: previewState.marketCategoryFilter = modelData } }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Selected only"); subtle: !previewState.marketSelectedOnly; onClicked: previewState.marketSelectedOnly = !previewState.marketSelectedOnly }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("AI candidates"); subtle: !previewState.marketAiCandidatesOnly; onClicked: previewState.marketAiCandidatesOnly = !previewState.marketAiCandidatesOnly }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Excluded / blacklist"); subtle: !previewState.marketExcludedOnly; onClicked: previewState.marketExcludedOnly = !previewState.marketExcludedOnly }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Select all visible"); onClicked: previewState.selectAllVisiblePairs() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Clear selected"); subtle: true; onClicked: previewState.clearSelectedPairs() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Select top 20"); onClicked: previewState.selectTop20Pairs() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Blacklist selected"); subtle: true; onClicked: previewState.blacklistSelectedPairs() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Whitelist selected"); onClicked: previewState.whitelistSelectedPairs() }
            }
            RowLayout {
                Layout.fillWidth: true
                Label { text: qsTr("total pairs: %1 • visible pairs: %2 • selected pairs: %3 • whitelisted: %4 • blacklisted: %5 • AI candidates: %6").arg(previewState.previewMarketPairs.length).arg(previewState.visiblePairsCount()).arg(previewState.selectedPairsCount()).arg(previewState.whitelistedPairsCount()).arg(previewState.blacklistedPairsCount()).arg(previewState.aiCandidatesCount()); color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
            }
            GridView {
                objectName: "tradingUniversePairsGrid"
                Layout.fillWidth: true
                Layout.preferredHeight: 430
                clip: true
                cellWidth: 230
                cellHeight: 82
                model: previewState.filteredMarketPairs()
                delegate: Rectangle {
                    required property string modelData
                    readonly property bool selected: previewState.hasValue(previewState.selectedPairs, modelData)
                    readonly property bool blacklisted: previewState.hasValue(previewState.blacklistPairs, modelData)
                    width: 210
                    height: 72
                    radius: 14
                    color: selected ? Qt.rgba(0.33, 0.78, 1.0, 0.18) : (blacklisted ? Qt.rgba(1, 0.32, 0.32, 0.10) : designSystem.color("surfaceMuted"))
                    border.color: selected ? designSystem.color("accent") : (blacklisted ? designSystem.color("critical") : designSystem.color("border"))
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 9
                        RowLayout {
                            Layout.fillWidth: true
                            Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true }
                            Label { text: previewState.isAiCandidate(modelData) ? qsTr("AI") : qsTr("PAIR"); color: previewState.isAiCandidate(modelData) ? designSystem.color("accent") : designSystem.color("textSecondary"); font.pixelSize: 11 }
                        }
                        Label { text: previewState.pairCategory(modelData) + " • " + (blacklisted ? qsTr("excluded/blacklist") : (selected ? qsTr("selected/whitelist") : qsTr("eligible"))); color: designSystem.color("textSecondary"); font.pixelSize: 11; Layout.fillWidth: true }
                    }
                    MouseArea { anchors.fill: parent; acceptedButtons: Qt.LeftButton | Qt.RightButton; onClicked: function(mouse) { if (mouse.button === Qt.RightButton) previewState.toggleBlacklist(modelData); else previewState.togglePair(modelData) } }
                }
            }
        }
    }
}
