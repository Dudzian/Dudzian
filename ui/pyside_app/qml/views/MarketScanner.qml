import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "marketScannerRoot"
    property var previewState
    contentWidth: availableWidth
    clip: true
    property var filterModes: ["All", "AI candidates", "Trade candidates", "Watchlist", "Rejected", "Blocked", "High liquidity", "Low risk", "Top score"]
    property var sortModes: ["AI score", "Risk score", "Liquidity", "Volume", "Spread", "Trend strength"]

    ColumnLayout {
        width: root.availableWidth
        spacing: 14
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "marketScannerTitle"; text: qsTr("Okazje / Market Scanner"); font.bold: true; font.pixelSize: 26; color: designSystem.color("textPrimary") }
                Label { text: qsTr("Safe preview scanner ocenia lokalny katalog par, typuje AI candidates, odrzuca słabe setupy i wyjaśnia decyzje bez realnych połączeń API."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                Label { text: qsTr("Watchlist = obserwacja. Whitelist = dopuszczone pary. Te listy są rozdzielone. / Watchlist is for observation. Whitelist is for allowed pairs. These lists are separate."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }
        Components.PreviewCard {
            objectName: "marketScannerSafetyBoundaryCard"
            designSystem: root.designSystem
            title: qsTr("Safe preview scanner / Skaner działa lokalnie w preview")
            description: qsTr("Safe preview scanner • Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • No real orders • No network/API calls • Local preview catalog only")
            Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: ["Skaner działa lokalnie w preview", "Live trading wyłączony", "Połączenie z giełdą wyłączone", "Składanie zleceń wyłączone", "Klucze API nie są wymagane", "Brak prawdziwych zleceń", "Brak realnych połączeń API", "Tylko lokalny katalog preview"]; delegate: Rectangle { required property string modelData; width: Math.max(190, safetyLabel.implicitWidth + 22); height: 34; radius: 12; color: Qt.rgba(0.33, 0.78, 1.0, 0.12); border.color: designSystem.color("border"); Label { id: safetyLabel; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.pixelSize: 12 } } } }
        }
        GridLayout {
            objectName: "marketScannerStatusGrid"
            Layout.fillWidth: true; columns: width > 1120 ? 4 : 2; rowSpacing: 10; columnSpacing: 10
            Repeater { model: [({ title: "status skanera", value: previewState.scannerStatus }), ({ title: "skanowane pary", value: String(previewState.scannerUniverseCount) }), ({ title: "kandydaci AI", value: String(previewState.scannerCandidateCount) }), ({ title: "odrzucone pary", value: String(previewState.scannerRejectedCount) }), ({ title: "watchlist", value: String(previewState.scannerWatchlistCount) }), ({ title: "najlepsza okazja", value: previewState.scannerBestOpportunity }), ({ title: "ostatni scan tick", value: previewState.scannerLastScanAt }), ({ title: "scenariusz rynku", value: previewState.simulationScenario })]; delegate: Components.PreviewCard { required property var modelData; designSystem: root.designSystem; title: modelData.title; description: modelData.value } }
        }
        Components.PreviewCard {
            objectName: "marketScannerControlsCard"
            designSystem: root.designSystem
            title: qsTr("Filters, sort and threshold controls")
            description: qsTr("All • AI candidates • Trade candidates • Watchlist • Rejected • Blocked • High liquidity • Low risk • Top score. Sort: AI score • Risk score • Liquidity • Volume • Spread • Trend strength.")
            Flow { Layout.fillWidth: true; spacing: 8
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Start scanner"); helpText: previewState.tooltipText("Start scanner"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: previewState.startMarketScannerPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Pause scanner"); helpText: previewState.tooltipText("Pause scanner"); onClicked: previewState.pauseMarketScannerPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Stop"); helpText: previewState.tooltipText("Stop"); onClicked: previewState.stopMarketScannerPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Reset"); helpText: previewState.tooltipText("Reset"); onClicked: previewState.resetMarketScannerPreview() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Run scan tick"); helpText: previewState.tooltipText("Run scan tick"); onClicked: previewState.runMarketScannerTick() }
                Components.IconButton { designSystem: root.designSystem; text: qsTr("Run scan burst"); helpText: previewState.tooltipText("Run scan burst"); onClicked: previewState.runMarketScannerBurst(5) }
            }
            RowLayout { Layout.fillWidth: true; spacing: 8; Label { text: qsTr("Filter"); color: designSystem.color("textPrimary") } ComboBox { objectName: "marketScannerFilterModeControl"; model: root.filterModes; currentIndex: Math.max(0, root.filterModes.indexOf(previewState.scannerFilterMode)); onActivated: previewState.setScannerFilterMode(root.filterModes[index]); Layout.preferredWidth: 180 } Label { text: qsTr("Sort"); color: designSystem.color("textPrimary") } ComboBox { objectName: "marketScannerSortModeControl"; model: root.sortModes; currentIndex: Math.max(0, root.sortModes.indexOf(previewState.scannerSortMode)); onActivated: previewState.setScannerSortMode(root.sortModes[index]); Layout.preferredWidth: 170 } }
            GridLayout { objectName: "marketScannerThresholdControls"; Layout.fillWidth: true; columns: width > 760 ? 3 : 1
                RowLayout { Label { text: qsTr("min AI score"); color: designSystem.color("textPrimary") } Slider { from: 0; to: 100; value: previewState.scannerMinAiScore; onMoved: previewState.setScannerThreshold("minAiScore", value); Layout.preferredWidth: 150 } Label { text: String(Math.round(previewState.scannerMinAiScore)); color: designSystem.color("textSecondary") } }
                RowLayout { Label { text: qsTr("min liquidity score"); color: designSystem.color("textPrimary") } Slider { from: 0; to: 100; value: previewState.scannerMinLiquidityScore; onMoved: previewState.setScannerThreshold("minLiquidityScore", value); Layout.preferredWidth: 150 } Label { text: String(Math.round(previewState.scannerMinLiquidityScore)); color: designSystem.color("textSecondary") } }
                RowLayout { Label { text: qsTr("max risk score"); color: designSystem.color("textPrimary") } Slider { from: 0; to: 100; value: previewState.scannerMaxRiskScore; onMoved: previewState.setScannerThreshold("maxRiskScore", value); Layout.preferredWidth: 150 } Label { text: String(Math.round(previewState.scannerMaxRiskScore)); color: designSystem.color("textSecondary") } }
            }
        }
        Components.PreviewCard {
            objectName: "marketScannerOpportunitiesTable"
            designSystem: root.designSystem
            title: qsTr("Okazje — Market Scanner table")
            description: qsTr("Pair • Exchange • Price • Volume • Trend • Spread • Liquidity • AI score • Risk • Strategy • Recommendation • Reason")
            ColumnLayout { Layout.fillWidth: true; spacing: 4
                RowLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: ["Pair", "Exchange", "Price", "Volume", "Trend", "Spread", "Liquidity", "AI score", "Risk", "Strategy", "Recommendation", "Reason"]; delegate: Label { required property string modelData; text: modelData; color: designSystem.color("accent"); font.bold: true; font.pixelSize: 11; Layout.preferredWidth: modelData === "Reason" ? 220 : 80; elide: Text.ElideRight } } }
                Repeater { model: previewState.visibleScannerRows().slice(0, 12); delegate: Rectangle { required property var modelData; Layout.fillWidth: true; height: 42; radius: 10; color: previewState.scannerSelectedPair === modelData.pair ? Qt.rgba(0.33, 0.78, 1.0, 0.16) : designSystem.color("surfaceMuted"); border.color: designSystem.color("border"); MouseArea { anchors.fill: parent; onClicked: previewState.selectScannerPair(modelData.pair) } RowLayout { anchors.fill: parent; anchors.margins: 6; spacing: 6; Label { text: modelData.pair; color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 80 } Label { text: modelData.exchange; color: designSystem.color("textSecondary"); Layout.preferredWidth: 80; elide: Text.ElideRight } Label { text: modelData.price; color: designSystem.color("textPrimary"); Layout.preferredWidth: 80 } Label { text: modelData.volume; color: designSystem.color("textSecondary"); Layout.preferredWidth: 80 } Label { text: modelData.trend; color: designSystem.color("textPrimary"); Layout.preferredWidth: 80; elide: Text.ElideRight } Label { text: modelData.spread; color: designSystem.color("textSecondary"); Layout.preferredWidth: 80 } Label { text: String(modelData.liquidityScore); color: designSystem.color("textPrimary"); Layout.preferredWidth: 80 } Label { text: String(modelData.aiScore); color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 80 } Label { text: String(modelData.riskScore); color: modelData.riskScore <= previewState.scannerMaxRiskScore ? designSystem.color("textPrimary") : designSystem.color("warning"); Layout.preferredWidth: 80 } Label { text: modelData.strategyMatch; color: designSystem.color("textSecondary"); Layout.preferredWidth: 80; elide: Text.ElideRight } Label { text: modelData.recommendation; color: modelData.recommendation === "TRADE" ? designSystem.color("accent") : designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 80 } Label { text: modelData.reason; color: designSystem.color("textSecondary"); Layout.fillWidth: true; elide: Text.ElideRight } } } }
            }
        }
        Components.PreviewCard { objectName: "marketScannerExplanationPanel"; designSystem: root.designSystem; title: qsTr("Dlaczego bot wybrał / odrzucił tę parę?"); description: previewState.scannerExplanation; Flow { Layout.fillWidth: true; spacing: 8; Components.IconButton { designSystem: root.designSystem; text: qsTr("Explain candidate"); helpText: previewState.tooltipText("Explain candidate"); onClicked: previewState.explainScannerCandidateDecision(previewState.scannerSelectedPair) } Components.IconButton { designSystem: root.designSystem; text: qsTr("Watchlist +"); helpText: previewState.tooltipText("Watchlist"); onClicked: previewState.addScannerPairToWatchlist(previewState.scannerSelectedPair) } Components.IconButton { designSystem: root.designSystem; text: qsTr("Watchlist -"); helpText: previewState.tooltipText("Watchlist"); onClicked: previewState.removeScannerPairFromWatchlist(previewState.scannerSelectedPair) } Components.IconButton { designSystem: root.designSystem; text: qsTr("Blacklist"); helpText: previewState.tooltipText("Blacklist") + " • preview-local blocklist shared with Trading Universe"; onClicked: previewState.blacklistScannerPair(previewState.scannerSelectedPair) } } }
    }
}
