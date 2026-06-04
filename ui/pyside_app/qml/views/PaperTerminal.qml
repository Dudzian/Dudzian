import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Components.StyledScrollView {
    id: root
    objectName: "paperTerminalRoot"
    property var previewState
    property var designSystem
    contentWidth: availableWidth
    clip: true

    readonly property color buyTint: Qt.rgba(0.22, 0.78, 0.54, 0.18)
    readonly property color sellTint: Qt.rgba(1.0, 0.31, 0.37, 0.18)
    readonly property color buyText: Qt.rgba(0.56, 1.0, 0.77, 1.0)
    readonly property color sellText: Qt.rgba(1.0, 0.62, 0.66, 1.0)
    readonly property string activePair: previewState.selectedTerminalPair && previewState.selectedTerminalPair.length > 0 ? previewState.selectedTerminalPair : "BTC/USDT"
    readonly property real orderValue: Number(previewState.terminalTotal)
    readonly property string feeEstimate: isNaN(orderValue) ? "0.00 USDT" : (orderValue * 0.001).toFixed(2) + " USDT"
    readonly property string availableBalance: "100,000.00 USDT paper balance"
    readonly property int cockpitColumns: root.availableWidth >= 1180 ? 3 : (root.availableWidth >= 760 ? 2 : 1)
    readonly property real orderFormPreferredWidth: cockpitColumns === 3 ? 320 : 340
    readonly property real chartPreferredWidth: cockpitColumns === 3 ? Math.max(520, root.availableWidth - 700) : 620
    readonly property real orderBookPreferredWidth: cockpitColumns === 3 ? 320 : 340
    readonly property real orderBookScrollHeight: cockpitColumns === 3 ? 360 : 430

    ColumnLayout {
        width: parent.availableWidth
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle { objectName: "paperTerminalTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
            ColumnLayout {
                Layout.fillWidth: true
                Label { objectName: "paperTerminalTitle"; text: qsTr("Paper Terminal"); font.bold: true; font.pixelSize: 24; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                Label { text: qsTr("Produktowy kokpit tradingowy preview dla %1 — lokalny Paper Preview. Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • Runtime loop not started • No real orders.").arg(root.activePair); wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 16
            color: designSystem.color("surfaceElevated")
            border.color: designSystem.color("border")
            implicitHeight: safetyRow.implicitHeight + 24
            Flow {
                id: safetyRow
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                Repeater {
                    model: ["local-only paper bridge/state", "Paper Preview only", "Live trading disabled", "Exchange I/O disabled", "Order submission disabled", "API keys not required", "Runtime loop not started", "No real orders", "paper simulation only"]
                    delegate: Rectangle {
                        required property string modelData
                        radius: 999
                        color: designSystem.color("surfaceMuted")
                        border.color: designSystem.color("border")
                        implicitWidth: safetyText.implicitWidth + 20
                        implicitHeight: 30
                        Label { id: safetyText; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: true }
                    }
                }
            }
        }

        Components.PreviewCard {
            objectName: "paperTerminalPairSelector"
            designSystem: root.designSystem
            title: qsTr("Selektor pary")
            description: qsTr("Lokalny selektor par: wyszukiwarka i chipy czytają selectedPairs albo previewMarketPairs. Wybór aktualizuje tylko selectedTerminalPair; no API call i bez ładowania rynku.")
            Layout.fillWidth: true
            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Components.StyledTextField {
                    objectName: "paperTerminalPairSearchInput"
                    designSystem: root.designSystem
                    placeholderText: qsTr("Search pair")
                    text: previewState.terminalPairSearch
                    Layout.preferredWidth: 240
                    onTextEdited: previewState.terminalPairSearch = text
                }
                Label { text: qsTr("aktywna para: %1 • lokalny selektor • selectedPairs z fallbackiem do previewMarketPairs").arg(root.activePair); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: previewState.terminalPairCandidates()
                    delegate: Components.IconButton {
                        required property string modelData
                        designSystem: root.designSystem
                        text: modelData
                        backgroundColor: root.activePair === modelData ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                        foregroundColor: root.activePair === modelData ? designSystem.color("surface") : designSystem.color("textPrimary")
                        onClicked: previewState.setTerminalPair(modelData)
                    }
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            objectName: "paperTerminalResponsiveCockpitGrid"
            columns: root.cockpitColumns
            rowSpacing: 14
            columnSpacing: 14

            Components.PreviewCard {
                objectName: "paperTerminalOrderForm"
                designSystem: root.designSystem
                title: qsTr("Order Form")
                description: qsTr("BUY / SELL oraz LIMIT / MARKET to aktywne lokalne przełączniki. Submit dopisuje tylko lokalny paper row: no real order / paper simulation only.")
                Layout.fillWidth: true
                Layout.preferredWidth: root.orderFormPreferredWidth
                Layout.minimumWidth: 280
                Layout.alignment: Qt.AlignTop

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: ["BUY", "SELL"]
                        delegate: Components.IconButton {
                            required property string modelData
                            designSystem: root.designSystem
                            text: modelData
                            Layout.fillWidth: true
                            backgroundColor: previewState.terminalSide === modelData ? (modelData === "BUY" ? root.buyTint : root.sellTint) : designSystem.color("surfaceMuted")
                            foregroundColor: modelData === "BUY" ? root.buyText : root.sellText
                            onClicked: previewState.setTerminalSide(modelData)
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: ["LIMIT", "MARKET"]
                        delegate: Components.IconButton {
                            required property string modelData
                            designSystem: root.designSystem
                            text: modelData
                            Layout.fillWidth: true
                            backgroundColor: previewState.terminalOrderType === modelData ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                            foregroundColor: previewState.terminalOrderType === modelData ? designSystem.color("surface") : designSystem.color("textPrimary")
                            onClicked: previewState.setTerminalOrderType(modelData)
                        }
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Label { text: qsTr("selected pair"); color: designSystem.color("textSecondary") }
                    Label { text: root.activePair; color: designSystem.color("textPrimary"); font.bold: true }
                    Label { text: qsTr("lowest ask"); color: designSystem.color("textSecondary") }
                    Label { text: previewState.mockOrderBookAsks[previewState.mockOrderBookAsks.length - 1].price; color: root.sellText }
                    Label { text: qsTr("highest bid"); color: designSystem.color("textSecondary") }
                    Label { text: previewState.mockOrderBookBids[0].price; color: root.buyText }
                    Label { text: qsTr("last price"); color: designSystem.color("textSecondary") }
                    Label { text: previewState.terminalPrice; color: designSystem.color("textPrimary") }
                }

                Label { text: qsTr("price input"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                Components.StyledTextField { designSystem: root.designSystem; text: previewState.terminalPrice; placeholderText: qsTr("Price"); Layout.fillWidth: true; onTextEdited: previewState.setTerminalPrice(text) }
                Label { text: qsTr("amount input"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                Components.StyledTextField { designSystem: root.designSystem; text: previewState.terminalAmount; placeholderText: qsTr("Amount"); Layout.fillWidth: true; onTextEdited: previewState.setTerminalAmount(text) }
                Label { text: qsTr("total input"); color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                Components.StyledTextField { designSystem: root.designSystem; text: previewState.terminalTotal; placeholderText: qsTr("Total"); Layout.fillWidth: true; onTextEdited: previewState.terminalTotal = text }

                Flow {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater { model: [10, 25, 50, 75, 100]; delegate: Components.IconButton { required property int modelData; designSystem: root.designSystem; text: modelData + "%"; subtle: true; onClicked: previewState.applyTerminalPercent(modelData) } }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Label { text: qsTr("available balance preview"); color: designSystem.color("textSecondary") }
                    Label { text: root.availableBalance; color: designSystem.color("textPrimary"); font.bold: true; wrapMode: Text.WordWrap }
                    Label { text: qsTr("fee estimate preview"); color: designSystem.color("textSecondary") }
                    Label { text: root.feeEstimate; color: designSystem.color("textPrimary") }
                    Label { text: qsTr("order value preview"); color: designSystem.color("textSecondary") }
                    Label { text: isNaN(root.orderValue) ? previewState.terminalTotal : root.orderValue.toFixed(2) + " USDT"; color: designSystem.color("textPrimary") }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: ["TP preview " + previewState.terminalTakeProfit, "SL preview " + previewState.terminalStopLoss, "post-only local", "reduce-only local", "time-in-force GTC"]
                        delegate: Rectangle {
                            required property string modelData
                            radius: 999
                            color: designSystem.color("surfaceMuted")
                            border.color: designSystem.color("border")
                            implicitWidth: previewChipText.implicitWidth + 18
                            implicitHeight: 28
                            Label { id: previewChipText; anchors.centerIn: parent; text: modelData; color: designSystem.color("textSecondary"); font.pixelSize: 11 }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 12
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    implicitHeight: safetyLabel.implicitHeight + 18
                    Label { id: safetyLabel; anchors.fill: parent; anchors.margins: 9; text: qsTr("Safety: no real order / paper simulation only. Live trading disabled and order submission disabled."); color: designSystem.color("textPrimary"); font.bold: true; wrapMode: Text.WordWrap }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Label { text: qsTr("auto confirm orders (local preview-only)"); color: designSystem.color("textPrimary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    Components.StyledSwitch { designSystem: root.designSystem; checked: previewState.terminalAutoConfirm; onToggled: previewState.terminalAutoConfirm = checked }
                }
                Components.IconButton {
                    objectName: "paperTerminalSimulateOrderButton"
                    designSystem: root.designSystem
                    text: previewState.terminalSide === "BUY" ? qsTr("Simulate buy order") : qsTr("Simulate sell order")
                    helpText: previewState.tooltipText("Simulate buy/sell order")
                    Layout.fillWidth: true
                    backgroundColor: previewState.terminalSide === "BUY" ? root.buyTint : root.sellTint
                    foregroundColor: previewState.terminalSide === "BUY" ? root.buyText : root.sellText
                    onClicked: previewState.simulateTerminalOrder()
                }
            }

            Components.PreviewCard {
                objectName: "paperTerminalChartArea"
                designSystem: root.designSystem
                title: qsTr("Chart area")
                description: qsTr("Lokalny wykres preview, no network/API call. Aktywny timeframe jest stanem lokalnym: %1; mock OHLC / last price / volume dla %2.").arg(previewState.terminalTimeframe).arg(root.activePair)
                Layout.fillWidth: true
                Layout.preferredWidth: root.chartPreferredWidth
                Layout.minimumWidth: 360
                Layout.alignment: Qt.AlignTop

                RowLayout {
                    Layout.fillWidth: true
                    Label { text: root.activePair + " • " + previewState.terminalTimeframe + " • Paper Exchange Preview"; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    Repeater { model: ["Indicators", "Templates", "Save", "Settings"]; delegate: Components.IconButton { required property string modelData; designSystem: root.designSystem; text: modelData; subtle: true } }
                }
                Flow {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                        delegate: Components.IconButton {
                            required property string modelData
                            designSystem: root.designSystem
                            text: modelData
                            backgroundColor: previewState.terminalTimeframe === modelData ? designSystem.color("accent") : designSystem.color("surfaceMuted")
                            foregroundColor: previewState.terminalTimeframe === modelData ? designSystem.color("surface") : designSystem.color("textPrimary")
                            onClicked: previewState.setTerminalTimeframe(modelData)
                        }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: width > 720 ? 4 : 2
                    rowSpacing: 8
                    columnSpacing: 8
                    Repeater {
                        model: [
                            { label: "mock OHLC", value: "O 68,180 • H 68,302 • L 68,178 • C 68,240" },
                            { label: "last price", value: previewState.terminalPrice },
                            { label: "volume", value: "1,284.42 BTC paper" },
                            { label: "chart source", value: "local preview chart" }
                        ]
                        delegate: Rectangle {
                            required property var modelData
                            Layout.fillWidth: true
                            radius: 12
                            color: designSystem.color("surfaceMuted")
                            border.color: designSystem.color("border")
                            implicitHeight: 54
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 8
                                Label {
                                    text: modelData.label
                                    color: designSystem.color("textSecondary")
                                    font.pixelSize: 11
                                }
                                Label {
                                    text: modelData.value
                                    color: designSystem.color("textPrimary")
                                    font.bold: true
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 360
                    radius: 14
                    color: designSystem.color("surfaceMuted")
                    border.color: designSystem.color("border")
                    clip: true
                    Repeater { model: 8; delegate: Rectangle { x: 20; y: 42 + index * 34; width: parent.width - 40; height: 1; color: Qt.rgba(1, 1, 1, 0.06) } }
                    Repeater { model: 11; delegate: Rectangle { x: 36 + index * Math.max(44, (parent.width - 90) / 10); y: 32; width: 1; height: parent.height - 70; color: Qt.rgba(1, 1, 1, 0.035) } }
                    Repeater {
                        model: [
                            { x: 36, open: 145, close: 104, high: 78, low: 174, vol: 44 }, { x: 78, open: 112, close: 168, high: 94, low: 194, vol: 72 },
                            { x: 120, open: 164, close: 122, high: 96, low: 184, vol: 58 }, { x: 162, open: 128, close: 88, high: 64, low: 148, vol: 38 },
                            { x: 204, open: 92, close: 154, high: 82, low: 178, vol: 82 }, { x: 246, open: 158, close: 128, high: 106, low: 188, vol: 66 },
                            { x: 288, open: 132, close: 86, high: 70, low: 154, vol: 48 }, { x: 330, open: 90, close: 142, high: 78, low: 168, vol: 74 },
                            { x: 372, open: 146, close: 98, high: 76, low: 166, vol: 62 }, { x: 414, open: 102, close: 72, high: 56, low: 132, vol: 36 },
                            { x: 456, open: 76, close: 128, high: 66, low: 158, vol: 70 }, { x: 498, open: 132, close: 92, high: 74, low: 152, vol: 54 }
                        ]
                        delegate: Item {
                            required property var modelData
                            x: Math.min(modelData.x, parent.width - 46)
                            y: 26
                            width: 24
                            height: 300
                            Rectangle { width: 2; height: modelData.low - modelData.high; x: 11; y: modelData.high; color: modelData.close < modelData.open ? root.buyText : root.sellText }
                            Rectangle { width: 18; height: Math.max(12, Math.abs(modelData.close - modelData.open)); x: 3; y: Math.min(modelData.open, modelData.close); radius: 3; color: modelData.close < modelData.open ? root.buyTint : root.sellTint; border.color: modelData.close < modelData.open ? root.buyText : root.sellText }
                            Rectangle { width: 18; height: modelData.vol; x: 3; y: 310 - height; radius: 2; color: Qt.rgba(0.35, 0.77, 1.0, 0.24) }
                        }
                    }
                    Canvas {
                        anchors.fill: parent
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            ctx.strokeStyle = "rgba(85,199,255,0.62)"
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            ctx.moveTo(28, 200)
                            ctx.bezierCurveTo(width * 0.18, 132, width * 0.28, 180, width * 0.38, 112)
                            ctx.bezierCurveTo(width * 0.52, 58, width * 0.62, 146, width * 0.74, 92)
                            ctx.bezierCurveTo(width * 0.84, 46, width * 0.92, 106, width - 28, 68)
                            ctx.stroke()
                        }
                    }
                    Label { anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 12; text: qsTr("mock candles / trend line / volume • local preview chart • no network/API call"); color: designSystem.color("textSecondary") }
                }
            }

            Components.PreviewCard {
                objectName: "paperTerminalOrderBook"
                designSystem: root.designSystem
                title: qsTr("Order Book")
                description: qsTr("10 ask levels and 10 bid levels. Price clicks copy to the local Order Form; action chips mutate only local UI state.")
                Layout.fillWidth: true
                Layout.preferredWidth: root.orderBookPreferredWidth
                Layout.minimumWidth: 280
                Layout.alignment: Qt.AlignTop

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: root.activePair
                        color: designSystem.color("textPrimary")
                        font.bold: true
                        Layout.fillWidth: true
                    }
                    Label {
                        text: qsTr("last price row: %1").arg(previewState.terminalPrice)
                        color: designSystem.color("textSecondary")
                    }
                }

                ScrollView {
                    objectName: "paperTerminalOrderBookScroll"
                    Layout.fillWidth: true
                    Layout.preferredHeight: root.orderBookScrollHeight
                    contentWidth: availableWidth
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                    ScrollBar.vertical.policy: ScrollBar.AsNeeded

                    ColumnLayout {
                        width: parent.availableWidth
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Label {
                                text: qsTr("Price")
                                color: designSystem.color("textSecondary")
                                Layout.preferredWidth: 82
                            }
                            Label {
                                text: qsTr("Amount")
                                color: designSystem.color("textSecondary")
                                Layout.preferredWidth: 76
                            }
                            Label {
                                text: qsTr("Total")
                                color: designSystem.color("textSecondary")
                                Layout.preferredWidth: 82
                            }
                            Label {
                                text: qsTr("Action")
                                color: designSystem.color("textSecondary")
                                Layout.fillWidth: true
                            }
                        }
                        Repeater { model: previewState.mockOrderBookAsks; delegate: orderBookRowDelegate }
                        Rectangle { Layout.fillWidth: true; radius: 10; color: designSystem.color("surfaceMuted"); border.color: designSystem.color("border"); implicitHeight: 38; Label { anchors.centerIn: parent; text: qsTr("Spread row 8.90 USDT • Last price row %1 • Paper Preview").arg(previewState.terminalPrice); color: designSystem.color("textPrimary"); font.bold: true } }
                        Repeater { model: previewState.mockOrderBookBids; delegate: orderBookRowDelegate }
                    }
                }
            }
        }

        Components.PreviewCard {
            objectName: "paperTerminalBottomTabs"
            designSystem: root.designSystem
            title: qsTr("Positions / Orders / History / Reserved / Strategy / Log / Messages")
            description: qsTr("Local tab strip with non-empty paper content and active style. Positions, Orders, History, Reserved, Strategy, Log, Messages never call external services.")
            Layout.fillWidth: true

            Flow {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: ["Positions", "Orders", "History", "Reserved", "Strategy", "Log", "Messages"]
                    delegate: Components.IconButton { required property string modelData; designSystem: root.designSystem; text: modelData; backgroundColor: previewState.terminalSelectedBottomTab === modelData ? designSystem.color("accent") : designSystem.color("surfaceMuted"); foregroundColor: previewState.terminalSelectedBottomTab === modelData ? designSystem.color("surface") : designSystem.color("textPrimary"); onClicked: previewState.selectTerminalBottomTab(modelData) }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: terminalBottomRows()
                    delegate: Rectangle {
                        required property var modelData
                        Layout.fillWidth: true
                        radius: 12
                        color: designSystem.color("surfaceMuted")
                        border.color: designSystem.color("border")
                        implicitHeight: bottomRow.implicitHeight + 18
                        RowLayout {
                            id: bottomRow
                            anchors.fill: parent
                            anchors.margins: 9
                            spacing: 10
                            Label { text: modelData.c1; color: designSystem.color("textPrimary"); font.bold: true; Layout.preferredWidth: 160; elide: Text.ElideRight }
                            Label { text: modelData.c2; color: designSystem.color("textSecondary"); Layout.preferredWidth: 190; elide: Text.ElideRight }
                            Label { text: modelData.c3; color: designSystem.color("textSecondary"); Layout.preferredWidth: 190; elide: Text.ElideRight }
                            Label { text: modelData.c4; color: modelData.c4 && modelData.c4.indexOf("+") >= 0 ? root.buyText : designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: orderBookRowDelegate
        Rectangle {
            required property var modelData
            Layout.fillWidth: true
            radius: 10
            color: modelData.action.indexOf("ask") >= 0 ? root.sellTint : root.buyTint
            border.color: designSystem.color("border")
            implicitHeight: 34
            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                Label { text: modelData.price; color: modelData.action.indexOf("ask") >= 0 ? root.sellText : root.buyText; font.bold: true; Layout.preferredWidth: 82 }
                Label { text: modelData.amount; color: designSystem.color("textPrimary"); Layout.preferredWidth: 76 }
                Label { text: modelData.total; color: designSystem.color("textSecondary"); Layout.preferredWidth: 82 }
                Components.IconButton { designSystem: root.designSystem; text: modelData.action; subtle: true; Layout.fillWidth: true; onClicked: previewState.useOrderBookPrice(modelData.price) }
            }
            MouseArea { anchors.fill: parent; acceptedButtons: Qt.LeftButton; onClicked: previewState.useOrderBookPrice(modelData.price) }
        }
    }

    function terminalBottomRows() {
        if (previewState.terminalSelectedBottomTab === "Positions")
            return previewState.paperOpenPositions.map(function(row) { return { c1: row.pair, c2: row.side + " • " + row.size, c3: row.entry ? "entry " + row.entry : "local paper position", c4: (row.pnl || "0.00") + " • " + (row.status || row.label || "paper preview") } })
        if (previewState.terminalSelectedBottomTab === "Orders")
            return previewState.paperOrderRows.map(function(row) { return { c1: (row.time || row.timestamp) + " • " + row.pair, c2: row.side ? row.side + " " + row.type : row.action, c3: row.price ? row.price + " • " + row.amount : "confidence " + row.confidence, c4: row.status + " • " + row.reason } })
        if (previewState.terminalSelectedBottomTab === "History")
            return previewState.paperClosedTrades.map(function(row) { return { c1: (row.time || row.timestamp || "paper history") + " • " + row.pair, c2: row.side, c3: row.price ? row.price + " • " + row.amount : "closed local paper trade", c4: row.result || row.pnl || row.label } })
        if (previewState.terminalSelectedBottomTab === "Reserved")
            return previewState.mockTerminalReservedBalances.map(function(row) { return { c1: row.asset, c2: "reserved " + row.reserved, c3: row.reason, c4: row.status } })
        if (previewState.terminalSelectedBottomTab === "Strategy")
            return [
                { c1: "Active strategies", c2: previewState.activeStrategies.join(", "), c3: "routing governor preview", c4: previewState.lastGovernorDecision },
                { c1: "Execution guard", c2: "Paper route only", c3: "Runtime loop not started", c4: "order submission disabled" }
            ]
        if (previewState.terminalSelectedBottomTab === "Log")
            return previewState.terminalLogRows.map(function(row) { return { c1: row.time, c2: "local UI event log", c3: "paper terminal", c4: row.message } })
        if (previewState.terminalSelectedBottomTab === "Messages")
            return [
                { c1: "Safety/system messages", c2: "Live trading disabled", c3: "Exchange I/O disabled", c4: "Order submission disabled • API keys not required • no real orders" },
                { c1: "Preview notices", c2: "Paper Preview only", c3: "Runtime loop not started", c4: "All controls are local-only mock UI state" }
            ]
        return [
            { c1: "Reserved", c2: "Preview capacity", c3: "No external integration", c4: "Reserved tab keeps local paper workspace space" },
            { c1: "Safe boundary", c2: "No real orders", c3: "No exchange I/O", c4: "No secrets, no key material, no runtime loop" }
        ]
    }
}
