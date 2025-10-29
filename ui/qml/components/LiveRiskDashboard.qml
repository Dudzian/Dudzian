import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Item {
    id: root
    property var riskModel
    property var riskHistoryModel
    property var capitalAllocationModel: null
    property var capitalAllocation: []
    property var capitalAllocationEntries: []
    property var dynamicPolicyMetadata: null

    implicitWidth: 720
    implicitHeight: 560

    ListModel { id: limitsModel }

    function formatPercent(value) {
        if (value === undefined || value === null)
            return "—"
        return Number(value * 100).toLocaleString(Qt.locale(), "f", 2) + "%"
    }

    function formatNumber(value, digits) {
        if (value === undefined || value === null)
            return "—"
        var precision = digits !== undefined ? digits : 2
        return Number(value).toLocaleString(Qt.locale(), "f", precision)
    }

    function updateLimits() {
        limitsModel.clear()
        var limits = null
        if (root.riskModel) {
            if (root.riskModel.limits !== undefined)
                limits = root.riskModel.limits
            else if (root.riskModel.getLimits)
                limits = root.riskModel.getLimits()
        }
        if (!limits)
            return
        function pushLimit(key, label, formatter) {
            if (limits[key] === undefined)
                return
            var value = limits[key]
            var formatted = formatter ? formatter(value) : formatNumber(value, 2)
            limitsModel.append({ label: label, value: formatted })
        }
        pushLimit("max_positions", qsTr("Liczba pozycji"), function(value) {
            return formatNumber(value, 0)
        })
        pushLimit("max_leverage", qsTr("Maksymalna dźwignia"), formatNumber)
        pushLimit("max_position_pct", qsTr("Limit ekspozycji"), formatPercent)
        pushLimit("daily_loss_limit", qsTr("Limit dzienny"), formatPercent)
        pushLimit("drawdown_limit", qsTr("Limit obsunięcia"), formatPercent)
        pushLimit("target_volatility", qsTr("Docelowa zmienność"), formatPercent)
        pushLimit("stop_loss_atr_multiple", qsTr("Stop loss (ATR)"), formatNumber)
    }

    function updateCapitalEntries() {
        if (root.capitalAllocationModel)
            return
        var source = root.capitalAllocation
        if (!source) {
            root.capitalAllocationEntries = []
            return
        }
        if (Array.isArray(source)) {
            root.capitalAllocationEntries = source
            return
        }
        var entries = []
        for (var key in source) {
            if (!source.hasOwnProperty(key))
                continue
            var value = source[key]
            if (value === null || value === undefined)
                continue
            if (typeof value === "object") {
                var payload = {}
                for (var prop in value) {
                    if (value.hasOwnProperty(prop))
                        payload[prop] = value[prop]
                }
                if (!payload.strategy && !payload.segment)
                    payload.strategy = key
                if (payload.weight === undefined)
                    payload.weight = Number(value.weight !== undefined ? value.weight : 0)
                entries.push(payload)
            } else {
                entries.push({ strategy: key, weight: Number(value) })
            }
        }
        root.capitalAllocationEntries = entries
    }

    onRiskModelChanged: updateLimits()
    onCapitalAllocationChanged: updateCapitalEntries()
    onCapitalAllocationModelChanged: updateCapitalEntries()
    Component.onCompleted: updateLimits()
    Component.onCompleted: updateCapitalEntries()

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        Frame {
            Layout.fillWidth: true
            visible: limitsModel.count > 0
            padding: 16
            background: Rectangle {
                radius: 8
                color: Qt.rgba(0.14, 0.16, 0.2, 0.9)
            }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                Label {
                    text: qsTr("Limity profilu ryzyka")
                    font.bold: true
                    font.pixelSize: 16
                }

                Repeater {
                    model: limitsModel
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Label {
                            text: model.label
                            color: Qt.rgba(0.85, 0.9, 0.95, 1)
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: model.value
                            font.bold: true
                        }
                    }
                }
            }
        }

        RiskMonitorPanel {
            Layout.fillWidth: true
            Layout.preferredHeight: 340
            model: root.riskModel
            historyModel: root.riskHistoryModel
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            padding: 16
            background: Rectangle {
                radius: 8
                color: Qt.rgba(0.14, 0.16, 0.2, 0.85)
            }

            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                Label {
                    text: qsTr("Alokacje kapitału")
                    font.bold: true
                    font.pixelSize: 16
                }

                Label {
                    visible: root.dynamicPolicyMetadata && root.dynamicPolicyMetadata.source
                    text: qsTr("Źródło: %1").arg(root.dynamicPolicyMetadata ? root.dynamicPolicyMetadata.source : "")
                    color: Qt.rgba(0.7, 0.75, 0.82, 1)
                }

                Label {
                    visible: root.dynamicPolicyMetadata && root.dynamicPolicyMetadata.applied_at
                    text: root.dynamicPolicyMetadata && root.dynamicPolicyMetadata.applied_at
                          ? qsTr("Zastosowano: %1").arg(root.dynamicPolicyMetadata.applied_at)
                          : ""
                    color: Qt.rgba(0.7, 0.75, 0.82, 1)
                }

                Label {
                    visible: !capitalView.hasModel
                    text: qsTr("Brak danych o bieżącej alokacji kapitału")
                    color: Qt.rgba(0.7, 0.75, 0.82, 1)
                }

                ListView {
                    id: capitalView
                    property bool hasModel: !!model && ((model.count !== undefined && model.count > 0) || (model.length !== undefined && model.length > 0))
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    ScrollBar.vertical: ScrollBar {}
                    model: root.capitalAllocationModel ? root.capitalAllocationModel : root.capitalAllocationEntries

                    delegate: ColumnLayout {
                        width: ListView.view ? ListView.view.width : parent.width
                        spacing: 6

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Label {
                                text: (modelData.segment || modelData.symbol || modelData.strategy || qsTr("Segment"))
                                font.bold: true
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                text: formatPercent(modelData.weight || 0)
                                font.bold: true
                            }
                        }

                        ProgressBar {
                            Layout.fillWidth: true
                            from: 0
                            to: 1
                            value: Math.min(1, Math.max(0, modelData.weight || 0))
                        }

                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 4

                            Label { text: qsTr("Cel") }
                            Label { text: formatPercent(modelData.targetWeight || 0) }

                            Label { text: qsTr("Odchylenie") }
                            Label {
                                text: formatPercent((modelData.deltaWeight !== undefined ? modelData.deltaWeight : ((modelData.weight || 0) - (modelData.targetWeight || 0))))
                            }

                            Label { text: qsTr("Notional") }
                            Label { text: formatNumber(modelData.notional || modelData.currentValue || 0, 0) }

                            Label { text: qsTr("Dźwignia") }
                            Label { text: formatNumber(modelData.leverage !== undefined ? modelData.leverage : 1.0, 2) }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Qt.rgba(1, 1, 1, 0.1)
                            visible: index < (ListView.view.count - 1)
                        }
                    }
                }
            }
        }
    }
}
