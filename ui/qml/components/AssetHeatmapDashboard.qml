import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Pane {
    id: root
    property var cells: []
    property string title: qsTr("Heatmapa aktywów")
    padding: 12

    background: Rectangle {
        color: Qt.rgba(0, 0, 0, 0.25)
        radius: 8
        border.color: Qt.rgba(1, 1, 1, 0.08)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            Label {
                text: root.title
                font.pixelSize: 16
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Label {
                text: root.totalValueText()
                color: Qt.rgba(0.8, 0.8, 0.8, 0.8)
            }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: flow.implicitWidth
            contentHeight: flow.implicitHeight
            clip: true

            Flow {
                id: flow
                width: Math.max(parent ? parent.width : 0, implicitWidth)
                spacing: 8

                Repeater {
                    model: root.cells || []
                    delegate: Rectangle {
                        required property var modelData
                        width: Math.max(140, Math.min(220, flow.width / 2))
                        height: 64
                        radius: 6
                        color: root.colorForValue(Number(modelData.value))
                        border.color: Qt.rgba(1, 1, 1, 0.06)
                        opacity: 0.95

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 4

                            Label {
                                text: modelData.label || modelData.asset || "?"
                                font.pixelSize: 14
                                font.bold: true
                                color: Qt.rgba(1, 1, 1, 0.9)
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }

                            Label {
                                text: root.formatValue(Number(modelData.value))
                                color: Qt.rgba(1, 1, 1, 0.8)
                                Layout.fillWidth: true
                            }
                        }

                        ToolTip.visible: hovered
                        ToolTip.delay: 400
                        ToolTip.text: root.contributionTooltip(modelData)

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            onEntered: parent.hovered = true
                            onExited: parent.hovered = false
                        }

                        property bool hovered: false
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: !cells || cells.length === 0
            text: qsTr("Brak danych ekspozycji dla bieżących filtrów")
            color: palette.mid
        }
    }

    property real maxAbsValue: 0
    property real totalValue: 0

    function refreshMetrics() {
        const data = cells || []
        let maxVal = 0
        let total = 0
        for (let i = 0; i < data.length; ++i) {
            const value = Number(data[i] && data[i].value)
            if (isNaN(value))
                continue
            if (Math.abs(value) > maxVal)
                maxVal = Math.abs(value)
            total += value
        }
        maxAbsValue = maxVal
        totalValue = total
    }

    function colorForValue(value) {
        if (!isFinite(value) || maxAbsValue <= 0)
            return Qt.rgba(0.25, 0.25, 0.3, 0.6)
        const ratio = Math.min(Math.abs(value) / maxAbsValue, 1.0)
        if (value >= 0)
            return Qt.rgba(0.2, 0.6, 0.4, 0.5 + 0.4 * ratio)
        return Qt.rgba(0.8, 0.3, 0.3, 0.5 + 0.4 * ratio)
    }

    function formatValue(value) {
        if (!isFinite(value))
            return qsTr("—")
        return qsTr("%1").arg(Number(value).toLocaleString(Qt.locale(), "f", 2))
    }

    function totalValueText() {
        if (!isFinite(totalValue))
            return qsTr("—")
        return qsTr("Suma: %1").arg(Number(totalValue).toLocaleString(Qt.locale(), "f", 2))
    }

    function contributionTooltip(cell) {
        const sources = cell && cell.sources ? cell.sources : []
        if (!sources || sources.length === 0)
            return qsTr("Źródło: %1").arg(cell && cell.source ? cell.source : qsTr("n/d"))
        let lines = [qsTr("Łącznie: %1").arg(formatValue(cell.value))]
        for (let i = 0; i < sources.length; ++i) {
            const entry = sources[i] || {}
            const sourceLabel = entry.source || qsTr("n/d")
            const category = entry.category ? qsTr(" (%1)").arg(entry.category) : ""
            lines.push(qsTr("%1%2: %3").arg(sourceLabel).arg(category).arg(formatValue(entry.value)))
        }
        return lines.join("\n")
    }

    onCellsChanged: refreshMetrics()
    Component.onCompleted: refreshMetrics()
}
