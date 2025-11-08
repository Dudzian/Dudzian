import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import ".." as Design

Item {
    id: root
    implicitWidth: 480
    implicitHeight: 320

    property var rows: [] // [{ label: "BTC", buckets: [{ label: "Mon", value: 1.2 }, ...] }]
    property var bucketLabels: []
    property real maxAbsValue: 1.0

    onRowsChanged: recomputeDerived()

    function recomputeDerived() {
        var maxValue = 0.01
        var headers = []
        for (var i = 0; i < rows.length; ++i) {
            var row = rows[i]
            var buckets = row && row.buckets ? row.buckets : []
            for (var j = 0; j < buckets.length; ++j) {
                var bucket = buckets[j]
                var value = Number(bucket.value)
                if (!isNaN(value))
                    maxValue = Math.max(maxValue, Math.abs(value))
                if (bucket && bucket.label && headers.indexOf(bucket.label) === -1)
                    headers.push(bucket.label)
            }
        }
        maxAbsValue = maxValue
        bucketLabels = headers
    }

    Flickable {
        anchors.fill: parent
        contentWidth: Math.max(root.width, grid.implicitWidth)
        contentHeight: Math.max(root.height, grid.implicitHeight)
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}

        ColumnLayout {
            id: grid
            width: Math.max(parent.width, implicitWidth)
            spacing: 6

            RowLayout {
                spacing: 6
                Layout.fillWidth: true
                visible: root.bucketLabels.length > 0

                Item {
                    Layout.preferredWidth: 120
                    Layout.preferredHeight: 28
                }

                Repeater {
                    model: root.bucketLabels
                    delegate: Label {
                        text: modelData || ""
                        color: Design.Palette.textSecondary
                        font.pixelSize: Design.Typography.caption
                        horizontalAlignment: Text.AlignHCenter
                        Layout.preferredWidth: 72
                    }
                }
            }

            Repeater {
                model: root.rows
                delegate: RowLayout {
                    spacing: 6
                    Layout.fillWidth: true

                    Label {
                        text: modelData && modelData.label ? modelData.label : ""
                        color: Design.Palette.textSecondary
                        font.pixelSize: Design.Typography.body
                        Layout.preferredWidth: 120
                        elide: Text.ElideRight
                    }

                    Repeater {
                        model: modelData && modelData.buckets ? modelData.buckets : []
                        delegate: Rectangle {
                            readonly property real cellValue: Number(modelData.value)

                            Layout.preferredWidth: 72
                            Layout.preferredHeight: 48
                            radius: 6
                            border.color: Design.Palette.border
                            border.width: 1
                            color: heatColor(cellValue)

                            ToolTip.visible: hoverHandler.hovered
                            ToolTip.text: (modelData && modelData.label ? modelData.label : "")
                                           + "\n" + (isNaN(cellValue) ? "0.00" : cellValue.toFixed(2))

                            HoverHandler {
                                id: hoverHandler
                            }

                            Label {
                                anchors.centerIn: parent
                                text: isNaN(cellValue) ? "--" : cellValue.toFixed(2)
                                color: Design.Palette.textPrimary
                                font.pixelSize: Design.Typography.body
                                font.bold: true
                            }
                        }
                    }
                }
            }
        }
    }

    function heatColor(value) {
        if (isNaN(value))
            return Qt.rgba(0, 0, 0, 0)
        var normalized = Math.min(1.0, Math.abs(value) / Math.max(0.01, maxAbsValue))
        var positive = [0.16, 0.65, 0.48]
        var negative = [0.96, 0.35, 0.41]
        var alpha = 0.25 + 0.5 * normalized
        if (value >= 0)
            return Qt.rgba(positive[0], positive[1], positive[2], alpha)
        return Qt.rgba(negative[0], negative[1], negative[2], alpha)
    }
}
