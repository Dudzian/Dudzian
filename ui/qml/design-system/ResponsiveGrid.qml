import QtQuick
import QtQuick.Layouts

GridLayout {
    id: root
    property int minColumnWidth: 360
    property int maxColumns: 3
    property int minColumns: 1

    readonly property int computedColumns: {
        const effectiveWidth = width > 0 ? width : implicitWidth
        const available = Math.max(1, Math.floor(effectiveWidth / minColumnWidth))
        return Math.max(minColumns, Math.min(maxColumns, available))
    }

    columns: computedColumns
    rowSpacing: 16
    columnSpacing: 16
}
