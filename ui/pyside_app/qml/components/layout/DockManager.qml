import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "./" as LayoutParts

Item {
    id: dockRoot
    property var layoutController
    property var panelRegistry: ({})
    property var designSystem
    property real columnSpacing: 16
    property real panelSpacing: 12
    property var layoutState: layoutController ? layoutController.layout : []
    property int columnCount: layoutController ? Math.max(1, layoutController.columnCount) : 1

    function panelComponent(panelId) {
        if (panelRegistry && panelRegistry[panelId]) {
            return panelRegistry[panelId].component
        }
        return null
    }

    function panelTitle(panelId) {
        if (panelRegistry && panelRegistry[panelId]) {
            return panelRegistry[panelId].title
        }
        return panelId
    }

    function panelIcon(panelId) {
        if (panelRegistry && panelRegistry[panelId]) {
            return panelRegistry[panelId].icon || ""
        }
        return ""
    }

    function hidePanel(panelId) {
        if (layoutController) {
            layoutController.setPanelVisibility(panelId, false)
        }
    }

    function showPanel(panelId) {
        if (layoutController) {
            layoutController.setPanelVisibility(panelId, true)
        }
    }

    function columnPanels(columnIndex) {
        var result = []
        if (!layoutState) {
            return result
        }
        for (var i = 0; i < layoutState.length; ++i) {
            var entry = layoutState[i]
            if (entry.column === columnIndex && entry.visible !== false && panelComponent(entry.panelId)) {
                result.push(entry)
            }
        }
        result.sort(function(a, b) { return a.order - b.order })
        return result
    }

    function handleDrop(columnIndex, listView, dropEvent, columnHost) {
        if (!layoutController || !dropEvent || !dropEvent.source) {
            return
        }
        var panelId = dropEvent.source.panelId
        if (!panelId) {
            return
        }
        var localPoint = listView.mapFromItem(columnHost, dropEvent.x, dropEvent.y)
        var index = listView.indexAt(localPoint.x, localPoint.y)
        if (index < 0) {
            index = listView.count
        }
        layoutController.updatePanelPosition(panelId, columnIndex, index)
    }

    RowLayout {
        anchors.fill: parent
        spacing: dockRoot.columnSpacing

        Repeater {
            id: columnRepeater
            model: dockRoot.columnCount
            delegate: Item {
                id: columnHost
                Layout.fillWidth: true
                Layout.fillHeight: true
                property int columnIndex: index

                ListView {
                    id: columnView
                    anchors.fill: parent
                    spacing: dockRoot.panelSpacing
                    interactive: false
                    model: dockRoot.columnPanels(columnHost.columnIndex)
                    delegate: LayoutParts.PanelPlaceholder {
                        required property var modelData
                        width: ListView.view ? ListView.view.width : parent.width
                        height: implicitHeight
                        dockManager: dockRoot
                        designSystem: dockRoot.designSystem
                        panelId: modelData.panelId
                        panelOrder: modelData.order
                        columnIndex: modelData.column
                        title: dockRoot.panelTitle(modelData.panelId)
                        iconName: dockRoot.panelIcon(modelData.panelId)
                        contentComponent: dockRoot.panelComponent(modelData.panelId)
                    }
                }

                DropArea {
                    anchors.fill: parent
                    onDropped: function(event) {
                        dockRoot.handleDrop(columnHost.columnIndex, columnView, event, columnHost)
                    }
                }
            }
        }
    }
}
