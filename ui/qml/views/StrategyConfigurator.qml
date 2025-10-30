import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components/workbench" as Workbench

Item {
    id: root
    objectName: "strategyConfiguratorRoot"

    property alias viewModel: viewModel
    property var appController: null
    property var strategyController: null
    property var workbenchController: null
    property var riskModel: null
    property var licenseController: null
    property var strategyNames: []
    property var filteredStrategies: []
    property var currentStrategy: ({})
    property string searchText: ""

    implicitWidth: 960
    implicitHeight: 540

    Workbench.StrategyWorkbenchViewModel {
        id: viewModel
        objectName: "strategyConfiguratorViewModel"
        appController: root.appController ? root.appController : (typeof appController !== "undefined" ? appController : null)
        strategyController: root.strategyController ? root.strategyController : (typeof strategyController !== "undefined" ? strategyController : null)
        workbenchController: root.workbenchController ? root.workbenchController : (typeof workbenchController !== "undefined" ? workbenchController : null)
        riskModel: root.riskModel ? root.riskModel : (typeof riskModel !== "undefined" ? riskModel : null)
        licenseController: root.licenseController ? root.licenseController : (typeof licenseController !== "undefined" ? licenseController : null)
    }

    function rebuildFiltered() {
        var definitions = viewModel.catalogDefinitions || []
        var needle = searchText ? searchText.toLowerCase() : ""
        var results = []
        for (var i = 0; i < definitions.length; ++i) {
            var entry = definitions[i]
            if (!entry)
                continue
            if (needle.length > 0) {
                var haystack = (entry.name || "") + " " + (entry.engine || "")
                var tags = entry.tags ? entry.tags.join(" ") : ""
                haystack += " " + tags
                if (haystack.toLowerCase().indexOf(needle) === -1)
                    continue
            }
            results.push(entry)
        }
        filteredStrategies = results
        var names = []
        for (var j = 0; j < results.length; ++j) {
            var def = results[j]
            names.push(def.name || def.engine)
        }
        strategyNames = names
        if (strategyListView.count > 0 && strategyListView.currentIndex < 0)
            strategyListView.currentIndex = 0
        if (strategyListView.currentIndex >= 0 && strategyListView.currentIndex < filteredStrategies.length)
            currentStrategy = filteredStrategies[strategyListView.currentIndex]
        else
            currentStrategy = {}
    }

    function syncFromViewModel() {
        rebuildFiltered()
    }

    Component.onCompleted: {
        if (workbenchController && workbenchController.refreshCatalog)
            workbenchController.refreshCatalog()
        viewModel.syncCatalog()
        rebuildFiltered()
    }

    Connections {
        target: workbenchController
        function onCatalogChanged() {
            viewModel.syncCatalog()
            rebuildFiltered()
        }
    }

    Connections {
        target: viewModel
        function onCatalogDefinitionsChanged() { rebuildFiltered() }
    }

    onSearchTextChanged: rebuildFiltered()

    ColumnLayout {
        anchors.fill: parent
        spacing: 16
        padding: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Katalog strategii")
                font.bold: true
                font.pointSize: 16
            }

            TextField {
                id: searchField
                objectName: "strategySearchField"
                Layout.fillWidth: true
                placeholderText: qsTr("Filtruj po nazwie, silniku lub tagach…")
                text: searchText
                onTextChanged: root.searchText = text
            }

            Button {
                text: qsTr("Odśwież")
                enabled: !workbenchController || !workbenchController.busy
                onClicked: {
                    if (workbenchController && workbenchController.refreshCatalog)
                        workbenchController.refreshCatalog()
                    else
                        rebuildFiltered()
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            Frame {
                Layout.fillHeight: true
                Layout.preferredWidth: parent.width * 0.55
                background: Rectangle { color: Qt.darker(palette.base, 1.05); radius: 8 }

                ListView {
                    id: strategyListView
                    objectName: "strategyConfiguratorList"
                    anchors.fill: parent
                    anchors.margins: 12
                    model: filteredStrategies
                    clip: true
                    spacing: 8
                    delegate: Frame {
                        width: ListView.view.width
                        padding: 12
                        background: Rectangle {
                            radius: 6
                            color: ListView.isCurrentItem ? palette.highlight : Qt.darker(palette.window, 1.02)
                            border.color: Qt.darker(color, 1.1)
                            border.width: 1
                            opacity: ListView.isCurrentItem ? 0.95 : 0.9
                        }

                        property var meta: modelData

                        ColumnLayout {
                            spacing: 4
                            width: parent.width

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Label {
                                    text: meta.name || meta.engine
                                    font.bold: true
                                    color: ListView.isCurrentItem ? palette.highlightedText : palette.text
                                }
                                Label {
                                    text: meta.engine
                                    color: ListView.isCurrentItem ? palette.highlightedText : palette.mid
                                }
                            }

                            Label {
                                text: meta.description || ""
                                wrapMode: Text.WordWrap
                                color: ListView.isCurrentItem ? palette.highlightedText : palette.text
                            }

                            Flow {
                                width: parent.width
                                spacing: 6
                                Repeater {
                                    model: meta.tags || []
                                    delegate: Rectangle {
                                        radius: 4
                                        height: 20
                                        color: ListView.isCurrentItem ? Qt.lighter(palette.highlight, 1.2) : Qt.darker(palette.window, 1.08)
                                        border.color: Qt.darker(color, 1.2)
                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData
                                            font.pixelSize: 12
                                            color: ListView.isCurrentItem ? palette.highlightedText : palette.text
                                        }
                                    }
                                }
                            }

                            GridLayout {
                                columns: 2
                                Layout.fillWidth: true
                                columnSpacing: 8
                                rowSpacing: 2

                                Label { text: qsTr("Licencja:"); color: ListView.isCurrentItem ? palette.highlightedText : palette.mid }
                                Label { text: meta.license_tier || "–"; color: ListView.isCurrentItem ? palette.highlightedText : palette.text }

                                Label { text: qsTr("Ryzyko:"); color: ListView.isCurrentItem ? palette.highlightedText : palette.mid }
                                Label {
                                    text: meta.risk_classes ? meta.risk_classes.join(", ") : "–"
                                    color: ListView.isCurrentItem ? palette.highlightedText : palette.text
                                }

                                Label { text: qsTr("Dane:"); color: ListView.isCurrentItem ? palette.highlightedText : palette.mid }
                                Label {
                                    text: meta.required_data ? meta.required_data.join(", ") : "–"
                                    wrapMode: Text.WordWrap
                                    color: ListView.isCurrentItem ? palette.highlightedText : palette.text
                                }
                            }
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: strategyListView.currentIndex = index
                            hoverEnabled: true
                        }
                    }

                    onCurrentIndexChanged: {
                        if (currentIndex >= 0 && currentIndex < filteredStrategies.length)
                            currentStrategy = filteredStrategies[currentIndex]
                        else
                            currentStrategy = {}
                    }
                }
            }

            Frame {
                Layout.fillHeight: true
                Layout.fillWidth: true
                background: Rectangle { color: Qt.darker(palette.window, 1.04); radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8

                    Label {
                        text: currentStrategy.name || currentStrategy.engine || qsTr("Wybierz strategię")
                        font.bold: true
                        font.pointSize: 14
                    }

                    Label {
                        text: currentStrategy.description || qsTr("Brak opisu")
                        wrapMode: Text.WordWrap
                        color: palette.text
                    }

                    Label {
                        text: qsTr("Silnik: %1").arg(currentStrategy.engine || "–")
                        color: palette.mid
                    }

                    Label {
                        text: qsTr("Parametry:")
                        font.bold: true
                    }

                    TextArea {
                        objectName: "strategyParameterView"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        readOnly: true
                        wrapMode: Text.WordWrap
                        text: currentStrategy.parameters ? JSON.stringify(currentStrategy.parameters, null, 2) : qsTr("Brak parametrów")
                    }
                }
            }
        }
    }
}
