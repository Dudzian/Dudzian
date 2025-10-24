import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var viewsModel: typeof moduleViewsModel !== "undefined" ? moduleViewsModel : null
    property string selectedViewId: ""
    property string selectedViewName: ""
    property string selectedModuleId: ""
    property string selectedCategoryLabel: ""
    property url selectedSource: ""
    property string selectedCategory: ""
    property string searchQuery: ""
    property var categoryOptions: []
    property bool suppressCategorySync: false
    property bool suppressSearchSync: false
    readonly property var moduleDirectories: typeof appController !== "undefined" ? appController.uiModuleDirectories : []
    property bool reloadingModules: false
    property string reloadStatusMessage: ""
    property bool reloadStatusError: false
    property var reloadReport: ({})

    signal viewActivated(string viewId)

    ListModel { id: metadataModel }

    Timer {
        id: reloadStatusTimer
        interval: 5000
        repeat: false
        onTriggered: root.reloadStatusMessage = ""
    }

    function metadataValueToString(value) {
        if (value === null || value === undefined)
            return "";
        if (typeof value === "object")
            return JSON.stringify(value, null, 2);
        return String(value);
    }

    function refreshCategories() {
        var options = [ { label: qsTr("Wszystkie kategorie"), value: "" } ];
        if (viewsModel && typeof viewsModel.categories === "function") {
            var values = viewsModel.categories();
            for (var i = 0; i < values.length; ++i) {
                options.push({ label: values[i], value: values[i] });
            }
        }
        categoryOptions = options;
        var found = false;
        for (var j = 0; j < categoryOptions.length; ++j) {
            if (categoryOptions[j].value === selectedCategory) {
                found = true;
                break;
            }
        }
        if (!found)
            selectedCategory = "";

        if (typeof categoryCombo !== "undefined") {
            var targetIndex = 0;
            for (var k = 0; k < categoryOptions.length; ++k) {
                if (categoryOptions[k].value === selectedCategory) {
                    targetIndex = k;
                    break;
                }
            }
            if (categoryCombo.currentIndex !== targetIndex)
                categoryCombo.currentIndex = targetIndex;
        }
    }

    function syncCategoryFromModel() {
        if (!viewsModel)
            return;
        suppressCategorySync = true;
        selectedCategory = viewsModel.categoryFilter || "";
        suppressCategorySync = false;
    }

    function syncSearchFromModel() {
        if (!viewsModel)
            return;
        suppressSearchSync = true;
        searchQuery = viewsModel.searchFilter || "";
        suppressSearchSync = false;
    }

    function descriptorMatchesSelection(descriptor) {
        return descriptor && descriptor.id && descriptor.id === selectedViewId;
    }

    function applyDescriptor(descriptor, index) {
        if (!descriptor || !descriptor.id) {
            selectedViewId = "";
            selectedViewName = "";
            selectedModuleId = "";
            selectedCategoryLabel = "";
            selectedSource = "";
            metadataModel.clear();
            viewList.currentIndex = index !== undefined ? index : -1;
            return;
        }

        selectedViewId = descriptor.id;
        selectedViewName = descriptor.name || descriptor.id;
        selectedModuleId = descriptor.moduleId || "";
        selectedCategoryLabel = descriptor.category || "";
        var urlValue = descriptor.source;
        if (urlValue && urlValue.toString)
            urlValue = urlValue.toString();
        selectedSource = urlValue || "";

        metadataModel.clear();
        if (descriptor.metadata) {
            var keys = Object.keys(descriptor.metadata);
            keys.sort();
            for (var i = 0; i < keys.length; ++i) {
                metadataModel.append({ key: keys[i], value: metadataValueToString(descriptor.metadata[keys[i]]) });
            }
        }

        if (viewList.currentIndex !== index)
            viewList.currentIndex = index;

        viewActivated(descriptor.id);
    }

    function selectIndex(index) {
        if (!viewsModel || index < 0 || index >= viewList.count) {
            applyDescriptor(null, -1);
            return;
        }
        var descriptor = viewsModel.viewAt(index);
        applyDescriptor(descriptor, index);
    }

    function ensureSelectionValid() {
        if (!viewsModel || viewList.count === 0) {
            applyDescriptor(null, -1);
            return;
        }

        var descriptor = viewsModel.findById(selectedViewId);
        if (descriptorMatchesSelection(descriptor)) {
            // ensure currentIndex matches
            for (var i = 0; i < viewList.count; ++i) {
                var candidate = viewsModel.viewAt(i);
                if (descriptorMatchesSelection(candidate)) {
                    if (viewList.currentIndex !== i)
                        viewList.currentIndex = i;
                    applyDescriptor(candidate, i);
                    return;
                }
            }
        }

        var index = viewList.currentIndex >= 0 ? viewList.currentIndex : 0;
        if (index >= viewList.count)
            index = viewList.count - 1;
        selectIndex(index);
    }

    function triggerReload() {
        if (root.reloadingModules)
            return;
        if (typeof appController === "undefined" || !appController.reloadUiModules)
            return;
        root.reloadingModules = true;
        reloadStatusTimer.stop();
        root.reloadStatusError = false;
        root.reloadStatusMessage = qsTr("Trwa przeładowywanie modułów…");
        appController.reloadUiModules();
    }

    onSelectedCategoryChanged: {
        if (suppressCategorySync)
            return;
        if (viewsModel && viewsModel.categoryFilter !== selectedCategory)
            viewsModel.categoryFilter = selectedCategory;
        Qt.callLater(ensureSelectionValid);
    }

    onSearchQueryChanged: {
        if (suppressSearchSync)
            return;
        if (viewsModel && viewsModel.searchFilter !== searchQuery)
            viewsModel.searchFilter = searchQuery;
        Qt.callLater(ensureSelectionValid);
    }

    Connections {
        target: viewsModel
        function onCategoryFilterChanged() {
            syncCategoryFromModel();
        }
        function onSearchFilterChanged() {
            syncSearchFromModel();
            Qt.callLater(ensureSelectionValid);
        }
        function onModelReset() {
            refreshCategories();
            syncSearchFromModel();
            Qt.callLater(ensureSelectionValid);
        }
        function onRowsInserted() {
            refreshCategories();
            Qt.callLater(ensureSelectionValid);
        }
        function onRowsRemoved() {
            refreshCategories();
            Qt.callLater(ensureSelectionValid);
        }
    }

    Connections {
        target: typeof appController !== "undefined" ? appController : null
        function onUiModulesReloaded(success, report) {
            root.reloadingModules = false;
            reloadReport = report || {};
            var message = "";
            var error = false;
            var pluginCount = reloadReport.pluginsLoaded || 0;
            var viewCount = reloadReport.viewsRegistered || 0;
            if (moduleDirectories.length === 0) {
                message = success
                        ? qsTr("Brak skonfigurowanych katalogów modułów.")
                        : qsTr("Nie udało się przeładować modułów – brak katalogów.");
                error = !success;
            } else if (success) {
                if (pluginCount === 0 && viewCount === 0)
                    message = qsTr("Przeładowano moduły, ale nie znaleziono widoków.");
                else
                    message = qsTr("Przeładowano %1 pluginów i %2 widoków.")
                            .arg(pluginCount)
                            .arg(viewCount);
            } else {
                message = qsTr("Nie udało się przeładować wszystkich modułów – sprawdź logi aplikacji.");
                error = true;
            }
            root.reloadStatusMessage = message;
            root.reloadStatusError = error;
            if (message.length > 0)
                reloadStatusTimer.restart();
        }
        function onUiModuleDirectoriesChanged() {
            root.reloadingModules = false;
            reloadStatusTimer.stop();
            root.reloadStatusMessage = "";
            root.reloadStatusError = false;
            reloadReport = {};
        }
    }

    Component.onCompleted: {
        syncCategoryFromModel();
        syncSearchFromModel();
        refreshCategories();
        ensureSelectionValid();
        if (typeof appController !== "undefined" && appController.moduleManager && appController.moduleManager.lastLoadReport)
            reloadReport = appController.moduleManager.lastLoadReport();
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Widoki modułów")
                font.pixelSize: 20
                font.bold: true
                Layout.fillWidth: true
            }

            TextField {
                id: searchField
                Layout.preferredWidth: 240
                placeholderText: qsTr("Szukaj widoków…")
                text: root.searchQuery
                selectByMouse: true
                onTextChanged: root.searchQuery = text
                ToolTip.visible: hovered && text.length === 0
                ToolTip.text: qsTr("Filtruj po nazwie, identyfikatorze, module lub metadanych")
            }

            ComboBox {
                id: categoryCombo
                Layout.preferredWidth: 220
                textRole: "label"
                valueRole: "value"
                model: categoryOptions
                onActivated: selectedCategory = categoryCombo.currentValue
                Component.onCompleted: refreshCategories()
                Connections {
                    target: root
                    function onSelectedCategoryChanged() {
                        var modelIndex = -1
                        for (var i = 0; i < categoryOptions.length; ++i) {
                            if (categoryOptions[i].value === root.selectedCategory) {
                                modelIndex = i
                                break
                            }
                        }
                        if (modelIndex >= 0 && categoryCombo.currentIndex !== modelIndex)
                            categoryCombo.currentIndex = modelIndex
                        else if (modelIndex < 0 && categoryCombo.currentIndex !== 0)
                            categoryCombo.currentIndex = 0
                    }
                }
            }

            Button {
                id: reloadButton
                Layout.preferredWidth: 160
                text: root.reloadingModules ? qsTr("Przeładowywanie…") : qsTr("Przeładuj moduły")
                icon.name: "view-refresh"
                enabled: !root.reloadingModules && moduleDirectories.length > 0 && typeof appController !== "undefined" && !!appController.reloadUiModules
                onClicked: root.triggerReload()
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: moduleDirectories.length > 0
                    ? qsTr("Ponownie wczytaj pluginy UI z aktualnych katalogów")
                    : qsTr("Skonfiguruj katalogi modułów przed przeładowaniem")
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                Layout.fillWidth: true
                text: moduleDirectories.length > 0
                    ? qsTr("Skonfigurowane katalogi modułów:")
                    : qsTr("Brak skonfigurowanych katalogów modułów. Użyj flagi --ui-module-dir lub zmiennej BOT_CORE_UI_MODULE_DIRS.")
                color: moduleDirectories.length > 0 ? Qt.rgba(1, 1, 1, 0.7) : Qt.rgba(1, 0.6, 0.6, 1)
                wrapMode: Text.WordWrap
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                visible: moduleDirectories.length > 0

                Repeater {
                    model: moduleDirectories
                    delegate: Label {
                        Layout.fillWidth: true
                        text: modelData
                        wrapMode: Text.WrapAnywhere
                        color: Qt.rgba(1, 1, 1, 0.65)
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: reloadStatusMessage.length > 0
            text: reloadStatusMessage
            color: reloadStatusError ? Qt.rgba(1, 0.45, 0.45, 1) : Qt.rgba(0.55, 0.85, 0.6, 1)
            wrapMode: Text.WordWrap
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            visible: (reloadReport.loadedPlugins && reloadReport.loadedPlugins.length > 0)
                     || (reloadReport.failedPlugins && reloadReport.failedPlugins.length > 0)

            GroupBox {
                Layout.fillWidth: true
                visible: reloadReport.loadedPlugins && reloadReport.loadedPlugins.length > 0
                title: qsTr("Załadowane pluginy (%1)").arg(reloadReport.loadedPlugins ? reloadReport.loadedPlugins.length : 0)

                ScrollView {
                    implicitHeight: Math.min(160, loadedPluginsView.contentHeight)
                    ListView {
                        id: loadedPluginsView
                        width: parent.width
                        model: reloadReport.loadedPlugins || []
                        clip: true
                        delegate: Label {
                            width: ListView.view ? ListView.view.width : 0
                            text: modelData
                            wrapMode: Text.WrapAnywhere
                        }
                    }
                }
            }

            GroupBox {
                Layout.fillWidth: true
                visible: reloadReport.failedPlugins && reloadReport.failedPlugins.length > 0
                title: qsTr("Błędy pluginów (%1)").arg(reloadReport.failedPlugins ? reloadReport.failedPlugins.length : 0)

                ScrollView {
                    implicitHeight: Math.min(160, failedPluginsView.contentHeight)
                    ListView {
                        id: failedPluginsView
                        width: parent.width
                        model: reloadReport.failedPlugins || []
                        clip: true
                        delegate: Column {
                            width: ListView.view ? ListView.view.width : 0
                            spacing: 4

                            Label {
                                text: modelData.path
                                wrapMode: Text.WrapAnywhere
                                font.bold: true
                            }

                            Label {
                                text: modelData.message
                                wrapMode: Text.WordWrap
                                color: Qt.rgba(1, 0.6, 0.6, 1)
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            visible: (reloadReport.invalidEntries && reloadReport.invalidEntries.length > 0)
                     || (reloadReport.missingPaths && reloadReport.missingPaths.length > 0)

            GroupBox {
                Layout.fillWidth: true
                visible: reloadReport.invalidEntries && reloadReport.invalidEntries.length > 0
                title: qsTr("Pominięte pliki (%1)").arg(reloadReport.invalidEntries ? reloadReport.invalidEntries.length : 0)

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Repeater {
                        model: reloadReport.invalidEntries || []
                        delegate: Label {
                            Layout.fillWidth: true
                            text: modelData
                            wrapMode: Text.WrapAnywhere
                        }
                    }
                }
            }

            GroupBox {
                Layout.fillWidth: true
                visible: reloadReport.missingPaths && reloadReport.missingPaths.length > 0
                title: qsTr("Brakujące ścieżki (%1)").arg(reloadReport.missingPaths ? reloadReport.missingPaths.length : 0)

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Repeater {
                        model: reloadReport.missingPaths || []
                        delegate: Label {
                            Layout.fillWidth: true
                            text: modelData
                            wrapMode: Text.WrapAnywhere
                        }
                    }
                }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ScrollView {
                id: listScroll
                SplitView.preferredWidth: 320
                Layout.fillHeight: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ListView {
                    id: viewList
                    anchors.fill: parent
                    model: viewsModel
                    clip: true
                    delegate: ItemDelegate {
                        width: ListView.view ? ListView.view.width : 0
                        highlighted: root.selectedViewId === model.id
                        onClicked: root.selectIndex(index)
                        padding: 12

                        contentItem: ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            Label {
                                text: model.name && model.name.length ? model.name : model.id
                                font.bold: true
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }

                            Label {
                                text: model.moduleId
                                color: Qt.rgba(1, 1, 1, 0.65)
                                font.pointSize: 9
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }

                            Label {
                                text: model.category && model.category.length
                                      ? qsTr("Kategoria: %1").arg(model.category)
                                      : qsTr("Brak kategorii")
                                color: Qt.rgba(1, 1, 1, 0.55)
                                font.pointSize: 9
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                        }
                    }

                    footer: Label {
                        visible: viewList.count === 0
                        text: qsTr("Brak zarejestrowanych widoków modułów")
                        padding: 16
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        width: ListView.view ? ListView.view.width : 0
                    }
                }
            }

            Rectangle {
                width: 1
                color: Qt.rgba(1, 1, 1, 0.08)
            }

            Pane {
                id: detailPane
                Layout.fillWidth: true
                Layout.fillHeight: true
                padding: 16
                background: Rectangle {
                    color: Qt.darker(detailPane.palette.window, 1.05)
                    radius: 8
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    Label {
                        id: selectedTitle
                        text: selectedViewName.length ? selectedViewName : qsTr("Wybierz widok modułu")
                        font.pixelSize: 20
                        font.bold: true
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        visible: selectedModuleId.length > 0 || selectedCategoryLabel.length > 0
                        spacing: 8

                        Label {
                            text: selectedModuleId.length ? qsTr("Moduł: %1").arg(selectedModuleId) : ""
                            visible: selectedModuleId.length > 0
                            Layout.fillWidth: true
                        }

                        Rectangle {
                            id: categoryChip
                            visible: selectedCategoryLabel.length > 0
                            radius: 12
                            color: Qt.rgba(0.14, 0.58, 0.82, 0.25)
                            border.color: Qt.rgba(0.14, 0.58, 0.82, 0.6)
                            Layout.alignment: Qt.AlignVCenter

                            Label {
                                anchors.centerIn: parent
                                padding: 6
                                text: selectedCategoryLabel
                                color: Qt.rgba(0.14, 0.58, 0.82, 0.9)
                            }
                        }
                    }

                    Frame {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Loader {
                            id: viewLoader
                            anchors.fill: parent
                            active: selectedSource && String(selectedSource).length > 0
                            source: selectedSource
                        }

                        Label {
                            anchors.centerIn: parent
                            width: parent.width * 0.8
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.WordWrap
                            visible: {
                                if (!selectedSource || String(selectedSource).length === 0)
                                    return true;
                                return viewLoader.status !== Loader.Ready;
                            }
                            text: {
                                if (!selectedSource || String(selectedSource).length === 0)
                                    return qsTr("Wybierz widok po lewej stronie, aby go załadować");
                                if (viewLoader.status === Loader.Loading)
                                    return qsTr("Ładowanie widoku modułu…");
                                if (viewLoader.status === Loader.Error)
                                    return qsTr("Nie udało się wczytać widoku modułu");
                                return "";
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Metadane")
                        Layout.fillWidth: true
                        visible: metadataModel.count > 0

                        ScrollView {
                            implicitHeight: Math.min(metadataView.contentHeight, 160)
                            ListView {
                                id: metadataView
                                width: parent.width
                                model: metadataModel
                                delegate: RowLayout {
                                    width: ListView.view ? ListView.view.width : 0
                                    spacing: 12

                                    Label {
                                        text: model.key
                                        font.bold: true
                                        Layout.preferredWidth: 160
                                        wrapMode: Text.WordWrap
                                    }

                                    Label {
                                        text: model.value
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: metadataModel.count === 0 && selectedViewId.length > 0
                        text: qsTr("Brak metadanych dla wybranego widoku")
                        color: Qt.rgba(1, 1, 1, 0.6)
                    }
                }
            }
        }
    }
}
