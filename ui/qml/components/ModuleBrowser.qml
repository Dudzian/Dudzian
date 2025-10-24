import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var viewsModel: typeof moduleViewsModel !== "undefined" ? moduleViewsModel : null
    property var servicesModel: typeof moduleServicesModel !== "undefined" ? moduleServicesModel : null
    property string selectedViewId: ""
    property string selectedViewName: ""
    property string selectedModuleId: ""
    property string selectedCategoryLabel: ""
    property url selectedSource: ""
    property string selectedCategory: ""
    property string searchQuery: ""
    property string serviceSearchQuery: ""
    property var categoryOptions: []
    property bool suppressCategorySync: false
    property bool suppressSearchSync: false
    property bool suppressServiceSearchSync: false
    readonly property var moduleDirectories: typeof appController !== "undefined" ? appController.uiModuleDirectories : []
    property bool reloadingModules: false
    property string reloadStatusMessage: ""
    property bool reloadStatusError: false
    property var reloadReport: ({})
    property bool autoReloadEnabled: false
    property string newModuleDirectoryPath: ""
    property string directoryStatusMessage: ""
    property bool directoryStatusError: false

    signal viewActivated(string viewId)

    ListModel { id: metadataModel }

    Timer {
        id: reloadStatusTimer
        interval: 5000
        repeat: false
        onTriggered: root.reloadStatusMessage = ""
    }

    Timer {
        id: directoryStatusTimer
        interval: 5000
        repeat: false
        onTriggered: root.directoryStatusMessage = ""
    }

    function metadataValueToString(value) {
        if (value === null || value === undefined)
            return "";
        if (typeof value === "object")
            return JSON.stringify(value, null, 2);
        return String(value);
    }

    function metadataEntries(metadata) {
        if (!metadata)
            return [];
        var keys = Object.keys(metadata);
        keys.sort();
        var results = [];
        for (var i = 0; i < keys.length; ++i) {
            results.push({ key: keys[i], value: metadataValueToString(metadata[keys[i]]) });
        }
        return results;
    }

    function canManageModuleDirectories() {
        return typeof appController !== "undefined"
                && typeof appController.addUiModuleDirectory === "function"
                && typeof appController.removeUiModuleDirectory === "function";
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

    function syncServiceSearchFromModel() {
        if (!servicesModel)
            return;
        suppressServiceSearchSync = true;
        serviceSearchQuery = servicesModel.searchFilter || "";
        suppressServiceSearchSync = false;
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

    function submitNewModuleDirectory() {
        directoryStatusTimer.stop();

        if (!canManageModuleDirectories()) {
            directoryStatusMessage = qsTr("Zarządzanie katalogami modułów jest niedostępne w tej wersji aplikacji.");
            directoryStatusError = true;
            directoryStatusTimer.restart();
            return;
        }

        var value = (newModuleDirectoryPath || "").trim();
        if (value.length === 0) {
            directoryStatusMessage = qsTr("Podaj ścieżkę katalogu modułów.");
            directoryStatusError = true;
            directoryStatusTimer.restart();
            return;
        }

        var normalized = appController.addUiModuleDirectory(value);
        if (normalized && normalized.length > 0) {
            directoryStatusMessage = qsTr("Dodano katalog modułów: %1").arg(normalized);
            directoryStatusError = false;
            newModuleDirectoryPath = "";
        } else {
            directoryStatusMessage = qsTr("Nie dodano katalogu modułów – ścieżka jest nieprawidłowa lub już istnieje.");
            directoryStatusError = true;
        }

        directoryStatusTimer.restart();
    }

    function requestRemoveModuleDirectory(path) {
        directoryStatusTimer.stop();

        if (!canManageModuleDirectories()) {
            directoryStatusMessage = qsTr("Zarządzanie katalogami modułów jest niedostępne w tej wersji aplikacji.");
            directoryStatusError = true;
            directoryStatusTimer.restart();
            return;
        }

        if (!path || path.length === 0)
            return;

        if (appController.removeUiModuleDirectory(path)) {
            directoryStatusMessage = qsTr("Usunięto katalog modułów: %1").arg(path);
            directoryStatusError = false;
        } else {
            directoryStatusMessage = qsTr("Nie udało się usunąć katalogu modułów: %1").arg(path);
            directoryStatusError = true;
        }

        directoryStatusTimer.restart();
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

    onServiceSearchQueryChanged: {
        if (suppressServiceSearchSync)
            return;
        if (servicesModel && servicesModel.searchFilter !== serviceSearchQuery)
            servicesModel.searchFilter = serviceSearchQuery;
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
        function onUiModuleAutoReloadEnabledChanged(enabled) {
            root.autoReloadEnabled = enabled;
        }
    }

    Connections {
        target: servicesModel
        function onSearchFilterChanged() {
            syncServiceSearchFromModel();
        }
        function onModelReset() {
            syncServiceSearchFromModel();
        }
        function onRowsInserted() {
            Qt.callLater(syncServiceSearchFromModel);
        }
        function onRowsRemoved() {
            Qt.callLater(syncServiceSearchFromModel);
        }
    }

    Component.onCompleted: {
        syncCategoryFromModel();
        syncSearchFromModel();
        syncServiceSearchFromModel();
        refreshCategories();
        ensureSelectionValid();
        if (typeof appController !== "undefined" && appController.moduleManager && appController.moduleManager.lastLoadReport)
            reloadReport = appController.moduleManager.lastLoadReport();
        if (typeof appController !== "undefined" && appController.uiModuleAutoReloadEnabled !== undefined)
            root.autoReloadEnabled = appController.uiModuleAutoReloadEnabled;
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

            Switch {
                id: autoReloadSwitch
                Layout.preferredWidth: 200
                text: qsTr("Auto przeładuj")
                checked: root.autoReloadEnabled
                onToggled: {
                    if (root.autoReloadEnabled === checked)
                        return;
                    root.autoReloadEnabled = checked;
                    if (typeof appController !== "undefined" && appController.setUiModuleAutoReloadEnabled)
                        appController.setUiModuleAutoReloadEnabled(checked);
                }
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Automatycznie przeładuj pluginy po zmianach w katalogach modułów")
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

            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                visible: canManageModuleDirectories()

                TextField {
                    id: newModuleDirectoryField
                    Layout.fillWidth: true
                    placeholderText: qsTr("Dodaj katalog modułów…")
                    text: root.newModuleDirectoryPath
                    selectByMouse: true
                    onTextChanged: root.newModuleDirectoryPath = text
                    onAccepted: root.submitNewModuleDirectory()
                }

                Button {
                    id: addModuleDirectoryButton
                    text: qsTr("Dodaj")
                    icon.name: "list-add"
                    enabled: root.canManageModuleDirectories() && root.newModuleDirectoryPath.trim().length > 0
                    onClicked: root.submitNewModuleDirectory()
                    ToolTip.visible: hovered
                    ToolTip.delay: 400
                    ToolTip.text: qsTr("Dodaj katalog do listy i przeładuj pluginy UI")
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                visible: moduleDirectories.length > 0

                Repeater {
                    model: moduleDirectories
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            Layout.fillWidth: true
                            text: modelData
                            wrapMode: Text.WrapAnywhere
                            color: Qt.rgba(1, 1, 1, 0.65)
                        }

                        ToolButton {
                            visible: root.canManageModuleDirectories()
                            icon.name: "list-remove"
                            display: AbstractButton.IconOnly
                            onClicked: root.requestRemoveModuleDirectory(modelData)
                            ToolTip.visible: hovered
                            ToolTip.delay: 400
                            ToolTip.text: qsTr("Usuń katalog z listy i przeładuj moduły")
                        }
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: directoryStatusMessage.length > 0
            text: directoryStatusMessage
            color: directoryStatusError ? Qt.rgba(1, 0.45, 0.45, 1) : Qt.rgba(0.55, 0.8, 1.0, 1)
            wrapMode: Text.WordWrap
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

                    GroupBox {
                        title: qsTr("Zarejestrowane serwisy (%1)").arg(servicesModel ? servicesModel.rowCount() : 0)
                        Layout.fillWidth: true
                        visible: servicesModel

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            TextField {
                                id: serviceSearchField
                                Layout.fillWidth: true
                                placeholderText: qsTr("Filtruj serwisy…")
                                text: root.serviceSearchQuery
                                selectByMouse: true
                                onTextChanged: root.serviceSearchQuery = text
                            }

                            ScrollView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 220

                                ListView {
                                    id: servicesList
                                    width: parent ? parent.width : 0
                                    clip: true
                                    spacing: 12
                                    model: servicesModel ? servicesModel : null
                                    delegate: Column {
                                        width: ListView.view ? ListView.view.width : 0
                                        spacing: 4

                                        Label {
                                            text: name && name.length ? name : id
                                            font.bold: true
                                            wrapMode: Text.WordWrap
                                        }

                                        Label {
                                            text: qsTr("Identyfikator: %1").arg(id)
                                            color: Qt.rgba(1, 1, 1, 0.6)
                                            wrapMode: Text.WrapAnywhere
                                        }

                                        Label {
                                            text: moduleId && moduleId.length ? qsTr("Moduł: %1").arg(moduleId) : ""
                                            visible: moduleId && moduleId.length
                                            color: Qt.rgba(1, 1, 1, 0.55)
                                            wrapMode: Text.WrapAnywhere
                                        }

                                        Label {
                                            text: singleton ? qsTr("Tryb: singleton") : qsTr("Tryb: instancja na żądanie")
                                            color: Qt.rgba(1, 1, 1, 0.55)
                                        }

                                        Repeater {
                                            model: metadataEntries(metadata)
                                            delegate: Label {
                                                width: ListView.view ? ListView.view.width : 0
                                                text: qsTr("%1: %2").arg(modelData.key).arg(modelData.value)
                                                color: Qt.rgba(1, 1, 1, 0.5)
                                                wrapMode: Text.WordWrap
                                            }
                                        }

                                        Rectangle {
                                            height: 1
                                            color: Qt.rgba(1, 1, 1, 0.08)
                                            visible: index < servicesList.count - 1
                                            anchors.left: parent.left
                                            anchors.right: parent.right
                                        }
                                    }

                                    footer: Label {
                                        visible: (servicesModel ? servicesModel.rowCount() : 0) === 0
                                        text: qsTr("Brak zarejestrowanych serwisów")
                                        padding: 12
                                        horizontalAlignment: Text.AlignHCenter
                                        wrapMode: Text.WordWrap
                                        width: ListView.view ? ListView.view.width : 0
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
