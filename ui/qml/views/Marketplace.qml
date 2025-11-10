import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../components" as Components

Item {
    id: root
    objectName: "marketplaceView"

    property var appController: null
    property var presets: []
    property bool busy: false
    property string statusMessage: ""
    property string statusError: ""
    property string exportFormat: "yaml"
    property var selectedPreset: null
    property var marketplaceControllerRef: (typeof marketplaceController !== "undefined" ? marketplaceController : null)
    property var updateController: (typeof updateManager !== "undefined" ? updateManager : null)
    property string selectedCategory: ""
    property var categories: []

    function resolvedController() {
        if (marketplaceControllerRef)
            return marketplaceControllerRef
        if (root.appController)
            return root.appController
        if (typeof appController !== "undefined")
            return appController
        return null
    }

    function refreshPresets() {
        const ctrl = marketplaceControllerRef
        if (!ctrl) {
            statusError = qsTr("Brak połączenia z usługą marketplace.")
            statusMessage = ""
            presets = []
            return
        }
        busy = true
        if (ctrl.refreshPresets)
            ctrl.refreshPresets()
        categories = buildCategoryList(ctrl.categories || [])
        if (categories.length > 0)
            selectedCategory = categories[Math.max(0, categorySelector.currentIndex)].value
        updatePresetList()
        statusError = ctrl.lastError || ""
        if (!statusError || statusError.length === 0)
            statusMessage = qsTr("Załadowano %1 presetów").arg(presets.length)
        else
            statusMessage = ""
        busy = false
    }

    function updatePresetList() {
        const ctrl = marketplaceControllerRef
        if (!ctrl) {
            presets = []
            return
        }
        if (!selectedCategory || selectedCategory.length === 0 || !ctrl.presetsForCategory) {
            presets = ctrl.presets || []
        } else {
            presets = ctrl.presetsForCategory(selectedCategory)
        }
    }

    function buildCategoryList(source) {
        var list = [{ display: qsTr("Wszystkie kategorie"), value: "" }]
        if (source && source.length) {
            for (var i = 0; i < source.length; ++i)
                list.push({ display: source[i], value: source[i] })
        }
        return list
    }

    function applyBackendResult(result, successMessage, refreshAfterSuccess) {
        if (!result || typeof result.success === "undefined") {
            statusError = qsTr("Nieprawidłowa odpowiedź z usługi marketplace.")
            statusMessage = ""
            return false
        }
        if (!result.success) {
            const error = result.error || (result.issues ? result.issues.join("\n") : "")
            statusError = error && error.length > 0 ? error : qsTr("Operacja nie powiodła się.")
            statusMessage = ""
            return false
        }
        statusError = ""
        statusMessage = successMessage
        if (refreshAfterSuccess)
            refreshPresets()
        return true
    }

    function importPresetFromUrl(url) {
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.marketplaceImportPreset) {
            statusError = qsTr("Backend marketplace nie obsługuje importu.")
            statusMessage = ""
            return
        }
        busy = true
        try {
            const result = ctrl.marketplaceImportPreset(url)
            const name = result && result.preset ? (result.preset.name || result.preset.presetId || "") : ""
            applyBackendResult(result, qsTr("Zaimportowano preset %1.").arg(name), true)
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Import nie powiódł się.")
            statusMessage = ""
        }
        busy = false
    }

    function activatePreset(preset) {
        if (!preset)
            return
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.marketplaceActivatePreset) {
            statusError = qsTr("Backend marketplace nie obsługuje aktywacji.")
            statusMessage = ""
            return
        }
        busy = true
        try {
            const result = ctrl.marketplaceActivatePreset(preset.presetId || preset.preset_id || "")
            const label = preset.name || preset.presetId || preset.preset_id || ""
            applyBackendResult(result, qsTr("Aktywowano preset %1.").arg(label), true)
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Aktywacja nie powiodła się.")
            statusMessage = ""
        }
        busy = false
    }

    function removePreset(preset) {
        if (!preset)
            return
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.marketplaceRemovePreset) {
            statusError = qsTr("Backend marketplace nie obsługuje usuwania presetów.")
            statusMessage = ""
            return
        }
        busy = true
        try {
            const result = ctrl.marketplaceRemovePreset(preset.presetId || preset.preset_id || "")
            const label = preset.name || preset.presetId || preset.preset_id || ""
            applyBackendResult(result, qsTr("Usunięto preset %1.").arg(label), true)
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Usuwanie presetu nie powiodło się.")
            statusMessage = ""
        }
        busy = false
    }

    function exportPreset(preset, destinationUrl) {
        if (!preset)
            return
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.marketplaceExportPreset) {
            statusError = qsTr("Backend marketplace nie obsługuje eksportu.")
            statusMessage = ""
            return
        }
        busy = true
        try {
            const result = ctrl.marketplaceExportPreset(preset.presetId || preset.preset_id || "", exportFormat, destinationUrl)
            const label = preset.name || preset.presetId || preset.preset_id || ""
            if (applyBackendResult(result, qsTr("Wyeksportowano preset %1.").arg(label), false) && result && result.path) {
                statusMessage = qsTr("Wyeksportowano preset do %1").arg(result.path)
            }
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Eksport nie powiódł się.")
            statusMessage = ""
        }
        busy = false
    }

    function assignPresetToPortfolio(preset, portfolioId) {
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.assignPresetToPortfolio) {
            statusError = qsTr("Backend marketplace nie obsługuje przydziału portfeli.")
            statusMessage = ""
            return false
        }
        const normalizedPortfolio = (portfolioId || "").trim()
        if (normalizedPortfolio.length === 0)
            return false
        const presetId = preset && (preset.presetId || preset.preset_id || "")
        if (!presetId) {
            statusError = qsTr("Niepoprawny identyfikator presetu.")
            statusMessage = ""
            return false
        }
        busy = true
        let success = false
        try {
            success = ctrl.assignPresetToPortfolio(presetId, normalizedPortfolio)
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Przydzielenie portfela nie powiodło się.")
            statusMessage = ""
        }
        busy = false
        if (success) {
            statusError = ""
            refreshPresets()
            return true
        }
        if (!statusError || statusError.length === 0)
            statusError = ctrl.lastError || qsTr("Przydzielenie portfela nie powiodło się.")
        return false
    }

    function unassignPresetFromPortfolio(preset, portfolioId) {
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.unassignPresetFromPortfolio) {
            statusError = qsTr("Backend marketplace nie obsługuje usuwania przydziałów.")
            statusMessage = ""
            return false
        }
        const normalizedPortfolio = (portfolioId || "").trim()
        if (normalizedPortfolio.length === 0)
            return false
        const presetId = preset && (preset.presetId || preset.preset_id || "")
        if (!presetId) {
            statusError = qsTr("Niepoprawny identyfikator presetu.")
            statusMessage = ""
            return false
        }
        busy = true
        let success = false
        try {
            success = ctrl.unassignPresetFromPortfolio(presetId, normalizedPortfolio)
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Usunięcie przydziału nie powiodło się.")
            statusMessage = ""
        }
        busy = false
        if (success) {
            statusError = ""
            refreshPresets()
            return true
        }
        if (!statusError || statusError.length === 0)
            statusError = ctrl.lastError || qsTr("Usunięcie przydziału nie powiodło się.")
        return false
    }

    function triggerImport() {
        importDialog.open()
    }

    function triggerExport(preset) {
        selectedPreset = preset
        exportDialog.currentFile = preset && preset.name ? preset.name + "." + exportFormat : "preset." + exportFormat
        exportDialog.open()
    }

    Component.onCompleted: refreshPresets()

    Connections {
        target: marketplaceControllerRef
        ignoreUnknownSignals: true
        function onPresetsChanged() { updatePresetList() }
        function onCategoriesChanged() {
            categories = marketplaceControllerRef ? buildCategoryList(marketplaceControllerRef.categories || []) : buildCategoryList([])
            if (categorySelector.currentIndex < 0)
                categorySelector.currentIndex = 0
        }
        function onLastErrorChanged() { statusError = marketplaceControllerRef && marketplaceControllerRef.lastError ? marketplaceControllerRef.lastError : "" }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Marketplace presetów strategii")
                font.pixelSize: 22
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            ComboBox {
                id: categorySelector
                Layout.preferredWidth: 200
                model: categories
                textRole: "display"
                valueRole: "value"
                editable: false
                onActivated: {
                    selectedCategory = categorySelector.currentValue
                    updatePresetList()
                }
                onCurrentIndexChanged: {
                    if (currentIndex >= 0) {
                        selectedCategory = categorySelector.currentValue
                        updatePresetList()
                    }
                }
                Component.onCompleted: currentIndex = 0
            }

            ComboBox {
                id: formatSelector
                Layout.preferredWidth: 140
                model: ["yaml", "json"]
                currentIndex: exportFormat === "json" ? 1 : 0
                onCurrentIndexChanged: exportFormat = model[currentIndex]
                toolTip: qsTr("Format eksportu presetu")
            }

            ToolButton {
                icon.name: "document-import"
                text: qsTr("Importuj…")
                display: AbstractButton.TextBesideIcon
                onClicked: triggerImport()
            }

            ToolButton {
                icon.name: "view-refresh"
                text: qsTr("Odśwież")
                display: AbstractButton.TextBesideIcon
                enabled: !busy
                onClicked: refreshPresets()
            }
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: statusError.length > 0
            color: "firebrick"
            text: statusError
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: statusError.length === 0 && statusMessage.length > 0
            text: statusMessage
        }

        BusyIndicator {
            Layout.alignment: Qt.AlignLeft
            running: busy
            visible: busy
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: presetList
                width: parent.width
                clip: true
                spacing: 12
                model: presets

                delegate: Frame {
                    width: ListView.view.width
                    padding: 16
                    background: Rectangle {
                        radius: 6
                        color: Qt.darker(palette.window, 1.02)
                        border.color: Qt.darker(color, 1.2)
                        border.width: 1
                    }

                    property var preset: modelData

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 6

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: preset.name || preset.presetId || preset.preset_id
                                font.pixelSize: 18
                                font.bold: true
                            }

                            Label {
                                visible: (preset.version || "").length > 0
                                text: qsTr("wersja %1").arg(preset.version)
                                color: palette.mid
                            }

                            Label {
                                visible: (preset.profile || "").length > 0
                                text: qsTr("profil: %1").arg(preset.profile)
                                color: palette.mid
                            }
                        }

                        Flow {
                            width: parent.width
                            spacing: 6
                            Repeater {
                                model: preset.tags || []
                                delegate: Rectangle {
                                    radius: 4
                                    height: 20
                                    color: Qt.darker(palette.base, 1.05)
                                    border.color: palette.mid
                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData
                                        font.pixelSize: 12
            }
        }

        Components.UpdateManagerPanel {
            Layout.fillWidth: true
            manager: updateController
        }
    }
}

                        Label {
                            visible: preset.signatureVerified === false
                            text: qsTr("UWAGA: podpis niezweryfikowany")
                            color: "#c0392b"
                            font.bold: true
                        }

                        ColumnLayout {
                            visible: preset.issues && preset.issues.length > 0
                            Layout.fillWidth: true
                            spacing: 4

                            Label {
                                text: qsTr("Problemy walidacji:")
                                color: "#c0392b"
                                font.bold: true
                            }

                            Repeater {
                                model: preset.issues || []
                                delegate: Label {
                                    Layout.fillWidth: true
                                    text: String.fromUtf8("\u2022 ") + (typeof modelData === "string" ? modelData : JSON.stringify(modelData))
                                    color: "#c0392b"
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }

                        ColumnLayout {
                            visible: preset.warningMessages && preset.warningMessages.length > 0
                            Layout.fillWidth: true
                            spacing: 4

                            Label {
                                text: qsTr("Ostrzeżenia licencji:")
                                color: "#e67e22"
                                font.bold: true
                            }

                            Repeater {
                                model: preset.warningMessages || []
                                delegate: Label {
                                    Layout.fillWidth: true
                                    text: String.fromUtf8("\u2022 ") + modelData
                                    color: "#e67e22"
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Layout.topMargin: 8

                            Button {
                                text: qsTr("Aktywuj")
                                enabled: !busy
                                onClicked: activatePreset(preset)
                            }

                            Button {
                                text: qsTr("Eksportuj…")
                                enabled: !busy
                                onClicked: triggerExport(preset)
                            }

                            Button {
                                text: qsTr("Usuń")
                                enabled: !busy
                                onClicked: removePreset(preset)
                            }

                            Item { Layout.fillWidth: true }
                                    }
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            GroupBox {
                                Layout.fillWidth: true
                                title: qsTr("Zależności")
                                visible: preset.dependencies && preset.dependencies.length > 0

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2
                                    Repeater {
                                        model: preset.dependencies || []
                                        delegate: Label {
                                            Layout.fillWidth: true
                                            wrapMode: Text.WordWrap
                                            text: {
                                                const item = modelData || {}
                                                const name = item.presetId || item.preset_id || item.name || ""
                                                const constraints = item.constraints && item.constraints.length ? " (" + item.constraints.join(", ") + ")" : ""
                                                const optional = item.optional ? qsTr(" [opcjonalne]") : ""
                                                const capability = item.capability ? qsTr(" – wymaga %1").arg(item.capability) : ""
                                                const note = item.notes ? qsTr(": %1").arg(item.notes) : ""
                                                return String.fromUtf8("\u2022 ") + name + constraints + optional + capability + note
                                            }
                                        }
                                    }
                                }
                            }

                            GroupBox {
                                Layout.fillWidth: true
                                title: qsTr("Licencja")
                                visible: preset.license && preset.license.seat_summary

                                ColumnLayout {
                                    property var seatSummary: preset.license ? preset.license.seat_summary || {} : {}
                                    property var subscriptionSummary: preset.license ? preset.license.subscription_summary || {} : {}
                                    Layout.fillWidth: true
                                    spacing: 4

                                    Label {
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                        text: {
                                            const seat = parent.seatSummary || {}
                                            const total = seat.total !== undefined && seat.total !== null ? seat.total : qsTr("brak")
                                            const inUse = seat.in_use !== undefined && seat.in_use !== null ? seat.in_use : qsTr("brak")
                                            const available = seat.available !== undefined && seat.available !== null ? seat.available : qsTr("brak")
                                            return qsTr("Stanowiska: %1 zajętych z %2 (dostępne: %3)").arg(inUse).arg(total).arg(available)
                                        }
                                    }

                                    Flow {
                                        Layout.fillWidth: true
                                        spacing: 6
                                        Repeater {
                                            model: (parent.seatSummary && parent.seatSummary.assignments) || []
                                            delegate: Rectangle {
                                                radius: 4
                                                border.color: palette.mid
                                                color: Qt.darker(palette.base, 1.05)
                                                height: 22
                                                width: implicitWidth
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: modelData
                                                    font.pixelSize: 12
                                                }
                                            }
                                        }
                                        Label {
                                            visible: !(parent.seatSummary && parent.seatSummary.assignments && parent.seatSummary.assignments.length > 0)
                                            text: qsTr("Brak przypisanych urządzeń")
                                            color: palette.mid
                                            font.pixelSize: 12
                                        }
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                        visible: subscriptionSummary && (subscriptionSummary.status || subscriptionSummary.renews_at)
                                        text: {
                                            const sub = subscriptionSummary || {}
                                            const status = sub.status || qsTr("nieznany")
                                            const renews = sub.renews_at || qsTr("brak")
                                            return qsTr("Subskrypcja: %1 (odnowienie: %2)").arg(status).arg(renews)
                                        }
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                        visible: subscriptionSummary && (subscriptionSummary.period_start || subscriptionSummary.period_end)
                                        text: {
                                            const sub = subscriptionSummary || {}
                                            const start = sub.period_start || qsTr("brak")
                                            const end = sub.period_end || qsTr("brak")
                                            return qsTr("Okres rozliczeniowy: %1 – %2").arg(start).arg(end)
                                        }
                                    }
                                }
                            }

                            GroupBox {
                                Layout.fillWidth: true
                                title: qsTr("Aktualizacje")
                                visible: preset.updateChannels && preset.updateChannels.length > 0

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2

                                    Label {
                                        visible: preset.upgradeAvailable === true
                                        color: "#c0392b"
                                        font.bold: true
                                        wrapMode: Text.WordWrap
                                        text: qsTr("Dostępna aktualizacja do wersji %1 (kanał %2)")
                                            .arg(preset.upgradeVersion || preset.version || "")
                                            .arg(preset.preferredChannel || qsTr("domyślny"))
                                    }

                                    Repeater {
                                        model: preset.updateChannels || []
                                        delegate: Label {
                                            Layout.fillWidth: true
                                            wrapMode: Text.WordWrap
                                            text: {
                                                const entry = modelData || {}
                                                const name = entry.name || entry.channel || qsTr("kanał")
                                                const version = entry.version ? qsTr(" wersja %1").arg(entry.version) : ""
                                                const severity = entry.severity ? qsTr(" (%1)").arg(entry.severity) : ""
                                                const notes = entry.notes ? qsTr(": %1").arg(entry.notes) : ""
                                                return String.fromUtf8("\u2022 ") + name + version + severity + notes
                                            }
                                        }
                                    }
                                }
                            }

                            GroupBox {
                                Layout.fillWidth: true
                                title: qsTr("Przydziały do portfeli")

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 6

                                    Flow {
                                        Layout.fillWidth: true
                                        spacing: 6
                                        Repeater {
                                            model: preset.assignedPortfolios || []
                                            delegate: Rectangle {
                                                radius: 4
                                                border.color: palette.mid
                                                color: Qt.darker(palette.base, 1.05)
                                                height: 28
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.margins: 6
                                                    spacing: 6
                                                    Label {
                                                        text: modelData
                                                        font.pixelSize: 12
                                                    }
                                                    Button {
                                                        text: qsTr("Usuń")
                                                        icon.name: "list-remove"
                                                        onClicked: unassignPresetFromPortfolio(preset, modelData)
                                                    }
                                                }
                                            }
                                        }
                                        Label {
                                            visible: !preset.assignedPortfolios || preset.assignedPortfolios.length === 0
                                            text: qsTr("Brak przydzielonych portfeli")
                                            color: palette.mid
                                            font.pixelSize: 12
                                        }
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 6
                                        TextField {
                                            id: portfolioInput
                                            Layout.fillWidth: true
                                            placeholderText: qsTr("Identyfikator portfela")
                                            onAccepted: {
                                                if (assignPresetToPortfolio(preset, text))
                                                    text = ""
                                            }
                                        }
                                        Button {
                                            text: qsTr("Dodaj")
                                            enabled: (portfolioInput.text || "").trim().length > 0
                                            onClicked: {
                                                if (assignPresetToPortfolio(preset, portfolioInput.text))
                                                    portfolioInput.text = ""
                                            }
                                        }
                                    }
                                }
                            }
                        }
        }
    }

    FileDialog {
        id: importDialog
        title: qsTr("Wybierz plik presetu")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("Pliki strategii (*.json *.yaml *.yml *.zip)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (selectedFile)
                importPresetFromUrl(selectedFile)
        }
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Zapisz preset strategii")
        fileMode: FileDialog.SaveFile
        onAccepted: {
            if (selectedFile)
                exportPreset(selectedPreset, selectedFile)
        }
    }
}
