import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../design-system" as DesignSystem
import "../design-system/components" as DesignComponents
import "StrategyBuilderUtils.js" as StrategyUtils

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property var runtimeService: (typeof runtimeService !== "undefined" ? runtimeService : null)
    property string statusMessage: ""
    property string presetName: qsTr("Nowy preset")
    property var savedPresets: []
    property int editingIndex: -1
    property string editingLabel: ""
    property string editingType: ""
    property var editingValues: ({})
    property var editingTemplate: []

    property var blockParameterLabels: ({
        symbol: qsTr("Symbol instrumentu"),
        interval: qsTr("Interwał"),
        source: qsTr("Źródło danych"),
        risk_limit: qsTr("Limit ryzyka (%)"),
        lookback: qsTr("Okno (dni)"),
        threshold: qsTr("Próg sygnału"),
        allocation: qsTr("Alokacja (%)"),
        max_positions: qsTr("Maks. liczba pozycji"),
        venue: qsTr("Giełda"),
        account: qsTr("Konto"),
        slippage: qsTr("Poślizg (%)"),
        quantity: qsTr("Wielkość pozycji"),
        enabled: qsTr("Aktywny")
    })

    property var blockParameterChoices: ({
        source: [
            { value: "sandbox", label: qsTr("Sandbox") },
            { value: "live", label: qsTr("Rynek live") }
        ],
        interval: [
            { value: "1m", label: qsTr("1 minuta") },
            { value: "5m", label: qsTr("5 minut") },
            { value: "1h", label: qsTr("1 godzina") },
            { value: "4h", label: qsTr("4 godziny") },
            { value: "1d", label: qsTr("1 dzień") }
        ]
    })

    ListModel {
        id: paletteModel
        ListElement {
            type: "data_feed"
            label: qsTr("Strumień danych")
            description: qsTr("Wczytaj sygnały rynkowe z sandboxa")
        }
        ListElement {
            type: "filter"
            label: qsTr("Filtr ryzyka")
            description: qsTr("Zastosuj filtr ekspozycji / drawdown")
        }
        ListElement {
            type: "signal"
            label: qsTr("Generator sygnałów")
            description: qsTr("Konfiguruj logikę wejścia/wyjścia")
        }
        ListElement {
            type: "allocator"
            label: qsTr("Alokator")
            description: qsTr("Skaluj pozycję według profilu")
        }
        ListElement {
            type: "execution"
            label: qsTr("Egzekucja")
            description: qsTr("Wyślij zlecenia przez Bot Core")
        }
    }

    ListModel { id: canvasModel }

    function defaultParamsFor(type) {
        return StrategyUtils.defaultParams(type) || {}
    }

    function normalizedParams(type, params) {
        return StrategyUtils.mergeParams(type, params)
    }

    function parameterConfig(type) {
        return StrategyUtils.parameterConfig(type)
    }

    function summarizeParams(type, params) {
        if (!params)
            return qsTr("Brak parametrów")
        var keys = StrategyUtils.summaryOrder(type)
        if (!keys || keys.length === 0)
            keys = Object.keys(params)
        var parts = []
        for (var i = 0; i < keys.length; ++i) {
            var key = keys[i]
            if (!params.hasOwnProperty(key))
                continue
            var value = params[key]
            if (value === null || value === undefined || value === "")
                continue
            var label = blockParameterLabels[key] || key
            if (blockParameterChoices.hasOwnProperty(key)) {
                var choices = blockParameterChoices[key]
                for (var j = 0; j < choices.length; ++j) {
                    if (choices[j].value === value) {
                        value = choices[j].label
                        break
                    }
                }
            } else if (typeof value === "number") {
                value = Qt.formatLocaleNumber(value, value % 1 === 0 ? "f" : "f", value % 1 === 0 ? 0 : 2)
            } else if (typeof value === "boolean") {
                value = value ? qsTr("Tak") : qsTr("Nie")
            }
            parts.push(label + ": " + value)
        }
        if (parts.length === 0)
            return qsTr("Brak parametrów")
        return parts.join(" • ")
    }

    function openBlockEditor(index) {
        if (index < 0 || index >= canvasModel.count)
            return
        var entry = canvasModel.get(index)
        editingIndex = index
        editingLabel = entry.label || ""
        editingType = entry.type || ""
        editingTemplate = parameterConfig(editingType)
        editingValues = normalizedParams(editingType, entry.params || {})
        blockEditorDialog.open()
    }

    function cancelBlockEdit() {
        editingIndex = -1
        editingLabel = ""
        editingType = ""
        editingValues = {}
        editingTemplate = []
    }

    function commitBlockEdit() {
        if (editingIndex < 0 || editingIndex >= canvasModel.count)
            return
        var params = normalizedParams(editingType, editingValues)
        canvasModel.setProperty(editingIndex, "label", editingLabel)
        canvasModel.setProperty(editingIndex, "params", params)
        statusMessage = qsTr("Zaktualizowano parametry bloku (%1)").arg(editingLabel.length ? editingLabel : editingType)
        if (blockEditorDialog.visible)
            blockEditorDialog.close()
    }

    function serializePreset() {
        var blocks = []
        for (var i = 0; i < canvasModel.count; ++i) {
            var entry = canvasModel.get(i)
            blocks.push({
                type: entry.type,
                label: entry.label,
                params: normalizedParams(entry.type, entry.params || {})
            })
        }
        var name = (root.presetName || "").trim()
        if (!name.length)
            name = qsTr("Preset bez nazwy") + " " + Qt.formatDateTime(new Date(), "yyyyMMdd-hhmmss")
        return {
            name: name,
            blocks: blocks,
            created_at: new Date().toISOString(),
            metadata: {
                source: "strategy-builder",
                locale: Qt.locale().name
            }
        }
    }

    function savePreset() {
        if (!runtimeService || !runtimeService.saveStrategyPreset) {
            statusMessage = qsTr("Mostek runtime nie jest dostępny")
            return
        }
        var payload = serializePreset()
        var response = runtimeService.saveStrategyPreset(payload)
        if (response && response.ok) {
            statusMessage = qsTr("Zapisano preset strategii (%1)").arg(response.path || "local")
            refreshSavedPresets()
        } else {
            statusMessage = response && response.error ? response.error : qsTr("Nie udało się zapisać presetu")
        }
    }

    function refreshSavedPresets() {
        if (!runtimeService || !runtimeService.listStrategyPresets) {
            savedPresets = []
            if (savedPresetSelector)
                savedPresetSelector.currentIndex = -1
            return
        }
        var entries = runtimeService.listStrategyPresets()
        savedPresets = entries || []
        if (savedPresetSelector)
            savedPresetSelector.currentIndex = savedPresets.length > 0 ? 0 : -1
    }

    function clearCanvas() {
        canvasModel.clear()
        statusMessage = qsTr("Wyczyszczono płótno")
        if (blockEditorDialog.visible)
            blockEditorDialog.close()
        cancelBlockEdit()
    }

    function loadPresetFromSelection() {
        if (!runtimeService || !runtimeService.loadStrategyPreset) {
            statusMessage = qsTr("Mostek runtime nie jest dostępny")
            return
        }
        if (!savedPresets.length) {
            statusMessage = qsTr("Brak zapisanych presetów")
            return
        }
        var selection = savedPresetSelector.currentIndex >= 0 && savedPresetSelector.currentIndex < savedPresets.length
                ? savedPresets[savedPresetSelector.currentIndex]
                : null
        if (!selection) {
            statusMessage = qsTr("Brak zapisanych presetów")
            return
        }
        var request = {}
        if (selection.slug)
            request.slug = selection.slug
        if (selection.path)
            request.path = selection.path
        var response = runtimeService.loadStrategyPreset(request)
        if (!response || response.error) {
            statusMessage = response && response.error ? response.error : qsTr("Nie udało się wczytać presetu")
            return
        }
        canvasModel.clear()
        if (blockEditorDialog.visible)
            blockEditorDialog.close()
        else
            cancelBlockEdit()
        var blocks = response.blocks || []
        for (var i = 0; i < blocks.length; ++i) {
            var block = blocks[i]
            if (!block)
                continue
            var entry = {
                type: block.type,
                label: block.label,
                params: normalizedParams(block.type, block.params || {})
            }
            canvasModel.append(entry)
        }
        presetName = response.name || selection.name || qsTr("Preset bez nazwy")
        statusMessage = qsTr("Wczytano preset: %1").arg(presetName)
    }

    function deletePresetFromSelection() {
        if (!runtimeService || !runtimeService.deleteStrategyPreset) {
            statusMessage = qsTr("Mostek runtime nie jest dostępny")
            return
        }
        if (!savedPresets.length) {
            statusMessage = qsTr("Brak zapisanych presetów")
            return
        }
        var idx = savedPresetSelector.currentIndex
        if (idx < 0 || idx >= savedPresets.length) {
            statusMessage = qsTr("Nie wybrano presetu do usunięcia")
            return
        }
        var selection = savedPresets[idx]
        var request = {}
        if (selection.slug)
            request.slug = selection.slug
        if (selection.path)
            request.path = selection.path
        var response = runtimeService.deleteStrategyPreset(request)
        if (response && response.ok) {
            var label = selection.name || selection.slug || selection.path || ""
            statusMessage = qsTr("Usunięto preset: %1").arg(label)
            refreshSavedPresets()
        } else {
            statusMessage = response && response.error ? response.error : qsTr("Nie udało się usunąć presetu")
        }
    }

    onRuntimeServiceChanged: {
        if (!runtimeService)
            savedPresets = []
        else
            refreshSavedPresets()
    }

    Component.onCompleted: refreshSavedPresets()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Kreator strategii")
                color: DesignSystem.Palette.textPrimary
                font.pixelSize: DesignSystem.Typography.headlineMedium
                font.bold: true
            }

            TextField {
                id: presetNameField
                Layout.fillWidth: true
                placeholderText: qsTr("Nazwa presetu")
                text: root.presetName
                onTextChanged: root.presetName = text
            }

            Button {
                text: qsTr("Zapisz preset")
                onClicked: savePreset()
            }
        }

        Label {
            Layout.fillWidth: true
            text: statusMessage
            color: DesignSystem.Palette.textSecondary
            wrapMode: Text.WordWrap
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Zapisane presety")
                color: DesignSystem.Palette.textSecondary
                font.pixelSize: DesignSystem.Typography.body
            }

            ComboBox {
                id: savedPresetSelector
                Layout.fillWidth: true
                model: root.savedPresets
                textRole: "name"
                valueRole: "slug"
                enabled: savedPresets.length > 0
                placeholderText: qsTr("Brak zapisanych presetów")
            }

            Button {
                text: qsTr("Wczytaj preset")
                enabled: savedPresets.length > 0
                onClicked: loadPresetFromSelection()
            }

            Button {
                text: qsTr("Usuń preset")
                enabled: savedPresets.length > 0
                onClicked: deletePresetFromSelection()
            }

            Button {
                text: qsTr("Wyczyść płótno")
                onClicked: clearCanvas()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            DesignComponents.Card {
                Layout.preferredWidth: parent.width * 0.32
                Layout.fillHeight: true
                background.color: DesignSystem.Palette.elevated

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    Label {
                        text: qsTr("Bloki dostępne")
                        color: DesignSystem.Palette.textPrimary
                        font.pixelSize: DesignSystem.Typography.title
                        font.bold: true
                    }

                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        model: paletteModel
                        spacing: 8

                        delegate: DesignComponents.Card {
                            width: parent.width
                            background.color: Qt.rgba(0.14, 0.2, 0.28, 0.8)

                            Drag.mimeData: { "application/x-strategy-block": JSON.stringify({ type: type, label: label }) }
                            Drag.hotSpot.x: width / 2
                            Drag.hotSpot.y: height / 2
                            Drag.active: dragHandler.active

                            DragHandler {
                                id: dragHandler
                                target: null
                                onActiveChanged: {
                                    if (active)
                                        parent.Drag.startDrag()
                                }
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 4

                                Label {
                                    text: label
                                    color: DesignSystem.Palette.textPrimary
                                    font.pixelSize: DesignSystem.Typography.body
                                    font.bold: true
                                }
                                Label {
                                    text: description
                                    color: DesignSystem.Palette.textSecondary
                                    font.pixelSize: DesignSystem.Typography.caption
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                }
            }

            DesignComponents.Card {
                Layout.fillWidth: true
                Layout.fillHeight: true
                background.color: DesignSystem.Palette.surface

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    Label {
                        text: qsTr("Przepływ strategii")
                        color: DesignSystem.Palette.textPrimary
                        font.pixelSize: DesignSystem.Typography.title
                        font.bold: true
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: Qt.rgba(0, 0, 0, 0.1)
                        radius: 12

                        DropArea {
                            anchors.fill: parent
                            onDropped: function(event) {
                                if (!event.mimeData || !event.mimeData.hasFormat("application/x-strategy-block"))
                                    return
                                var data = event.mimeData.dataAsString("application/x-strategy-block")
                                try {
                                    var block = JSON.parse(data)
                                    block.params = defaultParamsFor(block.type)
                                    canvasModel.append(block)
                                } catch (e) {
                                    console.warn("Failed to parse dropped block", e)
                                }
                                event.acceptProposedAction()
                            }
                        }

                        ListView {
                            anchors.fill: parent
                            anchors.margins: 12
                            model: canvasModel
                            spacing: 12
                            delegate: DesignComponents.Card {
                                width: parent.width
                                background.color: Qt.rgba(0.18, 0.26, 0.34, 0.8)

                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: 6

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        Label {
                                            text: label
                                            color: DesignSystem.Palette.textPrimary
                                            font.pixelSize: DesignSystem.Typography.body
                                            font.bold: true
                                            Layout.fillWidth: true
                                        }

                                        ToolButton {
                                            icon.name: "list-remove"
                                            onClicked: canvasModel.remove(index)
                                        }
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        wrapMode: Text.WordWrap
                                        text: summarizeParams(type, params)
                                        color: DesignSystem.Palette.textSecondary
                                        font.pixelSize: DesignSystem.Typography.caption
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        Button {
                                            text: qsTr("Konfiguruj blok")
                                            onClicked: root.openBlockEditor(index)
                                        }

                                        Item { Layout.fillWidth: true }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: blockEditorDialog
        modal: true
        focus: true
        title: qsTr("Parametry bloku")
        standardButtons: Dialog.Save | Dialog.Cancel
        closePolicy: Popup.CloseOnEscape
        implicitWidth: 420
        onAccepted: commitBlockEdit()
        onRejected: cancelBlockEdit()
        onClosed: cancelBlockEdit()

        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 12

            Label {
                text: qsTr("Etykieta bloku")
                color: DesignSystem.Palette.textSecondary
                font.pixelSize: DesignSystem.Typography.caption
            }

            TextField {
                Layout.fillWidth: true
                text: root.editingLabel
                onTextChanged: root.editingLabel = text
            }

            Repeater {
                model: root.editingTemplate
                delegate: ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    property var entry: modelData
                    property string key: entry.key
                    property var options: blockParameterChoices.hasOwnProperty(entry.key)
                                        ? blockParameterChoices[entry.key]
                                        : (entry.options ? entry.options.map(function(v) { return { value: v, label: v } }) : [])

                    Label {
                        text: blockParameterLabels[key] || key
                        color: DesignSystem.Palette.textSecondary
                        font.pixelSize: DesignSystem.Typography.caption
                    }

                    Loader {
                        Layout.fillWidth: true
                        property var templateEntry: entry
                        property var templateOptions: options
                        sourceComponent: {
                            if (entry.type === "enum")
                                return enumEditor
                            if (entry.type === "number")
                                return numberEditor
                            if (entry.type === "boolean")
                                return booleanEditor
                            return stringEditor
                        }
                    }
                }
            }
        }

        Component {
            id: stringEditor
            TextField {
                Layout.fillWidth: true
                text: root.editingValues[parent.templateEntry.key] !== undefined ? root.editingValues[parent.templateEntry.key] : ""
                onTextChanged: root.editingValues[parent.templateEntry.key] = text
            }
        }

        Component {
            id: numberEditor
            TextField {
                Layout.fillWidth: true
                text: root.editingValues[parent.templateEntry.key] !== undefined ? String(root.editingValues[parent.templateEntry.key]) : ""
                inputMethodHints: Qt.ImhFormattedNumbersOnly
                validator: DoubleValidator {
                    bottom: parent.templateEntry.min !== undefined ? parent.templateEntry.min : -1e9
                    top: parent.templateEntry.max !== undefined ? parent.templateEntry.max : 1e9
                }
                onEditingFinished: {
                    var value = Number(text)
                    if (Number.isNaN(value))
                        value = parent.templateEntry.defaultValue !== undefined ? parent.templateEntry.defaultValue : 0
                    root.editingValues[parent.templateEntry.key] = value
                }
                onTextChanged: {
                    if (text.length === 0)
                        return
                    var value = Number(text)
                    if (!Number.isNaN(value))
                        root.editingValues[parent.templateEntry.key] = value
                }
            }
        }

        Component {
            id: enumEditor
            ComboBox {
                Layout.fillWidth: true
                model: parent.templateOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: {
                    var options = parent.templateOptions || []
                    var value = root.editingValues[parent.templateEntry.key]
                    for (var i = 0; i < options.length; ++i) {
                        if (options[i].value === value)
                            return i
                    }
                    return 0
                }
                onActivated: {
                    var opt = parent.templateOptions[index]
                    root.editingValues[parent.templateEntry.key] = opt ? opt.value : parent.templateEntry.defaultValue
                }
            }
        }

        Component {
            id: booleanEditor
            Switch {
                checked: root.editingValues[parent.templateEntry.key] !== undefined ? root.editingValues[parent.templateEntry.key] : !!parent.templateEntry.defaultValue
                text: checked ? qsTr("Tak") : qsTr("Nie")
                onToggled: root.editingValues[parent.templateEntry.key] = checked
            }
        }
    }
}
