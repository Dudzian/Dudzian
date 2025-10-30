import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

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

    function resolvedController() {
        if (root.appController)
            return root.appController
        if (typeof appController !== "undefined")
            return appController
        return null
    }

    function refreshPresets() {
        const ctrl = resolvedController()
        if (!ctrl || !ctrl.marketplaceListPresets) {
            statusError = qsTr("Brak połączenia z usługą marketplace.")
            statusMessage = ""
            presets = []
            return
        }
        busy = true
        try {
            const result = ctrl.marketplaceListPresets()
            presets = Array.isArray(result) ? result : []
            statusMessage = qsTr("Załadowano %1 presetów").arg(presets.length)
            statusError = ""
        } catch (error) {
            statusError = error ? error.toString() : qsTr("Nieznany błąd podczas pobierania presetów.")
            statusMessage = ""
            presets = []
        }
        busy = false
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

    function triggerImport() {
        importDialog.open()
    }

    function triggerExport(preset) {
        selectedPreset = preset
        exportDialog.currentFile = preset && preset.name ? preset.name + "." + exportFormat : "preset." + exportFormat
        exportDialog.open()
    }

    Component.onCompleted: refreshPresets()

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
                            }
                        }

                        Label {
                            visible: preset.signatureVerified === false
                            text: qsTr("UWAGA: podpis niezweryfikowany")
                            color: "#c0392b"
                            font.bold: true
                        }

                        Label {
                            visible: preset.issues && preset.issues.length > 0
                            text: qsTr("Problemy: %1").arg((preset.issues || []).join(", "))
                            color: "#c0392b"
                            wrapMode: Text.WordWrap
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
