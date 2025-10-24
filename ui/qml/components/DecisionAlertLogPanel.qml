import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "."

Item {
    id: root

    property var alertsModel: null
    property var alertsFilterModel: null
    property var decisionModel: null
    property var decisionFilterModel: null
    property bool _syncingControls: false

    signal exportCompleted(url file)

    function approvalOptions() {
        return [
            { text: qsTr("Wszystkie"), value: 0 },
            { text: qsTr("Tylko zatwierdzone"), value: 1 },
            { text: qsTr("Tylko oczekujące"), value: 2 }
        ]
    }

    function sideOptions() {
        return [
            { text: qsTr("Dowolna strona"), value: "" },
            { text: qsTr("Kupno"), value: "BUY" },
            { text: qsTr("Sprzedaż"), value: "SELL" }
        ]
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Eksportuj dziennik decyzji")
        nameFilters: [qsTr("Pliki CSV (*.csv)"), qsTr("Wszystkie pliki (*)")]
        fileMode: FileDialog.SaveFile
        onAccepted: {
            if (decisionFilterModel && typeof decisionFilterModel.exportFilteredToCsv === "function") {
                const ok = decisionFilterModel.exportFilteredToCsv(exportDialog.currentFile)
                if (ok)
                    root.exportCompleted(exportDialog.currentFile)
            }
        }
    }

    function updateDecisionFilterControls() {
        if (root._syncingControls)
            return

        root._syncingControls = true

        if (!decisionFilterModel) {
            searchField.text = ""
            strategyField.text = ""
            regimeField.text = ""
            telemetryNamespaceField.text = ""
            environmentField.text = ""
            portfolioField.text = ""
            riskProfileField.text = ""
            scheduleField.text = ""
            quantityField.text = ""
            priceField.text = ""
            minQuantityField.text = ""
            maxQuantityField.text = ""
            minPriceField.text = ""
            maxPriceField.text = ""
            symbolField.text = ""
            decisionStateField.text = ""
            decisionModeField.text = ""
            decisionReasonField.text = ""
            eventField.text = ""
            detailsField.text = ""
            approvalCombo.currentIndex = 0
            sideCombo.currentIndex = 0
            startFilterCheck.checked = false
            endFilterCheck.checked = false
            root._syncingControls = false
            return
        }

        const search = decisionFilterModel.searchText || ""
        if (searchField.text !== search)
            searchField.text = search

        const strategy = decisionFilterModel.strategyFilter || ""
        if (strategyField.text !== strategy)
            strategyField.text = strategy

        const regime = decisionFilterModel.regimeFilter || ""
        if (regimeField.text !== regime)
            regimeField.text = regime

        const telemetryNamespace = decisionFilterModel.telemetryNamespaceFilter || ""
        if (telemetryNamespaceField.text !== telemetryNamespace)
            telemetryNamespaceField.text = telemetryNamespace

        const environment = decisionFilterModel.environmentFilter || ""
        if (environmentField.text !== environment)
            environmentField.text = environment

        const portfolio = decisionFilterModel.portfolioFilter || ""
        if (portfolioField.text !== portfolio)
            portfolioField.text = portfolio

        const riskProfile = decisionFilterModel.riskProfileFilter || ""
        if (riskProfileField.text !== riskProfile)
            riskProfileField.text = riskProfile

        const schedule = decisionFilterModel.scheduleFilter || ""
        if (scheduleField.text !== schedule)
            scheduleField.text = schedule

        const quantity = decisionFilterModel.quantityFilter || ""
        if (quantityField.text !== quantity)
            quantityField.text = quantity

        const price = decisionFilterModel.priceFilter || ""
        if (priceField.text !== price)
            priceField.text = price

        const minQuantity = decisionFilterModel.minQuantityFilter
        const minQuantityText = (minQuantity === undefined || minQuantity === null) ? "" : String(minQuantity)
        if (minQuantityField.text !== minQuantityText)
            minQuantityField.text = minQuantityText

        const maxQuantity = decisionFilterModel.maxQuantityFilter
        const maxQuantityText = (maxQuantity === undefined || maxQuantity === null) ? "" : String(maxQuantity)
        if (maxQuantityField.text !== maxQuantityText)
            maxQuantityField.text = maxQuantityText

        const minPrice = decisionFilterModel.minPriceFilter
        const minPriceText = (minPrice === undefined || minPrice === null) ? "" : String(minPrice)
        if (minPriceField.text !== minPriceText)
            minPriceField.text = minPriceText

        const maxPrice = decisionFilterModel.maxPriceFilter
        const maxPriceText = (maxPrice === undefined || maxPrice === null) ? "" : String(maxPrice)
        if (maxPriceField.text !== maxPriceText)
            maxPriceField.text = maxPriceText

        const symbol = decisionFilterModel.symbolFilter || ""
        if (symbolField.text !== symbol)
            symbolField.text = symbol

        const state = decisionFilterModel.decisionStateFilter || ""
        if (decisionStateField.text !== state)
            decisionStateField.text = state

        const mode = decisionFilterModel.decisionModeFilter || ""
        if (decisionModeField.text !== mode)
            decisionModeField.text = mode

        const reason = decisionFilterModel.decisionReasonFilter || ""
        if (decisionReasonField.text !== reason)
            decisionReasonField.text = reason

        const eventName = decisionFilterModel.eventFilter || ""
        if (eventField.text !== eventName)
            eventField.text = eventName

        const details = decisionFilterModel.detailsFilter || ""
        if (detailsField.text !== details)
            detailsField.text = details

        const approval = typeof decisionFilterModel.approvalFilter === "number"
                ? decisionFilterModel.approvalFilter : 0
        let comboIndex = 0
        for (let i = 0; i < approvalCombo.model.length; ++i) {
            if (approvalCombo.model[i].value === approval) {
                comboIndex = i
                break
            }
        }
        if (approvalCombo.currentIndex !== comboIndex)
            approvalCombo.currentIndex = comboIndex

        const sideValue = decisionFilterModel.sideFilter || ""
        let sideIndex = 0
        for (let i = 0; i < sideCombo.model.length; ++i) {
            if (sideCombo.model[i].value === sideValue) {
                sideIndex = i
                break
            }
        }
        if (sideCombo.currentIndex !== sideIndex)
            sideCombo.currentIndex = sideIndex

        const start = decisionFilterModel.startTimeFilter
        const hasStart = !!(start && start.isValid && start.isValid())
        if (startFilterCheck.checked !== hasStart)
            startFilterCheck.checked = hasStart
        if (hasStart) {
            const startLocal = start.toLocalTime ? start.toLocalTime() : start
            const startDate = new Date(startLocal.toMSecsSinceEpoch())
            if (!startDatePicker.dateTime || Math.abs(startDatePicker.dateTime.getTime() - startDate.getTime()) > 1)
                startDatePicker.dateTime = startDate
        }

        const end = decisionFilterModel.endTimeFilter
        const hasEnd = !!(end && end.isValid && end.isValid())
        if (endFilterCheck.checked !== hasEnd)
            endFilterCheck.checked = hasEnd
        if (hasEnd) {
            const endLocal = end.toLocalTime ? end.toLocalTime() : end
            const endDate = new Date(endLocal.toMSecsSinceEpoch())
            if (!endDatePicker.dateTime || Math.abs(endDatePicker.dateTime.getTime() - endDate.getTime()) > 1)
                endDatePicker.dateTime = endDate
        }

        root._syncingControls = false
    }

    onDecisionFilterModelChanged: updateDecisionFilterControls()

    Connections {
        target: decisionFilterModel
        function onFilterChanged() {
            updateDecisionFilterControls()
        }
    }

    Component.onCompleted: updateDecisionFilterControls()

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Frame {
            SplitView.preferredWidth: 360
            Layout.fillHeight: true
            padding: 12
            background: Rectangle {
                color: Qt.darker(palette.window, 1.2)
                radius: 8
            }

            AlertCenterPanel {
                anchors.fill: parent
                summaryModel: alertsModel
                listModel: alertsFilterModel
            }
        }

        Frame {
            SplitView.fillWidth: true
            Layout.fillHeight: true
            padding: 12
            background: Rectangle {
                color: Qt.darker(palette.window, 1.15)
                radius: 8
            }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: qsTr("Dziennik decyzji")
                        font.bold: true
                        font.pixelSize: 18
                    }
                    Item { Layout.fillWidth: true }
                    Button {
                        text: qsTr("Wyczyść filtry")
                        icon.name: "edit-clear"
                        enabled: !!decisionFilterModel
                        onClicked: {
                            if (!decisionFilterModel)
                                return
                            decisionFilterModel.clearAllFilters()
                        }
                    }
                    Button {
                        text: qsTr("Eksportuj CSV")
                        icon.name: "document-save"
                        onClicked: exportDialog.open()
                        enabled: decisionFilterModel && decisionFilterModel.rowCount !== undefined && decisionFilterModel.rowCount() > 0
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: searchField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Szukaj decyzji…")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel && decisionFilterModel.searchText !== text)
                                decisionFilterModel.searchText = text
                        }
                    }

                    ComboBox {
                        id: approvalCombo
                        model: approvalOptions()
                        textRole: "text"
                        valueRole: "value"
                        onActivated: {
                            if (root._syncingControls)
                                return
                            const opt = model[index]
                            if (decisionFilterModel && opt)
                                decisionFilterModel.approvalFilter = opt.value
                        }
                    }

                    ComboBox {
                        id: sideCombo
                        Layout.preferredWidth: 140
                        model: sideOptions()
                        textRole: "text"
                        valueRole: "value"
                        onActivated: {
                            if (root._syncingControls)
                                return
                            const opt = model[index]
                            if (decisionFilterModel)
                                decisionFilterModel.sideFilter = opt ? opt.value : ""
                        }
                    }

                    TextField {
                        id: environmentField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Środowisko")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.environmentFilter = text
                        }
                    }

                    TextField {
                        id: portfolioField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Portfel")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.portfolioFilter = text
                        }
                    }

                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: decisionStateField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Stan decyzji")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.decisionStateFilter = text
                        }
                    }

                    TextField {
                        id: decisionModeField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Tryb")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.decisionModeFilter = text
                        }
                    }

                    TextField {
                        id: decisionReasonField
                        Layout.preferredWidth: 200
                        placeholderText: qsTr("Powód decyzji")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.decisionReasonFilter = text
                        }
                    }

                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: eventField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Zdarzenie")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.eventFilter = text
                        }
                    }

                    TextField {
                        id: detailsField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Szczegóły")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.detailsFilter = text
                        }
                    }

                    TextField {
                        id: quantityField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Wolumen")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.quantityFilter = text
                        }
                    }

                    TextField {
                        id: priceField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Cena")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.priceFilter = text
                        }
                    }

                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: minQuantityField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Min wolumen")
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        validator: DoubleValidator { notation: DoubleValidator.StandardNotation }
                        onEditingFinished: {
                            if (root._syncingControls)
                                return
                            if (!decisionFilterModel)
                                return
                            const trimmed = text.trim()
                            if (!trimmed.length) {
                                decisionFilterModel.minQuantityFilter = undefined
                                return
                            }
                            const value = Number(trimmed)
                            if (!Number.isNaN(value))
                                decisionFilterModel.minQuantityFilter = value
                        }
                    }

                    TextField {
                        id: maxQuantityField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Max wolumen")
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        validator: DoubleValidator { notation: DoubleValidator.StandardNotation }
                        onEditingFinished: {
                            if (root._syncingControls)
                                return
                            if (!decisionFilterModel)
                                return
                            const trimmed = text.trim()
                            if (!trimmed.length) {
                                decisionFilterModel.maxQuantityFilter = undefined
                                return
                            }
                            const value = Number(trimmed)
                            if (!Number.isNaN(value))
                                decisionFilterModel.maxQuantityFilter = value
                        }
                    }

                    TextField {
                        id: minPriceField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Min cena")
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        validator: DoubleValidator { notation: DoubleValidator.StandardNotation }
                        onEditingFinished: {
                            if (root._syncingControls)
                                return
                            if (!decisionFilterModel)
                                return
                            const trimmed = text.trim()
                            if (!trimmed.length) {
                                decisionFilterModel.minPriceFilter = undefined
                                return
                            }
                            const value = Number(trimmed)
                            if (!Number.isNaN(value))
                                decisionFilterModel.minPriceFilter = value
                        }
                    }

                    TextField {
                        id: maxPriceField
                        Layout.preferredWidth: 120
                        placeholderText: qsTr("Max cena")
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        validator: DoubleValidator { notation: DoubleValidator.StandardNotation }
                        onEditingFinished: {
                            if (root._syncingControls)
                                return
                            if (!decisionFilterModel)
                                return
                            const trimmed = text.trim()
                            if (!trimmed.length) {
                                decisionFilterModel.maxPriceFilter = undefined
                                return
                            }
                            const value = Number(trimmed)
                            if (!Number.isNaN(value))
                                decisionFilterModel.maxPriceFilter = value
                        }
                    }

                    TextField {
                        id: riskProfileField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Profil ryzyka")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.riskProfileFilter = text
                        }
                    }

                    TextField {
                        id: scheduleField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Harmonogram")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.scheduleFilter = text
                        }
                    }

                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: strategyField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Strategia")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.strategyFilter = text
                        }
                    }

                    TextField {
                        id: regimeField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Reżim")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.regimeFilter = text
                        }
                    }

                    TextField {
                        id: telemetryNamespaceField
                        Layout.preferredWidth: 200
                        placeholderText: qsTr("Przestrzeń telemetryczna")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.telemetryNamespaceFilter = text
                        }
                    }

                    TextField {
                        id: symbolField
                        Layout.preferredWidth: 160
                        placeholderText: qsTr("Symbol")
                        onTextChanged: {
                            if (root._syncingControls)
                                return
                            if (decisionFilterModel)
                                decisionFilterModel.symbolFilter = text
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    CheckBox {
                        id: startFilterCheck
                        text: qsTr("Od")
                        onToggled: {
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            if (checked) {
                                decisionFilterModel.startTimeFilter = startDatePicker.dateTime
                            } else {
                                decisionFilterModel.clearStartTimeFilter()
                            }
                        }
                    }

                    DateTimeEdit {
                        id: startDatePicker
                        Layout.fillWidth: true
                        enabled: startFilterCheck.checked
                        displayFormat: "yyyy-MM-dd HH:mm"
                        onDateTimeChanged: {
                            if (root._syncingControls || !decisionFilterModel || !startFilterCheck.checked)
                                return
                            decisionFilterModel.startTimeFilter = dateTime
                        }
                    }

                    ToolButton {
                        icon.name: "view-refresh"
                        display: AbstractButton.IconOnly
                        enabled: startFilterCheck.checked
                        onClicked: {
                            const now = new Date()
                            startDatePicker.dateTime = now
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            decisionFilterModel.startTimeFilter = now
                        }
                    }

                    ToolButton {
                        icon.name: "edit-clear"
                        display: AbstractButton.IconOnly
                        enabled: startFilterCheck.checked
                        onClicked: {
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            decisionFilterModel.clearStartTimeFilter()
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    CheckBox {
                        id: endFilterCheck
                        text: qsTr("Do")
                        onToggled: {
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            if (checked) {
                                decisionFilterModel.endTimeFilter = endDatePicker.dateTime
                            } else {
                                decisionFilterModel.clearEndTimeFilter()
                            }
                        }
                    }

                    DateTimeEdit {
                        id: endDatePicker
                        Layout.fillWidth: true
                        enabled: endFilterCheck.checked
                        displayFormat: "yyyy-MM-dd HH:mm"
                        onDateTimeChanged: {
                            if (root._syncingControls || !decisionFilterModel || !endFilterCheck.checked)
                                return
                            decisionFilterModel.endTimeFilter = dateTime
                        }
                    }

                    ToolButton {
                        icon.name: "view-refresh"
                        display: AbstractButton.IconOnly
                        enabled: endFilterCheck.checked
                        onClicked: {
                            const now = new Date()
                            endDatePicker.dateTime = now
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            decisionFilterModel.endTimeFilter = now
                        }
                    }

                    ToolButton {
                        icon.name: "edit-clear"
                        display: AbstractButton.IconOnly
                        enabled: endFilterCheck.checked
                        onClicked: {
                            if (root._syncingControls || !decisionFilterModel)
                                return
                            decisionFilterModel.clearEndTimeFilter()
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: Qt.rgba(1, 1, 1, 0.1)
                }

                ListView {
                    id: decisionView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 6
                    model: decisionFilterModel
                    delegate: Frame {
                        required property string timestampDisplay
                        required property string strategy
                        required property string symbol
                        required property string side
                        required property string decisionState
                        required property string decisionReason
                        required property bool approved
                        required property string decisionMode

                        Layout.fillWidth: true
                        padding: 10
                        background: Rectangle {
                            radius: 6
                            color: approved ? Qt.rgba(0.16, 0.32, 0.22, 0.7) : Qt.rgba(0.32, 0.18, 0.18, 0.7)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 4

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: timestampDisplay
                                    color: Qt.rgba(0.82, 0.84, 0.88, 1)
                                }
                                Label {
                                    text: strategy
                                    font.bold: true
                                }
                                Label {
                                    text: symbol + " (" + side + ")"
                                    color: Qt.rgba(0.86, 0.88, 0.92, 1)
                                }
                                Item { Layout.fillWidth: true }
                                Label {
                                    text: decisionMode
                                    color: Qt.rgba(0.76, 0.78, 0.82, 1)
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Label {
                                    text: qsTr("Stan: %1").arg(decisionState)
                                    color: Qt.rgba(0.86, 0.88, 0.92, 1)
                                }
                                Label {
                                    text: decisionReason
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                    PlaceholderMessage {
                        anchors.centerIn: parent
                        text: qsTr("Brak wpisów pasujących do filtra")
                        visible: !decisionFilterModel || decisionFilterModel.rowCount === undefined || decisionFilterModel.rowCount() === 0
                    }
                }
            }
        }
    }
}
