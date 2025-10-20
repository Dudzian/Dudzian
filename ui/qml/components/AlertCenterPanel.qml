import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var summaryModel: null
    property var listModel: summaryModel
    property bool _syncing: false

    function updateControlsFromModel() {
        root._syncing = true;
        if (!root.listModel || typeof root.listModel.severityFilter === "undefined") {
            severityCombo.currentIndex = 0;
            hideAckCheck.checked = false;
            searchField.text = "";
            sortCombo.currentIndex = 0;
            root._syncing = false;
            return;
        }
        const filter = root.listModel.severityFilter;
        for (let i = 0; i < severityCombo.model.length; ++i) {
            if (severityCombo.model[i].value === filter) {
                severityCombo.currentIndex = i;
                break;
            }
        }
        hideAckCheck.checked = root.listModel.hideAcknowledged;
        searchField.text = root.listModel && typeof root.listModel.searchText !== "undefined"
                ? root.listModel.searchText : "";
        if (root.listModel && typeof root.listModel.sortMode !== "undefined") {
            const sortMode = root.listModel.sortMode;
            for (let i = 0; i < sortCombo.model.length; ++i) {
                if (sortCombo.model[i].value === sortMode) {
                    sortCombo.currentIndex = i;
                    break;
                }
            }
        } else {
            sortCombo.currentIndex = 0;
        }
        root._syncing = false;
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        Label {
            text: qsTr("Aktywne alerty")
            font.pixelSize: 20
            font.bold: true
        }

        Label {
            visible: !root.summaryModel || root.summaryModel.count === 0
            text: qsTr("Brak aktywnych alertów bezpieczeństwa.")
            color: Qt.rgba(0.7, 0.72, 0.76, 1)
        }

        Label {
            visible: root.summaryModel && root.summaryModel.count > 0
            text: qsTr("Niepotwierdzone: %1").arg(root.summaryModel ? root.summaryModel.unacknowledgedCount : 0)
            color: root.summaryModel && root.summaryModel.hasUnacknowledgedAlerts
                   ? Qt.rgba(0.96, 0.68, 0.26, 1)
                   : Qt.rgba(0.76, 0.78, 0.82, 1)
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Widoczne: %1").arg(root.listModel ? root.listModel.count : 0)
                color: Qt.rgba(0.76, 0.78, 0.82, 1)
            }

            TextField {
                id: searchField
                Layout.fillWidth: true
                placeholderText: qsTr("Szukaj alertów…")
                enabled: root.listModel && typeof root.listModel.searchText !== "undefined"
                onTextChanged: {
                    if (root._syncing)
                        return
                    if (!root.listModel || typeof root.listModel.searchText === "undefined")
                        return
                    if (root.listModel.searchText === text)
                        return
                    root.listModel.searchText = text
                }
                inputMethodHints: Qt.ImhNoPredictiveText | Qt.ImhPreferLowercase
            }

            Item { Layout.fillWidth: true }

            ComboBox {
                id: sortCombo
                Layout.preferredWidth: 220
                enabled: root.listModel && typeof root.listModel.sortMode !== "undefined"
                model: [
                    { text: qsTr("Najnowsze najpierw"), value: 0 },
                    { text: qsTr("Najstarsze najpierw"), value: 1 },
                    { text: qsTr("Najwyższa waga"), value: 2 },
                    { text: qsTr("Najniższa waga"), value: 3 },
                    { text: qsTr("Tytuł A-Z"), value: 4 }
                ]
                textRole: "text"
                valueRole: "value"
                onActivated: {
                    if (root._syncing)
                        return
                    if (!root.listModel || typeof root.listModel.sortMode === "undefined")
                        return
                    const entry = model[index]
                    if (entry)
                        root.listModel.sortMode = entry.value
                }
            }

            ComboBox {
                id: severityCombo
                Layout.preferredWidth: 220
                enabled: root.listModel && typeof root.listModel.severityFilter !== "undefined"
                model: [
                    { text: qsTr("Wszystkie alerty"), value: 0 },
                    { text: qsTr("Ostrzeżenia i krytyczne"), value: 1 },
                    { text: qsTr("Tylko krytyczne"), value: 2 },
                    { text: qsTr("Tylko ostrzeżenia"), value: 3 }
                ]
                textRole: "text"
                valueRole: "value"
                onActivated: {
                    if (root._syncing)
                        return
                    if (!root.listModel || typeof root.listModel.severityFilter === "undefined")
                        return
                    const entry = model[index]
                    if (entry)
                        root.listModel.severityFilter = entry.value
                }
            }

            CheckBox {
                id: hideAckCheck
                text: qsTr("Ukryj potwierdzone")
                enabled: root.listModel && typeof root.listModel.hideAcknowledged !== "undefined"
                onToggled: {
                    if (root._syncing)
                        return
                    if (root.listModel && typeof root.listModel.hideAcknowledged !== "undefined")
                        root.listModel.hideAcknowledged = checked
                    else
                        checked = false
                }
            }
        }

        ListView {
            id: alertsView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 8
            model: root.listModel
            delegate: Frame {
                required property string id
                required property string title
                required property string description
                required property int severity
                required property bool acknowledged
                required property var timestamp

                Layout.fillWidth: true
                padding: 12
                background: Rectangle {
                    radius: 8
                    color: severity === 2
                               ? Qt.rgba(0.6, 0.07, 0.07, 0.65)
                               : severity === 1
                                     ? Qt.rgba(0.92, 0.6, 0.1, 0.5)
                                     : Qt.rgba(0.25, 0.3, 0.35, 0.4)
                    border.width: acknowledged ? 1 : 0
                    border.color: acknowledged ? Qt.rgba(0.8, 0.8, 0.85, 0.6) : "transparent"
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Label {
                            text: title
                            font.bold: true
                            color: acknowledged ? Qt.rgba(0.85, 0.86, 0.9, 0.8) : palette.windowText
                        }
                        Label {
                            Layout.fillWidth: true
                            visible: !!timestamp
                            text: timestamp ? Qt.formatDateTime(timestamp, "yyyy-MM-dd HH:mm:ss") : ""
                            color: Qt.rgba(0.76, 0.78, 0.82, 1)
                            horizontalAlignment: Text.AlignRight
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        text: description
                        color: acknowledged ? Qt.rgba(0.82, 0.84, 0.88, 0.8) : palette.windowText
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Button {
                            text: acknowledged ? qsTr("Potwierdzono") : qsTr("Potwierdź")
                            enabled: !acknowledged
                            onClicked: {
                                if (root.summaryModel)
                                    root.summaryModel.acknowledge(id)
                            }
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: severity === 2
                                      ? qsTr("Krytyczny")
                                      : severity === 1 ? qsTr("Ostrzeżenie") : qsTr("Informacja")
                            color: severity === 2
                                      ? Qt.rgba(1, 0.8, 0.8, 1)
                                      : severity === 1 ? Qt.rgba(1, 0.9, 0.7, 1) : Qt.rgba(0.8, 0.85, 0.9, 1)
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Potwierdź wszystkie")
                enabled: root.summaryModel && root.summaryModel.unacknowledgedCount > 0
                onClicked: {
                    if (root.summaryModel)
                        root.summaryModel.acknowledgeAll()
                }
            }

            Button {
                text: qsTr("Wyczyść potwierdzone")
                enabled: root.summaryModel && root.summaryModel.count > 0
                onClicked: {
                    if (root.summaryModel)
                        root.summaryModel.clearAcknowledged()
                }
            }
        }
    }

    Component.onCompleted: updateControlsFromModel()

    onListModelChanged: updateControlsFromModel()

    Connections {
        target: root.listModel && typeof root.listModel.filterChanged === "function" ? root.listModel : null
        function onFilterChanged() {
            root.updateControlsFromModel()
        }
    }
}
