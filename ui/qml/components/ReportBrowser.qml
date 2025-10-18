import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property var controller
    property var selectedReport: null

    function ensureSelection() {
        if (!controller)
            return
        const reports = controller.reports || []
        if (!selectedReport)
            return
        const selectedPath = selectedReport.path || selectedReport.relative_path
        for (let index = 0; index < reports.length; ++index) {
            const entry = reports[index]
            if (!entry)
                continue
            if (entry.path === selectedPath || entry.relative_path === selectedPath) {
                selectedReport = entry
                reportList.currentIndex = index
                return
            }
        }
        selectedReport = null
        reportList.currentIndex = -1
    }

    Connections {
        target: controller
        function onReportsChanged() {
            root.ensureSelection()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Dostępne raporty: %1").arg((controller && controller.reports ? controller.reports.length : 0))
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Odśwież")
                enabled: controller && !controller.busy
                onClicked: controller && controller.refresh()
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            ListView {
                id: reportList
                Layout.preferredWidth: parent.width * 0.45
                clip: true
                model: controller ? controller.reports : []
                currentIndex: -1

                delegate: Frame {
                    required property var modelData
                    required property int index
                    readonly property bool isSelected: root.selectedReport && root.selectedReport.path === modelData.path

                    Layout.fillWidth: true
                    padding: 8
                    background: Rectangle {
                        radius: 6
                        color: isSelected ? Qt.rgba(0.2, 0.5, 0.8, 0.25) : Qt.rgba(0.1, 0.1, 0.1, 0.08)
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 4

                        Label {
                            text: modelData.name || modelData.relative_path || modelData.path
                            font.bold: true
                        }

                        Label {
                            text: qsTr("Rozmiar: %1 B").arg(modelData.size_bytes || 0)
                            color: palette.mid
                            font.pointSize: font.pointSize - 1
                        }

                        Label {
                            text: qsTr("Zmodyfikowano: %1").arg(modelData.modified || qsTr("n/d"))
                            color: palette.mid
                            font.pointSize: font.pointSize - 1
                        }
                    }

                    TapHandler {
                        acceptedButtons: Qt.LeftButton
                        onTapped: {
                            root.selectedReport = modelData
                            reportList.currentIndex = index
                        }
                    }
                }
            }

            Frame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                padding: 12
                visible: root.selectedReport !== null

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    Label {
                        text: root.selectedReport ? (root.selectedReport.name || root.selectedReport.relative_path) : qsTr("Wybierz raport")
                        font.bold: true
                        font.pointSize: font.pointSize + 2
                    }

                    GridLayout {
                        columns: 2
                        columnSpacing: 8
                        rowSpacing: 6
                        Layout.fillWidth: true

                        Label { text: qsTr("Ścieżka:") }
                        Label { text: root.selectedReport ? root.selectedReport.path : ""; wrapMode: Text.WrapAnywhere }

                        Label { text: qsTr("Typ:") }
                        Label { text: root.selectedReport ? root.selectedReport.type : "" }

                        Label { text: qsTr("Rozmiar [B]:") }
                        Label { text: root.selectedReport ? root.selectedReport.size_bytes : "" }

                        Label { text: qsTr("Zmodyfikowano:") }
                        Label { text: root.selectedReport ? root.selectedReport.modified || qsTr("n/d") : "" }

                        Label { text: qsTr("Liczba wpisów:") }
                        Label { text: root.selectedReport ? (root.selectedReport.entries || 0) : "" }
                    }

                    Item { Layout.fillHeight: true }

                    RowLayout {
                        Layout.alignment: Qt.AlignRight
                        spacing: 8

                        Button {
                            text: qsTr("Usuń raport")
                            enabled: controller && !controller.busy && root.selectedReport
                            onClicked: deleteDialog.openWithReport(root.selectedReport)
                            icon.name: "edit-delete"
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: deleteDialog
        property var report

        function openWithReport(data) {
            report = data
            open()
        }

        title: qsTr("Potwierdź usunięcie")
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        onAccepted: {
            if (!controller || !report)
                return
            controller.deleteReport(report.path || report.relative_path)
        }

        contentItem: ColumnLayout {
            spacing: 12
            implicitWidth: 360

            Label {
                wrapMode: Text.WordWrap
                text: report
                    ? qsTr("Czy na pewno chcesz usunąć raport \"%1\" wraz z plikami?").arg(report.name || report.relative_path)
                    : qsTr("Brak wybranego raportu")
            }

            Label {
                visible: controller && controller.busy
                text: qsTr("Operacja w toku…")
                color: palette.mid
            }
        }
    }

    MessageDialog {
        id: errorDialog
        title: qsTr("Operacja nieudana")
        buttons: MessageDialog.Ok
    }

    Connections {
        target: controller
        function onReportOperationFailed(message) {
            errorDialog.text = message || qsTr("Nieznany błąd")
            errorDialog.open()
        }
    }
}
