import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    width: parent ? parent.width : 800
    height: column.implicitHeight

    property var model: (typeof resultsDashboard !== "undefined" ? resultsDashboard : null)

    ColumnLayout {
        id: column
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Repeater {
                model: [
                    { label: qsTr("Zwrot skumulowany"), value: model ? Qt.formatLocaleNumber(model.cumulativeReturn * 100, "f", 2) + "%" : "--" },
                    { label: qsTr("Maks. obsunięcie"), value: model ? Qt.formatLocaleNumber(model.maxDrawdown * 100, "f", 2) + "%" : "--" },
                    { label: qsTr("Sharpe"), value: model ? Qt.formatLocaleNumber(model.sharpeRatio, "f", 2) : "--" },
                    { label: qsTr("Volatility"), value: model ? Qt.formatLocaleNumber(model.annualizedVolatility * 100, "f", 2) + "%" : "--" },
                    { label: qsTr("Win rate"), value: model ? Qt.formatLocaleNumber(model.winRate * 100, "f", 2) + "%" : "--" }
                ]
                delegate: Frame {
                    Layout.fillWidth: true
                    Layout.preferredWidth: root.width / 5
                    background: Rectangle { radius: 10; color: Qt.rgba(1,1,1,0.06) }
                    Column {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        Label {
                            text: modelData.label
                            font.pixelSize: 12
                            color: palette.mid
                        }
                        Label {
                            text: modelData.value
                            font.pixelSize: 20
                            font.bold: true
                        }
                    }
                }
            }
        }

        GroupBox {
            Layout.fillWidth: true
            title: qsTr("Przebieg wartości portfela")
            background: Rectangle { radius: 12; color: Qt.rgba(1,1,1,0.04) }

            ListView {
                id: timelineView
                anchors.fill: parent
                model: model ? model.equityTimeline : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: 12
                    Label {
                        Layout.preferredWidth: 160
                        text: modelData.timestamp ? Qt.formatDateTime(new Date(modelData.timestamp), Qt.ISODate) : "--"
                    }
                    Label { Layout.preferredWidth: 120; text: Qt.formatLocaleNumber(modelData.portfolio, "f", 2); }
                    Label { Layout.preferredWidth: 120; text: Qt.formatLocaleNumber(modelData.drawdown * 100, "f", 2) + "%" }
                    Label {
                        text: modelData.breach ? qsTr("naruszenie") : ""
                        color: modelData.breach ? palette.negative : palette.mid
                        font.bold: modelData.breach
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }

        GroupBox {
            Layout.fillWidth: true
            title: qsTr("Ekspozycje krytyczne")
            background: Rectangle { radius: 12; color: Qt.rgba(1,1,1,0.04) }

            ListView {
                anchors.fill: parent
                model: model ? model.exposureHighlights() : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: 12
                    Label { Layout.preferredWidth: 160; text: modelData.code }
                    Label { Layout.preferredWidth: 120; text: Qt.formatLocaleNumber(modelData.current, "f", 2) }
                    Label { Layout.preferredWidth: 120; text: Qt.formatLocaleNumber(modelData.threshold, "f", 2) }
                    Rectangle {
                        width: 18
                        height: 18
                        radius: 9
                        color: modelData.breached ? palette.negative : palette.alternateBase
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }
}
