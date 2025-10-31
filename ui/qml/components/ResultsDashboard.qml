import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    width: parent ? parent.width : 800
    height: column.implicitHeight

    property var model: (typeof resultsDashboard !== "undefined" ? resultsDashboard : null)

    ColumnLayout {
        id: column
        objectName: "resultsDashboardColumn"
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingMd

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
                    background: Rectangle {
                        radius: Styles.AppTheme.radiusMedium
                        color: Styles.AppTheme.cardBackground(0.9)
                    }
                    Column {
                        anchors.fill: parent
                        anchors.margins: Styles.AppTheme.spacingSm
                        spacing: Styles.AppTheme.spacingXs
                        Label {
                            text: modelData.label
                            font.pixelSize: Styles.AppTheme.fontSizeCaption
                            font.family: Styles.AppTheme.fontFamily
                            color: Styles.AppTheme.textSecondary
                        }
                        Label {
                            text: modelData.value
                            font.pixelSize: Styles.AppTheme.fontSizeTitle
                            font.family: Styles.AppTheme.fontFamily
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }
                    }
                }
            }
        }

        GroupBox {
            objectName: "resultsEquityGroup"
            Layout.fillWidth: true
            title: qsTr("Przebieg wartości portfela")
            background: Rectangle {
                radius: Styles.AppTheme.radiusLarge
                color: Styles.AppTheme.cardBackground(0.82)
            }

            ListView {
                id: timelineView
                anchors.fill: parent
                model: model ? model.equityTimeline : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: Styles.AppTheme.spacingMd
                    Label {
                        Layout.preferredWidth: 160
                        text: modelData.timestamp ? Qt.formatDateTime(new Date(modelData.timestamp), Qt.ISODate) : "--"
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Label {
                        Layout.preferredWidth: 120
                        text: Qt.formatLocaleNumber(modelData.portfolio, "f", 2)
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textPrimary
                    }
                    Label {
                        Layout.preferredWidth: 120
                        text: Qt.formatLocaleNumber(modelData.drawdown * 100, "f", 2) + "%"
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Label {
                        text: modelData.breach ? qsTr("naruszenie") : ""
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: modelData.breach ? Styles.AppTheme.negative : Styles.AppTheme.textTertiary
                        font.bold: modelData.breach
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }

        GroupBox {
            objectName: "resultsExposureGroup"
            Layout.fillWidth: true
            title: qsTr("Ekspozycje krytyczne")
            background: Rectangle {
                radius: Styles.AppTheme.radiusLarge
                color: Styles.AppTheme.cardBackground(0.82)
            }

            ListView {
                anchors.fill: parent
                model: model ? model.exposureHighlights() : []
                delegate: RowLayout {
                    width: parent.width
                    spacing: Styles.AppTheme.spacingMd
                    Label {
                        Layout.preferredWidth: 160
                        text: modelData.code
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textPrimary
                    }
                    Label {
                        Layout.preferredWidth: 120
                        text: Qt.formatLocaleNumber(modelData.current, "f", 2)
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Label {
                        Layout.preferredWidth: 120
                        text: Qt.formatLocaleNumber(modelData.threshold, "f", 2)
                        font.family: Styles.AppTheme.fontFamily
                        font.pixelSize: Styles.AppTheme.fontSizeBody
                        color: Styles.AppTheme.textSecondary
                    }
                    Rectangle {
                        width: 18
                        height: 18
                        radius: 9
                        color: modelData.breached ? Styles.AppTheme.negative : Styles.AppTheme.surfaceSubtle
                    }
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }
}
