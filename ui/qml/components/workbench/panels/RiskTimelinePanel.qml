import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "riskTimelinePanel"
    property var riskTimeline: []
    readonly property var latestEntry: riskTimeline && riskTimeline.length > 0 ? riskTimeline[0] : null

    Layout.fillWidth: true
    Layout.fillHeight: true
    background: Rectangle {
        color: Qt.darker(palette.window, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            Label {
                text: qsTr("Historia ryzyka")
                font.pointSize: 15
                font.bold: true
            }

            Label {
                text: root.latestEntry ? qsTr("Zaktualizowano: %1").arg(Utils.formatTimestamp(root.latestEntry.timestamp))
                                        : qsTr("Brak zapisanych próbek")
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                color: palette.mid
            }
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true
            visible: root.latestEntry !== null

            Label { text: qsTr("Profil"); font.bold: true }
            Label { text: root.latestEntry && root.latestEntry.profileLabel ? root.latestEntry.profileLabel : qsTr("Live") }

            Label { text: qsTr("Wartość portfela"); font.bold: true }
            Label { text: Utils.formatNumber(root.latestEntry ? root.latestEntry.portfolioValue : 0, 2) }

            Label { text: qsTr("Drawdown"); font.bold: true }
            Label { text: Utils.formatPercent(root.latestEntry ? root.latestEntry.drawdown : 0) }

            Label { text: qsTr("Ekspozycja"); font.bold: true }
            Label { text: Utils.formatPercent(root.latestEntry ? root.latestEntry.exposureUtilization : 0) }

            Label { text: qsTr("Dźwignia"); font.bold: true }
            Label { text: Utils.formatNumber(root.latestEntry ? root.latestEntry.leverage : 0, 2) }

            Label { text: qsTr("Naruszenia"); font.bold: true }
            Label {
                text: root.latestEntry && root.latestEntry.breach
                      ? qsTr("Tak (%1)").arg(root.latestEntry.breachCount || 1)
                      : qsTr("Nie")
                color: root.latestEntry && root.latestEntry.breach ? Qt.rgba(0.9, 0.4, 0.3, 1) : palette.mid
            }
        }

        Label {
            text: qsTr("Brak dostępnej historii. Odśwież dane lub uruchom odczyt demo.")
            color: palette.mid
            wrapMode: Text.WordWrap
            visible: root.latestEntry === null
        }

        ListView {
            id: timelineView
            Layout.fillWidth: true
            Layout.preferredHeight: 220
            clip: true
            spacing: 12
            visible: root.latestEntry !== null
            model: root.riskTimeline ? root.riskTimeline : []

            delegate: Item {
                width: timelineView.width
                height: container.implicitHeight

                ColumnLayout {
                    id: container
                    anchors.fill: parent
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: Utils.formatTimestamp(modelData.timestamp)
                            font.bold: true
                            Layout.fillWidth: true
                        }
                        Label {
                            text: Utils.formatNumber(modelData.portfolioValue, 2)
                            horizontalAlignment: Text.AlignRight
                            Layout.alignment: Qt.AlignRight
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Label {
                            text: qsTr("DD: %1").arg(Utils.formatPercent(modelData.drawdown))
                            color: palette.mid
                        }
                        Label {
                            text: qsTr("Ekspozycja: %1").arg(Utils.formatPercent(modelData.exposureUtilization))
                            color: palette.mid
                        }
                        Label {
                            text: qsTr("Dźwignia: %1").arg(Utils.formatNumber(modelData.leverage, 2))
                            color: palette.mid
                        }
                        Label {
                            text: modelData.breach
                                  ? qsTr("Naruszenia: %1").arg(modelData.breachCount || 1)
                                  : qsTr("Naruszenia: brak")
                            color: modelData.breach ? Qt.rgba(0.9, 0.4, 0.3, 1) : palette.mid
                        }
                    }

                    Label {
                        text: modelData.notes && modelData.notes.length > 0
                              ? modelData.notes
                              : qsTr("Brak dodatkowych notatek")
                        color: palette.mid
                        wrapMode: Text.WordWrap
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: Qt.rgba(1, 1, 1, 0.1)
                        visible: index < (timelineView.count - 1)
                    }
                }
            }

            ScrollBar.vertical: ScrollBar { }
        }
    }
}
