import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    property var activityLog: []

    Layout.fillWidth: true
    Layout.preferredHeight: 260

    background: Rectangle {
        color: Qt.darker(palette.base, 1.05)
        radius: 8
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            text: qsTr("Dziennik aktywności")
            font.bold: true
            Layout.alignment: Qt.AlignLeft
        }

        ListView {
            id: logView
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: 200
            clip: true
            model: Array.isArray(root.activityLog) ? root.activityLog : []
            interactive: false
            boundsBehavior: Flickable.StopAtBounds
            spacing: 8

            delegate: Item {
                width: logView.width
                implicitHeight: entryLayout.implicitHeight + 8

                ColumnLayout {
                    id: entryLayout
                    anchors.fill: parent
                    spacing: 4

                    RowLayout {
                        spacing: 8
                        Layout.alignment: Qt.AlignLeft

                        Rectangle {
                            width: 10
                            height: 10
                            radius: 5
                            color: modelData.success ? palette.highlight : palette.negative
                            Layout.alignment: Qt.AlignVCenter
                        }

                        Label {
                            text: Utils.formatTimestamp(modelData.timestamp)
                            font.pixelSize: 12
                            color: palette.mid
                        }

                        Label {
                            text: modelData.type || qsTr("aktywność")
                            font.pixelSize: 12
                            color: palette.mid
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }

                    Label {
                        text: modelData.message
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Label {
                        text: modelData.details && modelData.details.context
                              ? modelData.details.context
                              : (modelData.details && modelData.details.note ? modelData.details.note : "")
                        visible: text.length > 0
                        font.pixelSize: 12
                        color: palette.mid
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }

            footer: Item {
                width: logView.width
                height: logView.count > 0 ? 0 : 40
                Label {
                    anchors.centerIn: parent
                    text: qsTr("Brak aktywności do wyświetlenia")
                    color: palette.mid
                }
            }
        }
    }
}
