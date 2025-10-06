import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

ApplicationWindow {
    id: window
    property alias performanceGuard: guardModel.guard
    property alias sidePanel: sidePanel
    property alias chartView: chartView
    title: qsTr("Bot Trading Shell")
    minimumWidth: 960
    minimumHeight: 540
    visible: true

    color: Qt.darker(palette.window, 1.05)

    readonly property color accentColor: Qt.rgba(0.14, 0.58, 0.82, 1)

    header: ToolBar {
        contentHeight: 48
        RowLayout {
            anchors.fill: parent
            spacing: 12

            Label {
                text: qsTr("Connection: %1").arg(appController.connectionStatus)
                font.bold: true
                Layout.alignment: Qt.AlignVCenter
            }

            Rectangle {
                width: 1
                color: palette.mid
                height: parent.height * 0.6
                Layout.margins: 8
            }

            Label {
                text: qsTr("FPS target: %1").arg(performanceGuard.fpsTarget)
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Reconnect")
                onClicked: {
                    appController.stop()
                    appController.start()
                }
            }
        }
    }

    background: Rectangle {
        color: Qt.darker(palette.window, 1.15)
    }

    Item {
        id: guardModel
        property PerformanceGuard guard: appController.performanceGuard

        Connections {
            target: appController
            function onPerformanceGuardChanged() {
                guard = appController.performanceGuard
            }
        }
    }

    Item {
        anchors.fill: parent
        anchors.topMargin: header.height
        anchors.bottomMargin: footer.implicitHeight

        RowLayout {
            anchors.fill: parent
            spacing: 0

            SidePanel {
                id: sidePanel
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                performanceGuard: guardModel.guard
            }

            Rectangle {
                width: 1
                color: Qt.darker(palette.window, 1.2)
                Layout.fillHeight: true
            }

            CandlestickChartView {
                id: chartView
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: ohlcvModel
                performanceGuard: guardModel.guard
            }
        }
    }

    footer: StatusFooter {
        id: footer
        Layout.fillWidth: true
    }
}
