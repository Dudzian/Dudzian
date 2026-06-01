import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components

Item {
    id: root
    objectName: "operatorDashboardRoot"
    property var designSystem
    property bool defaultDashboard: true
    implicitWidth: 900
    implicitHeight: 620

    function statusAccent(token) {
        if (!root.designSystem)
            return "#f5f7ff"
        switch (token) {
        case "accent":
            return root.designSystem.color("accent")
        case "warning":
            return root.designSystem.color("warning")
        default:
            return root.designSystem.color("textPrimary")
        }
    }

    Rectangle {
        anchors.fill: parent
        radius: 24
        color: designSystem ? designSystem.color("surface") : Qt.rgba(0.12, 0.12, 0.12, 1)
        border.color: designSystem ? designSystem.color("border") : Qt.rgba(1, 1, 1, 0.08)
        border.width: 1
    }

    Components.StyledScrollView {
        id: dashboardScroll
        designSystem: root.designSystem
        anchors.fill: parent
        anchors.margins: 18
        clip: true
        contentWidth: availableWidth

        ColumnLayout {
            id: dashboardContent
            width: dashboardScroll.availableWidth
            spacing: 14

            Label {
                objectName: "operatorDashboardTitle"
                text: qsTr("Dashboard operatora")
                font.bold: true
                font.pointSize: 22
                wrapMode: Text.WordWrap
                color: root.designSystem ? root.designSystem.color("textPrimary") : "white"
                Layout.fillWidth: true
            }

            Label {
                text: qsTr("Tryb demo/offline — podłączony lokalny preview bridge. Live trading pozostaje wyłączony.")
                wrapMode: Text.WordWrap
                color: root.designSystem ? root.designSystem.color("textSecondary") : "#cbd5e1"
                Layout.fillWidth: true
            }

            GridLayout {
                objectName: "operatorDashboardSafetySummary"
                Layout.fillWidth: true
                columns: 2
                columnSpacing: 12
                rowSpacing: 12

                Repeater {
                    model: [
                        { title: qsTr("Tryb: Demo / Paper"), lines: [qsTr("Endpoint: in-process"), qsTr("Cloud runtime: wyłączony")], accent: "textPrimary" },
                        { title: qsTr("Exchange I/O disabled"), lines: [qsTr("Order submission disabled"), qsTr("Runtime loop not started")], accent: "warning" },
                        { title: qsTr("API keys required: false"), lines: [qsTr("Active strategy: Demo Momentum Guard"), qsTr("Last decision: HOLD / NO ORDER")], accent: "accent" },
                        { title: qsTr("Live trading: blocked / disabled"), lines: [qsTr("Live disabled"), qsTr("Kill switch: armed / preview")], accent: "warning" }
                    ]
                    delegate: Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 136
                        radius: 16
                        color: root.designSystem ? root.designSystem.color("surfaceMuted") : Qt.rgba(1, 1, 1, 0.06)
                        border.color: root.designSystem ? root.designSystem.color("border") : Qt.rgba(1, 1, 1, 0.1)
                        border.width: 1

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 6

                            Label {
                                text: modelData.title
                                font.bold: true
                                wrapMode: Text.WordWrap
                                color: root.statusAccent(modelData.accent)
                                Layout.fillWidth: true
                            }
                            Repeater {
                                model: modelData.lines
                                delegate: Label {
                                    text: modelData
                                    wrapMode: Text.WordWrap
                                    color: root.designSystem ? root.designSystem.color("textSecondary") : "#cbd5e1"
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }
            }

            Rectangle {
                objectName: "operatorDashboardFeed"
                Layout.fillWidth: true
                Layout.preferredHeight: feedColumn.implicitHeight + 28
                radius: 16
                color: root.designSystem ? root.designSystem.color("surfaceMuted") : Qt.rgba(1, 1, 1, 0.06)
                border.color: root.designSystem ? root.designSystem.color("border") : Qt.rgba(1, 1, 1, 0.1)
                border.width: 1

                ColumnLayout {
                    id: feedColumn
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8

                    Label {
                        text: qsTr("Demo feed")
                        font.bold: true
                        color: root.designSystem ? root.designSystem.color("textPrimary") : "white"
                    }
                    Label { text: qsTr("BTC/USDT demo row | HOLD | confidence 0.62 | no order"); color: root.designSystem ? root.designSystem.color("textPrimary") : "white"; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    Label { text: qsTr("ETH/USDT demo row | WAIT | confidence 0.55 | no order"); color: root.designSystem ? root.designSystem.color("textPrimary") : "white"; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    Label { text: qsTr("SOL/USDT demo row | BLOCKED LIVE | reason: demo mode"); color: root.designSystem ? root.designSystem.color("warning") : "#fbbf24"; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                }
            }

            Rectangle {
                objectName: "operatorDashboardRiskControls"
                Layout.fillWidth: true
                Layout.preferredHeight: riskColumn.implicitHeight + 28
                radius: 16
                color: root.designSystem ? root.designSystem.color("surfaceMuted") : Qt.rgba(1, 1, 1, 0.06)
                border.color: root.designSystem ? root.designSystem.color("border") : Qt.rgba(1, 1, 1, 0.1)
                border.width: 1

                ColumnLayout {
                    id: riskColumn
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8

                    Label {
                        text: qsTr("Kontrola ryzyka")
                        font.bold: true
                        color: root.designSystem ? root.designSystem.color("textPrimary") : "white"
                    }
                    Label { text: qsTr("Live disabled"); color: root.designSystem ? root.designSystem.color("warning") : "#fbbf24" }
                    Label { text: qsTr("Max drawdown guard: demo only"); color: root.designSystem ? root.designSystem.color("textSecondary") : "#cbd5e1" }
                    Label { text: qsTr("Kill switch: armed / preview"); color: root.designSystem ? root.designSystem.color("accent") : "#22c55e" }
                }
            }
        }
    }
}
