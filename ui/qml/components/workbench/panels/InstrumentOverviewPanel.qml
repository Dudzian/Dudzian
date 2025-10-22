import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import "../StrategyWorkbenchUtils.js" as Utils

Frame {
    id: root
    objectName: "instrumentPanel"
    property var instrumentDetails: ({})
    property var runtimeStatus: ({})

    readonly property var riskRefresh: runtimeStatus && runtimeStatus.riskRefresh
        ? runtimeStatus.riskRefresh
        : null

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

        Label {
            text: qsTr("Instrument i harmonogram ryzyka")
            font.pointSize: 15
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Giełda"); font.bold: true }
            Label { text: Utils.formatText(root.instrumentDetails.exchange, qsTr("brak")) }

            Label { text: qsTr("Symbol"); font.bold: true }
            Label { text: Utils.formatText(root.instrumentDetails.symbol, qsTr("brak")) }

            Label { text: qsTr("Para"); font.bold: true }
            Label { text: Utils.formatCurrencyPair(root.instrumentDetails.baseCurrency, root.instrumentDetails.quoteCurrency) }

            Label { text: qsTr("Instrument na giełdzie"); font.bold: true }
            Label {
                text: Utils.formatText(root.instrumentDetails.venueSymbol, qsTr("brak"))
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Interwał"); font.bold: true }
            Label { text: Utils.formatText(root.instrumentDetails.granularity, qsTr("brak")) }

            Label { text: qsTr("Status połączenia"); font.bold: true }
            Label {
                text: Utils.formatText(root.runtimeStatus.connection, qsTr("brak"))
                wrapMode: Text.WordWrap
            }
        }

        ColumnLayout {
            spacing: 6
            Layout.fillWidth: true

            Label {
                text: qsTr("Harmonogram odświeżania ryzyka")
                font.bold: true
            }

            Label {
                text: riskRefresh
                    ? qsTr("Włączone: %1").arg(Utils.formatBoolean(riskRefresh.enabled))
                    : qsTr("Brak danych o odświeżaniu")
            }

            Label {
                visible: !!riskRefresh
                text: qsTr("Interwał: %1")
                    .arg(Utils.formatDuration(riskRefresh ? riskRefresh.intervalSeconds : null))
            }

            Label {
                visible: !!riskRefresh
                text: qsTr("Kolejne za: %1")
                    .arg(Utils.formatDuration(riskRefresh ? riskRefresh.nextRefreshInSeconds : null))
            }

            Label {
                visible: !!riskRefresh && !!riskRefresh.nextRefreshDueAt
                text: qsTr("Następna aktualizacja o: %1")
                    .arg(Utils.formatText(riskRefresh ? riskRefresh.nextRefreshDueAt : null, qsTr("n/d")))
            }

            Label {
                visible: !!riskRefresh && !!riskRefresh.lastUpdateAt
                text: qsTr("Ostatnia aktualizacja: %1")
                    .arg(Utils.formatText(riskRefresh ? riskRefresh.lastUpdateAt : null, qsTr("n/d")))
            }

            Label {
                visible: !!riskRefresh && !!riskRefresh.lastRequestAt
                text: qsTr("Ostatnie żądanie: %1")
                    .arg(Utils.formatText(riskRefresh ? riskRefresh.lastRequestAt : null, qsTr("n/d")))
            }
        }
    }
}
