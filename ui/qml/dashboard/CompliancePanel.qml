import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Rectangle {
    id: root
    Layout.fillWidth: true
    Layout.fillHeight: true
    color: Styles.AppTheme.surfaceStrong
    radius: 8
    border.color: Styles.AppTheme.surfaceSubtle
    border.width: 1

    Item {
        id: wiring
        property var telemetryProvider: null
        property var complianceController: null
    }

    property var telemetryProviderOverride: null
    property var complianceControllerOverride: null
    readonly property var provider: telemetryProviderOverride !== null
                                  ? telemetryProviderOverride
                                  : (typeof telemetryProvider !== "undefined" ? telemetryProvider : null)
    readonly property var controller: complianceControllerOverride !== null
                                    ? complianceControllerOverride
                                    : (typeof complianceController !== "undefined" ? complianceController : null)
    readonly property bool hasController: controller !== null
    readonly property bool busy: controller ? controller.busy : false

    Component.onCompleted: {
        if (wiring.telemetryProvider === null
                && telemetryProviderOverride === null
                && typeof telemetryProvider !== "undefined") {
            wiring.telemetryProvider = telemetryProvider
        }
        if (wiring.complianceController === null
                && complianceControllerOverride === null
                && typeof complianceController !== "undefined") {
            wiring.complianceController = complianceController
        }
    }

    function statusLabel(value) {
        if (!value)
            return qsTr("n/d")
        switch (value) {
        case "ok":
            return qsTr("OK")
        case "warning":
            return qsTr("Ostrzeżenie")
        case "critical":
        case "high":
        case "error":
            return qsTr("Błąd")
        case "info":
            return qsTr("Informacja")
        default:
            return value
        }
    }

    function statusColor(value) {
        if (value === "ok")
            return Styles.AppTheme.textPrimary
        if (value === "warning")
            return Qt.rgba(0.95, 0.65, 0.2, 1)
        if (value === "critical" || value === "high" || value === "error")
            return Qt.rgba(0.9, 0.25, 0.3, 1)
        return Styles.AppTheme.textSecondary
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Text {
                objectName: "compliancePanelTitle"
                text: qsTr("Zgodność (KYC/AML)")
                font.bold: true
                font.pointSize: 15
                color: Styles.AppTheme.textPrimary
            }

            Item { Layout.fillWidth: true }

            BusyIndicator {
                running: root.busy
                visible: running
                width: 24
                height: 24
            }

            Button {
                id: auditButton
                objectName: "complianceAuditButton"
                text: qsTr("Przeprowadź audyt")
                enabled: root.controller !== null
                onClicked: root.controller ? root.controller.refreshAudit() : null
            }
        }

        Rectangle {
            id: errorBanner
            objectName: "complianceErrorBanner"
            Layout.fillWidth: true
            visible: root.controller && root.controller.errorMessage.length > 0
            implicitHeight: visible ? 32 : 0
            radius: 6
            color: Qt.rgba(0.75, 0.25, 0.28, 0.9)

            Text {
                anchors.centerIn: parent
                color: "white"
                font.pointSize: 11
                text: root.controller ? root.controller.errorMessage : ""
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: width > 700 ? 3 : 1
            columnSpacing: 12
            rowSpacing: 8

            Repeater {
                model: [
                    { title: qsTr("KYC"), key: "kycStatus" },
                    { title: qsTr("AML"), key: "amlStatus" },
                    { title: qsTr("Limity transakcji"), key: "transactionStatus" }
                ]
                delegate: Frame {
                    objectName: "complianceStatusFrame_" + modelData.key
                    Layout.fillWidth: true
                    background: Rectangle {
                        radius: 6
                        color: Qt.rgba(0.12, 0.14, 0.18, 0.85)
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6

                        Text {
                            objectName: "complianceStatusLabel_" + modelData.key
                            text: modelData.title
                            font.bold: true
                            color: Styles.AppTheme.textPrimary
                        }

                        Text {
                            objectName: "complianceStatusValue_" + modelData.key
                            text: root.controller ? statusLabel(root.controller.summary[modelData.key]) : qsTr("n/d")
                            color: root.controller ? statusColor(root.controller.summary[modelData.key]) : Styles.AppTheme.textSecondary
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                objectName: "complianceTotalViolations"
                text: qsTr("Łączna liczba naruszeń: %1").arg(root.provider ? Number(root.provider.complianceSummary.totalViolations || 0).toFixed(0) : "0")
                color: Styles.AppTheme.textSecondary
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Text {
                    objectName: "complianceFrequentViolations"
                    text: qsTr("Najczęstsze naruszenia: %1").arg(root.provider && root.provider.complianceSummary.byRule ? Object.keys(root.provider.complianceSummary.byRule).slice(0, 2).join(", ") : qsTr("brak"))
                    color: Styles.AppTheme.textSecondary
                    wrapMode: Text.WordWrap
                }
            }
        }

        Text {
            objectName: "complianceLastAudit"
            Layout.fillWidth: true
            text: root.controller && root.controller.lastUpdated.length > 0
                  ? qsTr("Ostatni audyt: %1").arg(root.controller.lastUpdated)
                  : qsTr("Ostatni audyt: n/d")
            color: Styles.AppTheme.textSecondary
            font.pointSize: 11
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                width: parent.width
                spacing: 8

                Label {
                    objectName: "complianceFindingsHeader"
                    text: qsTr("Naruszenia")
                    font.bold: true
                    color: Styles.AppTheme.textPrimary
                }

                Repeater {
                    model: root.controller ? root.controller.findings : []
                    delegate: Frame {
                        objectName: "complianceFindingFrame"
                        Layout.fillWidth: true
                        background: Rectangle {
                            radius: 6
                            color: Qt.rgba(0.12, 0.14, 0.18, 0.7)
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 4

                            Text {
                                objectName: "complianceFindingRule"
                                text: qsTr("Reguła: %1").arg(model.ruleId)
                                font.bold: true
                                color: Styles.AppTheme.textPrimary
                            }

                            Text {
                                objectName: "complianceFindingMessage"
                                text: qsTr("Opis: %1").arg(model.message)
                                color: Styles.AppTheme.textSecondary
                                wrapMode: Text.WordWrap
                            }

                            Text {
                                objectName: "complianceFindingMetadata"
                                visible: !!model.metadata
                                text: model.metadata ? qsTr("Szczegóły: %1").arg(JSON.stringify(model.metadata)) : ""
                                color: Styles.AppTheme.textSecondary
                                wrapMode: Text.WordWrap
                            }

                            Text {
                                objectName: "complianceFindingSeverity"
                                text: qsTr("Poziom: %1").arg(model.severity)
                                color: statusColor(model.severity)
                            }
                        }
                    }
                }

                Label {
                    objectName: "complianceNoFindings"
                    visible: !root.controller || root.controller.findings.length === 0
                    text: qsTr("Brak wykrytych naruszeń")
                    color: Styles.AppTheme.textSecondary
                }

                Label {
                    objectName: "complianceRecommendationsHeader"
                    text: qsTr("Rekomendacje")
                    font.bold: true
                    color: Styles.AppTheme.textPrimary
                }

                Repeater {
                    model: root.controller ? root.controller.recommendations : []
                    delegate: Text {
                        objectName: "complianceRecommendation"
                        Layout.fillWidth: true
                        text: modelData
                        wrapMode: Text.WordWrap
                        color: Styles.AppTheme.textSecondary
                    }
                }
            }
        }
    }
}
