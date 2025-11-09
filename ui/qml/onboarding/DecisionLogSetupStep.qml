import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

import "../styles" as Styles

Item {
    id: root

    property var appController: (typeof appController !== "undefined" ? appController : null)
    property alias selectedPath: pathField.text
    property bool ready: false

    signal completionStateChanged(bool ready)
    signal logPathApplied(string path)

    function trId(id, fallback) {
        const translated = qsTrId(id)
        return translated === id ? fallback : translated
    }

    function updateReadyState() {
        const state = demoToggle.checked || pathField.text.length > 0
        ready = state
        completionStateChanged(state)
    }

    function syncFromController() {
        if (!appController)
            return
        const controllerPath = appController.decisionLogPath || ""
        if (controllerPath !== pathField.text)
            pathField.text = controllerPath
        logPathApplied(controllerPath)
        updateReadyState()
    }

    function applySelectedPath(path) {
        const candidate = path || ""
        const normalized = candidate && candidate.toString ? candidate.toString() : candidate
        pathField.text = normalized
        if (normalized.length > 0 && appController && typeof appController.setDecisionLogPath === "function")
            appController.setDecisionLogPath(candidate)
        logPathApplied(normalized)
        updateReadyState()
    }

    onAppControllerChanged: syncFromController()

    Component.onCompleted: {
        syncFromController()
        updateReadyState()
    }

    Connections {
        target: appController
        ignoreUnknownSignals: true
        function onDecisionLogPathChanged() {
            syncFromController()
        }
    }

    FileDialog {
        id: decisionLogFileDialog
        title: trId("licenseWizard.decisionLog.dialog.title", "Wybierz plik dziennika decyzji")
        nameFilters: ["JSON Lines (*.jsonl *.json)"]
        onAccepted: {
            if (!selectedFile)
                return
            applySelectedPath(selectedFile)
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd
        padding: Styles.AppTheme.spacingXl

        Label {
            text: trId("licenseWizard.step.decisionLogTitle", "Krok 3 – konfiguracja decision logu")
            font.pixelSize: Styles.AppTheme.fontSizeTitle
            font.bold: true
            color: Styles.AppTheme.textPrimary
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: trId(
                "licenseWizard.step.decisionLogDescription",
                "Wybierz źródło dziennika decyzji. Możesz skorzystać z wbudowanych danych demonstracyjnych lub wskazać własny plik JSONL wygenerowany przez runtime.",
            )
            font.pixelSize: Styles.AppTheme.fontSizeBody
            color: Styles.AppTheme.textSecondary
        }

        CheckBox {
            id: demoToggle
            objectName: "decisionLogDemoToggle"
            checked: true
            text: trId("licenseWizard.decisionLog.useDemo", "Użyj wbudowanego dziennika demonstracyjnego")
            onToggled: {
                if (checked)
                    pathField.text = ""
                if (checked)
                    logPathApplied("")
                updateReadyState()
            }
        }

        Frame {
            Layout.fillWidth: true
            enabled: !demoToggle.checked
            opacity: demoToggle.checked ? 0.6 : 1.0
            background: Rectangle {
                color: Styles.AppTheme.surfaceSubtle
                radius: Styles.AppTheme.radiusMedium
                border.color: Styles.AppTheme.surfaceBorder
                border.width: 1
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingLg
                spacing: Styles.AppTheme.spacingSm

                Label {
                    text: trId(
                        "licenseWizard.decisionLog.customTitle",
                        "Wskaż lokalizację własnego dziennika decyzji",
                    )
                    font.pixelSize: Styles.AppTheme.fontSizeSubtitle
                    color: Styles.AppTheme.textPrimary
                }

                TextField {
                    id: pathField
                    objectName: "decisionLogPathField"
                    Layout.fillWidth: true
                    placeholderText: trId(
                        "licenseWizard.decisionLog.pathPlaceholder",
                        "np. /var/log/bot_core/decision_log.jsonl",
                    )
                    readOnly: true
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Styles.AppTheme.spacingSm

                    Button {
                        objectName: "decisionLogBrowseButton"
                        text: trId("licenseWizard.decisionLog.browse", "Wybierz plik…")
                        onClicked: decisionLogFileDialog.open()
                    }

                    Button {
                        objectName: "decisionLogApplyButton"
                        enabled: pathField.text.length > 0 && appController && typeof appController.setDecisionLogPath === "function"
                        text: trId("licenseWizard.decisionLog.apply", "Zastosuj ścieżkę")
                        onClicked: applySelectedPath(pathField.text)
                    }

                    Button {
                        objectName: "decisionLogClearButton"
                        text: trId("licenseWizard.decisionLog.clear", "Wyczyść")
                        onClicked: {
                            pathField.text = ""
                            updateReadyState()
                        }
                    }
                }
            }
        }

        Label {
            objectName: "decisionLogStatusMessage"
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            color: ready ? Styles.AppTheme.success : Styles.AppTheme.textSecondary
            text: ready
                  ? trId("licenseWizard.decisionLog.ready", "Decision log został skonfigurowany.")
                  : trId("licenseWizard.decisionLog.pending", "Wybierz źródło decision logu, aby kontynuować.")
        }
    }
}

