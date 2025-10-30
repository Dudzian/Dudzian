import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtCore
import "./LocalSecurityStore.js" as SecurityStore

Item {
    id: root
    property var licenseControllerRef: typeof licenseController !== "undefined" ? licenseController : null
    property var activationControllerRef: typeof activationController !== "undefined" ? activationController : null
    property var appControllerRef: typeof appController !== "undefined" ? appController : null
    property alias auditModel: auditEntries
    property int auditTotal: 0
    property bool scheduleAutoEnabled: false

    implicitWidth: parent ? parent.width : 640

    ListModel {
        id: auditEntries
    }

    function refreshAudit() {
        auditEntries.clear()
        var records = SecurityStore.fetchAudit(50)
        for (var i = 0; i < records.length; ++i)
            auditEntries.append(records[i])
        auditTotal = SecurityStore.totalAuditCount()
    }

    function clearAudit() {
        SecurityStore.clearAudit()
        refreshAudit()
    }

    function auditCount() {
        return auditTotal
    }

    function currentFingerprint() {
        if (licenseControllerRef && licenseControllerRef.licenseFingerprint)
            return licenseControllerRef.licenseFingerprint
        var fingerprintDoc = activationControllerRef ? activationControllerRef.fingerprint : null
        if (fingerprintDoc && fingerprintDoc.payload)
            return fingerprintDoc.payload.fingerprint || ""
        return ""
    }

    function storeActiveLicense() {
        if (!licenseControllerRef || !licenseControllerRef.licenseActive)
            return false
        var snapshot = {
            fingerprint: licenseControllerRef.licenseFingerprint,
            edition: licenseControllerRef.licenseEdition,
            license_id: licenseControllerRef.licenseLicenseId,
            maintenance_until: licenseControllerRef.licenseMaintenanceUntil,
            recorded_at: new Date().toISOString()
        }
        var stored = SecurityStore.addLicenseSnapshot(snapshot)
        if (stored)
            refreshAudit()
        return stored
    }

    function licenseSchedule() {
        if (appControllerRef && appControllerRef.licenseRefreshSchedule)
            return appControllerRef.licenseRefreshSchedule
        return ({})
    }

    function syncScheduleFromController() {
        var schedule = licenseSchedule()
        scheduleAutoEnabled = !!schedule.active
        if (intervalSpin && !intervalSpin.activeFocus) {
            var current = schedule.intervalSeconds
            if (!current || current <= 0)
                current = 600
            intervalSpin.value = current
        }
    }

    function formatRemaining(seconds) {
        if (seconds === undefined || seconds === null)
            return qsTr("n/d")
        if (seconds < 0)
            return qsTr("n/d")
        if (seconds < 60)
            return qsTr("%1 s").arg(Math.round(seconds))
        var minutes = Math.floor(seconds / 60)
        if (minutes < 60)
            return qsTr("%1 min").arg(minutes)
        var hours = Math.floor(minutes / 60)
        var remainingMinutes = minutes % 60
        if (remainingMinutes === 0)
            return qsTr("%1 h").arg(hours)
        return qsTr("%1 h %2 min").arg(hours).arg(remainingMinutes)
    }

    function lastSecurityError() {
        if (!appControllerRef || !appControllerRef.securityCache)
            return ""
        var cache = appControllerRef.securityCache
        return cache.lastError || ""
    }

    Component.onCompleted: {
        refreshAudit()
        syncScheduleFromController()
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 8
            Layout.fillWidth: true

            Label { text: qsTr("Status") }
            Label {
                text: {
                    if (!licenseControllerRef)
                        return qsTr("nieznany")
                    return licenseControllerRef.licenseActive ? qsTr("aktywna") : qsTr("nieaktywna")
                }
                font.bold: true
            }

            Label { text: qsTr("Fingerprint") }
            TextArea {
                Layout.fillWidth: true
                Layout.preferredHeight: 72
                wrapMode: TextEdit.Wrap
                readOnly: true
                text: currentFingerprint()
                font.family: "monospace"
            }

            Label { text: qsTr("Edycja") }
            Label {
                text: licenseControllerRef ? (licenseControllerRef.licenseEdition || qsTr("brak")) : qsTr("brak")
            }

            Label { text: qsTr("ID licencji") }
            Label {
                text: licenseControllerRef ? (licenseControllerRef.licenseLicenseId || qsTr("brak")) : qsTr("brak")
            }

            Label { text: qsTr("Utrzymanie do") }
            Label {
                text: licenseControllerRef ? (licenseControllerRef.licenseMaintenanceUntil || qsTr("brak")) : qsTr("brak")
            }

            Label { text: qsTr("Polityka licencji") }
            Label {
                text: {
                    if (!securityController || !securityController.licenseInfo)
                        return qsTr("n/d")
                    var policy = securityController.licenseInfo.policy || {}
                    var state = policy.state || "n/d"
                    var remaining = policy.days_remaining
                    if (remaining === undefined || remaining === null)
                        return state
                    return qsTr("%1 (%2 dni)").arg(state).arg(remaining)
                }
                font.bold: true
                color: {
                    if (!securityController || !securityController.licenseInfo)
                        return palette.text
                    var policy = securityController.licenseInfo.policy || {}
                    var state = (policy.state || "").toLowerCase()
                    if (state === "critical" || state === "expired")
                        return Qt.rgba(0.9, 0.35, 0.35, 1)
                    if (state === "warning")
                        return Qt.rgba(0.93, 0.74, 0.28, 1)
                    return palette.text
                }
            }

            Label { text: qsTr("Status harmonogramu") }
            Label {
                text: {
                    var schedule = licenseSchedule()
                    if (schedule && schedule.active !== undefined)
                        return schedule.active ? qsTr("aktywny") : qsTr("wstrzymany")
                    return qsTr("n/d")
                }
            }

            Label { text: qsTr("Ostatnie zapytanie") }
            Label {
                text: {
                    var schedule = licenseSchedule()
                    return schedule.lastRequestAt || qsTr("n/d")
                }
            }

            Label { text: qsTr("Ostatnie odświeżenie") }
            Label {
                text: {
                    var schedule = licenseSchedule()
                    return schedule.lastCompletedAt || qsTr("n/d")
                }
            }

            Label { text: qsTr("Następne odświeżenie") }
            Label {
                text: {
                    var schedule = licenseSchedule()
                    return schedule.nextRefreshDueAt || qsTr("n/d")
                }
            }

            Label { text: qsTr("Pozostały czas") }
            Label {
                text: {
                    var schedule = licenseSchedule()
                    return formatRemaining(schedule.nextRefreshInSeconds)
                }
            }

            Label { text: qsTr("Ostatni błąd") }
            Label {
                wrapMode: Text.WordWrap
                text: {
                    var errorText = lastSecurityError()
                    return errorText.length > 0 ? errorText : qsTr("brak")
                }
                color: lastSecurityError().length > 0
                       ? Qt.rgba(0.9, 0.35, 0.35, 1)
                       : palette.text
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            CheckBox {
                id: autoRefreshToggle
                text: qsTr("Automatyczne odświeżanie")
                checked: root.scheduleAutoEnabled
                enabled: !!appControllerRef
                onToggled: {
                    if (!appControllerRef)
                        return
                    if (!appControllerRef.setLicenseRefreshEnabled(checked)) {
                        root.scheduleAutoEnabled = !checked
                    } else {
                        root.scheduleAutoEnabled = checked
                    }
                }
            }

            Label {
                text: qsTr("Interwał (s)")
                enabled: autoRefreshToggle.checked
            }

            SpinBox {
                id: intervalSpin
                from: 60
                to: 86400
                stepSize: 60
                enabled: autoRefreshToggle.checked && !!appControllerRef
                value: 600
                onValueModified: {
                    if (!appControllerRef)
                        return
                    appControllerRef.setLicenseRefreshIntervalSeconds(value)
                }
                onActiveFocusChanged: {
                    if (!activeFocus)
                        syncScheduleFromController()
                }
            }

            Button {
                text: qsTr("Odśwież teraz")
                enabled: !!appControllerRef
                onClicked: appControllerRef && appControllerRef.triggerLicenseRefreshNow()
            }

            Item { Layout.fillWidth: true }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: qsTr("Wczytaj licencję…")
                enabled: !!licenseControllerRef
                onClicked: licenseFileDialog.open()
            }

            Button {
                text: qsTr("Automatyczna aktywacja")
                enabled: licenseControllerRef && !licenseControllerRef.provisioningInProgress
                onClicked: {
                    if (!licenseControllerRef)
                        return
                    var fingerprintDoc = activationControllerRef ? activationControllerRef.fingerprint : ({})
                    licenseControllerRef.autoProvision(fingerprintDoc)
                }
            }

            Button {
                text: qsTr("Zapisz fingerprint")
                enabled: licenseControllerRef && currentFingerprint().length > 0
                onClicked: licenseControllerRef.saveExpectedFingerprint(currentFingerprint())
            }

            Button {
                text: qsTr("Dodaj do audytu")
                enabled: licenseControllerRef && licenseControllerRef.licenseActive
                onClicked: storeActiveLicense()
            }

            Button {
                text: qsTr("Wyczyść audyt")
                enabled: auditEntries.count > 0
                onClicked: clearAudit()
            }

            Item { Layout.fillWidth: true }
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: licenseControllerRef && licenseControllerRef.statusMessage.length > 0
            text: licenseControllerRef ? licenseControllerRef.statusMessage : ""
            color: licenseControllerRef && licenseControllerRef.statusIsError
                    ? Qt.rgba(0.9, 0.35, 0.35, 1)
                    : Qt.rgba(0.35, 0.75, 0.45, 1)
        }

        BusyIndicator {
            running: licenseControllerRef && licenseControllerRef.provisioningInProgress
            visible: running
        }

        GroupBox {
            title: qsTr("Lokalna historia aktywacji")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 6

                ListView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.min(contentHeight, 200)
                    clip: true
                    model: auditEntries

                    delegate: Frame {
                        required property var modelData
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            color: Qt.rgba(0.16, 0.24, 0.35, 0.35)
                            radius: 6
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 2

                            Label {
                                text: qsTr("Fingerprint: %1").arg(modelData.fingerprint || qsTr("brak"))
                                font.family: "monospace"
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: qsTr("Edycja: %1").arg(modelData.edition || qsTr("brak"))
                            }

                            Label {
                                text: qsTr("ID licencji: %1").arg(modelData.license_id || qsTr("brak"))
                                color: palette.mid
                            }

                            Label {
                                text: qsTr("Utrzymanie do: %1").arg(modelData.maintenance_until || qsTr("brak"))
                                color: palette.mid
                            }

                            Label {
                                text: modelData.recorded_at || ""
                                color: palette.mid
                            }
                        }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    visible: auditEntries.count > 0
                    text: qsTr("Łącznie zapisów: %1").arg(auditTotal)
                    color: palette.mid
                }

                Label {
                    Layout.fillWidth: true
                    visible: auditEntries.count === 0
                    text: qsTr("Brak zapisanych wpisów audytowych.")
                    color: palette.mid
                }
            }
        }

        GroupBox {
            title: qsTr("Zdarzenia bezpieczeństwa i audyt")
            Layout.fillWidth: true
            Layout.preferredHeight: 240

            ColumnLayout {
                anchors.fill: parent
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Button {
                        text: qsTr("Odśwież")
                        enabled: securityController && !securityController.busy
                        onClicked: securityController && securityController.refresh()
                    }

                    Button {
                        text: qsTr("Eksportuj podpisany log")
                        enabled: securityController && !securityController.busy
                        onClicked: auditExportDialog.open()
                    }

                    Item { Layout.fillWidth: true }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: securityController ? securityController.auditLog : []
                    delegate: Frame {
                        required property var modelData
                        Layout.fillWidth: true
                        padding: 8
                        background: Rectangle {
                            color: Qt.rgba(0.09, 0.12, 0.18, 0.65)
                            radius: 6
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 2

                            Label {
                                text: (modelData.timestamp || "") + " · " + (modelData.category || "")
                                font.pixelSize: 12
                                color: Qt.rgba(0.7, 0.78, 0.88, 1)
                            }

                            Label {
                                text: modelData.message || ""
                                font.pixelSize: 13
                                wrapMode: Text.WordWrap
                                color: "white"
                            }

                            Label {
                                visible: modelData.details && Object.keys(modelData.details).length > 0
                                text: visible ? JSON.stringify(modelData.details) : ""
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                color: Qt.rgba(0.6, 0.72, 0.84, 1)
                            }
                        }
                    }
                }
            }
        }
    }

    FolderDialog {
        id: auditExportDialog
        title: qsTr("Wybierz katalog eksportu logów bezpieczeństwa")
        folder: QtCore.StandardPaths.writableLocation(QtCore.StandardPaths.DocumentsLocation)
        onAccepted: {
            if (securityController)
                securityController.exportSignedAuditLog(auditExportDialog.selectedFolder)
        }
    }

    FileDialog {
        id: licenseFileDialog
        title: qsTr("Wybierz plik licencji OEM")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("Dokumenty JSON (*.json *.jsonl)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (!licenseControllerRef)
                return
            if (selectedFile)
                licenseControllerRef.loadLicenseUrl(selectedFile)
        }
    }

    Connections {
        target: appControllerRef
        function onLicenseRefreshScheduleChanged() {
            syncScheduleFromController()
        }
    }
}
