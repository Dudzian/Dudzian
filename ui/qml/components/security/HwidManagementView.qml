import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

Item {
    id: root
    property var activationControllerRef: typeof activationController !== "undefined" ? activationController : null
    property var licenseControllerRef: typeof licenseController !== "undefined" ? licenseController : null
    property var appControllerRef: typeof appController !== "undefined" ? appController : null
    property bool scheduleAutoEnabled: false

    implicitWidth: parent ? parent.width : 640

    function fingerprintJson() {
        if (!activationControllerRef || !activationControllerRef.fingerprint)
            return "{}"
        return JSON.stringify(activationControllerRef.fingerprint, null, 2)
    }

    function resolvedFingerprint() {
        if (!activationControllerRef || !activationControllerRef.fingerprint)
            return ""
        var document = activationControllerRef.fingerprint
        if (document.fingerprint)
            return document.fingerprint
        if (document.payload && document.payload.fingerprint)
            return document.payload.fingerprint
        return ""
    }

    function fingerprintSchedule() {
        if (appControllerRef && appControllerRef.fingerprintRefreshSchedule)
            return appControllerRef.fingerprintRefreshSchedule
        if (appControllerRef && appControllerRef.securityCache)
            return appControllerRef.securityCache.fingerprintRefresh || {}
        return {}
    }

    function syncScheduleFromController() {
        var schedule = fingerprintSchedule()
        scheduleAutoEnabled = !!schedule.active
        if (intervalSpin && !intervalSpin.activeFocus) {
            var interval = schedule.intervalSeconds
            if (!interval || interval <= 0)
                interval = 86400
            intervalSpin.value = interval
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

    function lastFingerprintError() {
        var schedule = fingerprintSchedule()
        if (schedule && schedule.lastError)
            return schedule.lastError
        return ""
    }

    Component.onCompleted: syncScheduleFromController()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Fingerprint urządzenia")
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Odśwież")
                enabled: !!activationControllerRef
                onClicked: activationControllerRef.refresh()
            }

            Button {
                text: qsTr("Eksportuj…")
                enabled: !!activationControllerRef
                onClicked: exportDialog.open()
            }
        }

        TextArea {
            id: fingerprintText
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            text: fingerprintJson()
            wrapMode: TextEdit.Wrap
            readOnly: true
            font.family: "monospace"
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: qsTr("Fingerprint oczekiwany") }
            Label {
                text: licenseControllerRef ? (licenseControllerRef.expectedFingerprint || qsTr("brak")) : qsTr("brak")
                font.family: "monospace"
                wrapMode: Text.WordWrap
            }

            Label { text: qsTr("Fingerprint lokalny") }
            Label {
                text: resolvedFingerprint() || qsTr("brak")
                font.family: "monospace"
                wrapMode: Text.WordWrap
            }
        }

        GroupBox {
            title: qsTr("Harmonogram fingerprintu")
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                GridLayout {
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 6
                    Layout.fillWidth: true

                    property var schedule: fingerprintSchedule()

                    Label { text: qsTr("Status") }
                    Label {
                        text: schedule.active ? qsTr("aktywny") : qsTr("wstrzymany")
                        font.bold: true
                    }

                    Label { text: qsTr("Ostatnie zapytanie") }
                    Label { text: schedule.lastRequestAt || qsTr("n/d") }

                    Label { text: qsTr("Ostatnie odświeżenie") }
                    Label { text: schedule.lastCompletedAt || qsTr("n/d") }

                    Label { text: qsTr("Następne odświeżenie") }
                    Label { text: schedule.nextRefreshDueAt || qsTr("n/d") }

                    Label { text: qsTr("Pozostały czas") }
                    Label { text: formatRemaining(schedule.nextRefreshInSeconds) }

                    Label { text: qsTr("Ostatni błąd") }
                    Label {
                        wrapMode: Text.WordWrap
                        text: lastFingerprintError() || qsTr("brak")
                        color: lastFingerprintError().length > 0
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
                            if (!appControllerRef.setFingerprintRefreshEnabled(checked))
                                root.scheduleAutoEnabled = !checked
                            else
                                root.scheduleAutoEnabled = checked
                        }
                    }

                    Label {
                        text: qsTr("Interwał (s)")
                        enabled: autoRefreshToggle.checked
                    }

                    SpinBox {
                        id: intervalSpin
                        from: 300
                        to: 604800
                        stepSize: 60
                        enabled: autoRefreshToggle.checked && !!appControllerRef
                        value: 86400
                        onValueModified: {
                            if (!appControllerRef)
                                return
                            appControllerRef.setFingerprintRefreshIntervalSeconds(value)
                        }
                        onActiveFocusChanged: {
                            if (!activeFocus)
                                syncScheduleFromController()
                        }
                    }

                    Button {
                        text: qsTr("Odśwież teraz")
                        enabled: !!appControllerRef
                        onClicked: appControllerRef && appControllerRef.triggerFingerprintRefreshNow()
                    }

                    Item { Layout.fillWidth: true }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: overrideField
                Layout.fillWidth: true
                placeholderText: qsTr("Wpisz fingerprint do nadpisania")
            }

            Button {
                text: qsTr("Zapisz jako oczekiwany")
                enabled: licenseControllerRef && overrideField.text.length > 0
                onClicked: {
                    if (!licenseControllerRef)
                        return
                    licenseControllerRef.saveExpectedFingerprint(overrideField.text.trim())
                    overrideField.text = ""
                }
            }

            Button {
                text: qsTr("Nadpisz konfigurację")
                enabled: licenseControllerRef && overrideField.text.length > 0
                onClicked: {
                    if (!licenseControllerRef)
                        return
                    licenseControllerRef.overrideExpectedFingerprint(overrideField.text.trim())
                    overrideField.text = ""
                }
            }
        }
    }

    FileDialog {
        id: exportDialog
        title: qsTr("Zapisz dokument fingerprint")
        fileMode: FileDialog.SaveFile
        defaultSuffix: "json"
        nameFilters: [qsTr("Dokumenty JSON (*.json)"), qsTr("Wszystkie pliki (*)")]
        onAccepted: {
            if (!activationControllerRef)
                return
            if (selectedFile)
                activationControllerRef.exportFingerprint(selectedFile)
        }
    }

    Connections {
        target: appControllerRef
        function onFingerprintRefreshScheduleChanged() {
            syncScheduleFromController()
        }
        function onSecurityCacheChanged() {
            syncScheduleFromController()
        }
    }
}
