import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Drawer {
    id: adminPanel
    width: Math.min(parent ? parent.width * 0.42 : 480, 560)
    implicitHeight: parent ? parent.height : 720
    edge: Qt.RightEdge
    modal: false
    interactive: true
    closePolicy: Popup.CloseOnEscape

    property var instrumentForm: ({})
    property var guardForm: ({})
    property string statusMessage: ""
    property color statusColor: palette.highlight

    function syncForms() {
        if (typeof appController === "undefined")
            return
        instrumentForm = appController.instrumentConfigSnapshot()
        guardForm = appController.performanceGuardSnapshot()
    }

    function refreshData() {
        syncForms()
        if (typeof securityController !== "undefined")
            securityController.refresh()
        if (typeof reportController !== "undefined")
            reportController.refresh()
    }

    onOpened: refreshData()

    background: Rectangle {
        color: Qt.darker(adminPanel.palette.window, 1.2)
    }

    contentItem: TabView {
        id: tabs
        anchors.fill: parent
        currentIndex: 0

        Tab {
            title: qsTr("Strategia")

            Flickable {
                anchors.fill: parent
                contentWidth: width
                contentHeight: strategyLayout.implicitHeight
                clip: true

                ColumnLayout {
                    id: strategyLayout
                    width: parent.width
                    spacing: 16
                    padding: 16

                    GroupBox {
                        title: qsTr("Instrument i rynek")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 10
                            Layout.fillWidth: true

                            Label { text: qsTr("Giełda") }
                            TextField {
                                text: instrumentForm.exchange || ""
                                onEditingFinished: instrumentForm.exchange = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Symbol logiczny") }
                            TextField {
                                text: instrumentForm.symbol || ""
                                onEditingFinished: instrumentForm.symbol = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Symbol na giełdzie") }
                            TextField {
                                text: instrumentForm.venueSymbol || ""
                                onEditingFinished: instrumentForm.venueSymbol = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Waluta kwotowana") }
                            TextField {
                                text: instrumentForm.quoteCurrency || ""
                                onEditingFinished: instrumentForm.quoteCurrency = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Waluta bazowa") }
                            TextField {
                                text: instrumentForm.baseCurrency || ""
                                onEditingFinished: instrumentForm.baseCurrency = text
                                Layout.fillWidth: true
                            }

                            Label { text: qsTr("Interwał (ISO8601)") }
                            TextField {
                                text: instrumentForm.granularity || ""
                                onEditingFinished: instrumentForm.granularity = text
                                Layout.fillWidth: true
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Performance guard")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 10
                            Layout.fillWidth: true

                            Label { text: qsTr("Docelowy FPS") }
                            SpinBox {
                                value: guardForm.fpsTarget || 60
                                from: 15
                                to: 240
                                stepSize: 1
                                onValueModified: guardForm.fpsTarget = value
                            }

                            Label { text: qsTr("Reduce motion po (s)") }
                            SpinBox {
                                id: reduceMotionSpin
                                from: 0
                                to: 1000
                                stepSize: 5
                                editable: true
                                onValueModified: guardForm.reduceMotionAfter = value / 100
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value / 100, 'f', 2, locale)
                                }
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    var scaled = Math.round(number * 100)
                                    return Math.max(from, Math.min(to, scaled))
                                }
                                Binding {
                                    target: reduceMotionSpin
                                    property: "value"
                                    value: Math.round((guardForm.reduceMotionAfter !== undefined
                                                       ? guardForm.reduceMotionAfter
                                                       : 1) * 100)
                                    when: !reduceMotionSpin.activeFocus
                                }
                            }

                            Label { text: qsTr("Budżet janku (ms)") }
                            SpinBox {
                                id: jankBudgetSpin
                                from: 100
                                to: 10000
                                stepSize: 5
                                editable: true
                                onValueModified: guardForm.jankThresholdMs = value / 100
                                textFromValue: function(value, locale) {
                                    return Qt.formatLocaleNumber(value / 100, 'f', 2, locale)
                                }
                                valueFromText: function(text, locale) {
                                    var number = Number.fromLocaleString(locale, text)
                                    if (isNaN(number))
                                        number = parseFloat(text)
                                    if (isNaN(number))
                                        return value
                                    var scaled = Math.round(number * 100)
                                    return Math.max(from, Math.min(to, scaled))
                                }
                                Binding {
                                    target: jankBudgetSpin
                                    property: "value"
                                    value: Math.round((guardForm.jankThresholdMs !== undefined
                                                       ? guardForm.jankThresholdMs
                                                       : 18) * 100)
                                    when: !jankBudgetSpin.activeFocus
                                }
                            }

                            Label { text: qsTr("Limit nakładek") }
                            SpinBox {
                                value: guardForm.maxOverlayCount || 3
                                from: 0
                                to: 12
                                onValueModified: guardForm.maxOverlayCount = value
                            }

                            Label { text: qsTr("Wyłącz nakładki <FPS") }
                            SpinBox {
                                value: guardForm.disableSecondaryWhenBelow || 0
                                from: 0
                                to: 120
                                onValueModified: guardForm.disableSecondaryWhenBelow = value
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Button {
                            text: qsTr("Przywróć aktualne")
                            onClicked: syncForms()
                        }

                        Item { Layout.fillWidth: true }

                        Button {
                            text: qsTr("Zapisz zmiany")
                            highlighted: true
                            onClicked: {
                                const okInstrument = appController.updateInstrument(
                                            instrumentForm.exchange || "",
                                            instrumentForm.symbol || "",
                                            instrumentForm.venueSymbol || "",
                                            instrumentForm.quoteCurrency || "",
                                            instrumentForm.baseCurrency || "",
                                            instrumentForm.granularity || "")
                                const okGuard = appController.updatePerformanceGuard(
                                            guardForm.fpsTarget || 60,
                                            guardForm.reduceMotionAfter || 1,
                                            guardForm.jankThresholdMs || 18,
                                            guardForm.maxOverlayCount || 3,
                                            guardForm.disableSecondaryWhenBelow || 0)
                                if (okInstrument && okGuard) {
                                    statusMessage = qsTr("Zapisano konfigurację strategii")
                                    statusColor = Qt.rgba(0.3, 0.7, 0.4, 1)
                                } else {
                                    statusMessage = qsTr("Nie udało się zapisać konfiguracji")
                                    statusColor = Qt.rgba(0.9, 0.4, 0.3, 1)
                                }
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: statusMessage.length > 0
                        text: statusMessage
                        color: statusColor
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }

        Tab {
            title: qsTr("Monitorowanie")

            ReportBrowser {
                anchors.fill: parent
            }
        }

        Tab {
            title: qsTr("Licencje i profile")

            Flickable {
                anchors.fill: parent
                contentWidth: width
                contentHeight: securityLayout.implicitHeight
                clip: true

                ColumnLayout {
                    id: securityLayout
                    width: parent.width
                    spacing: 16
                    padding: 16

                    GroupBox {
                        title: qsTr("Licencja OEM")
                        Layout.fillWidth: true

                        GridLayout {
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 8
                            Layout.fillWidth: true

                            Label { text: qsTr("Status") }
                            Label { text: securityController && securityController.licenseInfo.status || qsTr("n/d") }

                            Label { text: qsTr("Edycja") }
                            Label { text: securityController && securityController.licenseInfo.edition || qsTr("n/d") }

                            Label { text: qsTr("Utrzymanie do") }
                            Label { text: securityController && securityController.licenseInfo.maintenance_until || qsTr("n/d") }

                            Label { text: qsTr("Trial") }
                            Label {
                                text: securityController && securityController.licenseInfo.trial_active
                                      ? (securityController.licenseInfo.trial_expires_at || qsTr("aktywny"))
                                      : qsTr("nieaktywny")
                            }

                            Label { text: qsTr("Odbiorca") }
                            Label {
                                wrapMode: Text.WrapAnywhere
                                text: {
                                    if (!securityController || !securityController.licenseInfo.holder)
                                        return qsTr("n/d");
                                    const holder = securityController.licenseInfo.holder;
                                    let base = holder.name || qsTr("n/d");
                                    if (holder.email)
                                        base += " (" + holder.email + ")";
                                    return base;
                                }
                            }

                            Label { text: qsTr("Seats") }
                            Label {
                                text: securityController && securityController.licenseInfo.seats !== undefined
                                      ? securityController.licenseInfo.seats
                                      : qsTr("n/d")
                            }

                            Label { text: qsTr("Fingerprint") }
                            Label {
                                text: securityController && securityController.licenseInfo.fingerprint || qsTr("n/d")
                                wrapMode: Text.WrapAnywhere
                            }

                            Label { text: qsTr("Moduły") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.modules)
                                        return qsTr("brak");
                                    const modules = securityController.licenseInfo.modules;
                                    return modules.length > 0 ? modules.join(", ") : qsTr("brak");
                                }
                            }

                            Label { text: qsTr("Środowiska") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.environments)
                                        return qsTr("brak");
                                    const envs = securityController.licenseInfo.environments;
                                    return envs.length > 0 ? envs.join(", ") : qsTr("brak");
                                }
                            }

                            Label { text: qsTr("Runtime") }
                            Label {
                                wrapMode: Text.WordWrap
                                text: {
                                    if (!securityController || !securityController.licenseInfo.runtime)
                                        return qsTr("brak");
                                    const runtime = securityController.licenseInfo.runtime;
                                    return runtime.length > 0 ? runtime.join(", ") : qsTr("brak");
                                }
                            }
                        }
                    }

                    GroupBox {
                        title: qsTr("Profile użytkowników")
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 12

                            ListView {
                                id: profileList
                                Layout.fillWidth: true
                                Layout.preferredHeight: 180
                                model: securityController ? securityController.userProfiles : []
                                clip: true

                                delegate: Frame {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    padding: 8
                                    background: Rectangle {
                                        color: Qt.rgba(0.2, 0.3, 0.5, 0.2)
                                        radius: 6
                                    }

                                    ColumnLayout {
                                        anchors.fill: parent
                                        spacing: 4

                                        Label {
                                            text: (modelData.display_name || modelData.user_id)
                                                  + " (" + (modelData.user_id || "-") + ")"
                                            font.bold: true
                                        }
                                        Label {
                                            text: qsTr("Role: %1").arg((modelData.roles || []).join(", "))
                                        }
                                        Label {
                                            text: qsTr("Aktualizacja: %1").arg(modelData.updated_at || "-")
                                            color: palette.mid
                                        }
                                    }

                                    TapHandler {
                                        acceptedButtons: Qt.LeftButton
                                        onTapped: {
                                            userField.text = modelData.user_id || ""
                                            nameField.text = modelData.display_name || ""
                                            rolesField.text = (modelData.roles || []).join(", ")
                                        }
                                    }
                                }
                            }

                            GridLayout {
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: qsTr("Użytkownik") }
                                TextField {
                                    id: userField
                                    Layout.fillWidth: true
                                    placeholderText: qsTr("Identyfikator użytkownika")
                                }

                                Label { text: qsTr("Nazwa wyświetlana") }
                                TextField {
                                    id: nameField
                                    Layout.fillWidth: true
                                }

                                Label { text: qsTr("Role (CSV)") }
                                TextField {
                                    id: rolesField
                                    Layout.fillWidth: true
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Button {
                                    text: qsTr("Zapisz profil")
                                    enabled: securityController && !securityController.busy
                                    onClicked: {
                                        const roles = rolesField.text.split(",").map(r => r.trim()).filter(r => r.length > 0)
                                        const ok = securityController.assignProfile(
                                                    userField.text,
                                                    roles,
                                                    nameField.text)
                                        if (ok) {
                                            userField.text = ""
                                            nameField.text = ""
                                            rolesField.text = ""
                                        }
                                    }
                                }

                                Button {
                                    text: qsTr("Usuń profil")
                                    enabled: securityController && !securityController.busy && userField.text.length > 0
                                    onClicked: securityController && securityController.removeProfile(userField.text)
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: qsTr("Odśwież")
                                    onClicked: securityController && securityController.refresh()
                                }
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }

    Connections {
        target: appController
        function onInstrumentChanged() { adminPanel.syncForms() }
        function onPerformanceGuardChanged() { adminPanel.syncForms() }
    }
}
