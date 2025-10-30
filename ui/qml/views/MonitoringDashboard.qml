import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    implicitWidth: 960
    implicitHeight: 600

    property string metricsEndpoint: (typeof appController !== "undefined" && appController && appController.metricsEndpoint)
                                      ? appController.metricsEndpoint
                                      : "http://127.0.0.1:9177/metrics"
    property bool autoRefresh: true
    property int refreshIntervalMs: 5000
    property var exchangeData: []
    property var strategyData: []
    property var securityData: []
    property string lastUpdated: ""
    property string statusMessage: ""
    readonly property color cardColor: Qt.rgba(0.12, 0.14, 0.18, 0.85)
    readonly property color mutedTextColor: Qt.rgba(0.65, 0.7, 0.78, 1)
    readonly property color negativeColor: Qt.rgba(0.85, 0.32, 0.38, 1)
    readonly property color positiveColor: Qt.rgba(0.25, 0.7, 0.5, 1)

    Timer {
        id: refreshTimer
        interval: Math.max(1500, root.refreshIntervalMs)
        running: root.autoRefresh
        repeat: true
        triggeredOnStart: true
        onTriggered: root.fetchMetrics()
    }

    function resolvedEndpoint() {
        var endpoint = root.metricsEndpoint || ""
        endpoint = endpoint.trim()
        if (endpoint.length === 0)
            return ""
        if (endpoint === "in-process")
            return ""
        if (endpoint.indexOf("http") === 0)
            return endpoint
        return "http://" + endpoint
    }

    function fetchMetrics() {
        var endpoint = resolvedEndpoint()
        if (!endpoint)
            return
        var xhr = new XMLHttpRequest()
        xhr.open("GET", endpoint)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE)
                return
            if (xhr.status === 200) {
                parseMetrics(xhr.responseText)
                statusMessage = qsTr("Ostatnia aktualizacja: %1").arg(new Date().toLocaleTimeString())
            } else {
                statusMessage = qsTr("Nie udało się pobrać metryk (%1)").arg(xhr.status)
            }
        }
        xhr.send()
    }

    function ensureExchange(stats, name) {
        if (!stats[name]) {
            stats[name] = {
                name: name,
                total: 0,
                success: 0,
                errors: 0,
                rateLimited: 0,
                health: 1
            }
        }
        return stats[name]
    }

    function ensureStrategy(stats, name) {
        if (!stats[name]) {
            stats[name] = {
                name: name,
                executed: 0,
                rejected: 0,
                warnings: 0,
                critical: 0
            }
        }
        return stats[name]
    }

    function ensureSecurity(stats, source) {
        if (!stats[source]) {
            stats[source] = {
                source: source,
                events: {},
                failures: 0
            }
        }
        return stats[source]
    }

    function parseLabelString(labelString) {
        var labels = {}
        if (!labelString)
            return labels
        var parts = labelString.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/)
        for (var i = 0; i < parts.length; ++i) {
            var chunk = parts[i]
            var idx = chunk.indexOf("=")
            if (idx === -1)
                continue
            var key = chunk.substring(0, idx).trim()
            var value = chunk.substring(idx + 1).trim()
            if (value.length >= 2 && value[0] === '"' && value[value.length - 1] === '"')
                value = value.substring(1, value.length - 1)
            labels[key] = value
        }
        return labels
    }

    function parseMetrics(text) {
        var exchangeStats = {}
        var strategyStats = {}
        var securityStats = {}
        var lines = text.split('\n')
        for (var i = 0; i < lines.length; ++i) {
            var line = lines[i]
            if (!line || line[0] === '#')
                continue
            var parts = line.split(/\s+/)
            if (parts.length < 2)
                continue
            var rawName = parts[0]
            var value = Number(parts[1])
            if (isNaN(value))
                value = 0
            var metricName = rawName
            var labelSpec = ""
            var braceIndex = rawName.indexOf('{')
            if (braceIndex !== -1) {
                metricName = rawName.substring(0, braceIndex)
                labelSpec = rawName.substring(braceIndex + 1, rawName.length - 1)
            }
            var labels = parseLabelString(labelSpec)
            if (metricName === "bot_exchange_requests_total") {
                var ex = ensureExchange(exchangeStats, labels.exchange || "unknown")
                ex.total += value
                if (labels.status === "ok")
                    ex.success += value
                else if (labels.status === "error")
                    ex.errors += value
            } else if (metricName === "bot_exchange_errors_total") {
                var exErrors = ensureExchange(exchangeStats, labels.exchange || "unknown")
                exErrors.errors = value
            } else if (metricName === "bot_exchange_rate_limited_total") {
                var exRate = ensureExchange(exchangeStats, labels.exchange || "unknown")
                exRate.rateLimited = value
            } else if (metricName === "bot_exchange_health_status") {
                var exHealth = ensureExchange(exchangeStats, labels.exchange || "unknown")
                exHealth.health = value
            } else if (metricName === "bot_strategy_decisions_total") {
                var st = ensureStrategy(strategyStats, labels.strategy || "strategy")
                if (labels.outcome === "executed")
                    st.executed += value
                else if (labels.outcome === "rejected")
                    st.rejected += value
                if (labels.severity === "critical")
                    st.critical += value
            } else if (metricName === "bot_strategy_alerts_total") {
                var stWarn = ensureStrategy(strategyStats, labels.strategy || "strategy")
                if (labels.severity === "warning" || labels.severity === "error")
                    stWarn.warnings += value
                if (labels.severity === "critical")
                    stWarn.critical += value
            } else if (metricName === "bot_security_events_total") {
                var sec = ensureSecurity(securityStats, labels.source || "core")
                var eventKey = labels.event || "event"
                if (!sec.events[eventKey])
                    sec.events[eventKey] = 0
                sec.events[eventKey] += value
            } else if (metricName === "bot_security_failures_total") {
                var secFail = ensureSecurity(securityStats, labels.source || "core")
                secFail.failures = value
            }
        }
        var exchanges = []
        for (var key in exchangeStats) {
            if (exchangeStats.hasOwnProperty(key))
                exchanges.push(exchangeStats[key])
        }
        exchanges.sort(function(a, b) { return a.name.localeCompare(b.name) })
        var strategies = []
        for (var key2 in strategyStats) {
            if (strategyStats.hasOwnProperty(key2))
                strategies.push(strategyStats[key2])
        }
        strategies.sort(function(a, b) { return a.name.localeCompare(b.name) })
        var securities = []
        for (var key3 in securityStats) {
            if (!securityStats.hasOwnProperty(key3))
                continue
            var entry = securityStats[key3]
            var eventsList = []
            for (var eventName in entry.events) {
                if (entry.events.hasOwnProperty(eventName))
                    eventsList.push({ name: eventName, count: entry.events[eventName] })
            }
            eventsList.sort(function(a, b) { return b.count - a.count })
            entry.eventList = eventsList
            securities.push(entry)
        }
        securities.sort(function(a, b) { return a.source.localeCompare(b.source) })
        exchangeData = exchanges
        strategyData = strategies
        securityData = securities
        lastUpdated = new Date().toLocaleTimeString()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Endpoint metryk: %1").arg(resolvedEndpoint() || qsTr("brak"))
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Odśwież")
                onClicked: fetchMetrics()
            }

            CheckBox {
                id: refreshToggle
                text: qsTr("Auto-odświeżanie")
                checked: root.autoRefresh
                onToggled: root.autoRefresh = checked
            }
        }

        Label {
            text: statusMessage.length > 0 ? statusMessage : qsTr("Ostatnia aktualizacja: %1").arg(lastUpdated || qsTr("n/d"))
            color: mutedTextColor
        }

        Frame {
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            padding: 16
            background: Rectangle { radius: 8; color: cardColor }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8
                Label {
                    text: qsTr("Giełdy")
                    font.pixelSize: 18
                    font.bold: true
                }
                TableView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: exchangeData
                    clip: true
                    rowSpacing: 4
                    columnSpacing: 12

                    delegate: RowLayout {
                        implicitHeight: 28
                        spacing: 12
                        Label { text: model.name; font.bold: true; Layout.preferredWidth: 120 }
                        Label { text: qsTr("Żądania: %1").arg(model.total.toFixed(0)); Layout.preferredWidth: 120 }
                        Label { text: qsTr("Sukcesy: %1").arg(model.success.toFixed(0)); Layout.preferredWidth: 120 }
                        Label {
                            text: qsTr("Błędy: %1").arg(model.errors.toFixed(0))
                            color: model.errors > 0 ? negativeColor : "white"
                            Layout.preferredWidth: 120
                        }
                        Label { text: qsTr("Rate-limit: %1").arg(model.rateLimited.toFixed(0)); Layout.preferredWidth: 140 }
                        Label {
                            text: model.health >= 1 ? qsTr("OK") : qsTr("Problemy")
                            color: model.health >= 1 ? positiveColor : negativeColor
                            Layout.preferredWidth: 120
                        }
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            padding: 16
            background: Rectangle { radius: 8; color: cardColor }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8
                Label {
                    text: qsTr("Strategie")
                    font.pixelSize: 18
                    font.bold: true
                }
                TableView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: strategyData
                    clip: true
                    rowSpacing: 4
                    columnSpacing: 12

                    delegate: RowLayout {
                        implicitHeight: 28
                        spacing: 12
                        Label { text: model.name; font.bold: true; Layout.preferredWidth: 160 }
                        Label { text: qsTr("Wykonane: %1").arg(model.executed.toFixed(0)); Layout.preferredWidth: 140 }
                        Label { text: qsTr("Odrzucone: %1").arg(model.rejected.toFixed(0)); Layout.preferredWidth: 140 }
                        Label { text: qsTr("Ostrzeżenia: %1").arg(model.warnings.toFixed(0)); Layout.preferredWidth: 140 }
                        Label {
                            text: qsTr("Krytyczne: %1").arg(model.critical.toFixed(0))
                            color: model.critical > 0 ? negativeColor : "white"
                            Layout.preferredWidth: 140
                        }
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            padding: 16
            background: Rectangle { radius: 8; color: cardColor }

            ColumnLayout {
                anchors.fill: parent
                spacing: 8
                Label {
                    text: qsTr("Bezpieczeństwo")
                    font.pixelSize: 18
                    font.bold: true
                }
                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: securityData
                    clip: true
                    spacing: 8
                    delegate: ColumnLayout {
                        width: ListView.view.width
                        spacing: 4
                        Label {
                            text: qsTr("Źródło: %1").arg(model.source)
                            font.bold: true
                        }
                        Label {
                            text: qsTr("Niepowodzenia: %1").arg(model.failures.toFixed(0))
                            color: model.failures > 0 ? negativeColor : "white"
                        }
                        Repeater {
                            model: model.eventList || []
                            delegate: Label {
                                text: qsTr("• %1: %2").arg(modelData.name).arg(modelData.count.toFixed(0))
                                color: mutedTextColor
                            }
                        }
                        Rectangle {
                            height: 1
                            color: Qt.rgba(0.5, 0.55, 0.6, 0.3)
                            visible: index < ListView.view.count - 1
                            Layout.fillWidth: true
                        }
                    }
                }
            }
        }
    }
}
