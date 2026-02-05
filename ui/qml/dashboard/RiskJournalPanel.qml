import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: root
    objectName: "riskJournalPanel"
    property var metrics: ({})
    property var timeline: []
    property var runtimeService: null
    property var lastOperatorAction: ({})
    property string strategyFilter: ""
    property string riskFilter: ""
    property var selectedEntry: null
    readonly property var strategyHistogram: {
        const metrics = root.metrics || {}
        const counts = metrics.strategyCounts || {}
        const entries = []
        for (const key in counts) {
            if (!Object.prototype.hasOwnProperty.call(counts, key))
                continue
            entries.push({ label: key, count: counts[key] })
        }
        entries.sort(function(a, b) {
            if (b.count === a.count)
                return a.label.localeCompare(b.label)
            return b.count - a.count
        })
        return entries
    }
    readonly property var riskFlagHistogram: {
        const counts = root.metricValue("riskFlagCounts", {})
        const entries = []
        for (const key in counts) {
            if (!Object.prototype.hasOwnProperty.call(counts, key))
                continue
            entries.push({ label: key, count: counts[key] })
        }
        entries.sort(function(a, b) {
            if (b.count === a.count)
                return a.label.localeCompare(b.label)
            return b.count - a.count
        })
        return entries
    }
    readonly property var stressFailureHistogram: {
        const counts = root.metricValue("stressFailureCounts", {})
        const entries = []
        for (const key in counts) {
            if (!Object.prototype.hasOwnProperty.call(counts, key))
                continue
            entries.push({ label: key, count: counts[key] })
        }
        entries.sort(function(a, b) {
            if (b.count === a.count)
                return a.label.localeCompare(b.label)
            return b.count - a.count
        })
        return entries
    }
    readonly property var strategySummaries: {
        const summaries = root.metricValue("strategySummaries", [])
        return Array.isArray(summaries) ? summaries : []
    }
    readonly property int strategyChipCount: strategyRepeater ? strategyRepeater.count : 0
    readonly property int riskFlagChipCount: riskFlagRepeater ? riskFlagRepeater.count : 0
    readonly property int stressFailureChipCount: stressFailureRepeater ? stressFailureRepeater.count : 0

    readonly property var availableStrategies: {
        const data = root.toJsArray(root.timeline)
        const values = []
        for (let i = 0; i < data.length; ++i) {
            const entry = data[i]
            if (!entry || !entry.strategy)
                continue
            if (values.indexOf(entry.strategy) === -1)
                values.push(entry.strategy)
        }
        values.sort()
        return values
    }

    readonly property var availableRiskSignals: {
        const result = []
        const riskFlags = root.metricValue("uniqueRiskFlags", [])
        const stressFailures = root.metricValue("uniqueStressFailures", [])
        const aggregate = riskFlags.concat(stressFailures)
        for (let i = 0; i < aggregate.length; ++i) {
            const item = aggregate[i]
            if (result.indexOf(item) === -1)
                result.push(item)
        }
        result.sort()
        return result
    }

    readonly property var filteredTimeline: {
        const data = root.toJsArray(root.timeline)
        if (!root.strategyFilter && !root.riskFilter)
            return data
        const result = []
        for (let i = 0; i < data.length; ++i) {
            const entry = data[i]
            if (!entry)
                continue
            if (root.strategyFilter && entry.strategy !== root.strategyFilter)
                continue
            if (root.riskFilter) {
                const riskFlags = root.toJsArray(entry.riskFlags)
                const stressFailures = root.toJsArray(entry.stressFailures)
                if (riskFlags.indexOf(root.riskFilter) === -1 && stressFailures.indexOf(root.riskFilter) === -1)
                    continue
            }
            result.push(entry)
        }
        return result
    }

    readonly property bool hasIncompleteEntries: (root.metricValue("incompleteEntries", 0) || 0) > 0

    function toJsArray(value) {
        const out = []
        if (!value)
            return out
        if (Array.isArray(value))
            return value
        if (value.length === undefined)
            return out
        const size = Number(value.length)
        if (!isFinite(size) || size <= 0)
            return out
        for (let i = 0; i < size; ++i)
            out.push(value[i])
        return out
    }

    function metricValue(key, defaultValue) {
        const metrics = root.metrics || {}
        if (metrics[key] !== undefined)
            return metrics[key]
        const snakeKey = key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()
        if (metrics[snakeKey] !== undefined)
            return metrics[snakeKey]
        return defaultValue
    }

    signal freezeRequested(var entry)
    signal unfreezeRequested(var entry)
    signal unblockRequested(var entry)

    function describeOperatorAction(action) {
        const descriptor = {
            text: qsTr("Ostatnia akcja operatora: brak"),
            color: Styles.AppTheme.textSecondary,
        }
        if (!action || !action.action)
            return descriptor

        let label = action.action
        let color = Styles.AppTheme.textSecondary
        if (action.action === "freeze") {
            label = qsTr("zamrożenie")
            color = Styles.AppTheme.warning
        } else if (action.action === "unfreeze") {
            label = qsTr("odmrożenie")
            color = Styles.AppTheme.accent
        } else if (action.action === "unblock") {
            label = qsTr("odblokowanie")
            color = Styles.AppTheme.positive
        }

        const reference = action.entry && (action.entry.event || action.entry.timestamp || action.entry.id || "")
        const timestamp = action.timestamp || ""
        const parts = [qsTr("Ostatnia akcja operatora: %1").arg(label)]
        if (reference)
            parts.push(qsTr("referencja %1").arg(reference))
        if (timestamp)
            parts.push(timestamp)

        return {
            text: parts.join(" • "),
            color: color,
        }
    }

    function describeLastRiskEvent(prefix, entry) {
        const fallback = qsTr("%1: brak").arg(prefix)
        if (!entry || !entry.timestamp)
            return fallback
        const parts = [qsTr("%1: %2").arg(prefix).arg(entry.timestamp)]
        if (entry.strategy)
            parts.push(qsTr("strategia %1").arg(entry.strategy))
        if (entry.event)
            parts.push(entry.event)
        if (entry.riskAction)
            parts.push(entry.riskAction)
        return parts.join(" • ")
    }

    function strategySeverityColor(summary) {
        const severity = summary && summary.severity || "neutral"
        if (severity === "block")
            return Styles.AppTheme.negative
        if (severity === "freeze")
            return Styles.AppTheme.warning
        if (severity === "override")
            return Styles.AppTheme.accent
        return Styles.AppTheme.surfaceStrong
    }

    function describeStrategySummary(summary) {
        if (!summary)
            return qsTr("Brak danych strategii")
        const parts = []
        if (summary.lastTimestamp)
            parts.push(summary.lastTimestamp)
        if (summary.lastEvent)
            parts.push(summary.lastEvent)
        if (summary.lastRiskAction)
            parts.push(summary.lastRiskAction)
        if (summary.lastRiskFlags && summary.lastRiskFlags.length)
            parts.push(qsTr("Flag: %1").arg(summary.lastRiskFlags.join(", ")))
        if (summary.lastStressFailures && summary.lastStressFailures.length)
            parts.push(qsTr("Stress: %1").arg(summary.lastStressFailures.join(", ")))
        return parts.length ? parts.join(" • ") : qsTr("Brak ostatnich zdarzeń")
    }

    function resetFilters() {
        strategyFilter = ""
        riskFilter = ""
        if (strategyFilterCombo)
            strategyFilterCombo.currentIndex = 0
        if (riskFilterCombo)
            riskFilterCombo.currentIndex = 0
    }

    function openDrilldown(entry) {
        if (!entry)
            return
        selectedEntry = entry
        drilldownDialog.open()
    }

    function triggerOperatorAction(action) {
        if (!selectedEntry)
            return
        let record = selectedEntry.record || selectedEntry
        if (runtimeService && typeof runtimeService[action] === "function") {
            runtimeService[action](record)
        }
        if (action === "requestFreeze")
            root.freezeRequested(record)
        else if (action === "requestUnfreeze")
            root.unfreezeRequested(record)
        else if (action === "requestUnblock")
            root.unblockRequested(record)
        drilldownDialog.close()
        selectedEntry = null
    }

    onTimelineChanged: resetFilters()

    ColumnLayout {
        anchors.fill: parent
        spacing: Styles.AppTheme.spacingMd

        Rectangle {
            Layout.fillWidth: true
            visible: root.hasIncompleteEntries
            radius: Styles.AppTheme.radiusSmall
            color: Qt.rgba(Styles.AppTheme.warning.r, Styles.AppTheme.warning.g, Styles.AppTheme.warning.b, 0.12)
            border.color: Styles.AppTheme.warning
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingSm
                spacing: Styles.AppTheme.spacingSm

                Label {
                    text: qsTr("Niekompletne wpisy Risk Journal – agregacje pomijają rekordy bez risk_flags/stress_overrides lub risk_action.")
                    color: Styles.AppTheme.warning
                    font.pointSize: Styles.AppTheme.fontSizeBody
                    wrapMode: Text.Wrap
                }

                Item { Layout.fillWidth: true }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            Rectangle {
                Layout.preferredWidth: 140
                Layout.preferredHeight: 70
                radius: Styles.AppTheme.radiusSmall
                color: Styles.AppTheme.surfaceSubtle

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingSm
                    spacing: 2

                    Label {
                        text: qsTr("Blokady")
                        color: Styles.AppTheme.textSecondary
                        font.pointSize: Styles.AppTheme.fontSizeCaption
                    }

                    Label {
                        text: String(root.metricValue("blockCount", 0))
                        color: Styles.AppTheme.textPrimary
                        font.pointSize: Styles.AppTheme.fontSizeSubtitle
                        font.bold: true
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 140
                Layout.preferredHeight: 70
                radius: Styles.AppTheme.radiusSmall
                color: Styles.AppTheme.surfaceSubtle

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingSm
                    spacing: 2

                    Label {
                        text: qsTr("Zamrożenia")
                        color: Styles.AppTheme.textSecondary
                        font.pointSize: Styles.AppTheme.fontSizeCaption
                    }

                    Label {
                        text: String(root.metricValue("freezeCount", 0))
                        color: Styles.AppTheme.textPrimary
                        font.pointSize: Styles.AppTheme.fontSizeSubtitle
                        font.bold: true
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 160
                Layout.preferredHeight: 70
                radius: Styles.AppTheme.radiusSmall
                color: Styles.AppTheme.surfaceSubtle

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Styles.AppTheme.spacingSm
                    spacing: 2

                    Label {
                        text: qsTr("Stress overrides")
                        color: Styles.AppTheme.textSecondary
                        font.pointSize: Styles.AppTheme.fontSizeCaption
                    }

                    Label {
                        text: String(root.metricValue("stressOverrideCount", 0))
                        color: Styles.AppTheme.textPrimary
                        font.pointSize: Styles.AppTheme.fontSizeSubtitle
                        font.bold: true
                    }
                }
            }

            Item { Layout.fillWidth: true }

            ColumnLayout {
                spacing: 2

                Label {
                    text: qsTr("Ostatnie stress failures: %1").arg(
                              (root.metricValue("latestStressFailures", []).length
                              ? root.metricValue("latestStressFailures", []).join(", ") : qsTr("brak")))
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Label {
                    text: qsTr("Zakres osi czasu: %1 – %2").arg(
                              root.metricValue("timelineStart", qsTr("n/d"))).arg(
                              root.metricValue("timelineEnd", qsTr("n/d")))
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Label {
                    id: lastOperatorActionLabel
                    objectName: "riskJournalLastOperatorAction"
                    readonly property var descriptor: root.describeOperatorAction(root.lastOperatorAction)
                    text: descriptor.text
                    color: descriptor.color
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Label {
                    objectName: "riskJournalLastBlock"
                    text: root.describeLastRiskEvent(qsTr("Ostatnia blokada"), root.metricValue("lastBlock", null))
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Label {
                    objectName: "riskJournalLastFreeze"
                    text: root.describeLastRiskEvent(qsTr("Ostatnia blokada strategiczna"), root.metricValue("lastFreeze", null))
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Label {
                    objectName: "riskJournalLastOverride"
                    text: root.describeLastRiskEvent(qsTr("Ostatni stress override"), root.metricValue("lastStressOverride", null))
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingSm

            ComboBox {
                id: strategyFilterCombo
                objectName: "riskJournalStrategyFilter"
                Layout.preferredWidth: 200
                model: [qsTr("Wszystkie strategie")].concat(root.availableStrategies)
                onActivated: {
                    if (currentIndex <= 0)
                        root.strategyFilter = ""
                    else
                        root.strategyFilter = model[currentIndex]
                }
            }

            ComboBox {
                id: riskFilterCombo
                objectName: "riskJournalRiskFilter"
                Layout.preferredWidth: 200
                model: [qsTr("Wszystkie ryzyka")].concat(root.availableRiskSignals)
                onActivated: {
                    if (currentIndex <= 0)
                        root.riskFilter = ""
                    else
                        root.riskFilter = model[currentIndex]
                }
            }

            Item { Layout.fillWidth: true }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingXs
            visible: root.strategyHistogram.length > 0 || root.riskFlagHistogram.length > 0 || root.stressFailureHistogram.length > 0

            ColumnLayout {
                Layout.fillWidth: true
                spacing: Styles.AppTheme.spacingXs
                visible: root.strategyHistogram.length > 0

                Label {
                    text: qsTr("Najczęściej flagowane strategie")
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Flow {
                    id: strategyChipFlow
                    objectName: "riskJournalStrategyChips"
                    Layout.fillWidth: true
                    width: parent.width
                    spacing: Styles.AppTheme.spacingXs

                    Repeater {
                        id: strategyRepeater
                        model: root.strategyHistogram
                        delegate: Rectangle {
                            required property var modelData
                            property int chipIndex: index
                            property string chipText: (modelData.label || "") + " (" + String(modelData.count || 0) + ")"
                            objectName: "riskJournalStrategyChip_" + chipIndex
                            radius: Styles.AppTheme.radiusMedium
                            color: Styles.AppTheme.surfaceSubtle
                            border.color: Styles.AppTheme.surfaceStrong
                            border.width: 1
                            height: 26

                            Text {
                                anchors.centerIn: parent
                                text: parent.chipText
                                color: Styles.AppTheme.textSecondary
                                font.pointSize: Styles.AppTheme.fontSizeCaption
                            }
                        }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: Styles.AppTheme.spacingXs
                visible: root.riskFlagHistogram.length > 0

                Label {
                    text: qsTr("Najczęstsze flagi ryzyka")
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Flow {
                    id: riskFlagChipFlow
                    objectName: "riskJournalRiskFlagChips"
                    Layout.fillWidth: true
                    width: parent.width
                    spacing: Styles.AppTheme.spacingXs

                    Repeater {
                        id: riskFlagRepeater
                        model: root.riskFlagHistogram
                        delegate: Rectangle {
                            required property var modelData
                            property int chipIndex: index
                            property string chipText: (modelData.label || "") + " (" + String(modelData.count || 0) + ")"
                            objectName: "riskJournalFlagChip_" + chipIndex
                            radius: Styles.AppTheme.radiusMedium
                            color: Styles.AppTheme.surfaceSubtle
                            border.color: Styles.AppTheme.surfaceStrong
                            border.width: 1
                            height: 26

                            Text {
                                anchors.centerIn: parent
                                text: parent.chipText
                                color: Styles.AppTheme.textSecondary
                                font.pointSize: Styles.AppTheme.fontSizeCaption
                            }
                        }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: Styles.AppTheme.spacingXs
                visible: root.stressFailureHistogram.length > 0

                Label {
                    text: qsTr("Najczęstsze stress failures")
                    color: Styles.AppTheme.textSecondary
                    font.pointSize: Styles.AppTheme.fontSizeCaption
                }

                Flow {
                    id: stressFailureChipFlow
                    objectName: "riskJournalStressFailureChips"
                    Layout.fillWidth: true
                    width: parent.width
                    spacing: Styles.AppTheme.spacingXs

                    Repeater {
                        id: stressFailureRepeater
                        model: root.stressFailureHistogram
                        delegate: Rectangle {
                            required property var modelData
                            property int chipIndex: index
                            property string chipText: (modelData.label || "") + " (" + String(modelData.count || 0) + ")"
                            objectName: "riskJournalStressChip_" + chipIndex
                            radius: Styles.AppTheme.radiusMedium
                            color: Styles.AppTheme.surfaceSubtle
                            border.color: Styles.AppTheme.surfaceStrong
                            border.width: 1
                            height: 26

                            Text {
                                anchors.centerIn: parent
                                text: parent.chipText
                                color: Styles.AppTheme.textSecondary
                                font.pointSize: Styles.AppTheme.fontSizeCaption
                            }
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: Styles.AppTheme.spacingXs
            visible: root.strategySummaries.length > 0

            Label {
                text: qsTr("Strategie wymagające uwagi")
                color: Styles.AppTheme.textSecondary
                font.pointSize: Styles.AppTheme.fontSizeCaption
            }

            Flow {
                id: strategySummaryFlow
                objectName: "riskJournalStrategySummaries"
                Layout.fillWidth: true
                width: parent.width
                spacing: Styles.AppTheme.spacingSm

                Repeater {
                    model: root.strategySummaries

                    delegate: Rectangle {
                        required property var modelData
                        property int summaryIndex: index
                        property string summaryStrategy: modelData.strategy || ""
                        property int summaryBlockCount: modelData.blockCount || 0
                        property int summaryFreezeCount: modelData.freezeCount || 0
                        property int summaryOverrideCount: modelData.stressOverrideCount || 0
                        property string summarySeverity: modelData.severity || "neutral"
                        objectName: "riskJournalStrategySummary_" + summaryIndex
                        width: Math.min(260, Math.max(200, strategySummaryFlow.width / 3))
                        implicitHeight: summaryLayout.implicitHeight + 2 * Styles.AppTheme.spacingSm
                        radius: Styles.AppTheme.radiusSmall
                        color: Styles.AppTheme.surfaceSubtle
                        border.width: 2
                        border.color: root.strategySeverityColor(modelData)

                        ColumnLayout {
                            id: summaryLayout
                            anchors.fill: parent
                            anchors.margins: Styles.AppTheme.spacingSm
                            spacing: Styles.AppTheme.spacingXs

                            Label {
                                text: summaryStrategy || qsTr("Brak strategii")
                                color: Styles.AppTheme.textPrimary
                                font.bold: true
                                font.pointSize: Styles.AppTheme.fontSizeBody
                            }

                            RowLayout {
                                spacing: Styles.AppTheme.spacingXs

                                Repeater {
                                    model: [
                                        { label: qsTr("Blokady"), value: summaryBlockCount },
                                        { label: qsTr("Zamrożenia"), value: summaryFreezeCount },
                                        { label: qsTr("Overrides"), value: summaryOverrideCount },
                                    ]

                                    delegate: Rectangle {
                                        required property var modelData
                                        radius: Styles.AppTheme.radiusMedium
                                        color: Styles.AppTheme.surfaceMuted
                                        border.color: Styles.AppTheme.surfaceStrong
                                        border.width: 1
                                        height: 24
                                        implicitWidth: 96

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData.label + ": " + String(modelData.value || 0)
                                            color: Styles.AppTheme.textSecondary
                                            font.pointSize: Styles.AppTheme.fontSizeCaption
                                        }
                                    }
                                }
                            }

                            Text {
                                text: root.describeStrategySummary(modelData)
                                color: Styles.AppTheme.textSecondary
                                wrapMode: Text.Wrap
                                font.pointSize: Styles.AppTheme.fontSizeCaption
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            id: activityChart
            Layout.fillWidth: true
            Layout.preferredHeight: 140
            radius: Styles.AppTheme.radiusSmall
            color: Styles.AppTheme.surfaceMuted
            border.color: Styles.AppTheme.surfaceSubtle

            Row {
                id: barsRow
                anchors.fill: parent
                anchors.margins: Styles.AppTheme.spacingMd
                spacing: 6

                Repeater {
                    model: root.filteredTimeline
                    delegate: Rectangle {
                        required property var modelData
                        width: Math.max(6, (barsRow.width / Math.max(1, root.filteredTimeline.length)) - 4)
                        anchors.bottom: parent.bottom
                        radius: 3
                        height: Math.max(12, (activityChart.height - 24) * Math.max(0.2, Math.min(1.0, modelData.activityScore || 0.4)))
                        color: modelData.isBlock ? Styles.AppTheme.negative : (modelData.isStressOverride ? Styles.AppTheme.warning : Styles.AppTheme.accentMuted)

                        HoverHandler { id: hoverHandler }
                        ToolTip.visible: hoverHandler.hovered
                        ToolTip.text: (modelData.timestamp || "") + "\n" + (modelData.event || "")
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: Styles.AppTheme.radiusSmall
            color: Styles.AppTheme.surfaceMuted
            border.color: Styles.AppTheme.surfaceSubtle

            StackLayout {
                anchors.fill: parent
                currentIndex: root.filteredTimeline.length > 0 ? 0 : 1

                Item {
                    ListView {
                        id: timelineList
                        objectName: "riskJournalTimelineList"
                        anchors.fill: parent
                        model: root.filteredTimeline
                        clip: true
                        delegate: Rectangle {
                            required property var modelData
                            width: ListView.view.width
                            height: 68
                            radius: Styles.AppTheme.radiusSmall
                            color: modelData.isIncomplete
                                   ? Qt.rgba(Styles.AppTheme.warning.r, Styles.AppTheme.warning.g, Styles.AppTheme.warning.b, 0.18)
                                   : (modelData.isBlock
                                      ? Qt.rgba(Styles.AppTheme.negative.r, Styles.AppTheme.negative.g, Styles.AppTheme.negative.b, 0.22)
                                      : (modelData.isStressOverride
                                         ? Qt.rgba(Styles.AppTheme.warning.r, Styles.AppTheme.warning.g, Styles.AppTheme.warning.b, 0.24)
                                         : Styles.AppTheme.surfaceSubtle))
                            border.color: modelData.isBlock ? Styles.AppTheme.negative : (modelData.isStressOverride || modelData.isIncomplete ? Styles.AppTheme.warning : Styles.AppTheme.surfaceSubtle)

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: Styles.AppTheme.spacingSm
                                spacing: 4

                                Text {
                                    text: (modelData.timestamp || "") + " • " + (modelData.event || qsTr("brak zdarzenia"))
                                    color: Styles.AppTheme.textPrimary
                                    font.pointSize: Styles.AppTheme.fontSizeBody
                                    font.bold: modelData.isBlock
                                }

                                Text {
                                    text: {
                                        const details = []
                                        if (modelData.strategy)
                                            details.push(qsTr("Strategia: %1").arg(modelData.strategy))
                                        if (modelData.riskFlags && modelData.riskFlags.length)
                                            details.push(qsTr("Flag: %1").arg(modelData.riskFlags.join(", ")))
                                        if (modelData.stressFailures && modelData.stressFailures.length)
                                            details.push(qsTr("Stress: %1").arg(modelData.stressFailures.join(", ")))
                                        if (modelData.isIncomplete)
                                            details.push(qsTr("Brak wymaganych pól: %1").arg((modelData.missingFields || []).join(", ")))
                                        return details.join(" • ")
                                    }
                                    color: Styles.AppTheme.textSecondary
                                    font.pointSize: Styles.AppTheme.fontSizeCaption
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: root.openDrilldown(modelData)
                            }
                        }

                        ScrollIndicator.vertical: ScrollIndicator {}
                    }
                }

                Item {
                    Label {
                        objectName: "riskJournalEmptyState"
                        anchors.centerIn: parent
                        text: qsTr("Brak wpisów spełniających filtr")
                        color: Styles.AppTheme.textSecondary
                    }
                }
            }
        }
    }

    Dialog {
        id: drilldownDialog
        objectName: "riskJournalDrilldownDialog"
        modal: true
        width: Math.min(parent ? parent.width - 48 : 520, 620)
        height: Math.min(parent ? parent.height - 48 : 520, 560)
        standardButtons: Dialog.Cancel
        title: selectedEntry && selectedEntry.event ? qsTr("Szczegóły: %1").arg(selectedEntry.event) : qsTr("Szczegóły wpisu")

        onRejected: root.selectedEntry = null
        onClosed: {
            if (!visible)
                root.selectedEntry = null
        }

        contentItem: ScrollView {
            anchors.fill: parent
            TextArea {
                readOnly: true
                wrapMode: TextEdit.NoWrap
                text: selectedEntry ? JSON.stringify(selectedEntry.record || selectedEntry, null, 2) : ""
                font.family: Styles.AppTheme.monoFontFamily
                font.pointSize: Styles.AppTheme.fontSizeBody
            }
        }

        footer: RowLayout {
            spacing: Styles.AppTheme.spacingSm
            Layout.margins: Styles.AppTheme.spacingSm

            Button {
                text: qsTr("Zamroź")
                enabled: !!root.selectedEntry
                onClicked: root.triggerOperatorAction("requestFreeze")
            }

            Button {
                text: qsTr("Odmroź")
                enabled: !!root.selectedEntry
                onClicked: root.triggerOperatorAction("requestUnfreeze")
            }

            Button {
                text: qsTr("Odblokuj")
                enabled: !!root.selectedEntry
                onClicked: root.triggerOperatorAction("requestUnblock")
            }

            Item { Layout.fillWidth: true }
        }
    }
}
