import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "." as Design
import "components" as Components

Components.Card {
    id: root
    objectName: "strategyAiPanel"
    title: qsTr("AI Governor i strategie adaptacyjne")
    subtitle: feedTransportSnapshot && feedTransportSnapshot.status
              ? qsTr("Transport %1 (%2)").arg(feedTransportSnapshot.status).arg(feedTransportSnapshot.mode || "local")
              : qsTr("Transport: brak danych")

    property var runtimeService: null
    property var feedTransportSnapshot: ({})
    property var aiRegimes: []
    property string adaptiveSummary: ""
    property string activationSummary: ""
    property var longPollMetricsModel: null
    readonly property var serviceLongPollMetricsModel: (runtimeService && runtimeService.longPollMetrics !== undefined && runtimeService.longPollMetrics !== null)
                                                     ? runtimeService.longPollMetrics
                                                     : null
    readonly property int injectedLongPollMetricsCount: root.modelItemCount(longPollMetricsModel)
    readonly property int serviceLongPollMetricsCount: root.modelItemCount(serviceLongPollMetricsModel)
    property bool longPollGuardWarningLogged: false
    readonly property var effectiveLongPollMetricsModel: {
        var injectedModel = longPollMetricsModel
        var serviceModel = serviceLongPollMetricsModel
        var injectedArray = []
        var serviceArray = []

        try {
            injectedArray = root._toPlainArray(injectedModel)
        } catch (error) {
            injectedArray = []
        }

        try {
            serviceArray = root._toPlainArray(serviceModel)
        } catch (error) {
            serviceArray = []
        }

        if (injectedArray.length > 0)
            return injectedArray

        if (serviceArray.length > 0)
            return serviceArray

        return []
    }

    function warnLongPollGuardOnce(context, detail) {
        if (root.longPollGuardWarningLogged)
            return
        root.longPollGuardWarningLogged = true
        console.warn("StrategyAiPanel long-poll guard fallback", context, detail)
    }

    function _seqCount(source) {
        try {
            if (!source)
                return 0

            if (typeof source.rowCount === "function") {
                try {
                    var fnRowCount = Number(source.rowCount())
                    if (isFinite(fnRowCount) && fnRowCount >= 0)
                        return fnRowCount
                } catch (error) {
                }
            }

            if (typeof source.count === "function") {
                try {
                    var fnCount = Number(source.count())
                    if (isFinite(fnCount) && fnCount >= 0)
                        return fnCount
                } catch (error) {
                }
            }

            if (typeof source.size === "function") {
                try {
                    var fnSize = Number(source.size())
                    if (isFinite(fnSize) && fnSize >= 0)
                        return fnSize
                } catch (error) {
                }
            }

            if (typeof source.length === "function") {
                try {
                    var fnLength = Number(source.length())
                    if (isFinite(fnLength) && fnLength >= 0)
                        return fnLength
                } catch (error) {
                }
            }

            if (typeof source.length === "number") {
                var numberLength = Number(source.length)
                if (isFinite(numberLength) && numberLength >= 0)
                    return numberLength
            }

            if (typeof source.count === "number") {
                var numberCount = Number(source.count)
                if (isFinite(numberCount) && numberCount >= 0)
                    return numberCount
            }

            if (typeof source.size === "number") {
                var numberSize = Number(source.size)
                if (isFinite(numberSize) && numberSize >= 0)
                    return numberSize
            }

            if (typeof source.property === "function") {
                try {
                    var propertyLength = Number(source.property("length"))
                    if (isFinite(propertyLength) && propertyLength >= 0)
                        return propertyLength
                } catch (error) {
                }
                try {
                    var propertyCount = Number(source.property("count"))
                    if (isFinite(propertyCount) && propertyCount >= 0)
                        return propertyCount
                } catch (error) {
                }
                try {
                    var propertySize = Number(source.property("size"))
                    if (isFinite(propertySize) && propertySize >= 0)
                        return propertySize
                } catch (error) {
                }
            }

            try {
                var keys = Object.keys(source)
                if (!keys || typeof keys.length !== "number")
                    return 0
                var numericKeys = keys.filter(function(key) {
                    return /^\d+$/.test(key)
                })
                return Array.isArray(numericKeys) ? numericKeys.length : 0
            } catch (error) {
                root.warnLongPollGuardOnce("_seqCount.keys", error)
                return 0
            }
        } catch (error) {
            root.warnLongPollGuardOnce("_seqCount", error)
            return 0
        }
    }

    function _seqAt(source, index) {
        if (!source)
            return undefined

        if (typeof source.get === "function") {
            try {
                return source.get(index)
            } catch (error) {
            }
        }

        if (typeof source.at === "function") {
            try {
                return source.at(index)
            } catch (error) {
            }
        }

        if (typeof source.value === "function") {
            try {
                return source.value(index)
            } catch (error) {
            }
        }

        if (typeof source.property === "function") {
            try {
                return source.property(String(index))
            } catch (error) {
            }

            try {
                return source.property(index)
            } catch (error) {
            }
        }

        try {
            return source[String(index)]
        } catch (error) {
        }

        try {
            return source[index]
        } catch (error) {
            return undefined
        }
    }

    function _toPlainArray(source) {
        var values = []

        if (!source)
            return values

        try {
            var size = 0
            try {
                size = root._seqCount(source)
            } catch (error) {
                root.warnLongPollGuardOnce("_toPlainArray.count", error)
                size = 0
            }

            var firstTransientEmpty = false
            if (size > 0) {
                for (var idx = 0; idx < size; idx++) {
                    var indexedEntry
                    try {
                        indexedEntry = root._seqAt(source, idx)
                    } catch (error) {
                        break
                    }

                    if (indexedEntry === undefined || indexedEntry === null) {
                        if (idx === 0)
                            firstTransientEmpty = true
                        break
                    }

                    values.push(indexedEntry)
                }
                if (values.length > 0)
                    return values
                if (!firstTransientEmpty)
                    return values
            }

            for (var probe = 0; probe < 256; probe++) {
                var entry
                try {
                    entry = root._seqAt(source, probe)
                } catch (error) {
                    break
                }

                if (entry === undefined || entry === null) {
                    if (probe === 0)
                        continue
                    break
                }

                values.push(entry)
            }
        } catch (error) {
            root.warnLongPollGuardOnce("_toPlainArray", error)
            return []
        }

        return values
    }

    function modelItemCount(model) {
        return root._seqCount(model)
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 16

        RowLayout {
            id: statusRow
            objectName: "strategyAiStatusRow"
            spacing: 12
            Layout.fillWidth: true

            Design.Icon {
                id: transportIcon
                iconName: feedTransportSnapshot && feedTransportSnapshot.mode === "grpc" ? "cloud" : "bolt"
                iconSize: 28
                iconColor: feedTransportSnapshot && feedTransportSnapshot.status === "connected"
                           ? Design.Palette.success
                           : Design.Palette.warning
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    id: statusLabel
                    objectName: "strategyAiTransportLabel"
                    text: feedTransportSnapshot && feedTransportSnapshot.label
                          ? qsTr("%1 • %2").arg(feedTransportSnapshot.label).arg(feedTransportSnapshot.status || qsTr("unknown"))
                          : qsTr("Źródło lokalne • %1").arg(feedTransportSnapshot && feedTransportSnapshot.status
                                                           ? feedTransportSnapshot.status
                                                           : qsTr("unknown"))
                    color: Design.Palette.textPrimary
                    font.pixelSize: Design.Typography.title
                    wrapMode: Text.WordWrap
                }

                Label {
                    text: feedTransportSnapshot && feedTransportSnapshot.latencyP95
                          ? qsTr("Latency p95: %1 ms • reconnects: %2")
                                .arg(Number(feedTransportSnapshot.latencyP95).toFixed(0))
                                .arg(feedTransportSnapshot.reconnects || 0)
                          : qsTr("Brak pomiarów latencji decyzji")
                    color: Design.Palette.textSecondary
                    font.pixelSize: Design.Typography.body
                }

                Label {
                    visible: !!(feedTransportSnapshot && feedTransportSnapshot.lastError && feedTransportSnapshot.lastError.length > 0)
                    text: qsTr("Ostatni błąd: %1").arg(feedTransportSnapshot.lastError)
                    color: Design.Palette.warning
                    font.pixelSize: Design.Typography.caption
                    wrapMode: Text.WordWrap
                }
            }

            Button {
                id: refreshButton
                text: qsTr("Odśwież AI")
                icon.name: "refresh"
                onClicked: {
                    if (root.runtimeService)
                        root.runtimeService.refreshRuntimeMetadata()
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 10

            Repeater {
                id: regimeRepeater
                model: aiRegimes || []
                delegate: Rectangle {
                    width: parent ? parent.width : 0
                    height: 64
                    radius: 14
                    color: Qt.rgba(1, 1, 1, 0.04)
                    border.color: Qt.rgba(1, 1, 1, 0.06)
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 10

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Label {
                                text: modelData && modelData.regime ? modelData.regime : qsTr("(brak nazwy)")
                                font.pixelSize: Design.Typography.title
                                color: Design.Palette.textPrimary
                                font.bold: true
                            }

                            Label {
                                text: modelData && modelData.bestStrategy
                                      ? qsTr("Najlepsza strategia: %1").arg(modelData.bestStrategy)
                                      : qsTr("Brak zwycięzcy")
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.body
                            }
                        }

                        ColumnLayout {
                            Layout.alignment: Qt.AlignVCenter
                            spacing: 2

                            Label {
                                text: modelData && modelData.meanReward !== undefined
                                      ? qsTr("μ=%1").arg(Number(modelData.meanReward).toFixed(2))
                                      : "—"
                                color: Design.Palette.textPrimary
                                font.pixelSize: Design.Typography.body
                            }

                            Label {
                                text: modelData && modelData.plays !== undefined
                                      ? qsTr("n=%1").arg(modelData.plays)
                                      : "—"
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.caption
                            }
                        }
                    }
                }
            }

            Label {
                visible: !aiRegimes || aiRegimes.length === 0
                text: qsTr("Brak danych polityk adaptacyjnych – uruchom retraining AI.")
                color: Design.Palette.textSecondary
                font.pixelSize: Design.Typography.body
                wrapMode: Text.WordWrap
            }
        }

        Label {
            objectName: "strategyAiActivationSummary"
            visible: activationSummary.length > 0
            text: activationSummary
            color: Design.Palette.textSecondary
            font.pixelSize: Design.Typography.body
            wrapMode: Text.WordWrap
        }

        Label {
            objectName: "strategyAiAdaptiveSummary"
            visible: adaptiveSummary.length > 0
            text: adaptiveSummary
            color: Design.Palette.textSecondary
            font.pixelSize: Design.Typography.caption
            wrapMode: Text.WordWrap
        }

        GroupBox {
            title: qsTr("Fallback long-poll")
            Layout.fillWidth: true

            ColumnLayout {
                spacing: 6

                readonly property var effectiveModel: root.effectiveLongPollMetricsModel

                Label {
                    objectName: root.objectName && root.objectName.startsWith("runtimeOverview")
                                ? "runtimeOverviewLongPollEmpty"
                                : "strategyAiLongPollEmpty"
                    visible: longPollRepeater.count === 0
                    text: qsTr("Fallback long-poll: brak aktywnych streamów")
                    color: Design.Palette.textSecondary
                    font.pixelSize: Design.Typography.body
                    wrapMode: Text.WordWrap
                }

                Repeater {
                    id: longPollRepeater
                    model: parent.effectiveModel

                    delegate: Rectangle {
                        id: longPollEntry
                        width: parent ? parent.width : 0
                        Layout.fillWidth: true
                        Layout.preferredHeight: longPollColumn.implicitHeight + 16
                        implicitHeight: longPollColumn.implicitHeight + 16
                        radius: 10
                        color: Qt.rgba(1, 1, 1, 0.03)
                        border.color: Qt.rgba(1, 1, 1, 0.06)
                        border.width: 1
                        readonly property string objectPrefix: root.objectName && root.objectName.startsWith("runtimeOverview")
                                                               ? "runtimeOverview"
                                                               : "strategyAi"
                        objectName: objectPrefix + "LongPollEntry"

                        readonly property var entryData: (modelData && typeof modelData === "object") ? modelData : ({})
                        readonly property var roleLabels: (typeof labels !== "undefined") ? labels : undefined
                        readonly property var roleRequestLatency: (typeof requestLatency !== "undefined") ? requestLatency : undefined
                        readonly property var roleHttpErrors: (typeof httpErrors !== "undefined") ? httpErrors : undefined
                        readonly property var roleReconnects: (typeof reconnects !== "undefined") ? reconnects : undefined
                        readonly property var safeLabels: (roleLabels && typeof roleLabels === "object")
                                                          ? roleLabels
                                                          : ((entryData.labels && typeof entryData.labels === "object") ? entryData.labels : ({}))
                        readonly property var latencyStats: (roleRequestLatency && typeof roleRequestLatency === "object")
                                                            ? roleRequestLatency
                                                            : ((entryData.requestLatency && typeof entryData.requestLatency === "object") ? entryData.requestLatency : ({}))
                        readonly property var httpErrorStats: (roleHttpErrors && typeof roleHttpErrors === "object")
                                                              ? roleHttpErrors
                                                              : ((entryData.httpErrors && typeof entryData.httpErrors === "object") ? entryData.httpErrors : ({}))
                        readonly property var reconnectStats: (roleReconnects && typeof roleReconnects === "object")
                                                              ? roleReconnects
                                                              : ((entryData.reconnects && typeof entryData.reconnects === "object") ? entryData.reconnects : ({}))
                        ColumnLayout {
                            id: longPollColumn
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 2

                            Label {
                                objectName: longPollEntry.objectPrefix + "LongPollHeader"
                                text: qsTr("%1 • %2 • %3")
                                      .arg(longPollEntry.safeLabels.adapter || qsTr("n/d"))
                                      .arg(longPollEntry.safeLabels.scope || qsTr("n/d"))
                                      .arg(longPollEntry.safeLabels.environment || qsTr("n/d"))
                                color: Design.Palette.textPrimary
                                font.pixelSize: Design.Typography.body
                                font.bold: true
                            }

                            Label {
                                objectName: longPollEntry.objectPrefix + "LongPollLatency"
                                text: {
                                    const hasP50 = typeof longPollEntry.latencyStats.p50 === "number"
                                    const hasP95 = typeof longPollEntry.latencyStats.p95 === "number"
                                    if (!hasP50 && !hasP95)
                                        return qsTr("Brak próbek latencji long-pollowych")
                                    const p50 = hasP50 ? Number(longPollEntry.latencyStats.p50).toFixed(3) : qsTr("n/d")
                                    const p95 = hasP95 ? Number(longPollEntry.latencyStats.p95).toFixed(3) : qsTr("n/d")
                                    return qsTr("Latencja p50: %1 s • p95: %2 s").arg(p50).arg(p95)
                                }
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.caption
                            }

                            Label {
                                objectName: longPollEntry.objectPrefix + "LongPollErrors"
                                text: {
                                    const total = typeof longPollEntry.httpErrorStats.total === "number"
                                            ? longPollEntry.httpErrorStats.total
                                            : 0
                                    if (total === 0)
                                        return qsTr("Błędy HTTP: brak prób w ostatnich próbkach")
                                    return qsTr("Błędy HTTP: %1").arg(total)
                                }
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.caption
                            }

                            Label {
                                objectName: longPollEntry.objectPrefix + "LongPollReconnects"
                                text: {
                                    const attempts = typeof longPollEntry.reconnectStats.attempts === "number"
                                            ? longPollEntry.reconnectStats.attempts
                                            : 0
                                    const failures = typeof longPollEntry.reconnectStats.failure === "number"
                                            ? longPollEntry.reconnectStats.failure
                                            : (typeof longPollEntry.reconnectStats.failures === "number"
                                               ? longPollEntry.reconnectStats.failures
                                               : 0)
                                    return qsTr("Reconnecty: próby %1 • błędy %2").arg(attempts).arg(failures)
                                }
                                color: Design.Palette.textSecondary
                                font.pixelSize: Design.Typography.caption
                            }
                        }
                    }
                }
            }
        }
    }
}
