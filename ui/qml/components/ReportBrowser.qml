import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "."

Item {
    id: browser
    property int currentIndex: -1
    property var currentReport: ({})
    property var filteredReports: []
    property var paginationInfo: ({})
    property var availableCategoryOptions: [{
            id: "",
            label: qsTr("Wszystkie kategorie"),
            stats: {
                id: "",
                label: qsTr("Wszystkie kategorie"),
                count: 0,
                totalSize: 0,
                exportCount: 0,
                latestUpdatedAt: "",
                earliestUpdatedAt: "",
                hasSummary: false,
                hasExports: false,
                invalidSummaryCount: 0,
                missingSummaryCount: 0
            }
        }]
    property string categoryFilter: ""
    property string summaryStatusFilter: "any"
    property string exportsFilter: "any"
    property string searchFilter: ""
    property int recentDaysFilter: 0
    property var sinceFilter: null
    property var untilFilter: null
    property int limitFilter: 0
    property int offsetFilter: 0
    property string sortKeyFilter: "updated_at"
    property string sortDirectionFilter: "desc"
    property var timeRangeOptions: [
        { label: qsTr("Cała historia"), value: 0 },
        { label: qsTr("Ostatnie 7 dni"), value: 7 },
        { label: qsTr("Ostatnie 30 dni"), value: 30 },
        { label: qsTr("Ostatnie 90 dni"), value: 90 }
    ]
    property var summaryStatusOptions: [
        { label: qsTr("Wszystkie raporty"), value: "any" },
        { label: qsTr("Z poprawnym podsumowaniem"), value: "valid" },
        { label: qsTr("Bez podsumowania"), value: "missing" },
        { label: qsTr("Z błędnym podsumowaniem"), value: "invalid" }
    ]
    property var exportsOptions: [
        { label: qsTr("Dowolne eksporty"), value: "any" },
        { label: qsTr("Tylko z eksportami"), value: "yes" },
        { label: qsTr("Bez eksportów"), value: "no" },
    ]
    property var archiveFormatOptions: [
        { label: qsTr("Katalog (bez kompresji)"), value: "directory" },
        { label: qsTr("Archiwum ZIP"), value: "zip" },
        { label: qsTr("Archiwum TAR.GZ"), value: "tar" }
    ]
    property var limitOptions: [
        { label: qsTr("Wszystkie raporty"), value: 0 },
        { label: qsTr("Ostatnie 25"), value: 25 },
        { label: qsTr("Ostatnie 50"), value: 50 },
        { label: qsTr("Ostatnie 100"), value: 100 },
    ]
    property var offsetOptions: [
        { label: qsTr("Bez przesunięcia"), value: 0 },
        { label: qsTr("Pomiń 25"), value: 25 },
        { label: qsTr("Pomiń 50"), value: 50 },
        { label: qsTr("Pomiń 100"), value: 100 },
    ]
    property var sortKeyOptions: [
        { label: qsTr("Najnowsze aktualizacje"), value: "updated_at" },
        { label: qsTr("Data utworzenia"), value: "created_at" },
        { label: qsTr("Nazwa raportu"), value: "name" },
        { label: qsTr("Rozmiar eksportów"), value: "size" },
    ]
    property var sortDirectionOptions: [
        { label: qsTr("Malejąco"), value: "desc" },
        { label: qsTr("Rosnąco"), value: "asc" },
    ]
    property bool hasAnyReports: false
    property var currentCategoryStats: null
    property var equityCurveData: []
    property var assetHeatmapData: []
    readonly property bool isBusy: reportController && reportController.busy === true
    readonly property bool filtersActive: categoryFilter.length > 0 || searchFilter.length > 0
        || recentDaysFilter > 0 || summaryStatusFilter !== "any" || exportsFilter !== "any" || limitFilter > 0
        || offsetFilter > 0 || sortKeyFilter !== "updated_at" || sortDirectionFilter !== "desc"
        || sinceFilter !== null || untilFilter !== null

    function formatTimestamp(value) {
        if (!value)
            return ""
        const parsed = new Date(value)
        if (isNaN(parsed.getTime()))
            return value
        return Qt.formatDateTime(parsed, Qt.DefaultLocaleLongDate)
    }

    function formatFileSize(bytes) {
        if (bytes === undefined || bytes === null || isNaN(bytes))
            return "0 B"
        let size = Number(bytes)
        const units = ["B", "KB", "MB", "GB", "TB"]
        let unitIndex = 0
        while (size >= 1024 && unitIndex < units.length - 1) {
            size = size / 1024
            unitIndex += 1
        }
        const precision = unitIndex === 0 || size >= 10 ? 0 : 1
        return size.toFixed(precision) + " " + units[unitIndex]
    }

    function copyToClipboard(text) {
        if (!text || !Qt.application || !Qt.application.clipboard)
            return
        Qt.application.clipboard.setText(text)
    }

    function updateDashboards() {
        equityCurveData = reportController && reportController.equityCurve ? reportController.equityCurve : []
        assetHeatmapData = reportController && reportController.assetHeatmap ? reportController.assetHeatmap : []
    }

    Component.onCompleted: updateDashboards()

    Connections {
        target: reportController
        function onEquityCurveChanged() { browser.updateDashboards() }
        function onAssetHeatmapChanged() { browser.updateDashboards() }
    }

    function normalizeDate(value) {
        if (value === undefined || value === null)
            return null
        if (value instanceof Date) {
            if (isNaN(value.getTime()))
                return null
            return new Date(value.getTime())
        }
        const parsed = new Date(value)
        if (isNaN(parsed.getTime()))
            return null
        return parsed
    }

    function datesEqual(first, second) {
        const firstValid = first instanceof Date && !isNaN(first.getTime())
        const secondValid = second instanceof Date && !isNaN(second.getTime())
        if (!firstValid && !secondValid)
            return true
        if (firstValid !== secondValid)
            return false
        return first.getTime() === second.getTime()
    }

    function hasSinceFilter() {
        return sinceFilter instanceof Date && !isNaN(sinceFilter.getTime())
    }

    function hasUntilFilter() {
        return untilFilter instanceof Date && !isNaN(untilFilter.getTime())
    }

    function formatDateOnly(value) {
        const normalized = normalizeDate(value)
        if (!normalized)
            return ""
        return Qt.formatDate(normalized, Qt.DefaultLocaleShortDate)
    }

    function applySinceDate(value, triggerRefresh) {
        const normalized = normalizeDate(value)
        const shouldRefresh = triggerRefresh === undefined ? true : triggerRefresh
        const changed = !datesEqual(sinceFilter, normalized)
        if (changed)
            sinceFilter = normalized ? new Date(normalized.getTime()) : null

        if (reportController) {
            if (normalized && reportController.sinceFilter !== undefined) {
                reportController.sinceFilter = new Date(normalized.getTime())
                if (recentDaysFilter !== 0 && reportController.recentDaysFilter !== undefined) {
                    if (recentDaysFilter !== 0)
                        recentDaysFilter = 0
                    reportController.recentDaysFilter = 0
                }
            } else if (!normalized && reportController.clearSinceFilter !== undefined) {
                reportController.clearSinceFilter()
            }
        }

        if (shouldRefresh && changed)
            browser.refresh()
        return changed
    }

    function clearSinceDate(triggerRefresh) {
        const shouldRefresh = triggerRefresh === undefined ? true : triggerRefresh
        const changed = applySinceDate(null, false)
        if (shouldRefresh && changed)
            browser.refresh()
    }

    function applyUntilDate(value, triggerRefresh) {
        const normalized = normalizeDate(value)
        const shouldRefresh = triggerRefresh === undefined ? true : triggerRefresh
        const changed = !datesEqual(untilFilter, normalized)
        if (changed)
            untilFilter = normalized ? new Date(normalized.getTime()) : null

        if (reportController) {
            if (normalized && reportController.untilFilter !== undefined) {
                reportController.untilFilter = new Date(normalized.getTime())
            } else if (!normalized && reportController.clearUntilFilter !== undefined) {
                reportController.clearUntilFilter()
            }
        }

        if (shouldRefresh && changed)
            browser.refresh()
        return changed
    }

    function clearUntilDate(triggerRefresh) {
        const shouldRefresh = triggerRefresh === undefined ? true : triggerRefresh
        const changed = applyUntilDate(null, false)
        if (shouldRefresh && changed)
            browser.refresh()
    }

    function ensureLimitOption(value) {
        const normalized = Number(value || 0)
        const seen = {}
        const options = []
        for (let i = 0; i < limitOptions.length; ++i) {
            const entry = limitOptions[i] || {}
            const entryValue = Number(entry.value || 0)
            if (seen[entryValue])
                continue
            seen[entryValue] = true
            const label = entry.label !== undefined && entry.label !== null && String(entry.label).length > 0
                ? String(entry.label)
                : (entryValue > 0 ? qsTr("Ostatnie %1").arg(entryValue) : qsTr("Wszystkie raporty"))
            options.push({ label: label, value: entryValue })
        }
        if (!seen[normalized]) {
            const label = normalized > 0 ? qsTr("Ostatnie %1").arg(normalized) : qsTr("Wszystkie raporty")
            options.push({ label: label, value: normalized })
            seen[normalized] = true
        }
        if (!seen[0])
            options.push({ label: qsTr("Wszystkie raporty"), value: 0 })
        options.sort(function(a, b) { return Number(a.value || 0) - Number(b.value || 0) })
        limitOptions = options
    }

    function ensureOffsetOption(value) {
        const normalized = Number(value || 0)
        const seen = {}
        const options = []
        for (let i = 0; i < offsetOptions.length; ++i) {
            const entry = offsetOptions[i] || {}
            const entryValue = Number(entry.value || 0)
            if (seen[entryValue])
                continue
            seen[entryValue] = true
            const label = entry.label !== undefined && entry.label !== null && String(entry.label).length > 0
                ? String(entry.label)
                : (entryValue > 0 ? qsTr("Pomiń %1").arg(entryValue) : qsTr("Bez przesunięcia"))
            options.push({ label: label, value: entryValue })
        }
        if (!seen[normalized]) {
            const label = normalized > 0 ? qsTr("Pomiń %1").arg(normalized) : qsTr("Bez przesunięcia")
            options.push({ label: label, value: normalized })
            seen[normalized] = true
        }
        if (!seen[0])
            options.push({ label: qsTr("Bez przesunięcia"), value: 0 })
        options.sort(function(a, b) { return Number(a.value || 0) - Number(b.value || 0) })
        offsetOptions = options
    }

    function paginationLimitValue() {
        if (paginationInfo && paginationInfo.limit !== undefined && paginationInfo.limit !== null)
            return Number(paginationInfo.limit)
        if (paginationInfo && paginationInfo.limit === null)
            return 0
        if (reportController && reportController.limit !== undefined && reportController.limit !== null)
            return Number(reportController.limit)
        return Number(limitFilter || 0)
    }

    function paginationOffsetValue() {
        if (paginationInfo && paginationInfo.offset !== undefined && paginationInfo.offset !== null)
            return Number(paginationInfo.offset)
        if (paginationInfo && paginationInfo.offset === null)
            return 0
        if (reportController && reportController.offset !== undefined && reportController.offset !== null)
            return Number(reportController.offset)
        return Number(offsetFilter || 0)
    }

    function paginationReturnedCount() {
        if (paginationInfo && paginationInfo.returned_count !== undefined && paginationInfo.returned_count !== null)
            return Number(paginationInfo.returned_count)
        if (paginationInfo && paginationInfo.returnedCount !== undefined && paginationInfo.returnedCount !== null)
            return Number(paginationInfo.returnedCount)
        const reports = reportController && reportController.reports ? reportController.reports : []
        return reports.length
    }

    function paginationTotalCount() {
        if (paginationInfo && paginationInfo.total_count !== undefined && paginationInfo.total_count !== null)
            return Number(paginationInfo.total_count)
        if (paginationInfo && paginationInfo.totalCount !== undefined && paginationInfo.totalCount !== null)
            return Number(paginationInfo.totalCount)
        const summary = reportController && reportController.overviewStats ? reportController.overviewStats : null
        if (summary) {
            if (summary.report_count !== undefined && summary.report_count !== null)
                return Number(summary.report_count)
            if (summary.count !== undefined && summary.count !== null)
                return Number(summary.count)
        }
        return filteredReports ? filteredReports.length : 0
    }

    function canGoNextPage() {
        const limit = paginationLimitValue()
        if (limit <= 0)
            return false
        const offset = paginationOffsetValue()
        const returned = paginationReturnedCount()
        const total = paginationTotalCount()
        if (returned <= 0)
            return false
        return offset + returned < total
    }

    function canGoPreviousPage() {
        const limit = paginationLimitValue()
        const offset = paginationOffsetValue()
        return limit > 0 && offset > 0
    }

    function goToNextPage() {
        if (!canGoNextPage())
            return
        const limit = paginationLimitValue()
        let offset = paginationOffsetValue()
        offset += limit
        ensureOffsetOption(offset)
        offsetFilter = offset
        if (reportController && reportController.offset !== undefined)
            reportController.offset = offset
        refresh()
    }

    function goToPreviousPage() {
        if (!canGoPreviousPage())
            return
        const limit = paginationLimitValue()
        let offset = paginationOffsetValue()
        offset = Math.max(0, offset - limit)
        ensureOffsetOption(offset)
        offsetFilter = offset
        if (reportController && reportController.offset !== undefined)
            reportController.offset = offset
        refresh()
    }

    function refresh() {
        if (typeof reportController !== "undefined")
            reportController.refresh()
    }

    function selectReport(index) {
        currentIndex = index
        if (!filteredReports || index < 0 || index >= filteredReports.length) {
            currentReport = ({})
            return
        }
        currentReport = filteredReports[index]
    }

    function categoryIndex(identifier) {
        if (!availableCategoryOptions)
            return 0
        for (let i = 0; i < availableCategoryOptions.length; ++i) {
            if (availableCategoryOptions[i].id === identifier)
                return i
        }
        return 0
    }

    function timeRangeIndex(days) {
        const normalized = Number(days || 0)
        for (let i = 0; i < timeRangeOptions.length; ++i) {
            if (Number(timeRangeOptions[i].value) === normalized)
                return i
        }
        return 0
    }

    function limitIndex(limitValue) {
        const normalized = Number(limitValue || 0)
        for (let i = 0; i < limitOptions.length; ++i) {
            if (Number(limitOptions[i].value) === normalized)
                return i
        }
        return 0
    }

    function offsetIndex(offsetValue) {
        const normalized = Number(offsetValue || 0)
        for (let i = 0; i < offsetOptions.length; ++i) {
            if (Number(offsetOptions[i].value) === normalized)
                return i
        }
        return 0
    }

    function sortKeyIndex(sortValue) {
        const normalized = String(sortValue || "updated_at").toLowerCase()
        for (let i = 0; i < sortKeyOptions.length; ++i) {
            if (String(sortKeyOptions[i].value).toLowerCase() === normalized)
                return i
        }
        return 0
    }

    function exportsIndex(filterValue) {
        const normalized = String(filterValue || "any").toLowerCase()
        for (let i = 0; i < exportsOptions.length; ++i) {
            if (String(exportsOptions[i].value).toLowerCase() === normalized)
                return i
        }
        return 0
    }

    function sortDirectionIndex(directionValue) {
        const normalized = String(directionValue || "desc").toLowerCase()
        for (let i = 0; i < sortDirectionOptions.length; ++i) {
            if (String(sortDirectionOptions[i].value).toLowerCase() === normalized)
                return i
        }
        return 0
    }

    function summaryStatusIndex(value) {
        const normalized = String(value || "any").toLowerCase()
        for (let i = 0; i < summaryStatusOptions.length; ++i) {
            const entryValue = String(summaryStatusOptions[i].value || "").toLowerCase()
            if (entryValue === normalized)
                return i
        }
        return 0
    }

    function archiveFormatIndex(value) {
        const normalized = String(value || "directory").toLowerCase()
        for (let i = 0; i < archiveFormatOptions.length; ++i) {
            const optionValue = String(archiveFormatOptions[i].value || "").toLowerCase()
            if (optionValue === normalized)
                return i
        }
        return 0
    }

    function rebuildFilters() {
        const reports = reportController && reportController.reports ? reportController.reports : []
        const pagination = reportController && reportController.overviewPagination ? reportController.overviewPagination : null
        paginationInfo = pagination && typeof pagination === "object" ? pagination : ({})
        ensureLimitOption(paginationLimitValue())
        ensureOffsetOption(paginationOffsetValue())
        const totalFromPagination = paginationTotalCount()
        hasAnyReports = reports.length > 0 || totalFromPagination > 0

        let categoryStats = []
        let totalCount = totalFromPagination
        let totalSize = 0
        let totalExports = 0
        let latestISO = ""
        let earliestISO = ""
        let totalInvalid = 0
        let totalMissing = 0
        const summary = reportController && reportController.overviewStats ? reportController.overviewStats : null

        if (reportController && reportController.categories && reportController.categories.length > 0) {
            const source = reportController.categories
            const seenCategories = {}
            for (let i = 0; i < source.length; ++i) {
                const rawEntry = source[i] || {}
                let identifier = ""
                if (typeof rawEntry === "string" || typeof rawEntry === "number")
                    identifier = String(rawEntry)
                else if (rawEntry.id !== undefined && rawEntry.id !== null)
                    identifier = String(rawEntry.id)
                else if (rawEntry.identifier !== undefined && rawEntry.identifier !== null)
                    identifier = String(rawEntry.identifier)

                if (seenCategories[identifier])
                    continue
                seenCategories[identifier] = true

                let baseLabel = ""
                if (typeof rawEntry === "string" || typeof rawEntry === "number")
                    baseLabel = String(rawEntry)
                else if (rawEntry.label !== undefined && rawEntry.label !== null)
                    baseLabel = String(rawEntry.label)
                else if (rawEntry.name !== undefined && rawEntry.name !== null)
                    baseLabel = String(rawEntry.name)

                if (!baseLabel || baseLabel.length === 0)
                    baseLabel = identifier.length > 0 ? identifier : qsTr("Bez kategorii")

                const countValue = Number(rawEntry.count !== undefined ? rawEntry.count : rawEntry.total || 0)
                const sizeValue = Number(rawEntry.total_size !== undefined ? rawEntry.total_size : rawEntry.totalSize || 0)
                const exportCountValue = Number(rawEntry.export_count !== undefined
                    ? rawEntry.export_count
                    : rawEntry.exportCount || 0)
                const latestValue = rawEntry.latest_updated_at || rawEntry.latestUpdatedAt || ""
                const earliestValue = rawEntry.earliest_updated_at || rawEntry.earliestUpdatedAt || ""
                const hasSummaryValue = !!(rawEntry.has_summary || rawEntry.hasSummary)
                const hasExportsValue = !!(rawEntry.has_exports || rawEntry.hasExports)
                const invalidValue = Number(rawEntry.invalid_summary_count !== undefined
                        ? rawEntry.invalid_summary_count : rawEntry.invalidSummaryCount || 0)
                const missingValue = Number(rawEntry.missing_summary_count !== undefined
                    ? rawEntry.missing_summary_count : rawEntry.missingSummaryCount || 0)

                categoryStats.push({
                    id: identifier,
                    stats: {
                        id: identifier,
                        label: baseLabel,
                        count: countValue,
                        totalSize: sizeValue,
                        exportCount: exportCountValue,
                        latestUpdatedAt: latestValue,
                        earliestUpdatedAt: earliestValue,
                        hasSummary: hasSummaryValue,
                        hasExports: hasExportsValue,
                        invalidSummaryCount: invalidValue,
                        missingSummaryCount: missingValue
                    },
                })

                totalCount += countValue
                totalSize += sizeValue
                totalExports += exportCountValue
                if (latestValue && (!latestISO || new Date(latestValue) > new Date(latestISO)))
                    latestISO = latestValue
                if (earliestValue && (!earliestISO || new Date(earliestValue) < new Date(earliestISO)))
                    earliestISO = earliestValue
                totalInvalid += invalidValue
                totalMissing += missingValue
            }
        } else {
            const buckets = {}
            for (let i = 0; i < reports.length; ++i) {
                const entry = reports[i] || {}
                const identifier = entry.category || ""
                if (!buckets[identifier]) {
                    const baseLabel = identifier.length > 0 ? identifier : qsTr("Bez kategorii")
                    buckets[identifier] = {
                        id: identifier,
                        label: baseLabel,
                        count: 0,
                        totalSize: 0,
                        exportCount: 0,
                        latestUpdatedAt: "",
                        earliestUpdatedAt: "",
                        hasSummary: false,
                        hasExports: false,
                        invalidSummaryCount: 0,
                        missingSummaryCount: 0,
                    }
                }

                const bucket = buckets[identifier]
                bucket.count += 1

                const updatedAt = entry.updated_at || ""
                if (updatedAt && (!bucket.latestUpdatedAt || new Date(updatedAt) > new Date(bucket.latestUpdatedAt)))
                    bucket.latestUpdatedAt = updatedAt

                const createdAt = entry.created_at || entry.createdAt || updatedAt || ""
                if (createdAt && (!bucket.earliestUpdatedAt || new Date(createdAt) < new Date(bucket.earliestUpdatedAt)))
                    bucket.earliestUpdatedAt = createdAt

                const exportsList = entry.exports || []
                if (exportsList.length > 0)
                    bucket.hasExports = true
                bucket.exportCount += Number(entry.export_count || entry.exportCount || exportsList.length)

                const summaryError = entry.summary_error || entry.summaryError || ""
                const summaryPath = entry.summary_path || entry.summaryPath || ""
                let entryHasSummary = false
                if (entry.has_summary !== undefined && entry.has_summary !== null)
                    entryHasSummary = Boolean(entry.has_summary)
                else if (summaryPath && !summaryError)
                    entryHasSummary = true
                else if (entry.summary !== undefined && entry.summary !== null)
                    entryHasSummary = true

                if (entryHasSummary)
                    bucket.hasSummary = true
                else if (!summaryPath)
                    bucket.missingSummaryCount += 1

                if (summaryError)
                    bucket.invalidSummaryCount += 1

                for (let j = 0; j < exportsList.length; ++j)
                    bucket.totalSize += Number(exportsList[j].size || 0)
            }

            for (const key in buckets) {
                if (!Object.prototype.hasOwnProperty.call(buckets, key))
                    continue
                const bucket = buckets[key]
                categoryStats.push({ id: bucket.id, stats: bucket })
                totalCount += bucket.count
                totalSize += bucket.totalSize
                totalExports += bucket.exportCount
                if (bucket.latestUpdatedAt && (!latestISO || new Date(bucket.latestUpdatedAt) > new Date(latestISO)))
                    latestISO = bucket.latestUpdatedAt
                if (bucket.earliestUpdatedAt && (!earliestISO || new Date(bucket.earliestUpdatedAt) < new Date(earliestISO)))
                    earliestISO = bucket.earliestUpdatedAt
                totalInvalid += bucket.invalidSummaryCount
                totalMissing += bucket.missingSummaryCount
            }
        }

        if (summary) {
            if (summary.report_count !== undefined && summary.report_count !== null)
                totalCount = Number(summary.report_count)
            else if (summary.count !== undefined && summary.count !== null)
                totalCount = Number(summary.count)

            if (summary.total_size !== undefined && summary.total_size !== null)
                totalSize = Number(summary.total_size)
            else if (summary.totalSize !== undefined && summary.totalSize !== null)
                totalSize = Number(summary.totalSize)

            if (summary.export_count !== undefined && summary.export_count !== null)
                totalExports = Number(summary.export_count)
            else if (summary.exportCount !== undefined && summary.exportCount !== null)
                totalExports = Number(summary.exportCount)

            const latestValue = summary.latest_updated_at !== undefined && summary.latest_updated_at !== null
                    ? summary.latest_updated_at : summary.latestUpdatedAt
            if (latestValue !== undefined && latestValue !== null && String(latestValue).length > 0)
                latestISO = String(latestValue)
            else if (latestValue === null)
                latestISO = ""

            const earliestValue = summary.earliest_updated_at !== undefined && summary.earliest_updated_at !== null
                    ? summary.earliest_updated_at : summary.earliestUpdatedAt
            if (earliestValue !== undefined && earliestValue !== null && String(earliestValue).length > 0)
                earliestISO = String(earliestValue)
            else if (earliestValue === null)
                earliestISO = ""

            if (summary.invalid_summary_count !== undefined && summary.invalid_summary_count !== null)
                totalInvalid = Number(summary.invalid_summary_count)
            else if (summary.invalidSummaryCount !== undefined && summary.invalidSummaryCount !== null)
                totalInvalid = Number(summary.invalidSummaryCount)

            if (summary.missing_summary_count !== undefined && summary.missing_summary_count !== null)
                totalMissing = Number(summary.missing_summary_count)
            else if (summary.missingSummaryCount !== undefined && summary.missingSummaryCount !== null)
                totalMissing = Number(summary.missingSummaryCount)
        }

        categoryStats.sort(function(a, b) {
            return a.stats.label.localeCompare(b.stats.label)
        })

        let overallHasSummary = categoryStats.some(function(item) { return item.stats.hasSummary })
        let overallHasExports = categoryStats.some(function(item) { return item.stats.hasExports })
        if (summary) {
            if (summary.has_summary !== undefined && summary.has_summary !== null)
                overallHasSummary = Boolean(summary.has_summary)
            else if (summary.hasSummary !== undefined && summary.hasSummary !== null)
                overallHasSummary = Boolean(summary.hasSummary)

            if (summary.has_exports !== undefined && summary.has_exports !== null)
                overallHasExports = Boolean(summary.has_exports)
            else if (summary.hasExports !== undefined && summary.hasExports !== null)
                overallHasExports = Boolean(summary.hasExports)
        }

        const options = []
        const overallLabel = totalCount > 0 ? qsTr("Wszystkie kategorie (%1)").arg(totalCount) : qsTr("Wszystkie kategorie")
        const overallStats = {
            id: "",
            label: qsTr("Wszystkie kategorie"),
            count: totalCount,
            totalSize: totalSize,
            exportCount: totalExports,
            latestUpdatedAt: latestISO,
            earliestUpdatedAt: earliestISO,
            hasSummary: overallHasSummary,
            hasExports: overallHasExports,
            invalidSummaryCount: totalInvalid,
            missingSummaryCount: totalMissing,
        }
        if (summary) {
            let categoryCountValue = null
            if (summary.category_count !== undefined && summary.category_count !== null)
                categoryCountValue = Number(summary.category_count)
            else if (summary.categoryCount !== undefined && summary.categoryCount !== null)
                categoryCountValue = Number(summary.categoryCount)
            if (categoryCountValue !== null)
                overallStats.categoryCount = categoryCountValue
        }
        options.push({ id: "", label: overallLabel, stats: overallStats })

        for (let i = 0; i < categoryStats.length; ++i) {
            const entry = categoryStats[i]
            const stats = entry.stats
            const decoratedLabel = stats.count > 0 ? qsTr("%1 (%2)").arg(stats.label).arg(stats.count) : stats.label
            options.push({ id: entry.id, label: decoratedLabel, stats: stats })
        }
        availableCategoryOptions = options

        const categoryExists = categoryStats.some(function(entry) { return entry.id === categoryFilter })
        if (categoryFilter && !categoryExists) {
            categoryFilter = ""
            return
        }

        let statsForCurrent = null
        for (let i = 0; i < options.length; ++i) {
            if (options[i].id === categoryFilter) {
                statsForCurrent = options[i].stats || null
                break
            }
        }
        if (!statsForCurrent && options.length > 0)
            statsForCurrent = options[0].stats || null
        currentCategoryStats = statsForCurrent

        const activeSearch = String(searchFilter || "").toLowerCase()
        const filtered = []
        for (let i = 0; i < reports.length; ++i) {
            const entry = reports[i]
            if (categoryFilter && entry.category !== categoryFilter)
                continue
            if (activeSearch.length > 0) {
                const haystack = [
                    entry.display_name || "",
                    entry.relative_path || "",
                    entry.category || "",
                ]
                    .join(" ")
                    .toLowerCase()
                if (haystack.indexOf(activeSearch) === -1)
                    continue
            }
            filtered.push(entry)
        }

        filteredReports = filtered

        const previousRelative = currentReport && currentReport.relative_path ? currentReport.relative_path : ""
        let targetIndex = -1
        if (previousRelative) {
            for (let i = 0; i < filtered.length; ++i) {
                if (filtered[i].relative_path === previousRelative) {
                    targetIndex = i
                    break
                }
            }
        }
        if (targetIndex === -1 && filtered.length > 0)
            targetIndex = 0

        selectReport(targetIndex)
    }

    onCategoryFilterChanged: rebuildFilters()
    onSearchFilterChanged: rebuildFilters()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        padding: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label {
                text: qsTr("Pipeline raportów")
                font.pixelSize: 20
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            ComboBox {
                id: categoryCombo
                Layout.preferredWidth: 200
                model: browser.availableCategoryOptions
                textRole: "label"
                valueRole: "id"
                currentIndex: browser.categoryIndex(browser.categoryFilter)
                onActivated: {
                    const value = categoryCombo.currentValue || ""
                    if (browser.categoryFilter === value)
                        return
                    browser.categoryFilter = value
                    if (reportController && reportController.categoryFilter !== undefined)
                        reportController.categoryFilter = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: summaryStatusCombo
                Layout.preferredWidth: 220
                model: browser.summaryStatusOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.summaryStatusIndex(browser.summaryStatusFilter)
                onActivated: {
                    const value = summaryStatusCombo.currentValue || "any"
                    if (browser.summaryStatusFilter === value)
                        return
                    browser.summaryStatusFilter = value
                    if (reportController && reportController.summaryStatusFilter !== undefined)
                        reportController.summaryStatusFilter = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: exportsCombo
                Layout.preferredWidth: 200
                model: browser.exportsOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.exportsIndex(browser.exportsFilter)
                onActivated: {
                    const value = exportsCombo.currentValue || "any"
                    if (browser.exportsFilter === value)
                        return
                    browser.exportsFilter = value
                    if (reportController && reportController.exportsFilter !== undefined)
                        reportController.exportsFilter = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: timeRangeCombo
                Layout.preferredWidth: 180
                model: browser.timeRangeOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.timeRangeIndex(browser.recentDaysFilter)
                onActivated: {
                    const value = Number(timeRangeCombo.currentValue || 0)
                    if (browser.recentDaysFilter === value)
                        return
                    browser.recentDaysFilter = value
                    if (reportController && reportController.recentDaysFilter !== undefined)
                        reportController.recentDaysFilter = value
                    browser.refresh()
                }
            }

            RowLayout {
                Layout.preferredWidth: 220
                Layout.alignment: Qt.AlignVCenter
                spacing: 4

                Button {
                    id: sinceButton
                    Layout.fillWidth: true
                    text: browser.hasSinceFilter()
                        ? qsTr("Od: %1").arg(browser.formatDateOnly(browser.sinceFilter))
                        : qsTr("Data od")
                    enabled: !browser.isBusy
                    onClicked: sinceDialog.open()
                }

                ToolButton {
                    text: "✕"
                    visible: browser.hasSinceFilter()
                    enabled: !browser.isBusy
                    onClicked: browser.clearSinceDate()
                }
            }

            RowLayout {
                Layout.preferredWidth: 220
                Layout.alignment: Qt.AlignVCenter
                spacing: 4

                Button {
                    id: untilButton
                    Layout.fillWidth: true
                    text: browser.hasUntilFilter()
                        ? qsTr("Do: %1").arg(browser.formatDateOnly(browser.untilFilter))
                        : qsTr("Data do")
                    enabled: !browser.isBusy
                    onClicked: untilDialog.open()
                }

                ToolButton {
                    text: "✕"
                    visible: browser.hasUntilFilter()
                    enabled: !browser.isBusy
                    onClicked: browser.clearUntilDate()
                }
            }

            ComboBox {
                id: limitCombo
                Layout.preferredWidth: 160
                model: browser.limitOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.limitIndex(browser.limitFilter)
                onActivated: {
                    const value = Number(limitCombo.currentValue || 0)
                    if (browser.limitFilter === value)
                        return
                    browser.limitFilter = value
                    if (reportController && reportController.limit !== undefined)
                        reportController.limit = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: offsetCombo
                Layout.preferredWidth: 180
                model: browser.offsetOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.offsetIndex(browser.offsetFilter)
                onActivated: {
                    const value = Number(offsetCombo.currentValue || 0)
                    if (browser.offsetFilter === value)
                        return
                    browser.offsetFilter = value
                    if (reportController && reportController.offset !== undefined)
                        reportController.offset = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: sortKeyCombo
                Layout.preferredWidth: 200
                model: browser.sortKeyOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.sortKeyIndex(browser.sortKeyFilter)
                onActivated: {
                    const value = String(sortKeyCombo.currentValue || "updated_at")
                    if (browser.sortKeyFilter === value)
                        return
                    browser.sortKeyFilter = value
                    if (reportController && reportController.sortKey !== undefined)
                        reportController.sortKey = value
                    browser.refresh()
                }
            }

            ComboBox {
                id: sortDirectionCombo
                Layout.preferredWidth: 160
                model: browser.sortDirectionOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.sortDirectionIndex(browser.sortDirectionFilter)
                onActivated: {
                    const value = String(sortDirectionCombo.currentValue || "desc")
                    if (browser.sortDirectionFilter === value)
                        return
                    browser.sortDirectionFilter = value
                    if (reportController && reportController.sortDirection !== undefined)
                        reportController.sortDirection = value
                    browser.refresh()
                }
            }

            TextField {
                id: searchField
                Layout.preferredWidth: 200
                placeholderText: qsTr("Szukaj")
                inputMethodHints: Qt.ImhNoPredictiveText | Qt.ImhNoAutoUppercase
                onTextChanged: searchDebounce.restart()
            }

            Button {
                text: qsTr("Wyczyść filtry")
                enabled: browser.filtersActive && !browser.isBusy
                onClicked: {
                    searchDebounce.stop()
                    const hadSearch = browser.searchFilter.length > 0
                    const hadTimeFilter = browser.recentDaysFilter > 0
                    const hadCategoryFilter = browser.categoryFilter.length > 0
                    const hadSummaryStatus = browser.summaryStatusFilter !== "any"
                    const hadExportsFilter = browser.exportsFilter !== "any"
                    const hadLimitFilter = browser.limitFilter > 0
                    const hadOffsetFilter = browser.offsetFilter > 0
                    const hadSortKey = browser.sortKeyFilter !== "updated_at"
                    const hadSortDirection = browser.sortDirectionFilter !== "desc"
                    const hadSinceFilter = browser.hasSinceFilter()
                    const hadUntilFilter = browser.hasUntilFilter()
                    browser.categoryFilter = ""
                    browser.searchFilter = ""
                    searchField.text = ""
                    if (reportController && reportController.categoryFilter !== undefined)
                        reportController.categoryFilter = ""
                    if (reportController && reportController.searchQuery !== undefined)
                        reportController.searchQuery = ""
                    if (hadTimeFilter) {
                        browser.recentDaysFilter = 0
                        if (reportController && reportController.recentDaysFilter !== undefined)
                            reportController.recentDaysFilter = 0
                    }
                    if (hadSummaryStatus) {
                        browser.summaryStatusFilter = "any"
                        if (reportController && reportController.summaryStatusFilter !== undefined)
                            reportController.summaryStatusFilter = "any"
                    }
                    if (hadExportsFilter) {
                        browser.exportsFilter = "any"
                        if (reportController && reportController.exportsFilter !== undefined)
                            reportController.exportsFilter = "any"
                    }
                    if (hadLimitFilter) {
                        browser.limitFilter = 0
                        if (reportController && reportController.limit !== undefined)
                            reportController.limit = 0
                    }
                    if (hadOffsetFilter) {
                        browser.offsetFilter = 0
                        if (reportController && reportController.offset !== undefined)
                            reportController.offset = 0
                    }
                    if (hadSortKey) {
                        browser.sortKeyFilter = "updated_at"
                        if (reportController && reportController.sortKey !== undefined)
                            reportController.sortKey = "updated_at"
                    }
                    if (hadSortDirection) {
                        browser.sortDirectionFilter = "desc"
                        if (reportController && reportController.sortDirection !== undefined)
                            reportController.sortDirection = "desc"
                    }
                    if (hadSinceFilter)
                        browser.applySinceDate(null, false)
                    if (hadUntilFilter)
                        browser.applyUntilDate(null, false)
                    if (hadSearch || hadTimeFilter || hadCategoryFilter || hadSummaryStatus || hadExportsFilter || hadLimitFilter
                        || hadOffsetFilter || hadSortKey || hadSortDirection || hadSinceFilter || hadUntilFilter)
                        browser.refresh()
                }
            }

            Button {
                text: qsTr("Usuń filtrowane")
                enabled: reportController && reportController.purgeReports !== undefined
                    && reportController.previewPurgeReports !== undefined
                    && !browser.isBusy
                onClicked: {
                    if (!reportController)
                        return
                    purgeDialog.open()
                }
            }

            Button {
                text: qsTr("Archiwizuj filtrowane")
                enabled: reportController && reportController.archiveReports !== undefined
                    && reportController.previewArchiveReports !== undefined
                    && !browser.isBusy
                onClicked: {
                    if (!reportController)
                        return
                    archiveDialog.open()
                }
            }

            Button {
                text: qsTr("Odśwież")
                enabled: !browser.isBusy
                onClicked: browser.refresh()
            }

            BusyIndicator {
                Layout.preferredWidth: 24
                Layout.preferredHeight: 24
                running: browser.isBusy
                visible: browser.isBusy
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: browser.paginationTotalCount() > 0
                && (browser.paginationLimitValue() > 0 || browser.paginationOffsetValue() > 0
                    || browser.paginationReturnedCount() < browser.paginationTotalCount())

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                color: palette.mid
                text: {
                    const total = browser.paginationTotalCount()
                    const returned = browser.paginationReturnedCount()
                    const offset = browser.paginationOffsetValue()
                    const limit = browser.paginationLimitValue()
                    if (total === 0)
                        return qsTr("Brak raportów do wyświetlenia.")
                    if (returned === 0)
                        return qsTr("Brak raportów na tej stronie (offset %1 z %2 dostępnych).").arg(offset).arg(total)
                    if (limit > 0)
                        return qsTr("Wyświetlono %1 z %2 raportów (offset %3).").arg(returned).arg(total).arg(offset)
                    return qsTr("Wyświetlono %1 raportów z %2 dostępnych.").arg(returned).arg(total)
                }
            }

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Poprzednia strona")
                enabled: browser.canGoPreviousPage() && !browser.isBusy
                visible: browser.paginationLimitValue() > 0 || browser.paginationOffsetValue() > 0
                onClicked: browser.goToPreviousPage()
            }

            Button {
                text: qsTr("Następna strona")
                enabled: browser.canGoNextPage() && !browser.isBusy
                visible: browser.paginationLimitValue() > 0
                onClicked: browser.goToNextPage()
            }
        }

        Label {
            Layout.fillWidth: true
            visible: reportController && reportController.lastNotification
                && reportController.lastNotification.length > 0
            text: reportController ? reportController.lastNotification : ""
            color: Qt.rgba(0.2, 0.6, 0.3, 1)
            wrapMode: Text.WordWrap
        }

        Label {
            Layout.fillWidth: true
            visible: reportController && reportController.lastError.length > 0
            text: reportController ? reportController.lastError : ""
            color: Qt.rgba(0.9, 0.4, 0.3, 1)
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            visible: browser.equityCurveData.length > 0 || browser.assetHeatmapData.length > 0

            EquityCurveDashboard {
                Layout.fillWidth: true
                Layout.preferredHeight: 260
                visible: browser.equityCurveData.length > 0
                points: browser.equityCurveData
            }

            AssetHeatmapDashboard {
                Layout.fillWidth: true
                Layout.preferredHeight: 260
                Layout.preferredWidth: parent ? parent.width / 2 : 0
                visible: browser.assetHeatmapData.length > 0
                cells: browser.assetHeatmapData
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            ListView {
                id: reportList
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                clip: true
                model: browser.filteredReports
                currentIndex: browser.currentIndex
                highlightFollowsCurrentItem: true
                highlightMoveDuration: 120
                highlight: Rectangle {
                    color: Qt.rgba(0.2, 0.4, 0.6, 0.25)
                    radius: 6
                }

                delegate: ItemDelegate {
                    required property var modelData
                    text: (modelData.display_name || modelData.relative_path)
                    width: parent ? parent.width : implicitWidth
                    highlighted: index === reportList.currentIndex
                    onClicked: {
                        browser.selectReport(index)
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: reportList.count === 0 && !browser.isBusy
                    text: browser.hasAnyReports ? qsTr("Brak raportów spełniających filtr") : qsTr("Brak raportów")
                    color: palette.mid
                }

                ScrollBar.vertical: ScrollBar {}
            }

            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    width: parent.width
                    spacing: 8
                    padding: 12

                    Label {
                        text: browser.currentReport.display_name || qsTr("Wybierz raport")
                        font.pixelSize: 18
                        font.bold: true
                    }

                    Label {
                        visible: browser.currentReport.updated_at !== undefined && browser.currentReport.updated_at !== ""
                        text: browser.formatTimestamp(browser.currentReport.updated_at)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentReport.created_at !== undefined && browser.currentReport.created_at !== ""
                        text: qsTr("Pierwsza aktualizacja: %1").arg(browser.formatTimestamp(browser.currentReport.created_at))
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentReport.category !== undefined && browser.currentReport.category !== ""
                        text: qsTr("Kategoria: %1").arg(browser.currentReport.category)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats !== null
                        text: browser.categoryFilter.length > 0
                            ? qsTr("Raporty w kategorii: %1").arg(browser.currentCategoryStats.count || 0)
                            : qsTr("Łącznie raportów: %1").arg(browser.currentCategoryStats.count || 0)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats
                            && (!browser.categoryFilter || browser.categoryFilter.length === 0)
                            && browser.currentCategoryStats.categoryCount !== undefined
                        text: qsTr("Liczba kategorii: %1").arg(browser.currentCategoryStats.categoryCount || 0)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && browser.currentCategoryStats.totalSize !== undefined
                        text: qsTr("Łączny rozmiar eksportów: %1").arg(
                            browser.formatFileSize(browser.currentCategoryStats ? browser.currentCategoryStats.totalSize || 0 : 0))
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && browser.currentCategoryStats.exportCount !== undefined
                        text: qsTr("Łączna liczba eksportów: %1").arg(
                            browser.currentCategoryStats ? browser.currentCategoryStats.exportCount || 0 : 0)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && browser.currentCategoryStats.latestUpdatedAt
                        text: qsTr("Ostatnia aktualizacja: %1").arg(
                            browser.formatTimestamp(browser.currentCategoryStats.latestUpdatedAt))
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && browser.currentCategoryStats.earliestUpdatedAt
                        text: qsTr("Pierwsza dostępna aktualizacja: %1").arg(
                            browser.formatTimestamp(browser.currentCategoryStats.earliestUpdatedAt))
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && browser.categoryFilter.length > 0
                        text: browser.currentCategoryStats && browser.currentCategoryStats.hasSummary
                            ? qsTr("W tej kategorii dostępne są podsumowania raportów.")
                            : qsTr("Brak podsumowań w tej kategorii.")
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats
                            && (browser.currentCategoryStats.missingSummaryCount || 0) > 0
                        text: qsTr("Raporty bez podsumowania: %1").arg(
                            browser.currentCategoryStats.missingSummaryCount || 0)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentCategoryStats && (browser.currentCategoryStats.invalidSummaryCount || 0) > 0
                        text: qsTr("Nieprawidłowe podsumowania: %1").arg(browser.currentCategoryStats.invalidSummaryCount || 0)
                        color: Qt.rgba(0.9, 0.4, 0.3, 1)
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        visible: browser.currentReport.absolute_path !== undefined && browser.currentReport.absolute_path !== ""

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("Ścieżka: %1").arg(browser.currentReport.absolute_path)
                            elide: Label.ElideMiddle
                            color: palette.mid
                        }

                        Button {
                            text: qsTr("Kopiuj ścieżkę")
                            enabled: !browser.isBusy
                            onClicked: browser.copyToClipboard(browser.currentReport.absolute_path)
                        }

                        Button {
                            text: qsTr("Pokaż w katalogu")
                            enabled: !browser.isBusy && reportController && reportController.revealReport !== undefined
                                && ((browser.currentReport.relative_path && browser.currentReport.relative_path.length > 0)
                                    || (browser.currentReport.summary_path && browser.currentReport.summary_path.length > 0))
                            onClicked: {
                                if (!reportController)
                                    return
                                const relative = browser.currentReport.relative_path || ""
                                if (relative.length === 0 && browser.currentReport.summary_path)
                                    reportController.revealReport(browser.currentReport.summary_path)
                                else
                                    reportController.revealReport(relative)
                            }
                        }

                        Button {
                            text: qsTr("Usuń")
                            enabled: !browser.isBusy && reportController && reportController.deleteReport !== undefined
                                && browser.currentReport.relative_path && browser.currentReport.relative_path.length > 0
                            onClicked: {
                                if (!reportController)
                                    return
                                deleteDialog.reportPath = browser.currentReport.relative_path || ""
                                deleteDialog.reportLabel = browser.currentReport.display_name
                                    || browser.currentReport.relative_path || ""
                                deleteDialog.open()
                            }
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        visible: browser.filteredReports.length === 0 && browser.hasAnyReports && !browser.isBusy
                        text: browser.paginationOffsetValue() > 0
                            ? qsTr("Brak raportów na tej stronie. Użyj nawigacji lub zresetuj przesunięcie.")
                            : qsTr("Brak raportów spełniających aktywne filtry.")
                        color: palette.mid
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        visible: browser.filteredReports.length === 0 && !browser.hasAnyReports && !browser.isBusy
                        text: qsTr("Panel raportów jest pusty. Uruchom pipeline raportujący, aby zobaczyć wyniki.")
                        color: palette.mid
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        visible: browser.currentReport.summary_error && browser.currentReport.summary_error.length > 0
                        text: qsTr("Błąd parsowania podsumowania: %1").arg(browser.currentReport.summary_error)
                        color: Qt.rgba(0.9, 0.4, 0.3, 1)
                    }

                    Label {
                        visible: browser.currentReport.export_count !== undefined
                        text: qsTr("Eksporty w raporcie: %1").arg(browser.currentReport.export_count || 0)
                        color: palette.mid
                    }

                    Label {
                        visible: browser.currentReport.total_size !== undefined
                        text: qsTr("Rozmiar raportu: %1").arg(
                            browser.formatFileSize(browser.currentReport.total_size || 0))
                        color: palette.mid
                    }

                    TextArea {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 140
                        readOnly: true
                        wrapMode: TextEdit.Wrap
                        visible: browser.currentReport.summary && (!browser.currentReport.summary_error || browser.currentReport.summary_error.length === 0)
                        text: browser.currentReport.summary ? JSON.stringify(browser.currentReport.summary, null, 2) : ""
                    }

                    ListView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.min(contentHeight, 200)
                        model: browser.currentReport.exports || []
                        clip: true

                        delegate: Frame {
                            required property var modelData
                            Layout.fillWidth: true
                            padding: 8

                            RowLayout {
                                anchors.fill: parent
                                spacing: 8

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2
                                    Label {
                                        text: modelData.relative_path
                                        font.bold: true
                                    }
                                    Label {
                                        text: qsTr("Rozmiar: %1").arg(browser.formatFileSize(modelData.size || 0))
                                        color: palette.mid
                                    }
                                }

                                Button {
                                    text: qsTr("Eksportuj")
                                    enabled: !browser.isBusy
                                    onClicked: {
                                        exportDialog.reportPath = modelData.relative_path
                                        exportDialog.open()
                                    }
                                }

                                Button {
                                    text: qsTr("Otwórz")
                                    enabled: !browser.isBusy && reportController && reportController.openExport !== undefined
                                        && modelData.absolute_path && modelData.absolute_path.length > 0
                                    onClicked: {
                                        if (!reportController)
                                            return
                                        reportController.openExport(modelData.relative_path)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: sinceDialog
        modal: true
        title: qsTr("Wybierz datę początkową")
        standardButtons: DialogButtonBox.Ok | DialogButtonBox.Cancel
        enabled: !browser.isBusy
        onOpened: {
            const current = browser.hasSinceFilter() ? browser.sinceFilter : new Date()
            sincePicker.date = current
        }
        onAccepted: browser.applySinceDate(sincePicker.date)
        contentItem: ColumnLayout {
            width: 280
            spacing: 12
            padding: 12

            DatePicker {
                id: sincePicker
                Layout.fillWidth: true
                from: new Date(2000, 0, 1)
                to: new Date(2100, 11, 31)
            }
        }
    }

    Dialog {
        id: untilDialog
        modal: true
        title: qsTr("Wybierz datę końcową")
        standardButtons: DialogButtonBox.Ok | DialogButtonBox.Cancel
        enabled: !browser.isBusy
        onOpened: {
            const current = browser.hasUntilFilter() ? browser.untilFilter : new Date()
            untilPicker.date = current
        }
        onAccepted: browser.applyUntilDate(untilPicker.date)
        contentItem: ColumnLayout {
            width: 280
            spacing: 12
            padding: 12

            DatePicker {
                id: untilPicker
                Layout.fillWidth: true
                from: new Date(2000, 0, 1)
                to: new Date(2100, 11, 31)
            }
        }
    }

    Dialog {
        id: deleteDialog
        modal: true
        title: qsTr("Usuń raport")
        property string reportPath: ""
        property string reportLabel: ""
        property var preview: null
        property string previewError: ""
        property string activeRequestPath: ""
        property string pendingPath: ""
        property string pendingLabel: ""
        onOpened: {
            if (!reportLabel || reportLabel.length === 0)
                reportLabel = reportPath
            previewError = ""
            preview = null
            activeRequestPath = reportPath
            pendingPath = ""
            pendingLabel = ""
            if (reportController && reportController.previewDeleteReport !== undefined && reportPath
                    && reportPath.length > 0)
                reportController.previewDeleteReport(reportPath)
        }
        onAccepted: {
            if (!reportController || !reportPath || reportPath.length === 0)
                return
            pendingPath = reportPath
            pendingLabel = reportLabel
            activeRequestPath = reportPath
            const started = reportController.deleteReport(reportPath)
            if (!started)
                Qt.callLater(function() { deleteDialog.open() })
            reportPath = ""
            reportLabel = ""
            preview = null
            previewError = ""
        }
        onRejected: {
            reportPath = ""
            reportLabel = ""
            preview = null
            previewError = ""
            activeRequestPath = ""
            pendingPath = ""
            pendingLabel = ""
        }
        contentItem: ColumnLayout {
            width: 320
            spacing: 12
            padding: 12

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: reportLabel && reportLabel.length > 0
                    ? qsTr("Czy na pewno chcesz usunąć raport \"%1\"? Operacja jest nieodwracalna.").arg(reportLabel)
                    : qsTr("Czy na pewno chcesz usunąć wybrany raport? Operacja jest nieodwracalna.")
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: reportPath && reportPath.length > 0
                text: qsTr("Ścieżka raportu: %1").arg(reportPath)
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: preview && preview.status === "preview"
                text: qsTr("Usunięcie obejmie %1 plików i %2 katalogów. Zwolnione miejsce: %3.")
                    .arg(preview.removed_files !== undefined ? preview.removed_files : 0)
                    .arg(preview.removed_directories !== undefined ? preview.removed_directories : 0)
                    .arg(browser.formatFileSize(preview.removed_size !== undefined ? preview.removed_size : 0))
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: preview && preview.status === "not_found"
                text: qsTr("Wybrany raport nie istnieje lub został już usunięty.")
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: previewError && previewError.length > 0
                text: previewError
                font.bold: true
            }
        }
        footer: DialogButtonBox {
            standardButtons: DialogButtonBox.Ok | DialogButtonBox.Cancel
            enabled: !browser.isBusy
            onAccepted: deleteDialog.accept()
            onRejected: deleteDialog.reject()
            Component.onCompleted: {
                if (button(StandardButton.Ok))
                    button(StandardButton.Ok).text = qsTr("Usuń")
                if (button(StandardButton.Cancel))
                    button(StandardButton.Cancel).text = qsTr("Anuluj")
            }
        }
        Connections {
            target: reportController
            function onDeletePreviewReady(path, result) {
                if (path !== deleteDialog.activeRequestPath)
                    return
                deleteDialog.preview = result
                if (result && result.status === "error" && result.error)
                    deleteDialog.previewError = String(result.error)
                else
                    deleteDialog.previewError = ""
            }
            function onDeleteFinished(path, success) {
                if (path !== deleteDialog.pendingPath && path !== deleteDialog.activeRequestPath)
                    return
                deleteDialog.activeRequestPath = ""
                if (!success) {
                    const restorePath = deleteDialog.pendingPath
                    const restoreLabel = deleteDialog.pendingLabel
                    deleteDialog.pendingPath = ""
                    deleteDialog.pendingLabel = ""
                    if (restorePath && restorePath.length > 0) {
                        deleteDialog.reportPath = restorePath
                        deleteDialog.reportLabel = restoreLabel
                    }
                    deleteDialog.preview = null
                    deleteDialog.previewError = ""
                    Qt.callLater(function() { deleteDialog.open() })
                } else {
                    deleteDialog.pendingPath = ""
                    deleteDialog.pendingLabel = ""
                }
            }
        }
    }

    Dialog {
        id: archiveDialog
        modal: true
        title: qsTr("Archiwizuj filtrowane raporty")
        property var preview: null
        property string previewError: ""
        property string destinationPath: ""
        property bool overwriteExisting: false
        property string archiveFormat: "directory"
        property string pendingDestination: ""
        property bool pendingOverwrite: false
        property string pendingFormat: "directory"
        enabled: !browser.isBusy

        function updatePreview() {
            preview = null
            previewError = ""
            if (!reportController || reportController.previewArchiveReports === undefined)
                return
            pendingDestination = String(destinationPath || "").trim()
            pendingOverwrite = overwriteExisting
            var normalizedFormat = String(archiveDialog.archiveFormat || "directory").trim().toLowerCase()
            if (normalizedFormat.length === 0)
                normalizedFormat = "directory"
            pendingFormat = normalizedFormat
            reportController.previewArchiveReports(destinationPath, overwriteExisting, archiveDialog.archiveFormat)
        }

        onOpened: {
            overwriteExisting = false
            preview = null
            previewError = ""
            if (reportController && reportController.archiveFormat !== undefined)
                archiveDialog.archiveFormat = reportController.archiveFormat
            else
                archiveDialog.archiveFormat = "directory"
            if (reportController && reportController.defaultArchiveDestination !== undefined)
                destinationPath = reportController.defaultArchiveDestination()
            else
                destinationPath = ""
            destinationField.text = destinationPath
            archiveFormatCombo.currentIndex = browser.archiveFormatIndex(archiveDialog.archiveFormat)
            updatePreview()
        }

        onAccepted: {
            if (!reportController)
                return
            const started = reportController.archiveReports(destinationPath, overwriteExisting, archiveDialog.archiveFormat)
            if (!started)
                Qt.callLater(function() { archiveDialog.open() })
            preview = null
            previewError = ""
        }

        onRejected: {
            preview = null
            previewError = ""
            pendingDestination = ""
            pendingOverwrite = false
            pendingFormat = "directory"
        }

        contentItem: ColumnLayout {
            width: 420
            spacing: 12
            padding: 12

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: qsTr("Zarchiwizuj raporty spełniające aktywne filtry do wskazanego katalogu docelowego.")
            }

            RowLayout {
                Layout.fillWidth: true

                TextField {
                    id: destinationField
                    Layout.fillWidth: true
                    text: archiveDialog.destinationPath
                    placeholderText: qsTr("Ścieżka katalogu archiwum")
                    enabled: !browser.isBusy
                    onEditingFinished: {
                        const value = text.trim()
                        if (value !== archiveDialog.destinationPath) {
                            archiveDialog.destinationPath = value
                            archiveDialog.updatePreview()
                        }
                    }
                }

                Button {
                    text: qsTr("Wybierz…")
                    enabled: !browser.isBusy
                    onClicked: archiveFolderDialog.open()
                }
            }

            Label {
                Layout.fillWidth: true
                text: qsTr("Format archiwum")
                color: palette.mid
            }

            ComboBox {
                id: archiveFormatCombo
                Layout.fillWidth: true
                model: browser.archiveFormatOptions
                textRole: "label"
                valueRole: "value"
                currentIndex: browser.archiveFormatIndex(archiveDialog.archiveFormat)
                enabled: !browser.isBusy
                onActivated: {
                    const value = archiveFormatCombo.currentValue || "directory"
                    if (archiveDialog.archiveFormat === value)
                        return
                    archiveDialog.archiveFormat = value
                    if (reportController && reportController.archiveFormat !== undefined && reportController.archiveFormat !== value)
                        reportController.archiveFormat = value
                    archiveDialog.updatePreview()
                }
            }

            CheckBox {
                text: qsTr("Nadpisz istniejące raporty")
                checked: archiveDialog.overwriteExisting
                enabled: !browser.isBusy
                onToggled: {
                    archiveDialog.overwriteExisting = checked
                    archiveDialog.updatePreview()
                }
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: archiveDialog.preview && (archiveDialog.preview.status === "preview"
                        || archiveDialog.preview.status === "partial_failure"
                        || archiveDialog.preview.status === "completed")
                text: qsTr("Wybrano %1 z %2 raportów do archiwizacji.")
                    .arg(archiveDialog.preview && archiveDialog.preview.planned_count !== undefined ? archiveDialog.preview.planned_count : 0)
                    .arg(archiveDialog.preview && archiveDialog.preview.matched_count !== undefined ? archiveDialog.preview.matched_count : 0)
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: archiveDialog.preview && archiveDialog.preview.status === "preview"
                text: qsTr("Operacja obejmie %1 plików i %2 katalogów. Zostanie skopiowane %3 danych.")
                    .arg(archiveDialog.preview && archiveDialog.preview.copied_files !== undefined ? archiveDialog.preview.copied_files : 0)
                    .arg(archiveDialog.preview && archiveDialog.preview.copied_directories !== undefined ? archiveDialog.preview.copied_directories : 0)
                    .arg(browser.formatFileSize(archiveDialog.preview && archiveDialog.preview.copied_size !== undefined ? archiveDialog.preview.copied_size : 0))
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: archiveDialog.preview && archiveDialog.preview.status === "empty"
                text: qsTr("Brak raportów spełniających aktualne filtry.")
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: archiveDialog.previewError && archiveDialog.previewError.length > 0
                text: archiveDialog.previewError
                font.bold: true
            }
        }

        footer: DialogButtonBox {
            standardButtons: DialogButtonBox.Ok | DialogButtonBox.Cancel
            enabled: !browser.isBusy
            onAccepted: archiveDialog.accept()
            onRejected: archiveDialog.reject()
            Component.onCompleted: {
                if (button(StandardButton.Ok))
                    button(StandardButton.Ok).text = qsTr("Archiwizuj")
                if (button(StandardButton.Cancel))
                    button(StandardButton.Cancel).text = qsTr("Anuluj")
            }
        }

        Connections {
            target: reportController
            function onArchiveFormatChanged() {
                if (!reportController || reportController.archiveFormat === undefined)
                    return
                archiveDialog.archiveFormat = reportController.archiveFormat
                archiveFormatCombo.currentIndex = browser.archiveFormatIndex(archiveDialog.archiveFormat)
                if (archiveDialog.visible)
                    archiveDialog.updatePreview()
            }
        }

        Connections {
            target: reportController
            function onArchivePreviewReady(destination, overwrite, format, result) {
                const normalizedDestination = String(destination || "").trim()
                var normalizedFormat = String(format || "directory").trim().toLowerCase()
                if (normalizedFormat.length === 0)
                    normalizedFormat = "directory"
                if (normalizedDestination !== archiveDialog.pendingDestination
                        || Boolean(overwrite) !== Boolean(archiveDialog.pendingOverwrite)
                        || normalizedFormat !== archiveDialog.pendingFormat)
                    return
                archiveDialog.preview = result
                if (result && result.status === "error" && result.error)
                    archiveDialog.previewError = String(result.error)
                else
                    archiveDialog.previewError = ""
            }
            function onArchiveFinished(success) {
                if (!success)
                    Qt.callLater(function() { archiveDialog.open() })
                else {
                    archiveDialog.preview = null
                    archiveDialog.previewError = ""
                }
            }
        }
    }

    Dialog {
        id: purgeDialog
        modal: true
        title: qsTr("Usuń filtrowane raporty")
        property var preview: null
        property string previewError: ""
        onOpened: {
            preview = null
            previewError = ""
            if (reportController && reportController.previewPurgeReports !== undefined)
                reportController.previewPurgeReports()
        }
        onAccepted: {
            if (!reportController)
                return
            const started = reportController.purgeReports()
            if (!started)
                Qt.callLater(function() { purgeDialog.open() })
            preview = null
            previewError = ""
        }
        onRejected: {
            preview = null
            previewError = ""
        }
        contentItem: ColumnLayout {
            width: 360
            spacing: 12
            padding: 12

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: qsTr("Czy na pewno chcesz usunąć raporty spełniające aktywne filtry? Operacja jest nieodwracalna.")
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: preview && (preview.status === "preview" || preview.status === "partial_failure"
                    || preview.status === "completed")
                text: qsTr("Wybrano %1 z %2 raportów do usunięcia.")
                    .arg(preview && preview.planned_count !== undefined ? preview.planned_count : 0)
                    .arg(preview && preview.matched_count !== undefined ? preview.matched_count : 0)
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: preview && preview.status === "preview"
                text: qsTr("Operacja obejmie %1 plików i %2 katalogów. Zwolnione miejsce: %3.")
                    .arg(preview && preview.removed_files !== undefined ? preview.removed_files : 0)
                    .arg(preview && preview.removed_directories !== undefined ? preview.removed_directories : 0)
                    .arg(browser.formatFileSize(preview && preview.removed_size !== undefined ? preview.removed_size : 0))
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: preview && preview.status === "empty"
                text: qsTr("Brak raportów spełniających aktualne filtry.")
                color: palette.mid
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: previewError && previewError.length > 0
                text: previewError
                font.bold: true
            }
        }
        footer: DialogButtonBox {
            standardButtons: DialogButtonBox.Ok | DialogButtonBox.Cancel
            enabled: !browser.isBusy
            onAccepted: purgeDialog.accept()
            onRejected: purgeDialog.reject()
            Component.onCompleted: {
                if (button(StandardButton.Ok))
                    button(StandardButton.Ok).text = qsTr("Usuń")
                if (button(StandardButton.Cancel))
                    button(StandardButton.Cancel).text = qsTr("Anuluj")
            }
        }
        Connections {
            target: reportController
            function onPurgePreviewReady(result) {
                if (!purgeDialog.visible)
                    return
                purgeDialog.preview = result
                if (result && result.status === "error" && result.error)
                    purgeDialog.previewError = String(result.error)
                else
                    purgeDialog.previewError = ""
            }
            function onPurgeFinished(success) {
                if (!success)
                    Qt.callLater(function() { purgeDialog.open() })
                else {
                    purgeDialog.preview = null
                    purgeDialog.previewError = ""
                }
            }
        }
    }

    FolderDialog {
        id: archiveFolderDialog
        title: qsTr("Wybierz katalog archiwum")
        onAccepted: {
            if (!selectedFolder)
                return
            var folder = ""
            if (selectedFolder && selectedFolder.toLocalFile)
                folder = selectedFolder.toLocalFile()
            else if (selectedFolder)
                folder = selectedFolder.toString()
            if (folder && folder.length > 0) {
                archiveDialog.destinationPath = folder
                destinationField.text = folder
                archiveDialog.updatePreview()
        }
    }

    Rectangle {
        anchors.fill: parent
        visible: browser.isBusy
        z: 9999
        color: Qt.rgba(0, 0, 0, 0.35)

        BusyIndicator {
            anchors.centerIn: parent
            running: browser.isBusy
            visible: browser.isBusy
            width: 56
            height: 56
        }

        MouseArea {
            anchors.fill: parent
            enabled: browser.isBusy
            cursorShape: Qt.WaitCursor
        }
    }
}

    FileDialog {
        id: exportDialog
        title: qsTr("Zapisz raport")
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("Archiwa (*.zip *.json *.jsonl *.csv *.md)"), qsTr("Wszystkie pliki (*)")]
        property string reportPath: ""
        onAccepted: {
            if (!reportPath || !selectedFile)
                return
            reportController.saveReportAs(reportPath, selectedFile)
        }
    }

    Timer {
        id: searchDebounce
        interval: 250
        repeat: false
        onTriggered: {
            const value = searchField.text.trim()
            const previous = browser.searchFilter
            if (previous !== value)
                browser.searchFilter = value
            if (reportController && reportController.searchQuery !== undefined
                    && reportController.searchQuery !== value)
                reportController.searchQuery = value
            if (previous !== value
                    || (reportController && reportController.searchQuery !== undefined
                        && reportController.searchQuery !== value))
                browser.refresh()
        }
    }

    Connections {
        target: reportController
        function onReportsChanged() {
            if (!reportController)
                return
            browser.rebuildFilters()
        }
        function onCategoriesChanged() {
            if (!reportController)
                return
            browser.rebuildFilters()
        }
        function onOverviewStatsChanged() {
            if (!reportController)
                return
            browser.rebuildFilters()
        }
        function onOverviewPaginationChanged() {
            if (!reportController)
                return
            browser.paginationInfo = reportController.overviewPagination || ({})
            browser.rebuildFilters()
        }
        function onRecentDaysFilterChanged() {
            if (!reportController)
                return
            const value = Number(reportController.recentDaysFilter || 0)
            if (browser.recentDaysFilter !== value)
                browser.recentDaysFilter = value
        }
        function onCategoryFilterChanged() {
            if (!reportController)
                return
            const value = reportController.categoryFilter || ""
            if (browser.categoryFilter !== value)
                browser.categoryFilter = value
        }
        function onSummaryStatusFilterChanged() {
            if (!reportController)
                return
            const value = reportController.summaryStatusFilter || "any"
            if (browser.summaryStatusFilter !== value)
                browser.summaryStatusFilter = value
        }
        function onExportsFilterChanged() {
            if (!reportController)
                return
            const value = reportController.exportsFilter || "any"
            if (browser.exportsFilter !== value)
                browser.exportsFilter = value
        }
        function onSearchQueryChanged() {
            if (!reportController)
                return
            const value = reportController.searchQuery || ""
            if (browser.searchFilter !== value)
                browser.searchFilter = value
            if (searchField.text !== value)
                searchField.text = value
        }
        function onLimitChanged() {
            if (!reportController)
                return
            const value = Number(reportController.limit || 0)
            browser.ensureLimitOption(value)
            if (browser.limitFilter !== value)
                browser.limitFilter = value
        }
        function onOffsetChanged() {
            if (!reportController)
                return
            const value = Number(reportController.offset || 0)
            browser.ensureOffsetOption(value)
            if (browser.offsetFilter !== value)
                browser.offsetFilter = value
        }
        function onSortKeyChanged() {
            if (!reportController)
                return
            const value = reportController.sortKey || "updated_at"
            if (browser.sortKeyFilter !== value)
                browser.sortKeyFilter = value
        }
        function onSortDirectionChanged() {
            if (!reportController)
                return
            const value = reportController.sortDirection || "desc"
            if (browser.sortDirectionFilter !== value)
                browser.sortDirectionFilter = value
        }
        function onSinceFilterChanged() {
            if (!reportController)
                return
            const value = browser.normalizeDate(reportController.sinceFilter)
            if (!browser.datesEqual(browser.sinceFilter, value))
                browser.sinceFilter = value ? new Date(value.getTime()) : null
        }
        function onUntilFilterChanged() {
            if (!reportController)
                return
            const value = browser.normalizeDate(reportController.untilFilter)
            if (!browser.datesEqual(browser.untilFilter, value))
                browser.untilFilter = value ? new Date(value.getTime()) : null
        }
    }

    Component.onCompleted: {
        if (reportController && reportController.recentDaysFilter !== undefined)
            browser.recentDaysFilter = reportController.recentDaysFilter
        if (reportController && reportController.categoryFilter !== undefined)
            browser.categoryFilter = reportController.categoryFilter || ""
        if (reportController && reportController.summaryStatusFilter !== undefined)
            browser.summaryStatusFilter = reportController.summaryStatusFilter || "any"
        if (reportController && reportController.exportsFilter !== undefined)
            browser.exportsFilter = reportController.exportsFilter || "any"
        if (reportController && reportController.searchQuery !== undefined) {
            browser.searchFilter = reportController.searchQuery || ""
            searchField.text = browser.searchFilter
        }
        if (reportController && reportController.limit !== undefined)
            browser.limitFilter = Number(reportController.limit || 0)
        if (reportController && reportController.offset !== undefined)
            browser.offsetFilter = Number(reportController.offset || 0)
        if (reportController && reportController.sortKey !== undefined)
            browser.sortKeyFilter = reportController.sortKey || "updated_at"
        if (reportController && reportController.sortDirection !== undefined)
            browser.sortDirectionFilter = reportController.sortDirection || "desc"
        if (reportController && reportController.sinceFilter !== undefined)
            browser.sinceFilter = browser.normalizeDate(reportController.sinceFilter)
        if (reportController && reportController.untilFilter !== undefined)
            browser.untilFilter = browser.normalizeDate(reportController.untilFilter)
        browser.ensureLimitOption(browser.limitFilter)
        browser.ensureOffsetOption(browser.offsetFilter)
        if (reportController && reportController.overviewPagination !== undefined)
            browser.paginationInfo = reportController.overviewPagination || ({})
        else
            browser.paginationInfo = ({})
        rebuildFilters()
        refresh()
    }
}
