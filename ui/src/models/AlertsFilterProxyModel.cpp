#include "AlertsFilterProxyModel.hpp"

AlertsFilterProxyModel::AlertsFilterProxyModel(QObject* parent)
    : QSortFilterProxyModel(parent)
{
    setDynamicSortFilter(true);
    sort(0);
}

void AlertsFilterProxyModel::setHideAcknowledged(bool hide)
{
    if (m_hideAcknowledged == hide)
        return;
    m_hideAcknowledged = hide;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void AlertsFilterProxyModel::setSeverityFilter(SeverityFilter filter)
{
    if (m_severityFilter == filter)
        return;
    m_severityFilter = filter;
    invalidateFilter();
    sort(0);
    Q_EMIT filterChanged();
}

void AlertsFilterProxyModel::setSearchText(const QString& text)
{
    if (m_searchText == text)
        return;

    m_searchText = text;
    const QString normalized = text.trimmed().toCaseFolded();
    const bool needleChanged = m_searchNeedle != normalized;
    m_searchNeedle = normalized;
    if (needleChanged) {
        invalidateFilter();
        sort(0);
    }
    Q_EMIT filterChanged();
}

void AlertsFilterProxyModel::setSortMode(SortMode mode)
{
    if (m_sortMode == mode)
        return;

    m_sortMode = mode;
    invalidate();
    sort(0);
    Q_EMIT filterChanged();
}

bool AlertsFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const
{
    if (!sourceModel())
        return false;

    const QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    if (!idx.isValid())
        return false;

    const int severity = idx.data(AlertsModel::SeverityRole).toInt();
    if (!matchesSeverity(severity))
        return false;

    if (m_hideAcknowledged) {
        const bool acknowledged = idx.data(AlertsModel::AcknowledgedRole).toBool();
        if (acknowledged)
            return false;
    }

    if (!m_searchNeedle.isEmpty()) {
        const QString id = idx.data(AlertsModel::IdRole).toString();
        const QString title = idx.data(AlertsModel::TitleRole).toString();
        const QString description = idx.data(AlertsModel::DescriptionRole).toString();
        const auto matches = [this](const QString& candidate) {
            return candidate.contains(m_searchNeedle, Qt::CaseInsensitive);
        };
        if (!matches(id) && !matches(title) && !matches(description))
            return false;
    }

    return true;
}

bool AlertsFilterProxyModel::matchesSeverity(int severity) const
{
    switch (m_severityFilter) {
    case AllSeverities:
        return true;
    case WarningsAndCritical:
        return severity == AlertsModel::Warning || severity == AlertsModel::Critical;
    case CriticalOnly:
        return severity == AlertsModel::Critical;
    case WarningOnly:
        return severity == AlertsModel::Warning;
    default:
        return true;
    }
}

namespace {
int compareTimestamps(const QDateTime& left, const QDateTime& right)
{
    const bool leftValid = left.isValid();
    const bool rightValid = right.isValid();
    if (leftValid != rightValid)
        return leftValid ? 1 : -1;
    if (!leftValid)
        return 0;

    const qint64 leftMs = left.toMSecsSinceEpoch();
    const qint64 rightMs = right.toMSecsSinceEpoch();
    if (leftMs < rightMs)
        return -1;
    if (leftMs > rightMs)
        return 1;
    return 0;
}
} // namespace

bool AlertsFilterProxyModel::lessThan(const QModelIndex& sourceLeft, const QModelIndex& sourceRight) const
{
    if (!sourceLeft.isValid() || !sourceRight.isValid())
        return QSortFilterProxyModel::lessThan(sourceLeft, sourceRight);

    const int leftSeverity = sourceLeft.data(AlertsModel::SeverityRole).toInt();
    const int rightSeverity = sourceRight.data(AlertsModel::SeverityRole).toInt();
    const QDateTime leftTimestamp = sourceLeft.data(AlertsModel::TimestampRole).toDateTime();
    const QDateTime rightTimestamp = sourceRight.data(AlertsModel::TimestampRole).toDateTime();
    const QString leftTitle = sourceLeft.data(AlertsModel::TitleRole).toString();
    const QString rightTitle = sourceRight.data(AlertsModel::TitleRole).toString();
    const QString leftId = sourceLeft.data(AlertsModel::IdRole).toString();
    const QString rightId = sourceRight.data(AlertsModel::IdRole).toString();

    const auto fallbackByTitle = [&]() {
        const int titleCompare = QString::localeAwareCompare(leftTitle, rightTitle);
        if (titleCompare != 0)
            return titleCompare < 0;
        const int timestampCompare = compareTimestamps(leftTimestamp, rightTimestamp);
        if (timestampCompare != 0)
            return timestampCompare > 0; // newer first as tie breaker
        return QString::localeAwareCompare(leftId, rightId) < 0;
    };

    const auto fallbackByTimestamp = [&](bool newestFirst) {
        const int timestampCompare = compareTimestamps(leftTimestamp, rightTimestamp);
        if (timestampCompare != 0)
            return newestFirst ? timestampCompare > 0 : timestampCompare < 0;
        if (leftSeverity != rightSeverity)
            return newestFirst ? leftSeverity > rightSeverity : leftSeverity < rightSeverity;
        return fallbackByTitle();
    };

    switch (m_sortMode) {
    case NewestFirst:
        return fallbackByTimestamp(true);
    case OldestFirst:
        return fallbackByTimestamp(false);
    case SeverityDescending:
        if (leftSeverity != rightSeverity)
            return leftSeverity > rightSeverity;
        return fallbackByTimestamp(true);
    case SeverityAscending:
        if (leftSeverity != rightSeverity)
            return leftSeverity < rightSeverity;
        return fallbackByTimestamp(true);
    case TitleAscending:
        return fallbackByTitle();
    default:
        return fallbackByTimestamp(true);
    }
}
