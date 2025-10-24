#include "MarketRegimeTimelineModel.hpp"

#include <QDateTime>

MarketRegimeTimelineModel::MarketRegimeTimelineModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int MarketRegimeTimelineModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid()) {
        return 0;
    }
    return m_snapshots.size();
}

QVariant MarketRegimeTimelineModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_snapshots.size()) {
        return {};
    }
    const auto& snapshot = m_snapshots.at(index.row());
    switch (role) {
    case TimestampRole:
        return snapshot.timestampMs;
    case TimestampDisplayRole:
        return QDateTime::fromMSecsSinceEpoch(snapshot.timestampMs, Qt::UTC).toLocalTime().toString(Qt::ISODate);
    case RegimeRole:
        return snapshot.regime;
    case TrendConfidenceRole:
        return snapshot.trendConfidence;
    case MeanReversionConfidenceRole:
        return snapshot.meanReversionConfidence;
    case DailyConfidenceRole:
        return snapshot.dailyConfidence;
    default:
        return {};
    }
}

QHash<int, QByteArray> MarketRegimeTimelineModel::roleNames() const
{
    return {
        {TimestampRole, "timestamp"},
        {TimestampDisplayRole, "timestampDisplay"},
        {RegimeRole, "regime"},
        {TrendConfidenceRole, "trendConfidence"},
        {MeanReversionConfidenceRole, "meanReversionConfidence"},
        {DailyConfidenceRole, "dailyConfidence"},
    };
}

QString MarketRegimeTimelineModel::latestRegime() const
{
    if (m_snapshots.isEmpty()) {
        return {};
    }
    return m_snapshots.constLast().regime;
}

void MarketRegimeTimelineModel::setMaximumSnapshots(int value)
{
    if (value < 0)
        value = 0;
    if (m_maxSnapshots == value)
        return;

    m_maxSnapshots = value;
    const bool trimmed = trimExcessSnapshots();

    Q_EMIT maximumSnapshotsChanged();
    if (trimmed) {
        Q_EMIT countChanged();
        Q_EMIT latestRegimeChanged();
    }
}

void MarketRegimeTimelineModel::resetWithSnapshots(const QVector<MarketRegimeSnapshotEntry>& snapshots)
{
    QVector<MarketRegimeSnapshotEntry> limited = snapshots;
    if (m_maxSnapshots > 0 && limited.size() > m_maxSnapshots) {
        const int start = limited.size() - m_maxSnapshots;
        limited = limited.mid(start);
    }

    beginResetModel();
    m_snapshots = limited;
    endResetModel();
    Q_EMIT countChanged();
    Q_EMIT latestRegimeChanged();
}

void MarketRegimeTimelineModel::appendSnapshot(const MarketRegimeSnapshotEntry& snapshot)
{
    const int previousCount = m_snapshots.size();
    const int insertRow = previousCount;
    beginInsertRows(QModelIndex(), insertRow, insertRow);
    m_snapshots.append(snapshot);
    endInsertRows();

    trimExcessSnapshots();

    if (m_snapshots.size() != previousCount)
        Q_EMIT countChanged();
    Q_EMIT latestRegimeChanged();
}

bool MarketRegimeTimelineModel::trimExcessSnapshots()
{
    if (m_maxSnapshots <= 0)
        return false;

    const int excess = m_snapshots.size() - m_maxSnapshots;
    if (excess <= 0)
        return false;

    beginRemoveRows(QModelIndex(), 0, excess - 1);
    m_snapshots.erase(m_snapshots.begin(), m_snapshots.begin() + excess);
    endRemoveRows();
    return true;
}

