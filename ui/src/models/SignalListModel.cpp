#include "SignalListModel.hpp"

#include <QDateTime>

SignalListModel::SignalListModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int SignalListModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid()) {
        return 0;
    }
    return m_events.size();
}

QVariant SignalListModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_events.size()) {
        return {};
    }
    const auto& event = m_events.at(index.row());
    switch (role) {
    case TimestampRole:
        return event.timestampMs;
    case TimestampDisplayRole:
        return QDateTime::fromMSecsSinceEpoch(event.timestampMs, Qt::UTC).toLocalTime().toString(Qt::ISODate);
    case CodeRole:
        return event.code;
    case DescriptionRole:
        return event.description;
    case ConfidenceRole:
        return event.confidence;
    case RegimeRole:
        return event.regime;
    default:
        return {};
    }
}

QHash<int, QByteArray> SignalListModel::roleNames() const
{
    return {
        {TimestampRole, "timestamp"},
        {TimestampDisplayRole, "timestampDisplay"},
        {CodeRole, "code"},
        {DescriptionRole, "description"},
        {ConfidenceRole, "confidence"},
        {RegimeRole, "regime"},
    };
}

void SignalListModel::resetWithSignals(const QVector<SignalEventEntry>& events)
{
    beginResetModel();
    m_events = events;
    endResetModel();
    Q_EMIT countChanged();
}

void SignalListModel::appendSignal(const SignalEventEntry& event)
{
    const int insertRow = m_events.size();
    beginInsertRows(QModelIndex(), insertRow, insertRow);
    m_events.append(event);
    endInsertRows();
    Q_EMIT countChanged();
}

