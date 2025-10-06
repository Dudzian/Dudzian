#include "OhlcvListModel.hpp"

#include <algorithm>

OhlcvListModel::OhlcvListModel(QObject* parent)
    : QAbstractListModel(parent) {
    qRegisterMetaType<OhlcvPoint>("OhlcvPoint");
}

int OhlcvListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }
    return m_candles.size();
}

QVariant OhlcvListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= m_candles.size()) {
        return {};
    }
    const auto& candle = m_candles.at(index.row());
    switch (role) {
    case TimestampRole:
        return candle.timestampMs;
    case OpenRole:
        return candle.open;
    case HighRole:
        return candle.high;
    case LowRole:
        return candle.low;
    case CloseRole:
        return candle.close;
    case VolumeRole:
        return candle.volume;
    case ClosedRole:
        return candle.closed;
    case SequenceRole:
        return QVariant::fromValue<qulonglong>(candle.sequence);
    default:
        return {};
    }
}

QHash<int, QByteArray> OhlcvListModel::roleNames() const {
    return {
        {TimestampRole, "timestamp"},
        {OpenRole, "open"},
        {HighRole, "high"},
        {LowRole, "low"},
        {CloseRole, "close"},
        {VolumeRole, "volume"},
        {ClosedRole, "closed"},
        {SequenceRole, "sequence"},
    };
}

void OhlcvListModel::resetWithHistory(const QList<OhlcvPoint>& candles) {
    beginResetModel();
    m_candles = QVector<OhlcvPoint>::fromList(candles);
    std::sort(m_candles.begin(), m_candles.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestampMs < rhs.timestampMs;
    });
    enforceMaximum();
    endResetModel();
}

void OhlcvListModel::applyIncrement(const OhlcvPoint& candle) {
    if (!m_candles.isEmpty()) {
        auto& last = m_candles.last();
        if (last.sequence == candle.sequence || last.timestampMs == candle.timestampMs) {
            last = candle;
            const auto row = m_candles.size() - 1;
            const QModelIndex idx = index(row);
            Q_EMIT dataChanged(idx, idx);
            return;
        }
    }

    const int insertRow = m_candles.size();
    beginInsertRows(QModelIndex(), insertRow, insertRow);
    m_candles.append(candle);
    enforceMaximum();
    endInsertRows();
}

void OhlcvListModel::setMaximumSamples(int value) {
    if (value <= 0 || value == m_maximumSamples) {
        return;
    }
    m_maximumSamples = value;
    enforceMaximum();
    Q_EMIT maximumSamplesChanged();
}

QVariant OhlcvListModel::latestClose() const {
    if (m_candles.isEmpty()) {
        return {};
    }
    return m_candles.last().close;
}

QVariant OhlcvListModel::timestampAt(int row) const {
    if (row < 0 || row >= m_candles.size()) {
        return {};
    }
    return m_candles.at(row).timestampMs;
}

QVariantMap OhlcvListModel::candleAt(int row) const {
    QVariantMap map;
    if (row < 0 || row >= m_candles.size()) {
        return map;
    }
    const auto& candle = m_candles.at(row);
    map.insert(QStringLiteral("timestamp"), candle.timestampMs);
    map.insert(QStringLiteral("open"), candle.open);
    map.insert(QStringLiteral("high"), candle.high);
    map.insert(QStringLiteral("low"), candle.low);
    map.insert(QStringLiteral("close"), candle.close);
    map.insert(QStringLiteral("volume"), candle.volume);
    map.insert(QStringLiteral("closed"), candle.closed);
    map.insert(QStringLiteral("sequence"), static_cast<qulonglong>(candle.sequence));
    return map;
}

void OhlcvListModel::enforceMaximum() {
    if (m_maximumSamples <= 0) {
        return;
    }
    if (m_candles.size() <= m_maximumSamples) {
        return;
    }
    const int removeCount = m_candles.size() - m_maximumSamples;
    beginRemoveRows(QModelIndex(), 0, removeCount - 1);
    m_candles.erase(m_candles.begin(), m_candles.begin() + removeCount);
    endRemoveRows();
}
