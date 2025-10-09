#include "OhlcvListModel.hpp"

#include <algorithm>
#include <cmath>

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
    recomputeIndicators();
    endResetModel();
}

void OhlcvListModel::applyIncrement(const OhlcvPoint& candle) {
    if (!m_candles.isEmpty()) {
        auto& last = m_candles.last();
        // Aktualizacja ostatniej świecy (ta sama sekwencja lub timestamp)
        if (last.sequence == candle.sequence || last.timestampMs == candle.timestampMs) {
            last = candle;
            const int row = m_candles.size() - 1;
            const QModelIndex idx = index(row);
            Q_EMIT dataChanged(idx, idx);
            recomputeIndicators();
            return;
        }
    }

    // Nowa świeca na końcu
    const int insertRow = m_candles.size();
    beginInsertRows(QModelIndex(), insertRow, insertRow);
    m_candles.append(candle);
    enforceMaximum();
    endInsertRows();
    recomputeIndicators();
}

void OhlcvListModel::setMaximumSamples(int value) {
    if (value <= 0 || value == m_maximumSamples) {
        return;
    }
    m_maximumSamples = value;
    enforceMaximum();
    recomputeIndicators();
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

QVariantList OhlcvListModel::overlaySeries(const QString& id) const {
    const QVector<double>* values = nullptr;
    if (id == QLatin1String("ema_fast")) {
        values = &m_emaFast;
    } else if (id == QLatin1String("ema_slow")) {
        values = &m_emaSlow;
    } else if (id == QLatin1String("vwap")) {
        values = &m_vwap;
    }

    QVariantList series;
    if (!values || values->size() != m_candles.size()) {
        return series;
    }

    series.reserve(values->size());
    for (int i = 0; i < values->size(); ++i) {
        const double value = values->at(i);
        if (std::isnan(value)) {
            continue;
        }
        QVariantMap point;
        point.insert(QStringLiteral("timestamp"), m_candles.at(i).timestampMs);
        point.insert(QStringLiteral("value"), value);
        series.append(point);
    }
    return series;
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

void OhlcvListModel::recomputeIndicators() {
    const int count = m_candles.size();
    m_emaFast.resize(count);
    m_emaSlow.resize(count);
    m_vwap.resize(count);

    if (count == 0) {
        return;
    }

    // Parametry nakładek (EMA/VWAP)
    const int fastPeriod = 12;
    const int slowPeriod = 26;
    const double fastMultiplier = 2.0 / (fastPeriod + 1.0);
    const double slowMultiplier = 2.0 / (slowPeriod + 1.0);

    double emaFast = m_candles.first().close;
    double emaSlow = m_candles.first().close;
    double cumulativePv = 0.0;
    double cumulativeVolume = 0.0;

    for (int i = 0; i < count; ++i) {
        const auto& candle = m_candles.at(i);
        const double price = candle.close;

        if (i == 0) {
            emaFast = price;
            emaSlow = price;
        } else {
            emaFast = (price - emaFast) * fastMultiplier + emaFast;
            emaSlow = (price - emaSlow) * slowMultiplier + emaSlow;
        }
        m_emaFast[i] = emaFast;
        m_emaSlow[i] = emaSlow;

        cumulativePv += price * candle.volume;
        cumulativeVolume += candle.volume;
        if (cumulativeVolume > 0.0) {
            m_vwap[i] = cumulativePv / cumulativeVolume;
        } else {
            m_vwap[i] = price;
        }
    }
}
