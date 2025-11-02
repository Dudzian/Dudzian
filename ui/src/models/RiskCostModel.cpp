#include "RiskCostModel.hpp"

#include <QLocale>
#include <QMetaType>

namespace {
using MetricDefinition = RiskCostModel::MetricDefinition;

MetricDefinition makeMetric(const QString& key, const QString& label, bool percent = false, int decimals = 2)
{
    MetricDefinition def;
    def.key = key;
    def.label = label;
    def.percent = percent;
    def.decimals = decimals;
    return def;
}

const QVector<MetricDefinition> kMetrics = {
    makeMetric(QStringLiteral("dailyRealizedPnl"), QObject::tr("Zrealizowany wynik (dzień)"), false, 2),
    makeMetric(QStringLiteral("grossNotional"), QObject::tr("Wartość brutto pozycji"), false, 2),
    makeMetric(QStringLiteral("activePositions"), QObject::tr("Aktywne pozycje"), false, 0),
    makeMetric(QStringLiteral("dailyLossPct"), QObject::tr("Strata dzienna"), true, 2),
    makeMetric(QStringLiteral("drawdownPct"), QObject::tr("Obsunięcie kapitału"), true, 2),
    makeMetric(QStringLiteral("averageCostBps"), QObject::tr("Średni koszt (bps)"), false, 2),
    makeMetric(QStringLiteral("totalCostBps"), QObject::tr("Łączny koszt (bps)"), false, 2),
};

} // namespace

RiskCostModel::RiskCostModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int RiskCostModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_entries.size();
}

QVariant RiskCostModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return {};
    if (index.row() < 0 || index.row() >= m_entries.size())
        return {};

    const Entry& entry = m_entries.at(index.row());
    switch (role) {
    case KeyRole:
        return entry.key;
    case LabelRole:
        return entry.label;
    case ValueRole:
        return entry.value;
    case FormattedRole:
        return entry.formatted;
    default:
        return {};
    }
}

QHash<int, QByteArray> RiskCostModel::roleNames() const
{
    return {
        {KeyRole, QByteArrayLiteral("key")},
        {LabelRole, QByteArrayLiteral("label")},
        {ValueRole, QByteArrayLiteral("value")},
        {FormattedRole, QByteArrayLiteral("formatted")},
    };
}

const QVector<RiskCostModel::MetricDefinition>& RiskCostModel::metricDefinitions()
{
    return kMetrics;
}

QString RiskCostModel::formatValue(const MetricDefinition& def, const QVariant& value)
{
    if (!value.isValid())
        return QStringLiteral("—");

    const QLocale locale;
    if (def.percent) {
        const double numeric = value.toDouble() * 100.0;
        return locale.toString(numeric, 'f', def.decimals) + QStringLiteral(" %");
    }

    if (value.typeId() == QMetaType::Int || value.typeId() == QMetaType::LongLong)
        return locale.toString(value.toLongLong());

    const double numeric = value.toDouble();
    return locale.toString(numeric, 'f', def.decimals);
}

void RiskCostModel::updateFromSnapshot(const RiskSnapshotData& snapshot)
{
    beginResetModel();
    m_entries.clear();
    m_summary.clear();

    const QVector<MetricDefinition>& defs = metricDefinitions();
    for (const MetricDefinition& def : defs) {
        QVariant metricValue;
        if (snapshot.statistics.contains(def.key))
            metricValue = snapshot.statistics.value(def.key);
        else if (snapshot.costBreakdown.contains(def.key))
            metricValue = snapshot.costBreakdown.value(def.key);

        if (!metricValue.isValid())
            metricValue = QVariant();

        Entry entry;
        entry.key = def.key;
        entry.label = def.label;
        entry.value = metricValue;
        entry.formatted = formatValue(def, metricValue);
        m_entries.append(entry);
        if (metricValue.isValid())
            m_summary.insert(entry.key, metricValue);
    }

    for (auto it = snapshot.statistics.constBegin(); it != snapshot.statistics.constEnd(); ++it) {
        if (m_summary.contains(it.key()))
            continue;
        m_summary.insert(it.key(), it.value());
    }
    for (auto it = snapshot.costBreakdown.constBegin(); it != snapshot.costBreakdown.constEnd(); ++it) {
        if (m_summary.contains(it.key()))
            continue;
        m_summary.insert(it.key(), it.value());
    }

    endResetModel();
    Q_EMIT summaryChanged();
}

void RiskCostModel::clear()
{
    beginResetModel();
    m_entries.clear();
    m_summary.clear();
    endResetModel();
    Q_EMIT summaryChanged();
}

