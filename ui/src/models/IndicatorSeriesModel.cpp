#include "IndicatorSeriesModel.hpp"

#include <QVariant>

IndicatorSeriesModel::IndicatorSeriesModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int IndicatorSeriesModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid()) {
        return 0;
    }
    return m_series.size();
}

QVariant IndicatorSeriesModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_series.size()) {
        return {};
    }
    const SeriesData& entry = m_series.at(index.row());
    switch (role) {
    case IdRole:
        return entry.definition.id;
    case LabelRole:
        return entry.definition.label;
    case ColorRole:
        return entry.definition.color;
    case SecondaryRole:
        return entry.definition.secondary;
    case SamplesRole: {
        QVariantList result;
        result.reserve(entry.samples.size());
        for (const auto& sample : entry.samples) {
            QVariantMap map;
            map.insert(QStringLiteral("timestamp"), sample.timestampMs);
            map.insert(QStringLiteral("value"), sample.value);
            result.append(map);
        }
        return result;
    }
    case LatestValueRole:
        if (!entry.samples.isEmpty()) {
            return entry.samples.constLast().value;
        }
        return QVariant();
    default:
        return {};
    }
}

QHash<int, QByteArray> IndicatorSeriesModel::roleNames() const
{
    return {
        {IdRole, "seriesId"},
        {LabelRole, "label"},
        {ColorRole, "color"},
        {SecondaryRole, "secondary"},
        {SamplesRole, "samples"},
        {LatestValueRole, "latestValue"},
    };
}

void IndicatorSeriesModel::setSeriesDefinitions(const QVector<IndicatorSeriesDefinition>& definitions)
{
    beginResetModel();
    m_series.clear();
    m_series.reserve(definitions.size());
    for (const auto& def : definitions) {
        SeriesData data;
        data.definition = def;
        m_series.append(data);
    }
    endResetModel();
    Q_EMIT countChanged();
}

void IndicatorSeriesModel::replaceSamples(const QString& id, const QVector<IndicatorSample>& samples)
{
    const int idx = indexForId(id);
    if (idx < 0) {
        return;
    }
    m_series[idx].samples = samples;
    const QModelIndex modelIndex = index(idx);
    Q_EMIT dataChanged(modelIndex, modelIndex, {SamplesRole, LatestValueRole});
}

void IndicatorSeriesModel::appendSample(const IndicatorSample& sample)
{
    const int idx = indexForId(sample.seriesId);
    if (idx < 0) {
        return;
    }
    auto& series = m_series[idx].samples;
    series.append(sample);
    const QModelIndex modelIndex = index(idx);
    Q_EMIT dataChanged(modelIndex, modelIndex, {SamplesRole, LatestValueRole});
}

int IndicatorSeriesModel::indexForId(const QString& id) const
{
    for (int i = 0; i < m_series.size(); ++i) {
        if (m_series.at(i).definition.id == id) {
            return i;
        }
    }
    return -1;
}

