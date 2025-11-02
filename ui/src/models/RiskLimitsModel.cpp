#include "RiskLimitsModel.hpp"

#include <QLocale>
#include <QtGlobal>

#include <algorithm>

namespace {
using Definition = RiskLimitsModel::Definition;

Definition makeDefinition(const QString& key,
                          const QString& label,
                          double defaultValue,
                          double minValue,
                          double maxValue,
                          double step,
                          bool isPercent,
                          bool editable = true)
{
    Definition def;
    def.key = key;
    def.label = label;
    def.defaultValue = defaultValue;
    def.minValue = minValue;
    def.maxValue = maxValue;
    def.step = step;
    def.isPercent = isPercent;
    def.editable = editable;
    return def;
}

const QVector<Definition> kDefinitions = {
    makeDefinition(QStringLiteral("max_positions"), QObject::tr("Liczba pozycji"), 0.0, 0.0, 200.0, 1.0, false),
    makeDefinition(QStringLiteral("max_leverage"), QObject::tr("Maksymalna dźwignia"), 1.0, 0.0, 100.0, 0.1, false),
    makeDefinition(QStringLiteral("max_position_pct"), QObject::tr("Limit ekspozycji"), 0.0, 0.0, 1.0, 0.01, true),
    makeDefinition(QStringLiteral("daily_loss_limit"), QObject::tr("Limit dzienny"), 0.0, 0.0, 1.0, 0.01, true),
    makeDefinition(QStringLiteral("drawdown_limit"), QObject::tr("Limit obsunięcia"), 0.0, 0.0, 1.0, 0.01, true),
    makeDefinition(QStringLiteral("target_volatility"), QObject::tr("Docelowa zmienność"), 0.0, 0.0, 1.0, 0.01, true),
    makeDefinition(QStringLiteral("stop_loss_atr_multiple"), QObject::tr("Stop loss (ATR)"), 0.0, 0.0, 10.0, 0.1, false),
};

} // namespace

RiskLimitsModel::RiskLimitsModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int RiskLimitsModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_entries.size();
}

QVariant RiskLimitsModel::data(const QModelIndex& index, int role) const
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
    case MinimumRole:
        return entry.minValue;
    case MaximumRole:
        return entry.maxValue;
    case StepRole:
        return entry.step;
    case PercentRole:
        return entry.isPercent;
    case EditableRole:
        return entry.editable;
    default:
        return {};
    }
}

QHash<int, QByteArray> RiskLimitsModel::roleNames() const
{
    return {
        {KeyRole, QByteArrayLiteral("key")},
        {LabelRole, QByteArrayLiteral("label")},
        {ValueRole, QByteArrayLiteral("value")},
        {MinimumRole, QByteArrayLiteral("minimum")},
        {MaximumRole, QByteArrayLiteral("maximum")},
        {StepRole, QByteArrayLiteral("step")},
        {PercentRole, QByteArrayLiteral("isPercent")},
        {EditableRole, QByteArrayLiteral("editable")},
    };
}

const QVector<RiskLimitsModel::Definition>& RiskLimitsModel::knownDefinitions()
{
    return kDefinitions;
}

void RiskLimitsModel::updateFromSnapshot(const RiskSnapshotData& snapshot)
{
    rebuildFromMap(snapshot.limits);
}

bool RiskLimitsModel::setLimitValue(const QString& key, double value)
{
    const int index = findEntryIndex(key);
    if (index < 0)
        return false;
    return setLimitValueAt(index, value);
}

bool RiskLimitsModel::setLimitValueAt(int index, double value)
{
    if (index < 0 || index >= m_entries.size())
        return false;

    Entry& entry = m_entries[index];
    if (!entry.editable)
        return false;

    double clamped = std::clamp(value, entry.minValue, entry.maxValue);
    if (!qFuzzyCompare(1.0 + clamped, 1.0 + entry.value)) {
        entry.value = clamped;
        m_limits.insert(entry.key, clamped);
        const QModelIndex modelIndex = createIndex(index, 0);
        Q_EMIT dataChanged(modelIndex, modelIndex, {ValueRole});
        Q_EMIT limitsChanged();
        Q_EMIT limitEdited(entry.key, entry.value);
    }
    return true;
}

void RiskLimitsModel::clear()
{
    beginResetModel();
    m_entries.clear();
    m_limits.clear();
    endResetModel();
    Q_EMIT limitsChanged();
}

int RiskLimitsModel::findEntryIndex(const QString& key) const
{
    for (int i = 0; i < m_entries.size(); ++i) {
        if (m_entries.at(i).key == key)
            return i;
    }
    return -1;
}

void RiskLimitsModel::rebuildFromMap(const QVariantMap& limits)
{
    beginResetModel();
    m_entries.clear();
    m_limits.clear();

    const auto defs = knownDefinitions();
    for (const Definition& def : defs) {
        Entry entry;
        entry.key = def.key;
        entry.label = def.label;
        const QVariant raw = limits.value(def.key);
        entry.value = raw.isValid() ? raw.toDouble() : def.defaultValue;
        entry.minValue = def.minValue;
        entry.maxValue = def.maxValue;
        entry.step = def.step;
        entry.isPercent = def.isPercent;
        entry.editable = def.editable;
        m_entries.append(entry);
        m_limits.insert(entry.key, entry.value);
    }

    for (auto it = limits.constBegin(); it != limits.constEnd(); ++it) {
        const QString key = it.key();
        if (findEntryIndex(key) >= 0)
            continue;
        Entry entry;
        entry.key = key;
        entry.label = key;
        entry.value = it.value().toDouble();
        entry.minValue = 0.0;
        entry.maxValue = std::max(entry.value * 5.0, entry.value + 1.0);
        entry.step = std::max(0.01, entry.maxValue / 100.0);
        entry.isPercent = false;
        entry.editable = true;
        m_entries.append(entry);
        m_limits.insert(entry.key, entry.value);
    }

    endResetModel();
    Q_EMIT limitsChanged();
}

