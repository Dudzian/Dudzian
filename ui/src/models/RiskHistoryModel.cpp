#include "RiskHistoryModel.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocale>
#include <QMetaType>
#include <QSaveFile>
#include <QStringConverter>
#include <QTextStream>
#include <QtGlobal>
#include <QVariant>
#include <QVariantMap>

#include <algorithm>
#include <utility>

namespace {
double computeExposureUtilization(const RiskExposureData& exposure)
{
    const double limit = exposure.thresholdValue > 0.0 ? exposure.thresholdValue : exposure.maxValue;
    if (limit <= 0.0) {
        return 0.0;
    }
    return exposure.currentValue / limit;
}

QVariantMap exposureToVariant(const RiskExposureData& exposure)
{
    QVariantMap map;
    map.insert(QStringLiteral("code"), exposure.code);
    map.insert(QStringLiteral("currentValue"), exposure.currentValue);
    map.insert(QStringLiteral("maxValue"), exposure.maxValue);
    map.insert(QStringLiteral("thresholdValue"), exposure.thresholdValue);
    map.insert(QStringLiteral("breached"), exposure.isBreached());
    map.insert(QStringLiteral("utilization"), computeExposureUtilization(exposure));
    return map;
}

RiskExposureData exposureFromJson(const QJsonObject& object)
{
    RiskExposureData exposure;
    exposure.code = object.value(QStringLiteral("code")).toString();
    exposure.currentValue = object.value(QStringLiteral("currentValue")).toDouble();
    exposure.maxValue = object.value(QStringLiteral("maxValue")).toDouble();
    exposure.thresholdValue = object.value(QStringLiteral("thresholdValue")).toDouble();
    return exposure;
}

QString escapeCsv(const QString& value)
{
    QString escaped = value;
    escaped.replace(QStringLiteral("\""), QStringLiteral("\"\""));
    if (escaped.contains(QLatin1Char(',')) || escaped.contains(QLatin1Char('\n')) || escaped.contains(QLatin1Char('"'))) {
        return QStringLiteral("\"%1\"").arg(escaped);
    }
    return escaped;
}
} // namespace

RiskHistoryModel::RiskHistoryModel(QObject* parent)
    : QAbstractListModel(parent)
{
    qRegisterMetaType<RiskSnapshotData>("RiskSnapshotData");
}

int RiskHistoryModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid()) {
        return 0;
    }
    return m_entries.size();
}

QVariant RiskHistoryModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid()) {
        return {};
    }
    if (index.row() < 0 || index.row() >= m_entries.size()) {
        return {};
    }

    const Entry& entry = m_entries.at(index.row());
    switch (role) {
    case TimestampRole:
        return entry.timestamp;
    case DrawdownRole:
        return entry.drawdown;
    case LeverageRole:
        return entry.leverage;
    case PortfolioValueRole:
        return entry.portfolioValue;
    case ProfileLabelRole:
        return entry.profileLabel;
    case HasBreachRole:
        return entry.hasBreach;
    case BreachCountRole:
        return entry.breachCount;
    case MaxExposureUtilizationRole:
        return entry.maxExposureUtilization;
    case ExposuresRole:
        return exposuresToVariantList(entry);
    default:
        return {};
    }
}

QHash<int, QByteArray> RiskHistoryModel::roleNames() const
{
    return {
        {TimestampRole, QByteArrayLiteral("timestamp")},
        {DrawdownRole, QByteArrayLiteral("drawdown")},
        {LeverageRole, QByteArrayLiteral("leverage")},
        {PortfolioValueRole, QByteArrayLiteral("portfolioValue")},
        {ProfileLabelRole, QByteArrayLiteral("profileLabel")},
        {HasBreachRole, QByteArrayLiteral("hasBreach")},
        {BreachCountRole, QByteArrayLiteral("breachCount")},
        {MaxExposureUtilizationRole, QByteArrayLiteral("maxExposureUtilization")},
        {ExposuresRole, QByteArrayLiteral("exposures")},
    };
}

void RiskHistoryModel::setMaximumEntries(int value)
{
    if (value < 1) {
        value = 1;
    }
    if (value == m_maxEntries) {
        return;
    }
    m_maxEntries = value;
    const bool trimmed = trimExcess();
    if (trimmed) {
        recalculateSummary();
    }
    Q_EMIT maximumEntriesChanged();
    if (trimmed) {
        Q_EMIT historyChanged();
    }
}

void RiskHistoryModel::recordSnapshot(const RiskSnapshotData& snapshot)
{
    if (!snapshot.hasData) {
        return;
    }

    Entry entry;
    entry.timestamp = snapshot.generatedAt.isValid() ? snapshot.generatedAt.toUTC() : QDateTime::currentDateTimeUtc();
    entry.profileLabel = snapshot.profileLabel;
    entry.drawdown = snapshot.currentDrawdown;
    entry.leverage = snapshot.usedLeverage;
    entry.portfolioValue = snapshot.portfolioValue;
    entry.exposures = snapshot.exposures;
    updateDerivedFields(entry);

    if (!m_entries.isEmpty() && m_entries.back().timestamp == entry.timestamp) {
        m_entries.back() = entry;
        const int row = m_entries.size() - 1;
        const QModelIndex idx = index(row, 0);
        Q_EMIT dataChanged(idx, idx,
                          {TimestampRole, DrawdownRole, LeverageRole, PortfolioValueRole, ProfileLabelRole,
                           HasBreachRole, BreachCountRole, MaxExposureUtilizationRole, ExposuresRole});
        recalculateSummary();
        Q_EMIT historyChanged();
        Q_EMIT snapshotRecorded(entry.timestamp);
        return;
    }

    const int row = m_entries.size();
    beginInsertRows(QModelIndex(), row, row);
    m_entries.append(entry);
    endInsertRows();

    trimExcess();
    recalculateSummary();
    Q_EMIT historyChanged();
    Q_EMIT snapshotRecorded(entry.timestamp);
}

void RiskHistoryModel::clear()
{
    if (m_entries.isEmpty()) {
        return;
    }

    beginResetModel();
    m_entries.clear();
    endResetModel();

    recalculateSummary();
    Q_EMIT historyChanged();
}

bool RiskHistoryModel::trimExcess()
{
    bool trimmed = false;
    while (m_entries.size() > m_maxEntries) {
        beginRemoveRows(QModelIndex(), 0, 0);
        m_entries.removeFirst();
        endRemoveRows();
        trimmed = true;
    }
    return trimmed;
}

void RiskHistoryModel::recalculateSummary()
{
    Summary summary;
    if (!m_entries.isEmpty()) {
        summary.maxDrawdown = m_entries.first().drawdown;
        summary.minDrawdown = m_entries.first().drawdown;
        summary.maxLeverage = m_entries.first().leverage;
        summary.minPortfolioValue = m_entries.first().portfolioValue;
        summary.maxPortfolioValue = m_entries.first().portfolioValue;

        double drawdownSum = 0.0;
        double leverageSum = 0.0;

        for (const Entry& entry : m_entries) {
            drawdownSum += entry.drawdown;
            leverageSum += entry.leverage;
            summary.maxDrawdown = std::max(summary.maxDrawdown, entry.drawdown);
            summary.minDrawdown = std::min(summary.minDrawdown, entry.drawdown);
            summary.maxLeverage = std::max(summary.maxLeverage, entry.leverage);
            summary.minPortfolioValue = std::min(summary.minPortfolioValue, entry.portfolioValue);
            summary.maxPortfolioValue = std::max(summary.maxPortfolioValue, entry.portfolioValue);
            summary.totalBreaches += entry.breachCount;
            summary.maxExposureUtilization = std::max(summary.maxExposureUtilization, entry.maxExposureUtilization);
        }

        const double count = static_cast<double>(m_entries.size());
        summary.averageDrawdown = count > 0 ? drawdownSum / count : 0.0;
        summary.averageLeverage = count > 0 ? leverageSum / count : 0.0;
        summary.anyBreach = summary.totalBreaches > 0;
}

    m_summary = summary;
    Q_EMIT summaryChanged();
}

QJsonArray RiskHistoryModel::toJson(int limit) const
{
    QJsonArray array;
    if (m_entries.isEmpty())
        return array;

    const int effectiveLimit = limit < 0 ? m_entries.size() : qMax(0, limit);
    const int startIndex = qBound(0, m_entries.size() - effectiveLimit, m_entries.size());
    for (int row = startIndex; row < m_entries.size(); ++row) {
        const Entry& entry = m_entries.at(row);
        QJsonObject object;
        if (entry.timestamp.isValid())
            object.insert(QStringLiteral("timestamp"), entry.timestamp.toUTC().toString(Qt::ISODateWithMs));
        else
            object.insert(QStringLiteral("timestamp"), QString());
        object.insert(QStringLiteral("drawdown"), entry.drawdown);
        object.insert(QStringLiteral("leverage"), entry.leverage);
        object.insert(QStringLiteral("portfolioValue"), entry.portfolioValue);
        object.insert(QStringLiteral("profileLabel"), entry.profileLabel);
        const QVariantList exposures = exposuresToVariantList(entry);
        if (!exposures.isEmpty()) {
            QJsonArray exposuresArray;
            exposuresArray.reserve(exposures.size());
            for (const QVariant& exposureVariant : exposures) {
                const QVariantMap exposureMap = exposureVariant.toMap();
                QJsonObject exposureObject;
                for (auto it = exposureMap.constBegin(); it != exposureMap.constEnd(); ++it) {
                    exposureObject.insert(it.key(), QJsonValue::fromVariant(it.value()));
                }
                exposuresArray.append(exposureObject);
            }
            object.insert(QStringLiteral("exposures"), exposuresArray);
        }
        array.append(object);
    }

    return array;
}

void RiskHistoryModel::restoreFromJson(const QJsonArray& array)
{
    QList<Entry> restored;
    restored.reserve(array.size());

    for (const QJsonValue& value : array) {
        if (!value.isObject())
            continue;

        const QJsonObject object = value.toObject();
        const QString timestampString = object.value(QStringLiteral("timestamp")).toString();
        QDateTime timestamp;
        if (!timestampString.isEmpty()) {
            timestamp = QDateTime::fromString(timestampString, Qt::ISODateWithMs);
            if (!timestamp.isValid())
                timestamp = QDateTime::fromString(timestampString, Qt::ISODate);
            if (timestamp.isValid())
                timestamp = timestamp.toUTC();
        }

        Entry entry;
        entry.timestamp = timestamp;
        entry.drawdown = object.value(QStringLiteral("drawdown")).toDouble();
        entry.leverage = object.value(QStringLiteral("leverage")).toDouble();
        entry.portfolioValue = object.value(QStringLiteral("portfolioValue")).toDouble();
        entry.profileLabel = object.value(QStringLiteral("profileLabel")).toString();
        const QJsonValue exposuresValue = object.value(QStringLiteral("exposures"));
        if (exposuresValue.isArray()) {
            const QJsonArray exposuresArray = exposuresValue.toArray();
            QList<RiskExposureData> exposures;
            exposures.reserve(exposuresArray.size());
            for (const QJsonValue& exposureValue : exposuresArray) {
                if (!exposureValue.isObject())
                    continue;
                exposures.append(exposureFromJson(exposureValue.toObject()));
            }
            entry.exposures = exposures;
        }
        updateDerivedFields(entry);

        restored.append(entry);
    }

    while (restored.size() > m_maxEntries)
        restored.removeFirst();

    beginResetModel();
    m_entries = restored;
    endResetModel();

    recalculateSummary();
    Q_EMIT historyChanged();
}

bool RiskHistoryModel::exportToCsv(const QString& filePath, int limit) const
{
    if (filePath.trimmed().isEmpty()) {
        return false;
    }

    if (limit == 0 || limit < -1) {
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream.setLocale(QLocale::c());
    stream.setEncoding(QStringConverter::Utf8);
    stream << QStringLiteral("timestamp,profile_label,drawdown,leverage,portfolio_value,breach_count,has_breach,max_exposure_utilization,exposures\n");

    const QLocale locale = QLocale::c();

    int startIndex = 0;
    if (limit > 0 && limit < m_entries.size()) {
        startIndex = m_entries.size() - limit;
    }

    for (int i = startIndex; i < m_entries.size(); ++i) {
        const Entry& entry = m_entries.at(i);
        const QString timestamp = entry.timestamp.isValid() ? entry.timestamp.toUTC().toString(Qt::ISODateWithMs)
                                                            : QString();
        stream << escapeCsv(timestamp) << QLatin1Char(',');
        stream << escapeCsv(entry.profileLabel) << QLatin1Char(',');
        stream << locale.toString(entry.drawdown, 'f', 6) << QLatin1Char(',');
        stream << locale.toString(entry.leverage, 'f', 6) << QLatin1Char(',');
        stream << locale.toString(entry.portfolioValue, 'f', 2) << QLatin1Char(',');
        stream << entry.breachCount << QLatin1Char(',');
        stream << (entry.hasBreach ? QStringLiteral("true") : QStringLiteral("false")) << QLatin1Char(',');
        stream << locale.toString(entry.maxExposureUtilization, 'f', 6) << QLatin1Char(',');

        QJsonArray exposuresArray;
        exposuresArray.reserve(entry.exposures.size());
        for (const RiskExposureData& exposure : entry.exposures) {
            QJsonObject object;
            object.insert(QStringLiteral("code"), exposure.code);
            object.insert(QStringLiteral("currentValue"), exposure.currentValue);
            object.insert(QStringLiteral("maxValue"), exposure.maxValue);
            object.insert(QStringLiteral("thresholdValue"), exposure.thresholdValue);
            object.insert(QStringLiteral("breached"), exposure.isBreached());
            object.insert(QStringLiteral("utilization"), computeExposureUtilization(exposure));
            exposuresArray.append(object);
        }
        const QString exposuresJson = QString::fromUtf8(QJsonDocument(exposuresArray).toJson(QJsonDocument::Compact));
        stream << escapeCsv(exposuresJson);
        stream << QLatin1Char('\n');
    }

    stream.flush();
    if (stream.status() != QTextStream::Ok) {
        return false;
    }

    return file.commit();
}

void RiskHistoryModel::updateDerivedFields(Entry& entry) const
{
    entry.breachCount = 0;
    entry.hasBreach = false;
    entry.maxExposureUtilization = 0.0;
    for (const RiskExposureData& exposure : std::as_const(entry.exposures)) {
        if (exposure.isBreached()) {
            entry.hasBreach = true;
            ++entry.breachCount;
        }
        entry.maxExposureUtilization = std::max(entry.maxExposureUtilization, computeExposureUtilization(exposure));
    }
}

QVariantList RiskHistoryModel::exposuresToVariantList(const Entry& entry) const
{
    QVariantList list;
    list.reserve(entry.exposures.size());
    for (const RiskExposureData& exposure : entry.exposures) {
        list.append(exposureToVariant(exposure));
    }
    return list;
}

