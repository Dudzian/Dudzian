#include "AlertsModel.hpp"

#include <QtMath>
#include <algorithm>

namespace {
constexpr double kWarningRatio = 0.8;
constexpr double kCriticalDrawdown = 0.08;   // 8%
constexpr double kWarningDrawdown = 0.05;    // 5%
constexpr double kCriticalLeverage = 8.0;
constexpr double kWarningLeverage = 5.0;

QString exposureAlertId(const QString& code)
{
    return QStringLiteral("exposure:%1").arg(code);
}

QString drawdownAlertId()
{
    return QStringLiteral("drawdown");
}

QString leverageAlertId()
{
    return QStringLiteral("leverage");
}

} // namespace

AlertsModel::AlertsModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int AlertsModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_alerts.size();
}

QVariant AlertsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_alerts.size())
        return {};
    const Alert& alert = m_alerts.at(index.row());
    switch (role) {
    case IdRole:
        return alert.id;
    case TitleRole:
        return alert.title;
    case DescriptionRole:
        return alert.description;
    case SeverityRole:
        return static_cast<int>(alert.severity);
    case TimestampRole:
        return alert.raisedAt;
    case AcknowledgedRole:
        return alert.acknowledged;
    default:
        return {};
    }
}

QHash<int, QByteArray> AlertsModel::roleNames() const
{
    return {
        {IdRole, QByteArrayLiteral("id")},
        {TitleRole, QByteArrayLiteral("title")},
        {DescriptionRole, QByteArrayLiteral("description")},
        {SeverityRole, QByteArrayLiteral("severity")},
        {TimestampRole, QByteArrayLiteral("timestamp")},
        {AcknowledgedRole, QByteArrayLiteral("acknowledged")},
    };
}

void AlertsModel::acknowledge(const QString& alertId)
{
    const int index = indexOfAlert(alertId);
    if (index < 0)
        return;
    Alert& alert = m_alerts[index];
    if (alert.acknowledged)
        return;
    alert.acknowledged = true;
    const int previousSize = m_acknowledgedIds.size();
    m_acknowledgedIds.insert(alertId);
    Q_EMIT dataChanged(this->index(index), this->index(index), {AcknowledgedRole});
    recomputeCounts();
    if (m_acknowledgedIds.size() != previousSize)
        Q_EMIT acknowledgementsChanged();
}

void AlertsModel::clearAcknowledged()
{
    bool ackChanged = false;
    for (int i = m_alerts.size() - 1; i >= 0; --i) {
        if (m_alerts.at(i).acknowledged && !m_alerts.at(i).sticky) {
            const QString id = m_alerts.at(i).id;
            beginRemoveRows(QModelIndex(), i, i);
            m_alerts.removeAt(i);
            endRemoveRows();
            if (m_acknowledgedIds.remove(id))
                ackChanged = true;
        }
    }
    recomputeCounts();
    if (ackChanged)
        Q_EMIT acknowledgementsChanged();
}

void AlertsModel::acknowledgeAll()
{
    if (m_alerts.isEmpty())
        return;

    bool ackChanged = false;
    for (int i = 0; i < m_alerts.size(); ++i) {
        Alert& alert = m_alerts[i];
        if (!alert.acknowledged) {
            alert.acknowledged = true;
            Q_EMIT dataChanged(this->index(i), this->index(i), {AcknowledgedRole});
            ackChanged = true;
        }
        if (!alert.id.isEmpty() && !m_acknowledgedIds.contains(alert.id)) {
            m_acknowledgedIds.insert(alert.id);
            ackChanged = true;
        }
    }

    if (!ackChanged)
        return;

    recomputeCounts();
    Q_EMIT acknowledgementsChanged();
}

void AlertsModel::raiseAlert(const QString& alertId,
                             const QString& title,
                             const QString& description,
                             Severity severity,
                             bool sticky)
{
    Alert alert;
    alert.id = alertId;
    alert.title = title;
    alert.description = description;
    alert.severity = severity;
    alert.raisedAt = QDateTime::currentDateTimeUtc();
    alert.sticky = sticky;

    const bool ackChanged = upsertAlert(alert);
    recomputeCounts();
    if (ackChanged)
        Q_EMIT acknowledgementsChanged();
}

void AlertsModel::clearAlert(const QString& alertId)
{
    const int index = indexOfAlert(alertId);
    if (index < 0)
        return;

    beginRemoveRows(QModelIndex(), index, index);
    const QString id = m_alerts.at(index).id;
    m_alerts.removeAt(index);
    endRemoveRows();

    const bool ackChanged = m_acknowledgedIds.remove(id);
    recomputeCounts();
    if (ackChanged)
        Q_EMIT acknowledgementsChanged();
}

void AlertsModel::updateFromRiskSnapshot(const RiskSnapshotData& snapshot)
{
    for (Alert& alert : m_alerts) {
        if (!isSecurityAlertId(alert.id))
            alert.stale = true;
    }

    bool ackChanged = false;

    // Drawdown alerts
    const double drawdown = snapshot.currentDrawdown;
    if (drawdown >= kCriticalDrawdown) {
        Alert alert;
        alert.id = drawdownAlertId();
        alert.title = QObject::tr("Krytyczny drawdown portfela");
        alert.description = QObject::tr("Aktualny drawdown: %1%%").arg(drawdown * 100.0, 0, 'f', 2);
        alert.severity = Critical;
        alert.raisedAt = QDateTime::currentDateTimeUtc();
        ackChanged = upsertAlert(alert) || ackChanged;
    } else if (drawdown >= kWarningDrawdown) {
        Alert alert;
        alert.id = drawdownAlertId();
        alert.title = QObject::tr("Wysoki drawdown portfela");
        alert.description = QObject::tr("Aktualny drawdown: %1%%").arg(drawdown * 100.0, 0, 'f', 2);
        alert.severity = Warning;
        alert.raisedAt = QDateTime::currentDateTimeUtc();
        ackChanged = upsertAlert(alert) || ackChanged;
    }

    // Leverage alerts
    const double leverage = snapshot.usedLeverage;
    if (leverage >= kCriticalLeverage) {
        Alert alert;
        alert.id = leverageAlertId();
        alert.title = QObject::tr("Przekroczona dźwignia portfela");
        alert.description = QObject::tr("Aktualna dźwignia: %1x").arg(leverage, 0, 'f', 2);
        alert.severity = Critical;
        alert.raisedAt = QDateTime::currentDateTimeUtc();
        ackChanged = upsertAlert(alert) || ackChanged;
    } else if (leverage >= kWarningLeverage) {
        Alert alert;
        alert.id = leverageAlertId();
        alert.title = QObject::tr("Podwyższona dźwignia portfela");
        alert.description = QObject::tr("Aktualna dźwignia: %1x").arg(leverage, 0, 'f', 2);
        alert.severity = Warning;
        alert.raisedAt = QDateTime::currentDateTimeUtc();
        ackChanged = upsertAlert(alert) || ackChanged;
    }

    // Exposure alerts
    for (const RiskExposureData& exposure : snapshot.exposures) {
        if (exposure.thresholdValue <= 0.0)
            continue;
        const double ratio = exposure.currentValue / exposure.thresholdValue;
        if (ratio < kWarningRatio)
            continue;

        Alert alert;
        alert.id = exposureAlertId(exposure.code);
        alert.title = QObject::tr("Limit ekspozycji: %1").arg(exposure.code);
        alert.description = QObject::tr("%1 / %2")
                                  .arg(exposure.currentValue, 0, 'f', 0)
                                  .arg(exposure.thresholdValue, 0, 'f', 0);
        alert.severity = ratio >= 1.0 ? Critical : Warning;
        alert.raisedAt = QDateTime::currentDateTimeUtc();
        ackChanged = upsertAlert(alert) || ackChanged;
    }

    ackChanged = pruneStale() || ackChanged;
    recomputeCounts();
    if (ackChanged)
        Q_EMIT acknowledgementsChanged();
}

void AlertsModel::reset()
{
    beginResetModel();
    m_alerts.clear();
    m_warningCount = 0;
    m_criticalCount = 0;
    m_unacknowledgedCount = 0;
    endResetModel();
    Q_EMIT countsChanged();
    if (!m_acknowledgedIds.isEmpty()) {
        m_acknowledgedIds.clear();
        Q_EMIT acknowledgementsChanged();
    }
}

int AlertsModel::indexOfAlert(const QString& id) const
{
    for (int i = 0; i < m_alerts.size(); ++i) {
        if (m_alerts.at(i).id == id)
            return i;
    }
    return -1;
}

bool AlertsModel::upsertAlert(const Alert& alert)
{
    const int index = indexOfAlert(alert.id);
    if (index < 0) {
        Alert inserted = alert;
        inserted.stale = false;
        inserted.acknowledged = m_acknowledgedIds.contains(alert.id);
        beginInsertRows(QModelIndex(), m_alerts.size(), m_alerts.size());
        m_alerts.append(inserted);
        endInsertRows();
        return false;
    }

    Alert& existing = m_alerts[index];
    const bool severityChanged = existing.severity != alert.severity;
    existing.title = alert.title;
    existing.description = alert.description;
    existing.severity = alert.severity;
    existing.stale = false;
    existing.sticky = alert.sticky;
    bool ackChanged = false;
    if (severityChanged) {
        if (existing.acknowledged) {
            existing.acknowledged = false;
            ackChanged = m_acknowledgedIds.remove(alert.id) || ackChanged;
        }
        existing.raisedAt = alert.raisedAt;
    } else if (!existing.acknowledged && m_acknowledgedIds.contains(alert.id)) {
        existing.acknowledged = true;
        ackChanged = true;
    }
    Q_EMIT dataChanged(this->index(index), this->index(index),
                      {TitleRole, DescriptionRole, SeverityRole, TimestampRole, AcknowledgedRole});
    return ackChanged;
}

bool AlertsModel::pruneStale()
{
    bool ackChanged = false;
    for (int i = m_alerts.size() - 1; i >= 0; --i) {
        if (m_alerts.at(i).stale && !m_alerts.at(i).sticky) {
            const QString id = m_alerts.at(i).id;
            beginRemoveRows(QModelIndex(), i, i);
            m_alerts.removeAt(i);
            endRemoveRows();
            if (m_acknowledgedIds.remove(id))
                ackChanged = true;
        }
    }
    return ackChanged;
}

bool AlertsModel::isSecurityAlertId(const QString& id)
{
    return id.startsWith(QStringLiteral("security:"));
}

void AlertsModel::recomputeCounts()
{
    int warnings = 0;
    int critical = 0;
    int unacknowledged = 0;
    for (const Alert& alert : std::as_const(m_alerts)) {
        if (alert.severity == Critical)
            ++critical;
        else if (alert.severity == Warning)
            ++warnings;
        if (!alert.acknowledged)
            ++unacknowledged;
    }
    if (warnings == m_warningCount && critical == m_criticalCount
        && unacknowledged == m_unacknowledgedCount)
        return;
    m_warningCount = warnings;
    m_criticalCount = critical;
    m_unacknowledgedCount = unacknowledged;
    Q_EMIT countsChanged();
}

QStringList AlertsModel::acknowledgedAlertIds() const
{
    QStringList ids;
    ids.reserve(m_acknowledgedIds.size());
    for (const QString& id : m_acknowledgedIds)
        ids.append(id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void AlertsModel::setAcknowledgedAlertIds(const QStringList& ids)
{
    QSet<QString> normalized;
    normalized.reserve(ids.size());
    for (const QString& id : ids) {
        const QString trimmed = id.trimmed();
        if (!trimmed.isEmpty())
            normalized.insert(trimmed);
    }

    if (normalized == m_acknowledgedIds)
        return;

    m_acknowledgedIds = normalized;

    bool ackChanged = false;
    for (int i = 0; i < m_alerts.size(); ++i) {
        Alert& alert = m_alerts[i];
        const bool shouldAck = m_acknowledgedIds.contains(alert.id);
        if (alert.acknowledged == shouldAck)
            continue;
        alert.acknowledged = shouldAck;
        Q_EMIT dataChanged(this->index(i), this->index(i), {AcknowledgedRole});
        ackChanged = true;
    }

    if (ackChanged) {
        recomputeCounts();
        Q_EMIT acknowledgementsChanged();
    }
}
