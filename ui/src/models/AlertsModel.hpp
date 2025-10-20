#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QList>
#include <QSet>
#include <QString>
#include <QStringList>

#include "RiskTypes.hpp"

class AlertsModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(int  criticalCount READ criticalCount NOTIFY countsChanged)
    Q_PROPERTY(int  warningCount READ warningCount NOTIFY countsChanged)
    Q_PROPERTY(int  unacknowledgedCount READ unacknowledgedCount NOTIFY countsChanged)
    Q_PROPERTY(bool hasActiveAlerts READ hasActiveAlerts NOTIFY countsChanged)
    Q_PROPERTY(bool hasUnacknowledgedAlerts READ hasUnacknowledgedAlerts NOTIFY countsChanged)

public:
    enum Roles {
        IdRole = Qt::UserRole + 1,
        TitleRole,
        DescriptionRole,
        SeverityRole,
        TimestampRole,
        AcknowledgedRole,
    };

    enum Severity {
        Info = 0,
        Warning = 1,
        Critical = 2,
    };
    Q_ENUM(Severity)

    explicit AlertsModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    int  criticalCount() const { return m_criticalCount; }
    int  warningCount() const { return m_warningCount; }
    int  unacknowledgedCount() const { return m_unacknowledgedCount; }
    bool hasActiveAlerts() const { return !m_alerts.isEmpty(); }
    bool hasUnacknowledgedAlerts() const { return m_unacknowledgedCount > 0; }

    Q_INVOKABLE void acknowledge(const QString& alertId);
    Q_INVOKABLE void clearAcknowledged();
    Q_INVOKABLE void acknowledgeAll();

    void updateFromRiskSnapshot(const RiskSnapshotData& snapshot);
    void reset();

    QStringList acknowledgedAlertIds() const;
    void setAcknowledgedAlertIds(const QStringList& ids);

signals:
    void countsChanged();
    void acknowledgementsChanged();

private:
    struct Alert {
        QString id;
        QString title;
        QString description;
        Severity severity = Info;
        QDateTime raisedAt;
        bool acknowledged = false;
        bool stale = false;
        bool sticky = false;
    };

    int indexOfAlert(const QString& id) const;
    bool upsertAlert(const Alert& alert);
    bool pruneStale();
    void recomputeCounts();

    QList<Alert> m_alerts;
    int m_warningCount = 0;
    int m_criticalCount = 0;
    int m_unacknowledgedCount = 0;
    QSet<QString> m_acknowledgedIds;
};
