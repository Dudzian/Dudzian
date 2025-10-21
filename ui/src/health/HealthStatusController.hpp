#pragma once

#include <QObject>
#include <QDateTime>
#include <QTimer>
#include <QFutureWatcher>
#include <QtGlobal>
#include <QStringList>

#include <memory>

#include "grpc/HealthClient.hpp"

class QThreadPool;

class HealthStatusController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool healthy READ healthy NOTIFY statusChanged)
    Q_PROPERTY(QString version READ version NOTIFY statusChanged)
    Q_PROPERTY(QString gitCommit READ gitCommit NOTIFY statusChanged)
    Q_PROPERTY(QString gitCommitShort READ gitCommitShort NOTIFY statusChanged)
    Q_PROPERTY(QString startedAt READ startedAt NOTIFY statusChanged)
    Q_PROPERTY(QString startedAtLocal READ startedAtLocal NOTIFY statusChanged)
    Q_PROPERTY(QString uptime READ uptime NOTIFY statusChanged)
    Q_PROPERTY(QString lastCheckedAt READ lastCheckedAt NOTIFY statusChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(int refreshIntervalSeconds READ refreshIntervalSeconds WRITE setRefreshIntervalSeconds NOTIFY refreshIntervalSecondsChanged)
    Q_PROPERTY(bool autoRefreshEnabled READ autoRefreshEnabled WRITE setAutoRefreshEnabled NOTIFY autoRefreshEnabledChanged)

public:
    explicit HealthStatusController(QObject* parent = nullptr);
    ~HealthStatusController() override;

    bool healthy() const { return m_healthy; }
    QString version() const { return m_version; }
    QString gitCommit() const { return m_gitCommit; }
    QString gitCommitShort() const;
    QString startedAt() const;
    QString startedAtLocal() const;
    QString uptime() const;
    QString lastCheckedAt() const;
    QString statusMessage() const { return m_statusMessage; }
    bool busy() const { return m_busy; }
    int refreshIntervalSeconds() const { return m_refreshIntervalSeconds; }
    bool autoRefreshEnabled() const { return m_autoRefreshEnabled; }

    Q_INVOKABLE void refresh();
    void setRefreshIntervalSeconds(int seconds);
    void setAutoRefreshEnabled(bool enabled);

    void setEndpoint(const QString& endpoint);
    void setTlsConfig(const GrpcTlsConfig& config);
    void setAuthToken(const QString& token);
    void setRbacRole(const QString& role);
    void setRbacScopes(const QStringList& scopes);

    void setHealthClientForTesting(const std::shared_ptr<HealthClientInterface>& client);
    void setThreadPoolForTesting(QThreadPool* pool);
    bool isTimerActiveForTesting() const { return m_refreshTimer.isActive(); }

signals:
    void statusChanged();
    void statusMessageChanged();
    void busyChanged();
    void refreshIntervalSecondsChanged();
    void autoRefreshEnabledChanged();

private slots:
    void handleRefreshFinished();
    void scheduleNextRefresh();

private:
    void startRefreshLocked();
    void applyResult(const HealthClientInterface::HealthCheckResult& result);
    void updateStatusMessage(bool ok, const QString& error);
    QString formatDateTime(const QDateTime& dateTime, Qt::TimeSpec spec) const;

    std::shared_ptr<HealthClientInterface> m_client;
    QFutureWatcher<HealthClientInterface::HealthCheckResult> m_watcher;
    QTimer             m_refreshTimer;
    QThreadPool*       m_threadPool = nullptr;

    bool        m_healthy = false;
    QString     m_version;
    QString     m_gitCommit;
    QDateTime   m_startedAtUtc;
    QDateTime   m_lastCheckedUtc;
    QString     m_statusMessage;
    bool        m_busy = false;
    int         m_refreshIntervalSeconds = 60;
    bool        m_autoRefreshEnabled = true;
};

