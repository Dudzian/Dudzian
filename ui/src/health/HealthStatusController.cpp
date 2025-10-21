#include "HealthStatusController.hpp"

#include <QtConcurrent>
#include <QLoggingCategory>
#include <QThreadPool>
#include <QStringList>

Q_LOGGING_CATEGORY(lcHealthStatus, "bot.shell.health.status")

namespace {

QString humanReadableDuration(qint64 seconds)
{
    if (seconds <= 0) {
        return QStringLiteral("0s");
    }

    const qint64 days = seconds / 86400;
    seconds %= 86400;
    const qint64 hours = seconds / 3600;
    seconds %= 3600;
    const qint64 minutes = seconds / 60;
    seconds %= 60;

    QStringList parts;
    if (days > 0) {
        parts.append(QObject::tr("%1d").arg(days));
    }
    if (hours > 0) {
        parts.append(QObject::tr("%1h").arg(hours));
    }
    if (minutes > 0) {
        parts.append(QObject::tr("%1m").arg(minutes));
    }
    if (seconds > 0 || parts.isEmpty()) {
        parts.append(QObject::tr("%1s").arg(seconds));
    }
    return parts.join(QLatin1Char(' '));
}

} // namespace

HealthStatusController::HealthStatusController(QObject* parent)
    : QObject(parent)
    , m_client(std::make_shared<HealthClient>())
{
    connect(&m_watcher, &QFutureWatcher<HealthClientInterface::HealthCheckResult>::finished,
            this, &HealthStatusController::handleRefreshFinished);
    connect(&m_refreshTimer, &QTimer::timeout, this, &HealthStatusController::scheduleNextRefresh);
    m_refreshTimer.setSingleShot(true);
}

HealthStatusController::~HealthStatusController() = default;

QString HealthStatusController::gitCommitShort() const
{
    return m_gitCommit.left(8);
}

QString HealthStatusController::startedAt() const
{
    return formatDateTime(m_startedAtUtc, Qt::UTC);
}

QString HealthStatusController::startedAtLocal() const
{
    return formatDateTime(m_startedAtUtc.toLocalTime(), Qt::LocalTime);
}

QString HealthStatusController::uptime() const
{
    if (!m_startedAtUtc.isValid()) {
        return {};
    }
    const qint64 seconds = m_startedAtUtc.secsTo(QDateTime::currentDateTimeUtc());
    return humanReadableDuration(seconds);
}

QString HealthStatusController::lastCheckedAt() const
{
    return formatDateTime(m_lastCheckedUtc.toLocalTime(), Qt::LocalTime);
}

void HealthStatusController::refresh()
{
    if (!m_client) {
        return;
    }
    if (m_busy) {
        return;
    }
    startRefreshLocked();
}

void HealthStatusController::setRefreshIntervalSeconds(int seconds)
{
    const int sanitized = qBound(5, seconds, 3600);
    if (sanitized == m_refreshIntervalSeconds) {
        return;
    }
    m_refreshIntervalSeconds = sanitized;
    Q_EMIT refreshIntervalSecondsChanged();
    if (m_autoRefreshEnabled && m_refreshTimer.isActive()) {
        m_refreshTimer.start(m_refreshIntervalSeconds * 1000);
    }
}

void HealthStatusController::setAutoRefreshEnabled(bool enabled)
{
    if (m_autoRefreshEnabled == enabled) {
        return;
    }
    m_autoRefreshEnabled = enabled;
    Q_EMIT autoRefreshEnabledChanged();
    if (!m_autoRefreshEnabled) {
        m_refreshTimer.stop();
    } else if (!m_busy) {
        m_refreshTimer.start(m_refreshIntervalSeconds * 1000);
    }
}

void HealthStatusController::setEndpoint(const QString& endpoint)
{
    if (m_client) {
        m_client->setEndpoint(endpoint);
    }
}

void HealthStatusController::setTlsConfig(const GrpcTlsConfig& config)
{
    if (m_client) {
        m_client->setTlsConfig(config);
    }
}

void HealthStatusController::setAuthToken(const QString& token)
{
    if (m_client) {
        m_client->setAuthToken(token);
    }
}

void HealthStatusController::setRbacRole(const QString& role)
{
    if (m_client) {
        m_client->setRbacRole(role);
    }
}

void HealthStatusController::setRbacScopes(const QStringList& scopes)
{
    if (m_client) {
        m_client->setRbacScopes(scopes);
    }
}

void HealthStatusController::setHealthClientForTesting(const std::shared_ptr<HealthClientInterface>& client)
{
    if (!client) {
        return;
    }
    m_client = client;
}

void HealthStatusController::setThreadPoolForTesting(QThreadPool* pool)
{
    m_threadPool = pool;
}

void HealthStatusController::handleRefreshFinished()
{
    if (!m_watcher.isFinished()) {
        return;
    }

    const auto result = m_watcher.result();
    applyResult(result);

    m_busy = false;
    Q_EMIT busyChanged();

    if (m_autoRefreshEnabled) {
        m_refreshTimer.start(m_refreshIntervalSeconds * 1000);
    }
}

void HealthStatusController::scheduleNextRefresh()
{
    if (m_autoRefreshEnabled && !m_busy) {
        startRefreshLocked();
    }
}

void HealthStatusController::startRefreshLocked()
{
    if (!m_client) {
        return;
    }
    m_busy = true;
    Q_EMIT busyChanged();

    auto future = QtConcurrent::run(m_threadPool ? m_threadPool : QThreadPool::globalInstance(), [client = m_client]() {
        return client->check();
    });
    m_watcher.setFuture(future);
}

void HealthStatusController::applyResult(const HealthClientInterface::HealthCheckResult& result)
{
    m_healthy = result.ok;
    m_version = result.version;
    m_gitCommit = result.gitCommit;
    m_startedAtUtc = result.startedAtUtc;
    m_lastCheckedUtc = QDateTime::currentDateTimeUtc();

    updateStatusMessage(result.ok, result.errorMessage);

    Q_EMIT statusChanged();
}

void HealthStatusController::updateStatusMessage(bool ok, const QString& error)
{
    QString message;
    if (ok) {
        if (!m_version.isEmpty()) {
            message = tr("HealthService OK (wersja %1)").arg(m_version);
        } else {
            message = tr("HealthService OK");
        }
    } else if (!error.isEmpty()) {
        message = tr("HealthService błąd: %1").arg(error);
    } else {
        message = tr("HealthService nie jest dostępny");
    }

    if (message == m_statusMessage) {
        return;
    }
    m_statusMessage = message;
    Q_EMIT statusMessageChanged();
}

QString HealthStatusController::formatDateTime(const QDateTime& dateTime, Qt::TimeSpec) const
{
    if (!dateTime.isValid()) {
        return {};
    }
    return dateTime.toString(Qt::ISODateWithMs);
}

