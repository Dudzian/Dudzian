#include "runtime/OfflineRuntimeBridge.hpp"

#include <QLoggingCategory>
#include <QtGlobal>

#include "runtime/OfflineRuntimeService.hpp"

Q_LOGGING_CATEGORY(lcOfflineBridge, "bot.shell.offline.bridge")

OfflineRuntimeBridge::OfflineRuntimeBridge(QObject* parent)
    : QObject(parent)
{
}

OfflineRuntimeBridge::~OfflineRuntimeBridge() = default;

void OfflineRuntimeBridge::setEndpoint(const QUrl& endpoint)
{
    m_endpoint = endpoint;
}

void OfflineRuntimeBridge::setInstrument(const TradingClient::InstrumentConfig& config)
{
    m_instrument = config;
    if (m_service)
        m_service->setInstrument(m_instrument);
}

void OfflineRuntimeBridge::setHistoryLimit(int limit)
{
    m_historyLimit = qMax(1, limit);
    if (m_service)
        m_service->setHistoryLimit(m_historyLimit);
}

void OfflineRuntimeBridge::setAutoRunEnabled(bool enabled)
{
    if (m_autoRunEnabled == enabled)
        return;
    m_autoRunEnabled = enabled;
    if (m_service)
        m_service->setAutoRunEnabled(m_autoRunEnabled);
}

void OfflineRuntimeBridge::setStrategyConfig(const QVariantMap& config)
{
    m_strategyConfig = config;
    if (m_service)
        m_service->setStrategyConfig(m_strategyConfig);
}

void OfflineRuntimeBridge::setDatasetPath(const QString& path)
{
    if (m_datasetPath == path)
        return;
    m_datasetPath = path;
    if (m_service)
        m_service->setDatasetPath(m_datasetPath);
}

QVariantMap OfflineRuntimeBridge::autoModeSnapshot() const
{
    if (!m_service)
        return QVariantMap();
    return m_service->buildAutoModeSnapshot();
}

QVariantMap OfflineRuntimeBridge::alertPreferences() const
{
    if (!m_service)
        return QVariantMap();
    return m_service->alertPreferences();
}

void OfflineRuntimeBridge::updateAlertPreferences(const QVariantMap& preferences)
{
    ensureService();
    if (!m_service)
        return;
    m_service->setAlertPreferences(preferences);
}

void OfflineRuntimeBridge::toggleAutoMode(bool enabled)
{
    if (enabled)
        startAutomation();
    else
        stopAutomation();
}

void OfflineRuntimeBridge::start()
{
    if (m_running)
        return;
    ensureService();
    configureService();
    if (!m_service) {
        qCWarning(lcOfflineBridge) << "Brak instancji OfflineRuntimeService";
        return;
    }
    m_service->start();
    m_running = true;
}

void OfflineRuntimeBridge::stop()
{
    if (!m_running)
        return;
    if (m_service)
        m_service->stop();
    m_running = false;
}

void OfflineRuntimeBridge::refreshRiskNow()
{
    if (!m_service)
        return;
    m_service->refreshRisk();
}

void OfflineRuntimeBridge::startAutomation()
{
    ensureService();
    if (!m_service)
        return;
    m_service->startAutomation();
}

void OfflineRuntimeBridge::stopAutomation()
{
    if (!m_service)
        return;
    m_service->stopAutomation();
}

void OfflineRuntimeBridge::ensureService()
{
    if (m_service)
        return;

    m_service = std::make_unique<OfflineRuntimeService>(this);
    connect(m_service.get(), &OfflineRuntimeService::connectionStateChanged,
            this, &OfflineRuntimeBridge::applyConnectionState);
    connect(m_service.get(), &OfflineRuntimeService::historyReady,
            this, &OfflineRuntimeBridge::historyReceived);
    connect(m_service.get(), &OfflineRuntimeService::riskReady,
            this, &OfflineRuntimeBridge::riskStateReceived);
    connect(m_service.get(), &OfflineRuntimeService::guardReady,
            this, &OfflineRuntimeBridge::performanceGuardUpdated);
    connect(m_service.get(), &OfflineRuntimeService::automationStateChanged,
            this, &OfflineRuntimeBridge::automationStateChanged);
    connect(m_service.get(), &OfflineRuntimeService::alertPreferencesChanged,
            this, &OfflineRuntimeBridge::alertPreferencesChanged);
}

void OfflineRuntimeBridge::configureService()
{
    if (!m_service)
        return;

    m_service->setInstrument(m_instrument);
    m_service->setHistoryLimit(m_historyLimit);
    m_service->setAutoRunEnabled(m_autoRunEnabled);
    m_service->setStrategyConfig(m_strategyConfig);
    m_service->setDatasetPath(m_datasetPath);
    Q_UNUSED(m_endpoint);
}

void OfflineRuntimeBridge::applyConnectionState(const QString& state)
{
    if (m_connectionState == state)
        return;
    m_connectionState = state;
    emit connectionStateChanged(state);
}
