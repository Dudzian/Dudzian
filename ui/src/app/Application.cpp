#include "Application.hpp"

#include <QByteArray>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QColor>
#include <QPalette>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QGuiApplication>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLoggingCategory>
#include <QStringList>
#include <QPoint>
#include <QRect>
#include <QQmlContext>
#include <QQuickWindow>
#include <QSaveFile>
#include <QScreen>
#include <QHash>
#include <QPair>
#include <QRegularExpression>
#include <QSysInfo>
#include <QSet>
#include <QUrl>
#include <QTimer>
#include <QScopeGuard>
#include <QtGlobal>
#include <optional>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "utils/PathUtils.hpp"
#include "license/LicenseActivationController.hpp"
#include "app/ActivationController.hpp"
#include "app/ConfigurationWizardController.hpp"
#include "app/UiModuleManager.hpp"
#include "app/UiModuleViewsModel.hpp"
#include "app/StrategyConfigController.hpp"
#include "app/StrategyWorkbenchController.hpp"
#include "app/MarketplaceController.hpp"
#include "app/PortfolioManagerController.hpp"
#include "runtime/OfflineRuntimeBridge.hpp"
#include "security/SecurityAdminController.hpp"
#include "support/SupportBundleController.hpp"
#include "health/HealthStatusController.hpp"
#include "reporting/ReportCenterController.hpp"
#include "update/OfflineUpdateManager.hpp"
#include "models/ResultsDashboardModel.hpp"
#include "grpc/BotCoreLocalService.hpp"
#include "grpc/HealthClient.hpp"
#include "grpc/MetricsClient.hpp"

Q_LOGGING_CATEGORY(lcAppMetrics, "bot.shell.app.metrics")

namespace {

constexpr double kDefaultRiskRefreshSeconds = 5.0;
constexpr int kMinRiskRefreshIntervalMs = 1000;
constexpr int kMaxRiskRefreshIntervalMs = 300000;
constexpr int kUiSettingsDebounceMs = 500;
constexpr auto kUiSettingsEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_PATH");
constexpr auto kUiSettingsDisableEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_DISABLE");
constexpr auto kRiskHistoryExportDirEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_DIR");
constexpr auto kRiskHistoryExportLimitEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_LIMIT");
constexpr auto kRiskHistoryExportLimitEnabledEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_LIMIT_ENABLED");
constexpr auto kRiskHistoryAutoExportEnabledEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT");
constexpr auto kRiskHistoryAutoExportIntervalEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_INTERVAL_MINUTES");
constexpr auto kRiskHistoryAutoExportBasenameEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_BASENAME");
constexpr auto kRiskHistoryAutoExportDirEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_DIR");
constexpr auto kRiskHistoryAutoExportLocalTimeEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_USE_LOCAL_TIME");
constexpr auto kDecisionLogPathEnv = QByteArrayLiteral("BOT_CORE_UI_DECISION_LOG");
constexpr auto kDecisionLogLimitEnv = QByteArrayLiteral("BOT_CORE_UI_DECISION_LOG_LIMIT");
constexpr auto kRegimeThresholdsEnv = QByteArrayLiteral("BOT_CORE_UI_REGIME_THRESHOLDS");
constexpr auto kRegimeTimelineLimitEnv = QByteArrayLiteral("BOT_CORE_UI_REGIME_TIMELINE_LIMIT");
constexpr auto kTransportModeEnv = QByteArrayLiteral("BOT_CORE_UI_TRANSPORT_MODE");
constexpr auto kTransportDatasetEnv = QByteArrayLiteral("BOT_CORE_UI_TRANSPORT_DATASET");

class InProcessMetricsClient final : public MetricsClientInterface
{
public:
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setTlsConfig(const TelemetryTlsConfig& config) override { m_tlsConfig = config; }
    void setAuthToken(const QString& token) override { m_authToken = token; }
    void setRbacRole(const QString& role) override { m_role = role; }

    bool pushSnapshot(const botcore::trading::v1::MetricsSnapshot& snapshot,
                      QString* errorMessage = nullptr) override
    {
        Q_UNUSED(errorMessage);
        m_snapshots.push_back(snapshot);
        return true;
    }

    const std::vector<botcore::trading::v1::MetricsSnapshot>& snapshots() const { return m_snapshots; }

private:
    QString m_endpoint;
    TelemetryTlsConfig m_tlsConfig;
    QString m_authToken;
    QString m_role;
    std::vector<botcore::trading::v1::MetricsSnapshot> m_snapshots;
};

class InProcessHealthClient final : public HealthClientInterface
{
public:
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setTlsConfig(const GrpcTlsConfig& config) override { m_tlsConfig = config; }
    void setAuthToken(const QString& token) override { m_authToken = token; }
    void setRbacRole(const QString& role) override { m_role = role; }
    void setRbacScopes(const QStringList& scopes) override { m_scopes = scopes; }

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const override { return {}; }

    HealthCheckResult check() override
    {
        HealthCheckResult result;
        result.ok = true;
        result.version = QStringLiteral("in-process");
        result.gitCommit = QStringLiteral("local");
        result.startedAtUtc = QDateTime::currentDateTimeUtc().addSecs(-3600);
        return result;
    }

private:
    QString m_endpoint;
    GrpcTlsConfig m_tlsConfig;
    QString m_authToken;
    QString m_role;
    QStringList m_scopes;
};

using bot::shell::utils::expandPath;
using bot::shell::utils::watchableDirectories;

std::optional<QString> envValue(const QByteArray& key)
{
    if (!qEnvironmentVariableIsSet(key.constData()))
        return std::nullopt;
    return qEnvironmentVariable(key.constData());
}

std::optional<bool> envBool(const QByteArray& key)
{
    const auto valueOpt = envValue(key);
    if (!valueOpt.has_value())
        return std::nullopt;
    const QString normalized = valueOpt->trimmed().toLower();
    if (normalized.isEmpty())
        return std::nullopt;
    if (normalized == QStringLiteral("1") || normalized == QStringLiteral("true") ||
        normalized == QStringLiteral("yes") || normalized == QStringLiteral("on"))
        return true;
    if (normalized == QStringLiteral("0") || normalized == QStringLiteral("false") ||
        normalized == QStringLiteral("no") || normalized == QStringLiteral("off"))
        return false;
    qCWarning(lcAppMetrics) << "Nieprawidłowa wartość" << *valueOpt
                            << "w zmiennej" << QString::fromUtf8(key)
                            << "– oczekiwano wartości boolowskiej (true/false).";
    return std::nullopt;
}

QString readTokenFile(const QString& rawPath, const QString& label = QStringLiteral("MetricsService"))
{
    if (rawPath.trimmed().isEmpty())
        return {};
    const QString path = expandPath(rawPath.trimmed());
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcAppMetrics) << "Plik z tokenem" << label << "nie istnieje:" << path;
        return {};
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać pliku z tokenem" << label << ':' << path
                                << file.errorString();
        return {};
    }
    const QByteArray data = file.readAll();
    QString token = QString::fromUtf8(data).trimmed();
    if (token.isEmpty()) {
        qCWarning(lcAppMetrics) << "Plik" << path << "nie zawiera tokenu autoryzacyjnego" << label;
        return {};
    }
    return token;
}

QStringList splitScopesList(const QString& raw)
{
    QString normalized = raw;
    normalized.replace(QLatin1Char(';'), QLatin1Char(','));
    QStringList parts = normalized.split(QRegularExpression(QStringLiteral("[\n\r\t, ]+")),
                                        Qt::SkipEmptyParts);
    QStringList unique;
    QSet<QString> seen;
    for (QString part : parts) {
        const QString trimmed = part.trimmed();
        if (trimmed.isEmpty())
            continue;
        if (seen.contains(trimmed))
            continue;
        seen.insert(trimmed);
        unique.append(trimmed);
    }
    return unique;
}

QString sanitizeAutoExportBasename(const QString& raw)
{
    QString sanitized = raw.trimmed();
    if (sanitized.isEmpty())
        return QStringLiteral("risk-history");

    static const QRegularExpression invalidCharacters(QStringLiteral("[^A-Za-z0-9_-]+"));
    sanitized.replace(invalidCharacters, QStringLiteral("_"));

    static const QRegularExpression repeatedUnderscore(QStringLiteral("_+"));
    sanitized.replace(repeatedUnderscore, QStringLiteral("_"));

    while (sanitized.startsWith(QLatin1Char('_')))
        sanitized.remove(0, 1);
    while (sanitized.endsWith(QLatin1Char('_')))
        sanitized.chop(1);

    if (sanitized.isEmpty())
        return QStringLiteral("risk-history");

    constexpr int kMaxLength = 80;
    if (sanitized.size() > kMaxLength)
        sanitized = sanitized.left(kMaxLength);

    return sanitized;
}

} // namespace

Application::Application(QQmlApplicationEngine& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine)
{
    m_offlineStatus = tr("Offline runtime: nieaktywny");
    m_activationController = std::make_unique<ActivationController>(this);
    connect(m_activationController.get(), &ActivationController::errorChanged, this,
            &Application::handleActivationErrorChanged);
    connect(m_activationController.get(), &ActivationController::fingerprintChanged, this,
            &Application::handleActivationFingerprintChanged);
    connect(m_activationController.get(), &ActivationController::licensesChanged, this,
            &Application::handleActivationLicensesChanged);
    connect(m_activationController.get(), &ActivationController::oemLicenseChanged, this,
            &Application::handleActivationOemLicenseChanged);

    // Startowe ustawienia instrumentu z klienta (mogą być nadpisane przez CLI)
    m_instrument = m_client.instrumentConfig();
    initializeSupportedExchanges();

    m_licenseController = std::make_unique<LicenseActivationController>();
    m_licenseController->setParent(this);
    const QString configDir = QDir::current().absoluteFilePath(QStringLiteral("config"));
    const QString defaultLicensePath = QDir::current().absoluteFilePath(QStringLiteral("var/licenses/active/license.json"));
    m_licenseController->setConfigDirectory(configDir);
    m_licenseController->setLicenseStoragePath(defaultLicensePath);

    m_securityController = std::make_unique<SecurityAdminController>(this);
    m_securityController->setProfilesPath(QDir::current().absoluteFilePath(QStringLiteral("config/user_profiles.json")));
    m_securityController->setLicensePath(defaultLicensePath);
    m_securityController->setLogPath(QDir::current().absoluteFilePath(QStringLiteral("logs/security_admin.log")));
    m_securityController->setTpmQuotePath(QDir::current().absoluteFilePath(QStringLiteral("var/security/tpm_quote.json")));
    m_securityController->setIntegrityManifestPath(QDir::current().absoluteFilePath(QStringLiteral("config/integrity_manifest.json")));
    connect(m_securityController.get(),
            &SecurityAdminController::securityAlertRaised,
            this,
            [this](const QString& id, int severity, const QString& title, const QString& message) {
                const int boundedSeverity = qBound(0, severity, 2);
                m_alertsModel.raiseAlert(id,
                                         title,
                                         message,
                                         static_cast<AlertsModel::Severity>(boundedSeverity),
                                         true);
            });
    connect(m_securityController.get(), &SecurityAdminController::tpmStatusChanged, this, &Application::updateSecurityCacheFromControllers);
    connect(m_securityController.get(), &SecurityAdminController::integrityReportChanged, this, &Application::updateSecurityCacheFromControllers);
    connect(m_securityController.get(), &SecurityAdminController::auditLogChanged, this, &Application::updateSecurityCacheFromControllers);

    m_reportController = std::make_unique<ReportCenterController>(this);
    m_reportController->setReportsDirectory(QDir::current().absoluteFilePath(QStringLiteral("var/reports")));
    m_reportController->setReportsRoot(QDir::current().absoluteFilePath(QStringLiteral("var/reports")));

    m_strategyController = std::make_unique<StrategyConfigController>(this);
    m_strategyController->setConfigPath(QDir::current().absoluteFilePath(QStringLiteral("config/core.yaml")));
    m_strategyController->setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py")));

    m_workbenchController = std::make_unique<StrategyWorkbenchController>(this);
    m_workbenchController->setConfigPath(QDir::current().absoluteFilePath(QStringLiteral("config/core.yaml")));
    m_workbenchController->setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py")));

    m_supportController = std::make_unique<SupportBundleController>(this);

    m_healthController = std::make_unique<HealthStatusController>(this);

    m_marketplaceController = std::make_unique<MarketplaceController>(this);
    m_marketplaceController->setBridgeScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_marketplace_bridge.py")));
    m_marketplaceController->setPresetsDirectory(QDir::current().absoluteFilePath(QStringLiteral("data/strategies")));
    m_marketplaceController->setLicensesPath(QDir::current().absoluteFilePath(QStringLiteral("var/marketplace_licenses.json")));

    m_portfolioController = std::make_unique<PortfolioManagerController>(this);
    m_portfolioController->setBridgeScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_portfolio_bridge.py")));
    m_portfolioController->setStorePath(QDir::current().absoluteFilePath(QStringLiteral("var/portfolio_links.json")));

    m_configurationWizard = std::make_unique<ConfigurationWizardController>(this);
    m_configurationWizard->setStrategyConfigController(m_strategyController.get());
    m_configurationWizard->setMarketplaceController(m_marketplaceController.get());
    m_configurationWizard->setRiskModel(&m_riskModel);
    m_configurationWizard->setAlertsModel(&m_alertsModel);

    m_updateManager = std::make_unique<OfflineUpdateManager>(this);
    m_updateManager->setPackagesDirectory(QDir::current().absoluteFilePath(QStringLiteral("var/updates/packages")));
    m_updateManager->setInstallDirectory(QDir::current().absoluteFilePath(QStringLiteral("var/updates/install")));
    m_updateManager->setStateFilePath(QDir::current().absoluteFilePath(QStringLiteral("var/updates/state.json")));
    m_updateManager->setLicenseController(m_licenseController.get());
    m_updateManager->setTpmEvidencePath(QDir::current().absoluteFilePath(QStringLiteral("var/security/tpm_quote.json")));

    m_resultsDashboard = std::make_unique<ResultsDashboardModel>(this);
    m_resultsDashboard->setRiskHistoryModel(&m_riskHistoryModel);
    m_resultsDashboard->setRiskStateModel(&m_riskModel);
    m_resultsDashboard->setDecisionLogModel(&m_decisionLogModel);

    connect(m_updateManager.get(),
            &OfflineUpdateManager::updateFailed,
            this,
            [this](const QString& id, const QString& reason) {
                const QString alertId = QStringLiteral("updates:%1").arg(id);
                m_alertsModel.raiseAlert(alertId,
                                         tr("Aktualizacja offline"),
                                         reason,
                                         AlertsModel::Warning,
                                         true);
            });
    connect(m_updateManager.get(),
            &OfflineUpdateManager::updateCompleted,
            this,
            [this](const QString& id) {
                const QString alertId = QStringLiteral("updates:%1").arg(id);
                m_alertsModel.raiseAlert(alertId,
                                         tr("Aktualizacja offline"),
                                         tr("Zainstalowano pakiet %1.").arg(id),
                                         AlertsModel::Info,
                                         false);
            });

    m_updateManager->refresh();

    m_moduleManager = std::make_unique<UiModuleManager>(this);
    m_moduleViewsModel = std::make_unique<UiModuleViewsModel>(this);
    m_moduleViewsModel->setModuleManager(m_moduleManager.get());

    m_decisionLogModel.setParent(this);
    m_decisionLogFilter.setSourceModel(&m_decisionLogModel);
    m_performanceTelemetry = std::make_unique<PerformanceTelemetryController>(this);
    m_performanceTelemetry->setPerformanceGuard(m_guard);

    const QVector<IndicatorSeriesDefinition> indicatorDefinitions = {
        {QStringLiteral("ema_fast"), tr("EMA 12"), QColor::fromRgbF(0.96, 0.74, 0.23, 1.0), false},
        {QStringLiteral("ema_slow"), tr("EMA 26"), QColor::fromRgbF(0.62, 0.81, 0.93, 1.0), true},
        {QStringLiteral("vwap"), tr("VWAP"), QColor::fromRgbF(0.74, 0.53, 0.96, 1.0), true},
    };
    m_indicatorModel.setSeriesDefinitions(indicatorDefinitions);

    m_regimeThresholdWatcher.setParent(this);
    connect(&m_regimeThresholdWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleRegimeThresholdPathChanged);
    connect(&m_regimeThresholdWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleRegimeThresholdPathChanged);

    m_tradingTokenWatcher.setParent(this);
    m_metricsTokenWatcher.setParent(this);
    m_healthTokenWatcher.setParent(this);
    m_tradingTlsWatcher.setParent(this);
    m_metricsTlsWatcher.setParent(this);
    m_healthTlsWatcher.setParent(this);

    connect(&m_tradingTokenWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleTradingTokenPathChanged);
    connect(&m_tradingTokenWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleTradingTokenPathChanged);
    connect(&m_metricsTokenWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleMetricsTokenPathChanged);
    connect(&m_metricsTokenWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleMetricsTokenPathChanged);
    connect(&m_healthTokenWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleHealthTokenPathChanged);
    connect(&m_healthTokenWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleHealthTokenPathChanged);
    connect(&m_tradingTlsWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleTradingTlsPathChanged);
    connect(&m_tradingTlsWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleTradingTlsPathChanged);
    connect(&m_metricsTlsWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleMetricsTlsPathChanged);
    connect(&m_metricsTlsWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleMetricsTlsPathChanged);
    connect(&m_healthTlsWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &Application::handleHealthTlsPathChanged);
    connect(&m_healthTlsWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &Application::handleHealthTlsPathChanged);

    m_regimeTimelineMaximumSnapshots = m_regimeTimelineModel.maximumSnapshots();
    connect(&m_regimeTimelineModel,
            &MarketRegimeTimelineModel::maximumSnapshotsChanged,
            this,
            [this]() {
                const int current = m_regimeTimelineModel.maximumSnapshots();
                if (m_regimeTimelineMaximumSnapshots == current)
                    return;
                m_regimeTimelineMaximumSnapshots = current;
                Q_EMIT regimeTimelineMaximumSnapshotsChanged();
                if (!m_loadingUiSettings)
                    scheduleUiSettingsPersist();
            });

    m_filteredAlertsModel.setSourceModel(&m_alertsModel);
    m_filteredAlertsModel.setSeverityFilter(AlertsFilterProxyModel::WarningsAndCritical);

    connect(&m_filteredAlertsModel, &AlertsFilterProxyModel::filterChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    initializeUiSettingsStorage();
    applyUiThemePalette();

    m_repoRoot = locateRepoRoot();

    if (!m_repoRoot.isEmpty()) {
        applyRegimeThresholdPath(QDir(m_repoRoot).absoluteFilePath(QStringLiteral("config/regime_thresholds.yaml")), false);
        setDecisionLogPathInternal(QDir(m_repoRoot).absoluteFilePath(QStringLiteral("logs/decision_journal")), false);
    } else {
        applyRegimeThresholdPath(QDir::current().absoluteFilePath(QStringLiteral("config/regime_thresholds.yaml")), false);
        setDecisionLogPathInternal(QDir::current().absoluteFilePath(QStringLiteral("logs/decision_journal")), false);
    }

    if (m_supportController) {
        if (!m_repoRoot.isEmpty())
            m_supportController->setScriptPath(QDir(m_repoRoot).absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py")));
        else
            m_supportController->setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py")));
        updateSupportBundleMetadata();
    }

    initializeSecurityRefresh();
    exposeToQml();

    // Podłącz okno po utworzeniu (dla FrameRateMonitor)
    connect(&m_engine, &QQmlApplicationEngine::objectCreated, this,
            [this](QObject* object, const QUrl&) { attachWindow(object); });

    // Połączenia sygnałów klienta (market data)
    connect(&m_client, &TradingClient::historyReceived, this, &Application::handleHistory);
    connect(&m_client, &TradingClient::candleReceived, this, &Application::handleCandle);
    connect(&m_client,
            &TradingClient::indicatorSnapshotReceived,
            this,
            [this](const QString& id, const QVector<IndicatorSample>& samples) {
                m_indicatorModel.replaceSamples(id, samples);
            });
    connect(&m_client,
            &TradingClient::indicatorSampleReceived,
            this,
            [this](const IndicatorSample& sample) { m_indicatorModel.appendSample(sample); });
    connect(&m_client,
            &TradingClient::signalHistoryReceived,
            this,
            [this](const QVector<SignalEventEntry>& events) { m_signalModel.resetWithSignals(events); });
    connect(&m_client,
            &TradingClient::signalEventReceived,
            this,
            [this](const SignalEventEntry& event) { m_signalModel.appendSignal(event); });
    connect(&m_client,
            &TradingClient::marketRegimeUpdated,
            this,
            [this](const MarketRegimeSnapshotEntry& snapshot) {
                if (m_regimeTimelineModel.rowCount() == 0) {
                    m_regimeTimelineModel.resetWithSnapshots({snapshot});
                } else {
                    m_regimeTimelineModel.appendSnapshot(snapshot);
                }
            });

    connect(&m_client, &TradingClient::connectionStateChanged, this,
            [this](const QString& status) {
                m_connectionStatus = status;
                Q_EMIT connectionStatusChanged();
                updateSupportBundleMetadata();
            });

    connect(&m_client, &TradingClient::performanceGuardUpdated, this,
            [this](const PerformanceGuard& guard) {
                m_guard = guard;
                Q_EMIT performanceGuardChanged();
                if (m_frameMonitor) {
                    m_frameMonitor->setPerformanceGuard(m_guard);
                }
                if (m_performanceTelemetry) {
                    m_performanceTelemetry->setPerformanceGuard(m_guard);
                }
                // Telemetria overlay może zależeć od nowych limitów
                reportOverlayTelemetry();
            });

    connect(&m_client, &TradingClient::streamingChanged, this, [this]() {
        const QString state = m_client.isStreaming()
                                  ? QStringLiteral("streaming")
                                  : QStringLiteral("idle");
        m_connectionStatus = state;
        Q_EMIT connectionStatusChanged();
        updateSupportBundleMetadata();
    });

    // Risk state (jeśli dostępne po stronie serwera)
    connect(&m_client, &TradingClient::riskStateReceived, this, &Application::handleRiskState);

    connect(&m_alertsModel, &AlertsModel::acknowledgementsChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::historyChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::maximumEntriesChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::snapshotRecorded, this,
            &Application::handleRiskHistorySnapshotRecorded);

    m_riskRefreshTimer.setTimerType(Qt::VeryCoarseTimer);
    m_riskRefreshTimer.setInterval(m_riskRefreshIntervalMs);
    m_riskRefreshTimer.setSingleShot(false);
    m_riskRefreshTimer.setParent(this);
    connect(&m_riskRefreshTimer, &QTimer::timeout, this, [this]() {
        if (!m_riskRefreshEnabled)
            return;

        m_lastRiskRefreshRequestUtc = QDateTime::currentDateTimeUtc();
        if (m_offlineMode) {
            if (m_offlineBridge)
                m_offlineBridge->refreshRiskNow();
        } else {
            m_client.refreshRiskState();
        }
        m_nextRiskRefreshUtc = m_lastRiskRefreshRequestUtc.addMSecs(m_riskRefreshIntervalMs);
        Q_EMIT riskRefreshScheduleChanged();
    });
}

Application::~Application()
{
    if (m_offlineBridge)
        m_offlineBridge->stop();
    if (m_localService)
        m_localService->stop();
}

QString Application::instrumentLabel() const {
    return m_instrument.symbol;
}

void Application::configureParser(QCommandLineParser& parser) const {
    parser.addHelpOption();
    parser.addOption({{"e", "endpoint"}, tr("Adres gRPC host:port"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"transport-mode", tr("Tryb transportu danych (grpc lub in-process)"),
                      tr("mode"), QStringLiteral("grpc")});
    parser.addOption({"transport-dataset", tr("Dataset OHLCV dla trybu in-process"),
                      tr("path"), QString()});
    parser.addOption({"exchange", tr("Nazwa giełdy"), tr("exchange"), QStringLiteral("BINANCE")});
    parser.addOption({"symbol", tr("Symbol logiczny"), tr("symbol"), QStringLiteral("BTC/USDT")});
    parser.addOption({"venue-symbol", tr("Symbol na giełdzie"), tr("venue"),
                      QStringLiteral("BTCUSDT")});
    parser.addOption({"quote", tr("Waluta kwotowana"), tr("quote"), QStringLiteral("USDT")});
    parser.addOption({"base", tr("Waluta bazowa"), tr("base"), QStringLiteral("BTC")});
    parser.addOption({"granularity", tr("ISO-8601 duration"), tr("granularity"),
                      QStringLiteral("PT1M")});
    parser.addOption({"history-limit", tr("Limit pobieranej historii"), tr("limit"),
                      QStringLiteral("500")});
    parser.addOption({"max-samples", tr("Maksymalna liczba świec w modelu"), tr("samples"),
                      QStringLiteral("10240")});
    parser.addOption({"regime-timeline-limit",
                      tr("Limit próbek osi czasu reżimu rynku (0 = bez limitu)"),
                      tr("count"),
                      QString()});
    parser.addOption({"regime-thresholds",
                      tr("Ścieżka progów MarketRegimeClassifier"),
                      tr("path"),
                      QString()});
    parser.addOption({"fps-target", tr("Docelowy FPS"), tr("fps"), QStringLiteral("60")});
    parser.addOption({"reduce-motion-after", tr("Czas (s) po którym ograniczamy animacje"),
                      tr("seconds"), QStringLiteral("1.0")});
    parser.addOption({"jank-threshold-ms", tr("Budżet janku w ms"), tr("ms"),
                      QStringLiteral("18.0")});
    parser.addOption({"max-overlay-count", tr("Limit nakładek na wykres"), tr("count"),
                      QStringLiteral("3")});
    parser.addOption({"overlay-disable-secondary-fps",
                      tr("Próg FPS wyłączający nakładki drugorzędne"), tr("fps"),
                      QStringLiteral("0")});
    parser.addOption({"risk-refresh-interval",
                      tr("Interwał automatycznego odświeżania ryzyka (s)"), tr("seconds"),
                      QString::number(kDefaultRiskRefreshSeconds, 'f', 1)});
    parser.addOption({"risk-refresh-disable",
                      tr("Wyłącz automatyczne pobieranie stanu ryzyka")});
    parser.addOption({"risk-history-export-dir",
                      tr("Katalog eksportu historii ryzyka"), tr("path"), QString()});
    parser.addOption({"risk-history-export-limit",
                      tr("Limit próbek eksportowanych do CSV"), tr("count"), QString()});
    parser.addOption({"risk-history-export-limit-disable",
                      tr("Wyłącza limit próbek eksportu historii ryzyka")});
    parser.addOption({"risk-history-auto-export",
                      tr("Włącza automatyczny eksport historii ryzyka")});
    parser.addOption({"risk-history-auto-export-disable",
                      tr("Wyłącza automatyczny eksport historii ryzyka")});
    parser.addOption({"risk-history-auto-export-interval",
                      tr("Interwał autoeksportu historii ryzyka (min)"), tr("minutes"), QString()});
    parser.addOption({"risk-history-auto-export-basename",
                      tr("Prefiks plików autoeksportu historii ryzyka"), tr("name"), QString()});
    parser.addOption({"risk-history-auto-export-local-time",
                      tr("Nazwy plików autoeksportu używają czasu lokalnego")});
    parser.addOption({"risk-history-auto-export-utc",
                      tr("Nazwy plików autoeksportu używają czasu UTC")});
    parser.addOption({"risk-history-auto-export-dir",
                      tr("Katalog docelowy automatycznego eksportu historii ryzyka"), tr("path"), QString()});
    parser.addOption({"decision-log",
                      tr("Ścieżka do decision logu (plik JSONL lub katalog)"), tr("path"), QString()});
    parser.addOption({"decision-log-limit",
                      tr("Limit liczby przechowywanych wpisów decision logu"), tr("count"), QString()});
    parser.addOption({"ui-module-dir",
                      tr("Katalog z pluginami UI (można powtórzyć)"), tr("path"), QString()});
    parser.addOption({"ui-settings-path", tr("Ścieżka pliku ustawień UI"), tr("path"), QString()});
    parser.addOption({"disable-ui-settings", tr("Wyłącza zapisywanie konfiguracji UI")});
    parser.addOption({"enable-ui-settings",
                      tr("Wymusza zapisywanie konfiguracji UI nawet przy dezaktywacji w zmiennych środowiskowych")});
    parser.addOption({"marketplace-bridge", tr("Ścieżka mostka marketplace (ui_marketplace_bridge.py)"), tr("path"), QString()});
    parser.addOption({"marketplace-presets-dir", tr("Katalog presetów marketplace"), tr("path"), QString()});
    parser.addOption({"marketplace-licenses-path", tr("Plik stanu licencji marketplace"), tr("path"), QString()});
    parser.addOption({"marketplace-signing-key", tr("Klucz podpisu presetów marketplace (KEY_ID=SECRET)"), tr("key"), QString()});
    parser.addOption({"marketplace-signing-key-file", tr("Plik JSON z kluczami podpisów marketplace"), tr("path"), QString()});
    parser.addOption({"marketplace-fingerprint", tr("Nadpisanie fingerprintu dla licencji marketplace"), tr("fingerprint"), QString()});
    parser.addOption({"portfolio-bridge", tr("Ścieżka mostka portfelowego (ui_portfolio_bridge.py)"), tr("path"), QString()});
    parser.addOption({"portfolio-store", tr("Plik konfiguracji portfeli multi-account"), tr("path"), QString()});

    parser.addOption({"screen-name", tr("Preferowany ekran (nazwa QScreen)"), tr("name")});
    parser.addOption({"screen-index", tr("Preferowany ekran (indeks)"), tr("index")});
    parser.addOption({"primary-screen", tr("Wymusza użycie ekranu podstawowego")});

    // Telemetria MetricsService
    parser.addOption({"metrics-endpoint", tr("Adres serwera MetricsService"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"metrics-tag", tr("Etykieta notatek telemetrii"), tr("tag"), QString()});
    parser.addOption({"metrics-auth-token", tr("Token autoryzacyjny MetricsService"), tr("token")});
    parser.addOption({"metrics-auth-token-file", tr("Ścieżka pliku z tokenem MetricsService"), tr("path")});
    parser.addOption({"metrics-rbac-role", tr("Rola RBAC przekazywana do MetricsService"), tr("role"), QString()});
    parser.addOption({"disable-metrics", tr("Wyłącza wysyłkę telemetrii")});
    parser.addOption({"no-metrics", tr("Alias: wyłącza wysyłkę telemetrii")});

    // Lokalny stub bot_core (gRPC)
    parser.addOption({"local-core", tr("Uruchamia lokalny stub API bot_core")});
    parser.addOption({"no-local-core", tr("Wyłącza lokalny stub API bot_core")});
    parser.addOption({"local-core-host", tr("Adres nasłuchu stubu"), tr("host"), QStringLiteral("127.0.0.1")});
    parser.addOption({"local-core-port", tr("Port stubu (0 = przydziel losowo)"), tr("port"), QStringLiteral("0")});
    parser.addOption({"local-core-python", tr("Interpreter Pythona stubu bot_core"), tr("path"), QString()});
    parser.addOption({"local-core-dataset", tr("Plik YAML z danymi stubu"), tr("path"), QString()});
    parser.addOption({"local-core-stream-repeat", tr("Powtarza strumień OHLCV stubu")});
    parser.addOption({"local-core-stream-interval", tr("Opóźnienie strumienia (s)"), tr("seconds"), QStringLiteral("0")});

    // Tryb offline (REST)
    parser.addOption({"offline-mode", tr("Uruchamia UI w trybie offline z lokalnym daemonem REST")});
    parser.addOption({"offline-endpoint", tr("Adres URL lokalnego daemona offline"), tr("url"),
                      QStringLiteral("http://127.0.0.1:58081")});
    parser.addOption({"offline-strategy-config", tr("Plik JSON z konfiguracją strategii offline"), tr("path"), QString()});
    parser.addOption({"offline-auto-run", tr("Automatycznie uruchamia tryb auto-run po starcie")});

    // TLS/mTLS gRPC (demon tradingowy)
    parser.addOption({"grpc-use-mtls", tr("Wymusza mTLS dla klienta tradingowego")});
    parser.addOption({"grpc-root-cert", tr("Root CA (PEM) dla kanału tradingowego"), tr("path"), QString()});
    parser.addOption({"grpc-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-auth-token", tr("Token autoryzacyjny TradingService"), tr("token")});
    parser.addOption({"grpc-auth-token-file", tr("Ścieżka pliku z tokenem TradingService"), tr("path")});
    parser.addOption({"grpc-rbac-role", tr("Rola RBAC przekazywana do TradingService"), tr("role"), QString()});
    parser.addOption({"grpc-rbac-scopes", tr("Lista scope RBAC (oddzielone przecinkami)") , tr("scopes"), QString()});
    parser.addOption({"grpc-target-name", tr("Override nazwy hosta TLS"), tr("name"), QString()});

    // TLS/mTLS dla TradingClient (ogólne przełączniki – mogą być nadpisane przez --grpc-*)
    parser.addOption({"use-tls", tr("Wymusza połączenie TLS z TradingService")});
    parser.addOption({"tls-root-cert", tr("Plik root CA (PEM) dla TradingService"), tr("path"), QString()});
    parser.addOption({"tls-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"tls-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"tls-server-name", tr("Override nazwy serwera TLS"), tr("name"), QString()});
    parser.addOption({"tls-pinned-sha256", tr("Oczekiwany fingerprint SHA-256 certyfikatu"), tr("hex"), QString()});
    parser.addOption({"tls-require-client-auth", tr("Wymaga dostarczenia certyfikatu klienta (mTLS)")});

    // TLS/mTLS dla MetricsService (opcjonalnie)
    parser.addOption({"metrics-use-tls", tr("Wymusza połączenie TLS z MetricsService")});
    parser.addOption({"metrics-root-cert", tr("Plik root CA (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-server-name", tr("Override nazwy serwera TLS"), tr("name"), QString()});
    parser.addOption({"metrics-server-sha256", tr("Oczekiwany odcisk SHA-256 certyfikatu serwera"), tr("hex"),
                      QString()});

    // HealthService
    parser.addOption({"health-endpoint", tr("Adres serwera HealthService"), tr("endpoint"),
                      QString()});
    parser.addOption({"health-auth-token", tr("Token autoryzacyjny HealthService"), tr("token")});
    parser.addOption({"health-auth-token-file", tr("Plik z tokenem HealthService"), tr("path")});
    parser.addOption({"health-rbac-role", tr("Rola RBAC HealthService"), tr("role"), QString()});
    parser.addOption({"health-rbac-scopes", tr("Scope RBAC HealthService"), tr("scopes"), QString()});
    parser.addOption({"health-use-tls", tr("Wymusza połączenie TLS z HealthService")});
    parser.addOption({"health-disable-tls", tr("Wyłącza TLS dla HealthService i dziedziczenie po TradingService")});
    parser.addOption({"health-tls-root-cert", tr("Plik root CA (PEM) dla HealthService"), tr("path"), QString()});
    parser.addOption({"health-tls-client-cert", tr("Certyfikat klienta (PEM) dla HealthService"), tr("path"), QString()});
    parser.addOption({"health-tls-client-key", tr("Klucz klienta (PEM) dla HealthService"), tr("path"), QString()});
    parser.addOption({"health-tls-server-name", tr("Override nazwy serwera TLS HealthService"), tr("name"), QString()});
    parser.addOption({"health-tls-target-name", tr("Override nazwy celu TLS HealthService"), tr("name"), QString()});
    parser.addOption({"health-tls-pinned-sha256", tr("Oczekiwany fingerprint SHA-256 certyfikatu HealthService"),
                      tr("hex"), QString()});
    parser.addOption({"health-tls-require-client-auth", tr("Wymaga certyfikatu klienta (mTLS) dla HealthService")});
    parser.addOption({"health-refresh-interval", tr("Interwał odświeżania HealthService (s)"), tr("seconds"),
                      QStringLiteral("60")});
    parser.addOption({"health-disable-auto-refresh", tr("Wyłącza automatyczne odświeżanie HealthService")});

    // Licencje OEM
    parser.addOption({"license-storage", tr("Ścieżka zapisu aktywowanej licencji OEM"), tr("path"),
                      QStringLiteral("var/licenses/active/license.json")});
    parser.addOption({"expected-fingerprint-path", tr("Ścieżka oczekiwanego fingerprintu OEM (JSON/tekst)"), tr("path"),
                      QString()});
    parser.addOption({"security-profiles-path", tr("Ścieżka profili użytkowników UI"), tr("path"), QString()});
    parser.addOption({"security-python", tr("Ścieżka do interpretera Pythona dla bridge"), tr("path"), QString()});
    parser.addOption({"security-log-path", tr("Plik logu zdarzeń administracyjnych"), tr("path"), QString()});
    parser.addOption({"reports-directory", tr("Katalog raportów pipeline"), tr("path"),
                      QStringLiteral("var/reports")});
    parser.addOption({"reporting-python", tr("Interpreter Pythona mostka raportów"), tr("path"), QString()});
    parser.addOption({"core-config", tr("Plik głównej konfiguracji core.yaml"), tr("path"),
                      QStringLiteral("config/core.yaml")});
    parser.addOption({"strategy-config-python", tr("Interpreter Pythona mostka konfiguracji strategii"), tr("path"),
                      QString()});
    parser.addOption({"strategy-config-bridge", tr("Ścieżka do scripts/ui_config_bridge.py"), tr("path"), QString()});
    parser.addOption({"support-bundle-python", tr("Interpreter Pythona eksportu pakietu wsparcia"), tr("path"), QString()});
    parser.addOption({"support-bundle-script", tr("Ścieżka do scripts/export_support_bundle.py"), tr("path"), QString()});
    parser.addOption({"support-bundle-output-dir", tr("Katalog docelowy pakietów wsparcia"), tr("path"), QString()});
    parser.addOption({"support-bundle-format", tr("Format pakietu wsparcia (tar.gz lub zip)"), tr("format"),
                      QStringLiteral("tar.gz")});
    parser.addOption({"support-bundle-basename", tr("Bazowa nazwa pliku pakietu wsparcia"), tr("name"),
                      QStringLiteral("support-bundle")});
    parser.addOption({"support-bundle-include", tr("Dodatkowa ścieżka pakietu wsparcia (label=path)"), tr("spec")});
    parser.addOption({"support-bundle-disable", tr("Wyłącz domyślny zasób pakietu wsparcia (np. logs)"), tr("label")});
    parser.addOption({"support-bundle-metadata", tr("Para metadata key=value dla pakietu wsparcia"), tr("pair")});
}

bool Application::applyParser(const QCommandLineParser& parser) {
    TradingClient::InstrumentConfig instrument;
    instrument.exchange = parser.value("exchange");
    instrument.symbol = parser.value("symbol");
    instrument.venueSymbol = parser.value("venue-symbol");
    instrument.quoteCurrency = parser.value("quote");
    instrument.baseCurrency = parser.value("base");
    instrument.granularityIso8601 = parser.value("granularity");

    m_offlineMode = parser.isSet("offline-mode");
    m_offlineEndpoint = parser.value("offline-endpoint").trimmed();
    if (m_offlineEndpoint.trimmed().isEmpty()) {
        m_offlineEndpoint = m_offlineMode ? QStringLiteral("inprocess://offline")
                                          : QStringLiteral("http://127.0.0.1:58081");
    }
    m_offlineAutoRun = parser.isSet("offline-auto-run");
    m_offlineStrategyConfig.clear();
    m_offlineAutomationRunning = false;
    m_offlineStatus = tr("Offline runtime: nieaktywny");
    const QString offlineConfigPath = parser.value("offline-strategy-config").trimmed();
    m_offlineStrategyPath.clear();
    if (!offlineConfigPath.isEmpty()) {
        const QString expanded = expandPath(offlineConfigPath);
        QFile configFile(expanded);
        if (!configFile.exists()) {
            qCWarning(lcAppMetrics)
                << "Plik konfiguracji offline nie istnieje:" << expanded;
        } else if (!configFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            qCWarning(lcAppMetrics)
                << "Nie udało się otworzyć konfiguracji offline" << expanded << configFile.errorString();
        } else {
            QJsonParseError error{};
            const QJsonDocument doc = QJsonDocument::fromJson(configFile.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !doc.isObject()) {
                qCWarning(lcAppMetrics)
                    << "Niepoprawny JSON konfiguracji offline" << expanded << error.errorString();
            } else {
                m_offlineStrategyConfig = doc.object().toVariantMap();
                m_offlineStrategyPath = expanded;
            }
        }
    }

    Q_EMIT offlineStrategyPathChanged();

    auto parseTransportMode = [](const QString& raw) -> std::optional<TradingClient::TransportMode> {
        const QString normalized = raw.trimmed().toLower();
        if (normalized.isEmpty() || normalized == QStringLiteral("grpc"))
            return TradingClient::TransportMode::Grpc;
        if (normalized == QStringLiteral("in-process") || normalized == QStringLiteral("inprocess")
            || normalized == QStringLiteral("local"))
            return TradingClient::TransportMode::InProcess;
        return std::nullopt;
    };

    QString transportModeRaw = parser.value("transport-mode").trimmed();
    if (transportModeRaw.isEmpty()) {
        if (const auto envMode = envValue(kTransportModeEnv); envMode.has_value())
            transportModeRaw = envMode->trimmed();
    }

    if (const auto modeOpt = parseTransportMode(transportModeRaw); modeOpt.has_value()) {
        m_transportMode = modeOpt.value();
    } else if (!transportModeRaw.isEmpty()) {
        qCWarning(lcAppMetrics) << "Nieznany tryb transportu" << transportModeRaw
                                << "– używam domyślnego 'grpc'.";
        m_transportMode = TradingClient::TransportMode::Grpc;
    } else {
        m_transportMode = TradingClient::TransportMode::Grpc;
    }
    m_client.setTransportMode(m_transportMode);

    QString datasetPath = parser.value("transport-dataset").trimmed();
    if (datasetPath.isEmpty()) {
        const QString localDataset = parser.value("local-core-dataset").trimmed();
        if (!localDataset.isEmpty())
            datasetPath = localDataset;
    }
    if (datasetPath.isEmpty()) {
        if (const auto envDataset = envValue(kTransportDatasetEnv); envDataset.has_value())
            datasetPath = envDataset->trimmed();
    }
    if (!datasetPath.isEmpty())
        datasetPath = expandPath(datasetPath);
    m_inProcessDatasetPath = datasetPath;
    m_client.setInProcessDatasetPath(m_inProcessDatasetPath);

    QString endpoint = parser.value("endpoint");
    if (m_transportMode == TradingClient::TransportMode::Grpc) {
        configureLocalBotCoreService(parser, endpoint);
    } else {
        m_localServiceEnabled = false;
        endpoint.clear();
    }
    m_client.setEndpoint(endpoint);
    m_client.setInstrument(instrument);
    m_instrument = instrument;
    Q_EMIT instrumentChanged();

    configureRegimeThresholds(parser);

    // Ogólny TLS (może być nadpisany przez sekcję gRPC)
    TradingClient::TlsConfig tlsConfig;
    if (m_transportMode == TradingClient::TransportMode::Grpc) {
        tlsConfig.enabled = parser.isSet("use-tls");
        tlsConfig.rootCertificatePath = parser.value("tls-root-cert");
        tlsConfig.clientCertificatePath = parser.value("tls-client-cert");
        tlsConfig.clientKeyPath = parser.value("tls-client-key");
        tlsConfig.serverNameOverride = parser.value("tls-server-name");
        tlsConfig.pinnedServerFingerprint = parser.value("tls-pinned-sha256");
        tlsConfig.requireClientAuth = parser.isSet("tls-require-client-auth");
    } else {
        tlsConfig.enabled = false;
    }
    m_client.setTlsConfig(tlsConfig);

    const int historyLimit = parser.value("history-limit").toInt();
    m_client.setHistoryLimit(historyLimit);
    if (historyLimit > 0)
        m_historyLimit = historyLimit;

    const QString regimeLimitRaw = parser.value("regime-timeline-limit").trimmed();
    if (!regimeLimitRaw.isEmpty()) {
        bool ok = false;
        const int regimeLimit = regimeLimitRaw.toInt(&ok);
        if (!ok || regimeLimit < 0) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy limit próbek reżimu rynku podany w CLI:" << regimeLimitRaw;
        } else {
            setRegimeTimelineMaximumSnapshots(regimeLimit);
        }
    } else if (const auto regimeEnv = envValue(kRegimeTimelineLimitEnv); regimeEnv.has_value()) {
        const QString trimmed = regimeEnv->trimmed();
        if (!trimmed.isEmpty()) {
            bool ok = false;
            const int envLimit = trimmed.toInt(&ok);
            if (!ok || envLimit < 0) {
                qCWarning(lcAppMetrics)
                    << "Nieprawidłowy limit próbek reżimu rynku podany w zmiennej"
                    << QString::fromUtf8(kRegimeTimelineLimitEnv) << ':' << trimmed;
            } else {
                setRegimeTimelineMaximumSnapshots(envLimit);
            }
        }
    }

    m_maxSamples = parser.value("max-samples").toInt();
    if (m_maxSamples <= 0)
        m_maxSamples = 10240;

    PerformanceGuard guard;
    guard.fpsTarget = parser.value("fps-target").toInt();
    guard.reduceMotionAfterSeconds = parser.value("reduce-motion-after").toDouble();
    guard.jankThresholdMs = parser.value("jank-threshold-ms").toDouble();
    guard.maxOverlayCount = parser.value("max-overlay-count").toInt();
    guard.disableSecondaryWhenFpsBelow = parser.value("overlay-disable-secondary-fps").toInt();

    m_client.setPerformanceGuard(guard);
    m_guard = guard;
    Q_EMIT performanceGuardChanged();

    if (m_frameMonitor) {
        m_frameMonitor->setPerformanceGuard(m_guard);
    }

    m_forcePrimaryScreen = false;
    m_preferredScreenName.clear();
    m_preferredScreenIndex = -1;
    m_screenWarningLogged = false;

    m_forcePrimaryScreen = parser.isSet("primary-screen");
    m_preferredScreenName = parser.value("screen-name").trimmed();
    if (!parser.value("screen-index").trimmed().isEmpty()) {
        bool ok = false;
        const int index = parser.value("screen-index").toInt(&ok);
        if (ok && index >= 0) {
            m_preferredScreenIndex = index;
        } else if (parser.isSet("screen-index")) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy indeks ekranu" << parser.value("screen-index")
                << "– oczekiwano liczby całkowitej >= 0.";
            m_preferredScreenIndex = -1;
        }
    } else if (parser.isSet("screen-index")) {
        m_preferredScreenIndex = -1;
    }

    applyScreenEnvironmentOverrides(parser);
    m_preferredScreenConfigured = m_forcePrimaryScreen || m_preferredScreenIndex >= 0
        || !m_preferredScreenName.isEmpty();

    QString tradingAuthToken;
    QString tradingAuthTokenFile;
    QString metricsAuthToken;
    QString metricsAuthTokenFile;
    QString healthAuthToken;
    QString healthAuthTokenFile;

    if (m_transportMode == TradingClient::TransportMode::Grpc) {
        TradingClient::TlsConfig tradingTls;
        tradingTls.enabled = parser.isSet("grpc-use-mtls");
        const QString cliRootCert = parser.value("grpc-root-cert").trimmed();
        if (!cliRootCert.isEmpty())
            tradingTls.rootCertificatePath = expandPath(cliRootCert);
        const QString cliClientCert = parser.value("grpc-client-cert").trimmed();
        if (!cliClientCert.isEmpty())
            tradingTls.clientCertificatePath = expandPath(cliClientCert);
        const QString cliClientKey = parser.value("grpc-client-key").trimmed();
        if (!cliClientKey.isEmpty())
            tradingTls.clientKeyPath = expandPath(cliClientKey);
        tradingTls.targetNameOverride = parser.value("grpc-target-name");
        if (m_localServiceEnabled) {
            tradingTls.enabled = false;
            tradingTls.rootCertificatePath.clear();
            tradingTls.clientCertificatePath.clear();
            tradingTls.clientKeyPath.clear();
            tradingTls.targetNameOverride.clear();
        }
        m_tradingTlsConfig = tradingTls;
        m_healthTlsConfig = m_tradingTlsConfig;

        const bool cliHealthTlsEnable = parser.isSet("health-use-tls");
        const bool cliHealthTlsDisable = parser.isSet("health-disable-tls");
        if (cliHealthTlsEnable && cliHealthTlsDisable) {
            qCWarning(lcAppMetrics)
                << "Podano jednocześnie --health-use-tls oraz --health-disable-tls. Priorytet ma wyłączenie TLS.";
        }
        if (cliHealthTlsEnable)
            m_healthTlsConfig.enabled = true;
        if (cliHealthTlsDisable)
            m_healthTlsConfig.enabled = false;

        if (parser.isSet("health-tls-require-client-auth"))
            m_healthTlsConfig.requireClientAuth = true;
        if (parser.isSet("health-tls-root-cert")) {
            const QString value = parser.value("health-tls-root-cert").trimmed();
            m_healthTlsConfig.rootCertificatePath = value.isEmpty() ? QString() : expandPath(value);
        }
        if (parser.isSet("health-tls-client-cert")) {
            const QString value = parser.value("health-tls-client-cert").trimmed();
            m_healthTlsConfig.clientCertificatePath = value.isEmpty() ? QString() : expandPath(value);
        }
        if (parser.isSet("health-tls-client-key")) {
            const QString value = parser.value("health-tls-client-key").trimmed();
            m_healthTlsConfig.clientKeyPath = value.isEmpty() ? QString() : expandPath(value);
        }
        if (parser.isSet("health-tls-server-name"))
            m_healthTlsConfig.serverNameOverride = parser.value("health-tls-server-name").trimmed();
        if (parser.isSet("health-tls-target-name"))
            m_healthTlsConfig.targetNameOverride = parser.value("health-tls-target-name").trimmed();
        if (parser.isSet("health-tls-pinned-sha256"))
            m_healthTlsConfig.pinnedServerFingerprint = parser.value("health-tls-pinned-sha256").trimmed();

        const QString cliTradingToken = parser.value("grpc-auth-token").trimmed();
        const QString cliTradingTokenFile = parser.value("grpc-auth-token-file").trimmed();
        const bool cliTradingTokenProvided = !cliTradingToken.isEmpty();
        const bool cliTradingTokenFileProvided = !cliTradingTokenFile.isEmpty();
        if (cliTradingTokenProvided && cliTradingTokenFileProvided) {
            qCWarning(lcAppMetrics)
                << "Podano jednocześnie --grpc-auth-token oraz --grpc-auth-token-file. Użyję tokenu przekazanego bezpośrednio.";
        }
        if (cliTradingTokenProvided) {
            tradingAuthToken = cliTradingToken;
        } else if (cliTradingTokenFileProvided) {
            tradingAuthTokenFile = expandPath(cliTradingTokenFile);
            tradingAuthToken = readTokenFile(tradingAuthTokenFile, QStringLiteral("TradingService"));
        }

        const QString cliTradingRole = parser.value("grpc-rbac-role").trimmed();
        if (!cliTradingRole.isEmpty())
            m_tradingRbacRole = cliTradingRole;
        else
            m_tradingRbacRole.clear();

        const QString cliTradingScopesRaw = parser.value("grpc-rbac-scopes").trimmed();
        if (!cliTradingScopesRaw.isEmpty())
            m_tradingRbacScopes = splitScopesList(cliTradingScopesRaw);
        else
            m_tradingRbacScopes.clear();

        QString metricsEndpointCli = parser.value("metrics-endpoint");
        if (metricsEndpointCli.trimmed().isEmpty())
            metricsEndpointCli = endpoint;
        applyMetricsEndpoint(metricsEndpointCli);
        m_metricsTag = parser.value("metrics-tag");
        m_metricsEnabled = !(parser.isSet("disable-metrics") || parser.isSet("no-metrics"));

        const QString cliMetricsToken = parser.value("metrics-auth-token").trimmed();
        const QString cliMetricsTokenFile = parser.value("metrics-auth-token-file").trimmed();
        const bool cliMetricsTokenProvided = !cliMetricsToken.isEmpty();
        const bool cliMetricsTokenFileProvided = !cliMetricsTokenFile.isEmpty();
        if (cliMetricsTokenProvided && cliMetricsTokenFileProvided) {
            qCWarning(lcAppMetrics)
                << "Podano jednocześnie --metrics-auth-token oraz --metrics-auth-token-file. Użyję tokenu przekazanego bezpośrednio.";
        }
        if (cliMetricsTokenProvided) {
            metricsAuthToken = cliMetricsToken;
        } else if (cliMetricsTokenFileProvided) {
            metricsAuthTokenFile = expandPath(cliMetricsTokenFile);
            metricsAuthToken = readTokenFile(metricsAuthTokenFile);
        }
        m_metricsRbacRole = parser.value("metrics-rbac-role").trimmed();

        const QString cliHealthEndpoint = parser.value("health-endpoint").trimmed();
        const bool cliHealthEndpointProvided = !cliHealthEndpoint.isEmpty();
        if (cliHealthEndpointProvided)
            m_healthEndpoint = cliHealthEndpoint;
        else
            m_healthEndpoint.clear();

        const QString cliHealthToken = parser.value("health-auth-token").trimmed();
        const QString cliHealthTokenFile = parser.value("health-auth-token-file").trimmed();
        const bool cliHealthTokenProvided = !cliHealthToken.isEmpty();
        const bool cliHealthTokenFileProvided = !cliHealthTokenFile.isEmpty();
        if (cliHealthTokenProvided && cliHealthTokenFileProvided) {
            qCWarning(lcAppMetrics)
                << "Podano jednocześnie --health-auth-token oraz --health-auth-token-file. Użyję tokenu przekazanego bezpośrednio.";
        }
        if (cliHealthTokenProvided) {
            healthAuthToken = cliHealthToken;
        } else if (cliHealthTokenFileProvided) {
            healthAuthTokenFile = expandPath(cliHealthTokenFile);
            healthAuthToken = readTokenFile(healthAuthTokenFile, QStringLiteral("HealthService"));
        }

        const QString cliHealthRole = parser.value("health-rbac-role").trimmed();
        if (!cliHealthRole.isEmpty())
            m_healthRbacRole = cliHealthRole;
        else
            m_healthRbacRole.clear();

        const QString cliHealthScopesRaw = parser.value("health-rbac-scopes").trimmed();
        if (!cliHealthScopesRaw.isEmpty())
            m_healthRbacScopes = splitScopesList(cliHealthScopesRaw);
        else
            m_healthRbacScopes.clear();

        applyTradingTlsEnvironmentOverrides(parser);
        configureTradingTlsWatchers();
        applyTradingAuthEnvironmentOverrides(parser,
                                             cliTradingTokenProvided,
                                             cliTradingTokenFileProvided,
                                             !m_tradingRbacRole.isEmpty(),
                                             !m_tradingRbacScopes.isEmpty(),
                                             tradingAuthToken,
                                             tradingAuthTokenFile);
        applyHealthEnvironmentOverrides(parser,
                                        cliHealthEndpointProvided,
                                        cliHealthTokenProvided,
                                        cliHealthTokenFileProvided,
                                        !m_healthRbacRole.isEmpty(),
                                        !m_healthRbacScopes.isEmpty(),
                                        parser.isSet("health-refresh-interval"),
                                        cliHealthTlsEnable,
                                        cliHealthTlsDisable,
                                        parser.isSet("health-tls-require-client-auth"),
                                        parser.isSet("health-tls-root-cert"),
                                        parser.isSet("health-tls-client-cert"),
                                        parser.isSet("health-tls-client-key"),
                                        parser.isSet("health-tls-server-name"),
                                        parser.isSet("health-tls-target-name"),
                                        parser.isSet("health-tls-pinned-sha256"),
                                        healthAuthToken,
                                        healthAuthTokenFile);
        configureHealthTlsWatchers();
    } else {
        m_tradingTlsConfig = TradingClient::TlsConfig{};
        m_healthTlsConfig = GrpcTlsConfig{};
        m_tradingRbacRole.clear();
        m_tradingRbacScopes.clear();
        QString metricsEndpointCli = parser.value("metrics-endpoint");
        if (metricsEndpointCli.trimmed().isEmpty())
            metricsEndpointCli = QStringLiteral("in-process");
        applyMetricsEndpoint(metricsEndpointCli);
        m_metricsTag = parser.value("metrics-tag");
        m_metricsEnabled = !(parser.isSet("disable-metrics") || parser.isSet("no-metrics"));
        m_metricsRbacRole.clear();
        metricsAuthToken.clear();
        metricsAuthTokenFile.clear();
        m_healthEndpoint = QStringLiteral("in-process");
        m_healthRbacRole.clear();
        m_healthRbacScopes.clear();
        healthAuthToken.clear();
        healthAuthTokenFile.clear();
    }

    m_tradingAuthToken = tradingAuthToken.trimmed();
    setTradingAuthTokenFile(tradingAuthTokenFile);
    applyTradingTlsConfig();
    m_client.setAuthToken(m_tradingAuthToken);
    m_client.setRbacRole(m_tradingRbacRole);
    m_client.setRbacScopes(m_tradingRbacScopes);

    m_healthAuthToken = healthAuthToken.trimmed();
    setHealthAuthTokenFile(healthAuthTokenFile);

    if (m_healthController) {
        if (m_transportMode == TradingClient::TransportMode::InProcess) {
            if (!m_inProcessHealthClient)
                m_inProcessHealthClient = std::make_shared<InProcessHealthClient>();
            m_healthController->setHealthClientForTesting(m_inProcessHealthClient);
        }
        const QString endpointForHealth = !m_healthEndpoint.isEmpty() ? m_healthEndpoint : endpoint;
        m_healthController->setEndpoint(endpointForHealth);
        applyHealthTlsConfig();

        QString healthTokenEffective = m_healthAuthToken.trimmed();
        if (healthTokenEffective.isEmpty()) {
            healthTokenEffective = m_tradingAuthToken;
        }
        m_healthController->setAuthToken(healthTokenEffective.trimmed());

        QString healthRoleEffective = m_healthRbacRole.trimmed();
        if (healthRoleEffective.isEmpty()) {
            healthRoleEffective = m_tradingRbacRole;
        }
        m_healthController->setRbacRole(healthRoleEffective.trimmed());

        QStringList healthScopesEffective = m_healthRbacScopes;
        if (healthScopesEffective.isEmpty()) {
            healthScopesEffective = m_tradingRbacScopes;
        }
        if (healthScopesEffective.isEmpty()) {
            healthScopesEffective = QStringList{QStringLiteral("health.read")};
        }
        m_healthController->setRbacScopes(healthScopesEffective);
        m_healthController->setRefreshIntervalSeconds(m_healthRefreshIntervalSeconds);
        m_healthController->setAutoRefreshEnabled(m_healthAutoRefreshEnabled);
    }
    m_client.setTlsConfig(m_tradingTlsConfig);
    m_client.setAuthToken(m_tradingAuthToken);
    m_client.setRbacRole(m_tradingRbacRole);
    m_client.setRbacScopes(m_tradingRbacScopes);

    bool riskRefreshEnabled = !parser.isSet("risk-refresh-disable");
    bool intervalOk = false;
    double riskRefreshSeconds = parser.value("risk-refresh-interval").toDouble(&intervalOk);
    if (!intervalOk || riskRefreshSeconds <= 0.0) {
        if (parser.isSet("risk-refresh-interval")) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy interwał --risk-refresh-interval:" << parser.value("risk-refresh-interval")
                << "– używam wartości domyślnej";
        }
        riskRefreshSeconds = kDefaultRiskRefreshSeconds;
    }

    if (!parser.isSet("risk-refresh-interval")) {
        if (const auto envInterval = envValue("BOT_CORE_UI_RISK_REFRESH_SECONDS")) {
            bool envOk = false;
            const double candidate = envInterval->toDouble(&envOk);
            if (envOk && candidate > 0.0) {
                riskRefreshSeconds = candidate;
            } else {
                qCWarning(lcAppMetrics)
                    << "Nieprawidłowa wartość BOT_CORE_UI_RISK_REFRESH_SECONDS:" << *envInterval
                    << "– oczekiwano liczby dodatniej";
            }
        }
    }

    if (!parser.isSet("risk-refresh-disable")) {
        if (const auto envDisable = envBool("BOT_CORE_UI_RISK_REFRESH_DISABLE"))
            riskRefreshEnabled = !envDisable.value();
    }

    configureRiskRefresh(riskRefreshEnabled, riskRefreshSeconds);

    applyRiskHistoryCliOverrides(parser);
    configureStrategyBridge(parser);
    configureSupportBundle(parser);
    configureDecisionLog(parser);
    configureUiModules(parser);
    applyMarketplaceEnvironmentOverrides(parser);
    if (m_marketplaceController) {
        const QString bridgePath = parser.value(QStringLiteral("marketplace-bridge")).trimmed();
        if (!bridgePath.isEmpty())
            m_marketplaceController->setBridgeScriptPath(expandPath(bridgePath));
        const QString presetsDir = parser.value(QStringLiteral("marketplace-presets-dir")).trimmed();
        if (!presetsDir.isEmpty())
            m_marketplaceController->setPresetsDirectory(expandPath(presetsDir));
        const QString licensesPath = parser.value(QStringLiteral("marketplace-licenses-path")).trimmed();
        if (!licensesPath.isEmpty())
            m_marketplaceController->setLicensesPath(expandPath(licensesPath));
        const QStringList cliSigningKeys = parser.values(QStringLiteral("marketplace-signing-key"));
        if (!cliSigningKeys.isEmpty())
            m_marketplaceController->setSigningKeys(cliSigningKeys);
        const QStringList cliSigningKeyFiles = parser.values(QStringLiteral("marketplace-signing-key-file"));
        if (!cliSigningKeyFiles.isEmpty())
            m_marketplaceController->setSigningKeyFiles(cliSigningKeyFiles);
        const QString fingerprintOverride = parser.value(QStringLiteral("marketplace-fingerprint")).trimmed();
        if (!fingerprintOverride.isEmpty())
            m_marketplaceController->setFingerprintOverride(fingerprintOverride);
    }

    if (m_portfolioController) {
        const QString portfolioBridge = parser.value(QStringLiteral("portfolio-bridge")).trimmed();
        if (!portfolioBridge.isEmpty())
            m_portfolioController->setBridgeScriptPath(expandPath(portfolioBridge));
        const QString portfolioStore = parser.value(QStringLiteral("portfolio-store")).trimmed();
        if (!portfolioStore.isEmpty())
            m_portfolioController->setStorePath(expandPath(portfolioStore));
    }

    // TLS config (MetricsService)
    TelemetryTlsConfig mtls;
    mtls.enabled = parser.isSet("metrics-use-tls");
    mtls.rootCertificatePath = parser.value("metrics-root-cert");
    mtls.clientCertificatePath = parser.value("metrics-client-cert");
    mtls.clientKeyPath = parser.value("metrics-client-key");
    mtls.serverNameOverride = parser.value("metrics-server-name");
    mtls.pinnedServerSha256 = parser.value("metrics-server-sha256");
    m_tlsConfig = mtls;

    // Konfiguracja licencji OEM (CLI/ENV)
    const QString cliLicensePath = parser.value("license-storage").trimmed();
    if (!cliLicensePath.isEmpty())
        m_licenseController->setLicenseStoragePath(expandPath(cliLicensePath));
    const QString cliExpectedFingerprint = parser.value("expected-fingerprint-path").trimmed();
    if (!cliExpectedFingerprint.isEmpty())
        m_licenseController->setFingerprintDocumentPath(expandPath(cliExpectedFingerprint));
    m_licenseController->initialize();

    if (m_transportMode == TradingClient::TransportMode::Grpc) {
        applyMetricsEnvironmentOverrides(parser,
                                         cliTokenProvided,
                                         cliTokenFileProvided,
                                         metricsAuthToken,
                                         metricsAuthTokenFile);
        configureMetricsTlsWatchers();
    }
    applyMetricsTlsConfig();
    m_metricsAuthToken = metricsAuthToken.trimmed();
    setMetricsAuthTokenFile(metricsAuthTokenFile);

    if (!validateTransportConfiguration(endpoint,
                                        m_inProcessDatasetPath,
                                        m_tradingTlsConfig,
                                        m_tlsConfig,
                                        m_healthTlsConfig,
                                        m_metricsEndpoint,
                                        m_healthEndpoint)) {
        return false;
    }

    if (m_securityController) {
        if (!parser.value("security-profiles-path").trimmed().isEmpty()) {
            m_securityController->setProfilesPath(expandPath(parser.value("security-profiles-path")));
        }
        if (!parser.value("security-python").trimmed().isEmpty()) {
            m_securityController->setPythonExecutable(parser.value("security-python"));
        }
        if (!parser.value("security-log-path").trimmed().isEmpty()) {
            m_securityController->setLogPath(expandPath(parser.value("security-log-path")));
        }
        m_securityController->refresh();
    }

    if (m_reportController) {
        const auto applyReportsDirectory = [this](const QString& candidate) {
            const QString trimmed = candidate.trimmed();
            if (trimmed.isEmpty())
                return;
            m_reportController->setReportsDirectory(expandPath(trimmed));
        };
        if (parser.isSet("reports-directory")) {
            applyReportsDirectory(parser.value("reports-directory"));
        } else {
            const auto envReports = envValue("BOT_CORE_UI_REPORTS_DIR");
            if (envReports.has_value()) {
                const QString envReportsValue = envReports->trimmed();
                if (!envReportsValue.isEmpty()) {
                    applyReportsDirectory(envReportsValue);
                } else {
                    applyReportsDirectory(parser.value("reports-directory"));
                }
            } else {
                applyReportsDirectory(parser.value("reports-directory"));
            }
        }

        const auto applyPythonExecutable = [this](const QString& candidate) {
            const QString trimmed = candidate.trimmed();
            if (trimmed.isEmpty())
                return;
            m_reportController->setPythonExecutable(expandPath(trimmed));
        };
        if (parser.isSet("reporting-python")) {
            applyPythonExecutable(parser.value("reporting-python"));
        } else {
            const auto envPython = envValue("BOT_CORE_UI_REPORTS_PYTHON");
            if (envPython.has_value()) {
                const QString envPythonValue = envPython->trimmed();
                if (!envPythonValue.isEmpty())
                    applyPythonExecutable(envPythonValue);
            }
        }

        m_reportController->refresh();
    }

    applyUiSettingsCliOverrides(parser);

    // Inicjalizacja/reportera + token
    ensureTelemetry();

    return true;
}

QString Application::locateRepoRoot() const
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists(QStringLiteral("bot_core")) && dir.exists(QStringLiteral("ui")))
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }

    dir = QDir(QDir::currentPath());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists(QStringLiteral("bot_core")) && dir.exists(QStringLiteral("ui")))
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }

    return QDir::currentPath();
}

void Application::configureRiskRefresh(bool enabled, double intervalSeconds)
{
    double sanitizedSeconds = intervalSeconds > 0.0 ? intervalSeconds : kDefaultRiskRefreshSeconds;
    const int rawMs = static_cast<int>(std::llround(sanitizedSeconds * 1000.0));
    const int clampedMs = qBound(kMinRiskRefreshIntervalMs, rawMs, kMaxRiskRefreshIntervalMs);

    m_riskRefreshIntervalMs = clampedMs;
    m_riskRefreshEnabled = enabled && sanitizedSeconds > 0.0;

    m_riskRefreshTimer.stop();
    m_riskRefreshTimer.setInterval(m_riskRefreshIntervalMs);
    if (!m_riskRefreshEnabled)
        m_nextRiskRefreshUtc = {};
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::applyRiskRefreshTimerState()
{
    if (m_riskRefreshEnabled && m_started) {
        const bool needsRestart = !m_riskRefreshTimer.isActive()
            || m_riskRefreshTimer.interval() != m_riskRefreshIntervalMs;
        if (needsRestart)
            m_riskRefreshTimer.start(m_riskRefreshIntervalMs);
        m_nextRiskRefreshUtc = QDateTime::currentDateTimeUtc().addMSecs(m_riskRefreshIntervalMs);
    } else {
        if (m_riskRefreshTimer.isActive())
            m_riskRefreshTimer.stop();
        m_nextRiskRefreshUtc = {};
    }
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::initializeUiSettingsStorage()
{
    if (const auto disableEnv = envBool(kUiSettingsDisableEnv); disableEnv.has_value())
        m_uiSettingsPersistenceEnabled = !disableEnv.value();

    if (m_uiSettingsPath.trimmed().isEmpty()) {
        QString candidate;
        if (const auto envPath = envValue(kUiSettingsEnv); envPath.has_value()) {
            const QString trimmed = envPath->trimmed();
            if (!trimmed.isEmpty())
                candidate = expandPath(trimmed);
        }

        if (candidate.isEmpty())
            candidate = QDir::current().absoluteFilePath(QStringLiteral("config/ui_prefs.json"));

        QFileInfo info(candidate);
        if (!info.isAbsolute())
            candidate = QDir::current().absoluteFilePath(candidate);

        m_uiSettingsPath = candidate;
    } else {
        QFileInfo info(m_uiSettingsPath);
        if (!info.isAbsolute())
            m_uiSettingsPath = QDir::current().absoluteFilePath(m_uiSettingsPath);
    }

    ensureUiSettingsTimerConfigured();

    if (!m_uiSettingsPersistenceEnabled)
        return;

    loadUiSettings();
}

void Application::ensureUiSettingsTimerConfigured()
{
    if (m_uiSettingsTimerConfigured)
        return;

    m_uiSettingsSaveTimer.setParent(this);
    m_uiSettingsSaveTimer.setSingleShot(true);
    m_uiSettingsSaveTimer.setInterval(kUiSettingsDebounceMs);
    connect(&m_uiSettingsSaveTimer, &QTimer::timeout, this, &Application::persistUiSettings);
    m_uiSettingsTimerConfigured = true;
}

void Application::applyUiSettingsCliOverrides(const QCommandLineParser& parser)
{
    if (parser.isSet("enable-ui-settings"))
        setUiSettingsPersistenceEnabled(true);
    if (parser.isSet("disable-ui-settings"))
        setUiSettingsPersistenceEnabled(false);

    const QString cliPath = parser.value("ui-settings-path").trimmed();
    if (!cliPath.isEmpty())
        setUiSettingsPath(cliPath);
}

void Application::applyRiskHistoryCliOverrides(const QCommandLineParser& parser)
{
    const bool previousLoading = m_loadingUiSettings;
    m_loadingUiSettings = true;

    const auto restoreLoadingFlag = qScopeGuard([this, previousLoading]() {
        m_loadingUiSettings = previousLoading;
    });

    const auto applyDirectory = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;

        QUrl url(trimmed);
        if (!url.isValid() || url.scheme().isEmpty()) {
            const QString expanded = expandPath(trimmed);
            const QString absolute = QDir(expanded).absolutePath();
            url = QUrl::fromLocalFile(absolute);
        } else if (url.isLocalFile()) {
            url = QUrl::fromLocalFile(QDir(url.toLocalFile()).absolutePath());
        }

        if (!url.isValid() || (!url.isLocalFile() && !url.scheme().isEmpty())) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy katalog eksportu historii ryzyka:" << raw;
            return;
        }

        setRiskHistoryExportLastDirectory(url);
    };

    const auto applyLimitValue = [this](const QString& raw) -> bool {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return false;

        bool ok = false;
        const int value = trimmed.toInt(&ok);
        if (!ok) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy limit eksportu historii ryzyka:" << raw;
            return false;
        }

        if (value <= 0) {
            setRiskHistoryExportLimitEnabled(false);
            return true;
        }

        setRiskHistoryExportLimitValue(value);
        setRiskHistoryExportLimitEnabled(true);
        return true;
    };

    const auto applyAutoInterval = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;

        bool ok = false;
        int minutes = trimmed.toInt(&ok);
        if (!ok) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy interwał autoeksportu historii ryzyka:" << raw;
            return;
        }

        minutes = qMax(1, minutes);
        setRiskHistoryAutoExportIntervalMinutes(minutes);
    };

    const auto applyBasename = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;
        setRiskHistoryAutoExportBasename(trimmed);
    };

    if (parser.isSet("risk-history-export-dir"))
        applyDirectory(parser.value("risk-history-export-dir"));
    else if (const auto envDir = envValue(kRiskHistoryExportDirEnv); envDir.has_value())
        applyDirectory(envDir->trimmed());

    if (parser.isSet("risk-history-auto-export-dir"))
        applyDirectory(parser.value("risk-history-auto-export-dir"));
    else if (const auto envAutoDir = envValue(kRiskHistoryAutoExportDirEnv); envAutoDir.has_value())
        applyDirectory(envAutoDir->trimmed());

    bool limitValueApplied = false;
    bool limitEnabledForced = false;

    if (parser.isSet("risk-history-export-limit")) {
        limitValueApplied = applyLimitValue(parser.value("risk-history-export-limit"));
        if (limitValueApplied)
            limitEnabledForced = true;
    }

    if (parser.isSet("risk-history-export-limit-disable")) {
        setRiskHistoryExportLimitEnabled(false);
        limitEnabledForced = true;
    }

    if (!limitValueApplied) {
        if (const auto envLimit = envValue(kRiskHistoryExportLimitEnv); envLimit.has_value())
            limitValueApplied = applyLimitValue(envLimit->trimmed());
    }

    if (!limitEnabledForced) {
        if (const auto envLimitEnabled = envBool(kRiskHistoryExportLimitEnabledEnv); envLimitEnabled.has_value()) {
            setRiskHistoryExportLimitEnabled(envLimitEnabled.value());
            limitEnabledForced = true;
        }
    }

    bool autoExportEnabledForced = false;
    if (parser.isSet("risk-history-auto-export")) {
        setRiskHistoryAutoExportEnabled(true);
        autoExportEnabledForced = true;
    }
    if (parser.isSet("risk-history-auto-export-disable")) {
        setRiskHistoryAutoExportEnabled(false);
        autoExportEnabledForced = true;
    }
    if (!autoExportEnabledForced) {
        if (const auto envAutoEnabled = envBool(kRiskHistoryAutoExportEnabledEnv); envAutoEnabled.has_value()) {
            setRiskHistoryAutoExportEnabled(envAutoEnabled.value());
            autoExportEnabledForced = true;
        }
    }

    bool autoExportTimeForced = false;
    if (parser.isSet("risk-history-auto-export-local-time")) {
        setRiskHistoryAutoExportUseLocalTime(true);
        autoExportTimeForced = true;
    }
    if (parser.isSet("risk-history-auto-export-utc")) {
        setRiskHistoryAutoExportUseLocalTime(false);
        autoExportTimeForced = true;
    }
    if (!autoExportTimeForced) {
        if (const auto envLocalTime = envBool(kRiskHistoryAutoExportLocalTimeEnv); envLocalTime.has_value()) {
            setRiskHistoryAutoExportUseLocalTime(envLocalTime.value());
            autoExportTimeForced = true;
        }
    }

    bool intervalApplied = false;
    if (parser.isSet("risk-history-auto-export-interval")) {
        applyAutoInterval(parser.value("risk-history-auto-export-interval"));
        intervalApplied = true;
    }
    if (!intervalApplied) {
        if (const auto envInterval = envValue(kRiskHistoryAutoExportIntervalEnv); envInterval.has_value())
            applyAutoInterval(envInterval->trimmed());
    }

    if (parser.isSet("risk-history-auto-export-basename"))
        applyBasename(parser.value("risk-history-auto-export-basename"));
    else if (const auto envBasename = envValue(kRiskHistoryAutoExportBasenameEnv); envBasename.has_value())
        applyBasename(envBasename->trimmed());
}

void Application::configureStrategyBridge(const QCommandLineParser& parser)
{
    if (!m_strategyController && !m_workbenchController)
        return;

    QString configPath = parser.value("core-config").trimmed();
    if (!parser.isSet("core-config")) {
        if (const auto envConfig = envValue(QByteArrayLiteral("BOT_CORE_UI_CORE_CONFIG_PATH")))
            configPath = envConfig->trimmed();
    }
    if (configPath.isEmpty())
        configPath = QStringLiteral("config/core.yaml");
    const QString normalizedConfigPath = expandPath(configPath);
    if (!normalizedConfigPath.isEmpty()) {
        if (m_strategyController)
            m_strategyController->setConfigPath(normalizedConfigPath);
        if (m_workbenchController)
            m_workbenchController->setConfigPath(normalizedConfigPath);
    }

    QString pythonExec = parser.value("strategy-config-python").trimmed();
    if (pythonExec.isEmpty()) {
        if (const auto envPython = envValue(QByteArrayLiteral("BOT_CORE_UI_STRATEGY_PYTHON")))
            pythonExec = envPython->trimmed();
    }
    if (!pythonExec.isEmpty()) {
        const QString normalizedPython = expandPath(pythonExec);
        if (!normalizedPython.isEmpty()) {
            if (m_strategyController)
                m_strategyController->setPythonExecutable(normalizedPython);
            if (m_workbenchController)
                m_workbenchController->setPythonExecutable(normalizedPython);
        }
    }

    QString bridgePath = parser.value("strategy-config-bridge").trimmed();
    if (bridgePath.isEmpty()) {
        if (const auto envBridge = envValue(QByteArrayLiteral("BOT_CORE_UI_STRATEGY_BRIDGE")))
            bridgePath = envBridge->trimmed();
    }
    if (bridgePath.isEmpty()) {
        if (!m_repoRoot.isEmpty())
            bridgePath = QDir(m_repoRoot).absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py"));
        else
            bridgePath = QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py"));
    }
    const QString normalizedBridge = expandPath(bridgePath);
    if (!normalizedBridge.isEmpty()) {
        if (m_strategyController)
            m_strategyController->setScriptPath(normalizedBridge);
        if (m_workbenchController)
            m_workbenchController->setScriptPath(normalizedBridge);
    }

    if (m_strategyController && !m_strategyController->refresh()) {
        const QString error = m_strategyController->lastError();
        if (!error.isEmpty())
            qCWarning(lcAppMetrics) << "Mostek konfiguracji strategii zwrócił błąd:" << error;
    }

    if (m_workbenchController && !m_workbenchController->refreshCatalog()) {
        const QString error = m_workbenchController->lastError();
        if (!error.isEmpty())
            qCWarning(lcAppMetrics) << "Mostek katalogu strategii zwrócił błąd:" << error;
    }
}

void Application::configureSupportBundle(const QCommandLineParser& parser)
{
    if (!m_supportController)
        return;

    const auto envTrimmed = [](const QByteArray& key) -> std::optional<QString> {
        if (const auto value = envValue(key); value.has_value())
            return value->trimmed();
        return std::nullopt;
    };

    const auto pickPathOption = [&](const QString& optionName, const QByteArray& envKey) -> QString {
        if (parser.isSet(optionName))
            return parser.value(optionName).trimmed();
        if (const auto envValue = envTrimmed(envKey); envValue.has_value())
            return envValue->trimmed();
        return {};
    };

    const auto normalizePathInput = [](const QString& raw) -> QString {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return {};
        return expandPath(trimmed);
    };

    if (const QString pythonExec = pickPathOption(QStringLiteral("support-bundle-python"),
                                                  QByteArrayLiteral("BOT_CORE_UI_SUPPORT_PYTHON"));
        !pythonExec.isEmpty()) {
        if (const QString normalized = normalizePathInput(pythonExec); !normalized.isEmpty())
            m_supportController->setPythonExecutable(normalized);
    }

    if (const QString scriptPath = pickPathOption(QStringLiteral("support-bundle-script"),
                                                  QByteArrayLiteral("BOT_CORE_UI_SUPPORT_SCRIPT"));
        !scriptPath.isEmpty()) {
        if (const QString normalized = normalizePathInput(scriptPath); !normalized.isEmpty())
            m_supportController->setScriptPath(normalized);
    }

    if (const QString outputDir = pickPathOption(QStringLiteral("support-bundle-output-dir"),
                                                 QByteArrayLiteral("BOT_CORE_UI_SUPPORT_OUTPUT_DIR"));
        !outputDir.isEmpty()) {
        if (const QString normalized = normalizePathInput(outputDir); !normalized.isEmpty())
            m_supportController->setOutputDirectory(normalized);
    }

    if (parser.isSet("support-bundle-format"))
        m_supportController->setFormat(parser.value("support-bundle-format"));
    else if (const auto envFormat = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_FORMAT")); envFormat.has_value())
        m_supportController->setFormat(envFormat.value());

    if (parser.isSet("support-bundle-basename"))
        m_supportController->setDefaultBasename(parser.value("support-bundle-basename"));
    else if (const auto envBasename = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_BASENAME")); envBasename.has_value())
        m_supportController->setDefaultBasename(envBasename.value());

    const auto splitSpecs = [](const QString& raw) {
        return raw.split(QRegularExpression(QStringLiteral("[;,\\n]")), Qt::SkipEmptyParts);
    };

    QStringList includeEnv;
    if (const auto envInclude = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_INCLUDE")); envInclude.has_value())
        includeEnv = splitSpecs(envInclude.value());
    const QStringList includeCli = parser.values(QStringLiteral("support-bundle-include"));

    QStringList disableEnv;
    if (const auto envDisable = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_DISABLE")); envDisable.has_value())
        disableEnv = splitSpecs(envDisable.value());
    const QStringList disableCli = parser.values(QStringLiteral("support-bundle-disable"));

    QStringList metadataEnv;
    if (const auto envMetadata = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_METADATA")); envMetadata.has_value())
        metadataEnv = splitSpecs(envMetadata.value());
    const QStringList metadataCli = parser.values(QStringLiteral("support-bundle-metadata"));

    QStringList extraOrder;
    QHash<QString, QPair<QString, QString>> extraMap;

    const auto storeExtra = [&](const QString& label, const QString& path) {
        const QString lower = label.toLower();
        if (!extraMap.contains(lower))
            extraOrder.append(lower);
        extraMap.insert(lower, qMakePair(label, path));
    };

    const auto handleInclude = [&](const QString& rawSpec) {
        const QString trimmed = rawSpec.trimmed();
        if (trimmed.isEmpty())
            return;
        const int eq = trimmed.indexOf('=');
        if (eq <= 0) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa ścieżka pakietu wsparcia" << trimmed;
            return;
        }
        const QString label = trimmed.left(eq).trimmed();
        const QString path = trimmed.mid(eq + 1).trimmed();
        if (label.isEmpty() || path.isEmpty()) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa ścieżka pakietu wsparcia" << trimmed;
            return;
        }
        const QString lower = label.toLower();
        const QString expandedPath = normalizePathInput(path);
        if (expandedPath.isEmpty()) {
            qCWarning(lcAppMetrics) << "Nie udało się znormalizować ścieżki pakietu wsparcia" << trimmed;
            return;
        }
        if (lower == QStringLiteral("logs")) {
            m_supportController->setLogsPath(expandedPath);
            m_supportController->setIncludeLogs(true);
        } else if (lower == QStringLiteral("reports")) {
            m_supportController->setReportsPath(expandedPath);
            m_supportController->setIncludeReports(true);
        } else if (lower == QStringLiteral("licenses")) {
            m_supportController->setLicensesPath(expandedPath);
            m_supportController->setIncludeLicenses(true);
        } else if (lower == QStringLiteral("metrics")) {
            m_supportController->setMetricsPath(expandedPath);
            m_supportController->setIncludeMetrics(true);
        } else if (lower == QStringLiteral("audit")) {
            m_supportController->setAuditPath(expandedPath);
            m_supportController->setIncludeAudit(true);
        } else {
            storeExtra(label, expandedPath);
        }
    };

    for (const QString& spec : includeEnv)
        handleInclude(spec);
    for (const QString& spec : includeCli)
        handleInclude(spec);

    QSet<QString> disabledCustom;
    const auto handleDisable = [&](const QString& rawLabel) {
        const QString label = rawLabel.trimmed().toLower();
        if (label.isEmpty())
            return;
        if (label == QStringLiteral("logs")) {
            m_supportController->setIncludeLogs(false);
        } else if (label == QStringLiteral("reports")) {
            m_supportController->setIncludeReports(false);
        } else if (label == QStringLiteral("licenses")) {
            m_supportController->setIncludeLicenses(false);
        } else if (label == QStringLiteral("metrics")) {
            m_supportController->setIncludeMetrics(false);
        } else if (label == QStringLiteral("audit")) {
            m_supportController->setIncludeAudit(false);
        } else {
            disabledCustom.insert(label);
        }
    };

    for (const QString& spec : disableEnv)
        handleDisable(spec);
    for (const QString& spec : disableCli)
        handleDisable(spec);

    if (!disabledCustom.isEmpty()) {
        QStringList filteredOrder;
        QHash<QString, QPair<QString, QString>> filteredMap;
        for (const QString& key : std::as_const(extraOrder)) {
            if (disabledCustom.contains(key))
                continue;
            filteredOrder.append(key);
            filteredMap.insert(key, extraMap.value(key));
        }
        extraOrder = filteredOrder;
        extraMap = filteredMap;
    }

    QStringList extraSpecs;
    extraSpecs.reserve(extraOrder.size());
    for (const QString& key : std::as_const(extraOrder)) {
        const auto pair = extraMap.value(key);
        extraSpecs.append(QStringLiteral("%1=%2").arg(pair.first, pair.second));
    }
    m_supportController->setExtraIncludeSpecs(extraSpecs);

    QVariantMap overrides;

    const auto applyMetadata = [&](const QString& rawSpec) {
        const QString trimmed = rawSpec.trimmed();
        if (trimmed.isEmpty())
            return;
        const int eq = trimmed.indexOf('=');
        if (eq <= 0) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa para metadata pakietu wsparcia" << trimmed;
            return;
        }
        const QString key = trimmed.left(eq).trimmed();
        const QString value = trimmed.mid(eq + 1).trimmed();
        if (key.isEmpty())
            return;
        overrides.insert(key, value);
    };

    for (const QString& spec : metadataEnv)
        applyMetadata(spec);
    for (const QString& spec : metadataCli)
        applyMetadata(spec);
    m_supportMetadataOverrides = overrides;
    updateSupportBundleMetadata();
}

void Application::updateSupportBundleMetadata()
{
    if (!m_supportController)
        return;

    QVariantMap metadata;
    metadata.insert(QStringLiteral("origin"), QStringLiteral("desktop_ui"));
    metadata.insert(QStringLiteral("instrument"), instrumentLabel());
    metadata.insert(QStringLiteral("exchange"), m_instrument.exchange);
    metadata.insert(QStringLiteral("symbol"), m_instrument.symbol);
    metadata.insert(QStringLiteral("connection_status"), m_connectionStatus);
    metadata.insert(QStringLiteral("app_version"), QCoreApplication::applicationVersion());
    metadata.insert(QStringLiteral("hostname"), QSysInfo::machineHostName());

    for (auto it = m_supportMetadataOverrides.constBegin(); it != m_supportMetadataOverrides.constEnd(); ++it) {
        const QString key = it.key().trimmed();
        if (key.isEmpty())
            continue;
        metadata.insert(key, it.value());
    }

    m_supportController->setMetadata(metadata);
}

void Application::configureUiModules(const QCommandLineParser& parser)
{
    if (!m_moduleManager)
        return;

    m_moduleManager->unloadPlugins();

    const auto normalize = [](const QString& raw) -> QString {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return {};
        const QString expanded = expandPath(trimmed);
        if (!expanded.isEmpty())
            return QFileInfo(expanded).absoluteFilePath();
        return QFileInfo(trimmed).absoluteFilePath();
    };

    QSet<QString> unique;
    QStringList directories;

    const QStringList cliDirs = parser.values(QStringLiteral("ui-module-dir"));
    for (const QString& value : cliDirs) {
        const QString normalized = normalize(value);
        if (normalized.isEmpty() || unique.contains(normalized))
            continue;
        unique.insert(normalized);
        directories.append(normalized);
    }

    if (cliDirs.isEmpty()) {
        if (const auto envDirs = envValue(QByteArrayLiteral("BOT_CORE_UI_MODULE_DIRS")); envDirs.has_value()) {
            const auto pieces = envDirs->split(QDir::listSeparator(), Qt::SkipEmptyParts);
            for (const QString& piece : pieces) {
                const QString normalized = normalize(piece);
                if (normalized.isEmpty() || unique.contains(normalized))
                    continue;
                unique.insert(normalized);
                directories.append(normalized);
            }
        }
    }

    if (directories.isEmpty()) {
        const QString binaryModules = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("modules"));
        const QString repoModules = QDir::current().absoluteFilePath(QStringLiteral("ui/modules"));
        for (const QString& candidate : {binaryModules, repoModules}) {
            const QString normalized = normalize(candidate);
            if (normalized.isEmpty() || unique.contains(normalized))
                continue;
            unique.insert(normalized);
            directories.append(normalized);
        }
    }

    m_uiModuleDirectories = directories;
    m_moduleManager->setPluginPaths(directories);
    if (!directories.isEmpty()) {
        if (!m_moduleManager->loadPlugins()) {
            qCWarning(lcAppMetrics) << "Nie wszystkie pluginy UI zostały poprawnie załadowane";
        }
    }
}

void Application::configureDecisionLog(const QCommandLineParser& parser)
{
    QString path = parser.value(QStringLiteral("decision-log")).trimmed();
    bool pathExplicit = false;
    if (!path.isEmpty()) {
        setDecisionLogPathInternal(path, true);
        pathExplicit = true;
    } else if (const auto envPath = envValue(kDecisionLogPathEnv); envPath.has_value()) {
        const QString trimmed = envPath->trimmed();
        if (!trimmed.isEmpty()) {
            setDecisionLogPathInternal(trimmed, true);
            pathExplicit = true;
        }
    }

    QString limitText;
    if (parser.isSet(QStringLiteral("decision-log-limit")))
        limitText = parser.value(QStringLiteral("decision-log-limit")).trimmed();
    else if (const auto envLimit = envValue(kDecisionLogLimitEnv); envLimit.has_value())
        limitText = envLimit->trimmed();

    if (!limitText.isEmpty()) {
        bool ok = false;
        const int limit = limitText.toInt(&ok);
        if (ok && limit > 0)
            m_decisionLogModel.setMaximumEntries(limit);
        else if (parser.isSet(QStringLiteral("decision-log-limit")))
            qCWarning(lcAppMetrics) << "Nieprawidłowa wartość --decision-log-limit:" << limitText;
    }

    if (!pathExplicit && m_decisionLogPath.isEmpty()) {
        const QString fallback = !m_repoRoot.isEmpty()
            ? QDir(m_repoRoot).absoluteFilePath(QStringLiteral("logs/decision_journal"))
            : QDir::current().absoluteFilePath(QStringLiteral("logs/decision_journal"));
        setDecisionLogPathInternal(fallback, false);
    }
}

void Application::setUiSettingsPersistenceEnabled(bool enabled)
{
    if (m_uiSettingsPersistenceEnabled == enabled)
        return;

    m_uiSettingsPersistenceEnabled = enabled;
    if (!enabled) {
        if (m_uiSettingsSaveTimer.isActive())
            m_uiSettingsSaveTimer.stop();
        return;
    }

    ensureUiSettingsTimerConfigured();
    loadUiSettings();
}

void Application::setUiSettingsPath(const QString& path, bool reload)
{
    QString candidate = path.trimmed();
    if (candidate.isEmpty())
        return;

    candidate = expandPath(candidate);
    QFileInfo info(candidate);
    if (!info.isAbsolute())
        candidate = QDir::current().absoluteFilePath(candidate);

    if (m_uiSettingsPath == candidate)
        return;

    m_uiSettingsPath = candidate;

    if (m_uiSettingsPersistenceEnabled && reload)
        loadUiSettings();
}

void Application::loadUiSettings()
{
    if (!m_uiSettingsPersistenceEnabled)
        return;
    if (m_uiSettingsPath.trimmed().isEmpty())
        return;

    const QString desiredPath = m_uiSettingsPath;
    QFile file(desiredPath);
    bool loadedFromLegacy = false;
    if (!file.exists()) {
        const QString legacyPath = QDir::current().absoluteFilePath(QStringLiteral("var/state/ui_settings.json"));
        if (!legacyPath.isEmpty() && legacyPath != desiredPath && QFile::exists(legacyPath)) {
            file.setFileName(legacyPath);
            loadedFromLegacy = true;
        } else {
            return;
        }
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać pliku ustawień UI" << m_uiSettingsPath
                                << file.errorString();
        return;
    }

    const QByteArray data = file.readAll();
    file.close();

    const QJsonDocument document = QJsonDocument::fromJson(data);
    if (!document.isObject()) {
        qCWarning(lcAppMetrics) << "Plik ustawień UI ma nieprawidłowy format JSON" << m_uiSettingsPath;
        return;
    }

    const QJsonObject root = document.object();

    const auto requireString = [](const QJsonObject& object, const QString& key) -> QString {
        const QJsonValue value = object.value(key);
        if (!value.isString())
            return {};
        return value.toString().trimmed();
    };

    m_loadingUiSettings = true;

    if (root.contains(QStringLiteral("instrument")) && root.value(QStringLiteral("instrument")).isObject()) {
        const QJsonObject instrument = root.value(QStringLiteral("instrument")).toObject();
        const QString exchange = requireString(instrument, QStringLiteral("exchange"));
        const QString symbol = requireString(instrument, QStringLiteral("symbol"));
        const QString venueSymbol = requireString(instrument, QStringLiteral("venueSymbol"));
        const QString quote = requireString(instrument, QStringLiteral("quoteCurrency"));
        const QString base = requireString(instrument, QStringLiteral("baseCurrency"));
        const QString granularity = requireString(instrument, QStringLiteral("granularity"));

        if (!exchange.isEmpty() && !symbol.isEmpty() && !venueSymbol.isEmpty() && !quote.isEmpty()
            && !base.isEmpty() && !granularity.isEmpty()) {
            updateInstrument(exchange, symbol, venueSymbol, quote, base, granularity);
        }
    }

    if (root.contains(QStringLiteral("performanceGuard"))
        && root.value(QStringLiteral("performanceGuard")).isObject()) {
        const QJsonObject guardObj = root.value(QStringLiteral("performanceGuard")).toObject();
        const int fpsTarget = guardObj.value(QStringLiteral("fpsTarget")).toInt(m_guard.fpsTarget);
        const double reduceMotionAfter = guardObj.value(QStringLiteral("reduceMotionAfter"))
                                             .toDouble(m_guard.reduceMotionAfterSeconds);
        const double jankThreshold = guardObj.value(QStringLiteral("jankThresholdMs"))
                                         .toDouble(m_guard.jankThresholdMs);
        const int overlays = guardObj.value(QStringLiteral("maxOverlayCount")).toInt(m_guard.maxOverlayCount);
        const int disableSecondary = guardObj.value(QStringLiteral("disableSecondaryWhenBelow"))
                                         .toInt(m_guard.disableSecondaryWhenFpsBelow);
        updatePerformanceGuard(fpsTarget, reduceMotionAfter, jankThreshold, overlays, disableSecondary);
    }

    if (root.contains(QStringLiteral("riskRefresh")) && root.value(QStringLiteral("riskRefresh")).isObject()) {
        const QJsonObject riskObj = root.value(QStringLiteral("riskRefresh")).toObject();
        const bool enabled = riskObj.value(QStringLiteral("enabled")).toBool(m_riskRefreshEnabled);
        const double intervalSeconds = riskObj.value(QStringLiteral("intervalSeconds"))
                                         .toDouble(static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
        updateRiskRefresh(enabled, intervalSeconds);
    }

    if (root.contains(QStringLiteral("marketRegimeTimeline"))
        && root.value(QStringLiteral("marketRegimeTimeline")).isObject()) {
        const QJsonObject regimeObj = root.value(QStringLiteral("marketRegimeTimeline")).toObject();
        const QJsonValue maxValue = regimeObj.value(QStringLiteral("maximumSnapshots"));
        if (maxValue.isDouble()) {
            const int limit = std::max(0, maxValue.toInt(m_regimeTimelineMaximumSnapshots));
            setRegimeTimelineMaximumSnapshots(limit);
        }
    }

    if (root.contains(QStringLiteral("alerts")) && root.value(QStringLiteral("alerts")).isObject()) {
        const QJsonObject alertsObj = root.value(QStringLiteral("alerts")).toObject();
        const bool hideAcknowledged = alertsObj.value(QStringLiteral("hideAcknowledged"))
                                          .toBool(m_filteredAlertsModel.hideAcknowledged());
        m_filteredAlertsModel.setHideAcknowledged(hideAcknowledged);

        const int severityValue = alertsObj.value(QStringLiteral("severityFilter"))
                                      .toInt(static_cast<int>(m_filteredAlertsModel.severityFilter()));
        if (severityValue >= AlertsFilterProxyModel::AllSeverities
            && severityValue <= AlertsFilterProxyModel::WarningOnly) {
            m_filteredAlertsModel.setSeverityFilter(
                static_cast<AlertsFilterProxyModel::SeverityFilter>(severityValue));
        }

        const int sortValue = alertsObj.value(QStringLiteral("sortMode"))
                                  .toInt(static_cast<int>(m_filteredAlertsModel.sortMode()));
        if (sortValue >= AlertsFilterProxyModel::NewestFirst
            && sortValue <= AlertsFilterProxyModel::TitleAscending) {
            m_filteredAlertsModel.setSortMode(static_cast<AlertsFilterProxyModel::SortMode>(sortValue));
        }

        if (alertsObj.contains(QStringLiteral("searchText")))
            m_filteredAlertsModel.setSearchText(alertsObj.value(QStringLiteral("searchText")).toString());

        const QJsonValue acknowledgedValue = alertsObj.value(QStringLiteral("acknowledgedIds"));
        if (acknowledgedValue.isArray()) {
            const QJsonArray ackArray = acknowledgedValue.toArray();
            QStringList ids;
            ids.reserve(ackArray.size());
            for (const QJsonValue& value : ackArray) {
                if (value.isString())
                    ids.append(value.toString());
            }
            m_alertsModel.setAcknowledgedAlertIds(ids);
        }
    }

    if (root.contains(QStringLiteral("preferences")) && root.value(QStringLiteral("preferences")).isObject()) {
        const QJsonObject prefs = root.value(QStringLiteral("preferences")).toObject();
        applyPersonalizationFromJson(prefs);
    }

    if (root.contains(QStringLiteral("uiModules")) && root.value(QStringLiteral("uiModules")).isObject()) {
        const QJsonObject modulesObj = root.value(QStringLiteral("uiModules")).toObject();

        if (modulesObj.contains(QStringLiteral("autoReload"))) {
            const bool autoReload = modulesObj.value(QStringLiteral("autoReload"))
                                       .toBool(m_uiModuleAutoReloadEnabled);
            setUiModuleAutoReloadEnabled(autoReload);
        }

        if (!m_uiModuleDirectoriesExplicit && modulesObj.value(QStringLiteral("directories")).isArray()) {
            const QJsonArray directoriesArray = modulesObj.value(QStringLiteral("directories")).toArray();
            QStringList directories;
            QSet<QString> seen;
            for (const QJsonValue& value : directoriesArray) {
                if (!value.isString())
                    continue;
                const QString normalized = normalizeUiModulePath(value.toString());
                if (normalized.isEmpty() || seen.contains(normalized))
                    continue;
                seen.insert(normalized);
                directories.append(normalized);
            }

            if (directories != m_uiModuleDirectories) {
                m_uiModuleDirectories = directories;
                emit uiModuleDirectoriesChanged(m_uiModuleDirectories);

                if (m_moduleManager) {
                    m_moduleManager->unloadPlugins();
                    m_moduleManager->setPluginPaths(m_uiModuleDirectories);
                    const bool loadSuccess = m_moduleManager->loadPlugins();
                    if (!loadSuccess) {
                        qCWarning(lcAppMetrics)
                            << "Nie wszystkie pluginy UI zostały poprawnie załadowane podczas przywracania ustawień.";
                    }
                    const QVariantMap report = m_moduleManager->lastLoadReport();
                    updateUiModuleWatchTargets(m_uiModuleDirectories,
                                               report.value(QStringLiteral("loadedPlugins")).toStringList());
                } else {
                    updateUiModuleWatchTargets(m_uiModuleDirectories, {});
                }
            }
        }

        if (modulesObj.contains(QStringLiteral("views"))
            && modulesObj.value(QStringLiteral("views")).isObject() && m_moduleViewsModel) {
            const QJsonObject viewsObj = modulesObj.value(QStringLiteral("views")).toObject();
            if (viewsObj.contains(QStringLiteral("categoryFilter"))) {
                const QString category = viewsObj.value(QStringLiteral("categoryFilter")).toString();
                m_moduleViewsModel->setCategoryFilter(category);
            }
            if (viewsObj.contains(QStringLiteral("searchFilter"))) {
                const QString search = viewsObj.value(QStringLiteral("searchFilter")).toString();
                m_moduleViewsModel->setSearchFilter(search);
            }
        }

        if (modulesObj.contains(QStringLiteral("services"))
            && modulesObj.value(QStringLiteral("services")).isObject() && m_moduleServicesModel) {
            const QJsonObject servicesObj = modulesObj.value(QStringLiteral("services")).toObject();
            if (servicesObj.contains(QStringLiteral("searchFilter"))) {
                const QString search = servicesObj.value(QStringLiteral("searchFilter")).toString();
                m_moduleServicesModel->setSearchFilter(search);
            }
        }
    }

    if (root.contains(QStringLiteral("riskHistory"))) {
        const QJsonValue riskHistoryValue = root.value(QStringLiteral("riskHistory"));
        if (riskHistoryValue.isArray()) {
            m_riskHistoryModel.restoreFromJson(riskHistoryValue.toArray());
        } else if (riskHistoryValue.isObject()) {
            const QJsonObject historyObject = riskHistoryValue.toObject();
            const QJsonValue maxValue = historyObject.value(QStringLiteral("maximumEntries"));
            if (maxValue.isDouble())
                m_riskHistoryModel.setMaximumEntries(maxValue.toInt(m_riskHistoryModel.maximumEntries()));

            const QJsonValue entriesValue = historyObject.value(QStringLiteral("entries"));
            if (entriesValue.isArray())
                m_riskHistoryModel.restoreFromJson(entriesValue.toArray());

            const QJsonValue exportValue = historyObject.value(QStringLiteral("export"));
            if (exportValue.isObject()) {
                const QJsonObject exportObject = exportValue.toObject();
                setRiskHistoryExportLimitEnabled(exportObject.value(QStringLiteral("limitEnabled"))
                                                    .toBool(m_riskHistoryExportLimitEnabled));

                const QJsonValue limitValue = exportObject.value(QStringLiteral("limitValue"));
                if (limitValue.isDouble())
                    setRiskHistoryExportLimitValue(std::max(1, limitValue.toInt(m_riskHistoryExportLimitValue)));

                if (exportObject.contains(QStringLiteral("lastDirectory"))) {
                    const QString lastDirValue = exportObject.value(QStringLiteral("lastDirectory"))
                                                       .toString();
                    const QString trimmed = lastDirValue.trimmed();
                    if (!trimmed.isEmpty()) {
                        QUrl directoryUrl(trimmed);
                        if (!directoryUrl.isValid() || directoryUrl.scheme().isEmpty())
                            directoryUrl = QUrl::fromLocalFile(expandPath(trimmed));
                        setRiskHistoryExportLastDirectory(directoryUrl);
                    }
                }

                const QJsonValue autoValue = exportObject.value(QStringLiteral("auto"));
                if (autoValue.isObject()) {
                    const QJsonObject autoObject = autoValue.toObject();
                    setRiskHistoryAutoExportEnabled(autoObject.value(QStringLiteral("enabled"))
                                                       .toBool(m_riskHistoryAutoExportEnabled));

                    const QJsonValue intervalValue = autoObject.value(QStringLiteral("intervalMinutes"));
                    if (intervalValue.isDouble()) {
                        const int intervalMinutes = std::max(1, intervalValue.toInt(m_riskHistoryAutoExportIntervalMinutes));
                        setRiskHistoryAutoExportIntervalMinutes(intervalMinutes);
                    }

                    if (autoObject.contains(QStringLiteral("basename")))
                        setRiskHistoryAutoExportBasename(autoObject.value(QStringLiteral("basename"))
                                                            .toString(m_riskHistoryAutoExportBasename));

                    if (autoObject.contains(QStringLiteral("useLocalTime")))
                        setRiskHistoryAutoExportUseLocalTime(autoObject.value(QStringLiteral("useLocalTime"))
                                                                .toBool(m_riskHistoryAutoExportUseLocalTime));

                    if (autoObject.contains(QStringLiteral("lastExportAt"))) {
                        const QString lastExportString = autoObject.value(QStringLiteral("lastExportAt")).toString();
                        QDateTime parsed = QDateTime::fromString(lastExportString, Qt::ISODateWithMs);
                        if (!parsed.isValid())
                            parsed = QDateTime::fromString(lastExportString, Qt::ISODate);
                        if (parsed.isValid()) {
                            parsed = parsed.toUTC();
                            m_lastRiskHistoryAutoExportUtc = parsed;
                            Q_EMIT riskHistoryLastAutoExportAtChanged();
                        }
                    }

                    if (autoObject.contains(QStringLiteral("lastPath"))) {
                        const QString lastPath = autoObject.value(QStringLiteral("lastPath")).toString();
                        if (!lastPath.trimmed().isEmpty()) {
                            QUrl pathUrl(lastPath);
                            if (!pathUrl.isValid() || pathUrl.scheme().isEmpty())
                                pathUrl = QUrl::fromLocalFile(expandPath(lastPath));
                            m_lastRiskHistoryAutoExportPath = pathUrl;
                            Q_EMIT riskHistoryLastAutoExportPathChanged();
                        }
                    }
                }
            }
        }
    }

    m_loadingUiSettings = false;

    if (loadedFromLegacy && desiredPath != file.fileName()) {
        m_uiSettingsPath = desiredPath;
        scheduleUiSettingsPersist();
    }
}

void Application::scheduleUiSettingsPersist()
{
    if (m_loadingUiSettings || !m_uiSettingsPersistenceEnabled || !m_uiSettingsTimerConfigured
        || m_uiSettingsPath.trimmed().isEmpty())
        return;

    m_uiSettingsSaveTimer.start();
}

void Application::persistUiSettings()
{
    if (!m_uiSettingsPersistenceEnabled || m_uiSettingsPath.trimmed().isEmpty())
        return;

    const QJsonObject payload = buildUiSettingsPayload();
    if (payload.isEmpty())
        return;

    const QFileInfo info(m_uiSettingsPath);
    QDir dir = info.dir();
    if (!dir.exists()) {
        if (!dir.mkpath(QStringLiteral("."))) {
            qCWarning(lcAppMetrics) << "Nie udało się utworzyć katalogu ustawień UI" << dir.absolutePath();
            return;
        }
    }

    QSaveFile file(m_uiSettingsPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się zapisać ustawień UI" << m_uiSettingsPath
                                << file.errorString();
        return;
    }

    const QJsonDocument document(payload);
    file.write(document.toJson(QJsonDocument::Compact));
    if (!file.commit()) {
        qCWarning(lcAppMetrics) << "Nie udało się zatwierdzić pliku ustawień UI" << m_uiSettingsPath
                                << file.errorString();
    } else {
        qCInfo(lcAppMetrics) << "Zapisano konfigurację UI do" << m_uiSettingsPath;
    }
}

void Application::ensureLicenseRefreshTimerConfigured()
{
    if (m_licenseRefreshTimerConfigured)
        return;

    m_licenseRefreshTimer.setTimerType(Qt::VeryCoarseTimer);
    m_licenseRefreshTimer.setSingleShot(false);
    m_licenseRefreshTimer.setParent(this);
    connect(&m_licenseRefreshTimer, &QTimer::timeout, this, &Application::refreshSecurityArtifacts);
    m_licenseRefreshTimerConfigured = true;
}

void Application::initializeSecurityRefresh()
{
    const QByteArray cacheEnv = qgetenv(kLicenseCachePathEnv.constData());
    if (!cacheEnv.isEmpty())
        m_licenseCachePath = expandPath(QString::fromUtf8(cacheEnv));
    else
        m_licenseCachePath = QDir::current().absoluteFilePath(QStringLiteral("var/cache/ui_license_snapshot.json"));

    if (!m_licenseCachePath.trimmed().isEmpty()) {
        QDir dir = QFileInfo(m_licenseCachePath).dir();
        if (!dir.exists())
            dir.mkpath(QStringLiteral("."));
    }

    loadSecurityCache();

    int intervalSeconds = kDefaultLicenseRefreshIntervalSeconds;
    if (qEnvironmentVariableIsSet(kLicenseRefreshIntervalEnv.constData())) {
        bool ok = false;
        const QByteArray raw = qgetenv(kLicenseRefreshIntervalEnv.constData());
        const int candidate = QString::fromUtf8(raw).toInt(&ok);
        if (ok)
            intervalSeconds = std::clamp(candidate, kMinLicenseRefreshIntervalSeconds, kMaxLicenseRefreshIntervalSeconds);
        else
            qCWarning(lcAppMetrics) << "Nieprawidłowa wartość" << QString::fromUtf8(raw)
                                    << "dla" << kLicenseRefreshIntervalEnv;
    }

    if (intervalSeconds <= 0) {
        m_licenseRefreshIntervalSeconds = 0;
        ensureLicenseRefreshTimerConfigured();
        m_licenseRefreshTimer.stop();
        m_nextLicenseRefreshUtc = QDateTime();
        updateSecurityCacheFromControllers();
        Q_EMIT licenseRefreshScheduleChanged();
        return;
    }

    ensureLicenseRefreshTimerConfigured();
    m_licenseRefreshIntervalSeconds = intervalSeconds;
    m_licenseRefreshTimer.setInterval(m_licenseRefreshIntervalSeconds * 1000);
    m_licenseRefreshTimer.start();
    QTimer::singleShot(0, this, &Application::refreshSecurityArtifacts);
}

void Application::refreshSecurityArtifacts()
{
    if (!m_activationController)
        return;

    m_lastLicenseRefreshRequestUtc = QDateTime::currentDateTimeUtc();
    m_activationController->refresh();
    if (m_licenseRefreshIntervalSeconds > 0)
        m_nextLicenseRefreshUtc = m_lastLicenseRefreshRequestUtc.addSecs(m_licenseRefreshIntervalSeconds);
    else
        m_nextLicenseRefreshUtc = QDateTime();
    Q_EMIT licenseRefreshScheduleChanged();
}

void Application::processSecurityArtifactsUpdate()
{
    if (!m_activationController)
        return;

    const bool loadingFromCache = m_loadingSecurityCache;
    const QString controllerError = m_activationController->lastError();
    const bool refreshSucceeded = controllerError.isEmpty();

    if (!loadingFromCache)
        m_lastSecurityError = controllerError;
    else if (!controllerError.isEmpty())
        m_lastSecurityError = controllerError;

    if (refreshSucceeded && !loadingFromCache) {
        m_lastLicenseRefreshUtc = QDateTime::currentDateTimeUtc();
        if (m_licenseRefreshIntervalSeconds > 0) {
            if (m_lastLicenseRefreshRequestUtc.isValid())
                m_nextLicenseRefreshUtc =
                    m_lastLicenseRefreshRequestUtc.addSecs(m_licenseRefreshIntervalSeconds);
            else
                m_nextLicenseRefreshUtc = m_lastLicenseRefreshUtc.addSecs(m_licenseRefreshIntervalSeconds);
        } else {
            m_nextLicenseRefreshUtc = QDateTime();
        }
    }

    updateSecurityCacheFromControllers();

    if (refreshSucceeded && !loadingFromCache) {
        clearSecurityAlert(QStringLiteral("security:license-refresh"));
        Q_EMIT licenseRefreshScheduleChanged();
    }
}

void Application::updateSecurityCacheFromControllers()
{
    QVariantMap cache;
    if (m_activationController) {
        cache.insert(QStringLiteral("fingerprint"), m_activationController->fingerprint());
        cache.insert(QStringLiteral("oemLicense"), m_activationController->oemLicense());
        cache.insert(QStringLiteral("licenseHistory"), QVariant::fromValue(m_activationController->licenses()));
        if (!m_lastSecurityError.isEmpty())
            cache.insert(QStringLiteral("lastError"), m_lastSecurityError);
    }
    if (m_securityController) {
        cache.insert(QStringLiteral("tpmStatus"), m_securityController->tpmStatus());
        cache.insert(QStringLiteral("integrityReport"), m_securityController->integrityReport());
        cache.insert(QStringLiteral("auditLog"), m_securityController->auditLog());
    }
    if (m_updateManager) {
        cache.insert(QStringLiteral("updatesAvailable"), !m_updateManager->availableUpdates().isEmpty());
        cache.insert(QStringLiteral("installedUpdates"), m_updateManager->installedUpdates());
    }
    cache.insert(QStringLiteral("refreshIntervalSeconds"), m_licenseRefreshIntervalSeconds);
    cache.insert(QStringLiteral("refreshActive"), m_licenseRefreshTimer.isActive());
    if (m_lastLicenseRefreshRequestUtc.isValid())
        cache.insert(QStringLiteral("lastRequestIso"), m_lastLicenseRefreshRequestUtc.toString(Qt::ISODateWithMs));
    if (m_lastLicenseRefreshUtc.isValid())
        cache.insert(QStringLiteral("lastRefreshIso"), m_lastLicenseRefreshUtc.toString(Qt::ISODateWithMs));
    if (m_nextLicenseRefreshUtc.isValid())
        cache.insert(QStringLiteral("nextRefreshIso"), m_nextLicenseRefreshUtc.toString(Qt::ISODateWithMs));

    if (cache == m_securityCache)
        return;

    m_securityCache = cache;
    Q_EMIT securityCacheChanged();
    persistSecurityCache();
}

void Application::loadSecurityCache()
{
    if (m_licenseCachePath.trimmed().isEmpty())
        return;

    QFile file(m_licenseCachePath);
    if (!file.exists())
        return;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać cache licencji" << m_licenseCachePath
                                << file.errorString();
        return;
    }

    const QByteArray data = file.readAll();
    file.close();

    QJsonParseError parseError{};
    const QJsonDocument document = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        qCWarning(lcAppMetrics) << "Cache licencji ma niepoprawny format" << m_licenseCachePath
                                << parseError.errorString();
        return;
    }

    m_securityCache = document.object().toVariantMap();

    m_lastSecurityError = m_securityCache.value(QStringLiteral("lastError")).toString();

    if (m_securityCache.contains(QStringLiteral("refreshIntervalSeconds"))) {
        const int cachedInterval = m_securityCache.value(QStringLiteral("refreshIntervalSeconds")).toInt();
        if (cachedInterval > 0)
            m_licenseRefreshIntervalSeconds = cachedInterval;
    }

    const bool cachedActive = m_securityCache.value(QStringLiteral("refreshActive")).toBool();

    const QString lastRefreshIso = m_securityCache.value(QStringLiteral("lastRefreshIso")).toString();
    if (!lastRefreshIso.isEmpty()) {
        m_lastLicenseRefreshUtc = QDateTime::fromString(lastRefreshIso, Qt::ISODateWithMs);
        if (!m_lastLicenseRefreshUtc.isValid())
            m_lastLicenseRefreshUtc = QDateTime::fromString(lastRefreshIso, Qt::ISODate);
    }
    const QString lastRequestIso = m_securityCache.value(QStringLiteral("lastRequestIso")).toString();
    if (!lastRequestIso.isEmpty()) {
        m_lastLicenseRefreshRequestUtc = QDateTime::fromString(lastRequestIso, Qt::ISODateWithMs);
        if (!m_lastLicenseRefreshRequestUtc.isValid())
            m_lastLicenseRefreshRequestUtc = QDateTime::fromString(lastRequestIso, Qt::ISODate);
    }
    const QString nextRefreshIso = m_securityCache.value(QStringLiteral("nextRefreshIso")).toString();
    if (!nextRefreshIso.isEmpty()) {
        m_nextLicenseRefreshUtc = QDateTime::fromString(nextRefreshIso, Qt::ISODateWithMs);
        if (!m_nextLicenseRefreshUtc.isValid())
            m_nextLicenseRefreshUtc = QDateTime::fromString(nextRefreshIso, Qt::ISODate);
    }

    if (cachedActive && m_licenseRefreshIntervalSeconds > 0) {
        ensureLicenseRefreshTimerConfigured();
        m_licenseRefreshTimer.setInterval(m_licenseRefreshIntervalSeconds * 1000);
        if (!m_licenseRefreshTimer.isActive())
            m_licenseRefreshTimer.start();
    }

    if (m_activationController) {
        const QVariantMap fingerprint = m_securityCache.value(QStringLiteral("fingerprint")).toMap();
        const QVariantMap oemLicense = m_securityCache.value(QStringLiteral("oemLicense")).toMap();
        const QVariantList history = m_securityCache.value(QStringLiteral("licenseHistory")).toList();
        m_loadingSecurityCache = true;
        const auto guard = qScopeGuard([this]() { m_loadingSecurityCache = false; });
        m_activationController->applyCachedState(fingerprint, oemLicense, history);
    }

    Q_EMIT securityCacheChanged();
    Q_EMIT licenseRefreshScheduleChanged();
}

void Application::persistSecurityCache()
{
    if (m_licenseCachePath.trimmed().isEmpty())
        return;

    QFileInfo info(m_licenseCachePath);
    QDir dir = info.dir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        qCWarning(lcAppMetrics) << "Nie udało się utworzyć katalogu cache licencji" << dir.absolutePath();
        return;
    }

    QSaveFile file(m_licenseCachePath);
    file.setDirectWriteFallback(true);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się zapisać cache licencji" << m_licenseCachePath
                                << file.errorString();
        return;
    }

    const QJsonObject object = QJsonObject::fromVariantMap(m_securityCache);
    const QByteArray payload = QJsonDocument(object).toJson(QJsonDocument::Compact);
    if (file.write(payload) != payload.size()) {
        qCWarning(lcAppMetrics) << "Nie udało się zapisać pełnego cache licencji" << m_licenseCachePath;
        return;
    }
    if (!file.commit())
        qCWarning(lcAppMetrics) << "Nie udało się zatwierdzić cache licencji" << m_licenseCachePath;
}

void Application::raiseSecurityAlert(const QString& id,
                                     AlertsModel::Severity severity,
                                     const QString& title,
                                     const QString& description)
{
    m_alertsModel.raiseAlert(id, title, description, severity, true);
}

void Application::clearSecurityAlert(const QString& id)
{
    m_alertsModel.clearAlert(id);
}

QJsonObject Application::buildUiSettingsPayload() const
{
    QJsonObject root;

    QJsonObject instrument;
    instrument.insert(QStringLiteral("exchange"), m_instrument.exchange);
    instrument.insert(QStringLiteral("symbol"), m_instrument.symbol);
    instrument.insert(QStringLiteral("venueSymbol"), m_instrument.venueSymbol);
    instrument.insert(QStringLiteral("quoteCurrency"), m_instrument.quoteCurrency);
    instrument.insert(QStringLiteral("baseCurrency"), m_instrument.baseCurrency);
    instrument.insert(QStringLiteral("granularity"), m_instrument.granularityIso8601);
    root.insert(QStringLiteral("instrument"), instrument);

    QJsonObject guard;
    guard.insert(QStringLiteral("fpsTarget"), m_guard.fpsTarget);
    guard.insert(QStringLiteral("reduceMotionAfter"), m_guard.reduceMotionAfterSeconds);
    guard.insert(QStringLiteral("jankThresholdMs"), m_guard.jankThresholdMs);
    guard.insert(QStringLiteral("maxOverlayCount"), m_guard.maxOverlayCount);
    guard.insert(QStringLiteral("disableSecondaryWhenBelow"), m_guard.disableSecondaryWhenFpsBelow);
    root.insert(QStringLiteral("performanceGuard"), guard);

    QJsonObject risk;
    risk.insert(QStringLiteral("enabled"), m_riskRefreshEnabled);
    risk.insert(QStringLiteral("intervalSeconds"), static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
    root.insert(QStringLiteral("riskRefresh"), risk);

    QJsonObject alerts;
    QJsonArray acknowledged;
    const QStringList ackIds = m_alertsModel.acknowledgedAlertIds();
    for (const QString& id : ackIds)
        acknowledged.append(id);
    alerts.insert(QStringLiteral("acknowledgedIds"), acknowledged);
    alerts.insert(QStringLiteral("hideAcknowledged"), m_filteredAlertsModel.hideAcknowledged());
    alerts.insert(QStringLiteral("severityFilter"),
                  static_cast<int>(m_filteredAlertsModel.severityFilter()));
    alerts.insert(QStringLiteral("sortMode"), static_cast<int>(m_filteredAlertsModel.sortMode()));
    alerts.insert(QStringLiteral("searchText"), m_filteredAlertsModel.searchText());
    root.insert(QStringLiteral("alerts"), alerts);

    QJsonObject regimeTimeline;
    regimeTimeline.insert(QStringLiteral("maximumSnapshots"), m_regimeTimelineMaximumSnapshots);
    root.insert(QStringLiteral("marketRegimeTimeline"), regimeTimeline);

    QJsonObject history;
    history.insert(QStringLiteral("maximumEntries"), m_riskHistoryModel.maximumEntries());
    const QJsonArray historyEntries = m_riskHistoryModel.toJson();
    if (!historyEntries.isEmpty())
        history.insert(QStringLiteral("entries"), historyEntries);

    QJsonObject exportPrefs;
    exportPrefs.insert(QStringLiteral("limitEnabled"), m_riskHistoryExportLimitEnabled);
    exportPrefs.insert(QStringLiteral("limitValue"), m_riskHistoryExportLimitValue);
    if (!m_riskHistoryExportLastDirectory.isEmpty())
        exportPrefs.insert(QStringLiteral("lastDirectory"),
                           m_riskHistoryExportLastDirectory.toString(QUrl::PreferLocalFile));
    QJsonObject autoPrefs;
    autoPrefs.insert(QStringLiteral("enabled"), m_riskHistoryAutoExportEnabled);
    autoPrefs.insert(QStringLiteral("intervalMinutes"), m_riskHistoryAutoExportIntervalMinutes);
    autoPrefs.insert(QStringLiteral("basename"), m_riskHistoryAutoExportBasename);
    autoPrefs.insert(QStringLiteral("useLocalTime"), m_riskHistoryAutoExportUseLocalTime);
    if (m_lastRiskHistoryAutoExportUtc.isValid())
        autoPrefs.insert(QStringLiteral("lastExportAt"),
                         m_lastRiskHistoryAutoExportUtc.toUTC().toString(Qt::ISODateWithMs));
    if (!m_lastRiskHistoryAutoExportPath.isEmpty())
        autoPrefs.insert(QStringLiteral("lastPath"),
                         m_lastRiskHistoryAutoExportPath.toString(QUrl::PreferLocalFile));
    exportPrefs.insert(QStringLiteral("auto"), autoPrefs);
    history.insert(QStringLiteral("export"), exportPrefs);
    root.insert(QStringLiteral("riskHistory"), history);

    const QVariantMap personalization = buildPersonalizationPayload();
    if (!personalization.isEmpty())
        root.insert(QStringLiteral("preferences"), QJsonObject::fromVariantMap(personalization));

    return root;
}

QVariantMap Application::buildPersonalizationPayload() const
{
    QVariantMap map;
    map.insert(QStringLiteral("theme"), m_uiTheme);
    map.insert(QStringLiteral("layout"), m_uiLayoutMode);
    map.insert(QStringLiteral("alert_toasts"), m_alertToastsEnabled);
    return map;
}

void Application::applyPersonalizationFromJson(const QJsonObject& object)
{
    if (object.contains(QStringLiteral("theme")) && object.value(QStringLiteral("theme")).isString())
        setUiTheme(object.value(QStringLiteral("theme")).toString());
    if (object.contains(QStringLiteral("layout")) && object.value(QStringLiteral("layout")).isString())
        setUiLayoutMode(object.value(QStringLiteral("layout")).toString());
    if (object.contains(QStringLiteral("alert_toasts")))
        setAlertToastsEnabled(object.value(QStringLiteral("alert_toasts")).toBool(m_alertToastsEnabled));
}

void Application::applyUiThemePalette()
{
    QPalette palette = QGuiApplication::palette();
    const QString theme = m_uiTheme.toLower();
    if (theme == QStringLiteral("light")) {
        palette.setColor(QPalette::Window, QColor(QStringLiteral("#f5f7fb")));
        palette.setColor(QPalette::WindowText, QColor(QStringLiteral("#1c1f2a")));
        palette.setColor(QPalette::Base, QColor(QStringLiteral("#ffffff")));
        palette.setColor(QPalette::AlternateBase, QColor(QStringLiteral("#e8ecf5")));
        palette.setColor(QPalette::ToolTipBase, QColor(QStringLiteral("#ffffff")));
        palette.setColor(QPalette::ToolTipText, QColor(QStringLiteral("#1c1f2a")));
        palette.setColor(QPalette::Text, QColor(QStringLiteral("#1c1f2a")));
        palette.setColor(QPalette::Button, QColor(QStringLiteral("#e3e7f2")));
        palette.setColor(QPalette::ButtonText, QColor(QStringLiteral("#1c1f2a")));
        palette.setColor(QPalette::Highlight, QColor(QStringLiteral("#1a73e8")));
        palette.setColor(QPalette::HighlightedText, QColor(QStringLiteral("#ffffff")));
    } else if (theme == QStringLiteral("midnight")) {
        palette.setColor(QPalette::Window, QColor(QStringLiteral("#0d1117")));
        palette.setColor(QPalette::WindowText, QColor(QStringLiteral("#e6edf3")));
        palette.setColor(QPalette::Base, QColor(QStringLiteral("#161b22")));
        palette.setColor(QPalette::AlternateBase, QColor(QStringLiteral("#0d1117")));
        palette.setColor(QPalette::ToolTipBase, QColor(QStringLiteral("#161b22")));
        palette.setColor(QPalette::ToolTipText, QColor(QStringLiteral("#e6edf3")));
        palette.setColor(QPalette::Text, QColor(QStringLiteral("#e6edf3")));
        palette.setColor(QPalette::Button, QColor(QStringLiteral("#21262d")));
        palette.setColor(QPalette::ButtonText, QColor(QStringLiteral("#e6edf3")));
        palette.setColor(QPalette::Highlight, QColor(QStringLiteral("#58a6ff")));
        palette.setColor(QPalette::HighlightedText, QColor(QStringLiteral("#0d1117")));
    } else {
        palette.setColor(QPalette::Window, QColor(QStringLiteral("#111722")));
        palette.setColor(QPalette::WindowText, QColor(QStringLiteral("#f0f4ff")));
        palette.setColor(QPalette::Base, QColor(QStringLiteral("#0e141f")));
        palette.setColor(QPalette::AlternateBase, QColor(QStringLiteral("#161d2b")));
        palette.setColor(QPalette::ToolTipBase, QColor(QStringLiteral("#111722")));
        palette.setColor(QPalette::ToolTipText, QColor(QStringLiteral("#f0f4ff")));
        palette.setColor(QPalette::Text, QColor(QStringLiteral("#f0f4ff")));
        palette.setColor(QPalette::Button, QColor(QStringLiteral("#182132")));
        palette.setColor(QPalette::ButtonText, QColor(QStringLiteral("#f0f4ff")));
        palette.setColor(QPalette::Highlight, QColor(QStringLiteral("#2196f3")));
        palette.setColor(QPalette::HighlightedText, QColor(QStringLiteral("#ffffff")));
    }
    QGuiApplication::setPalette(palette);
}

void Application::initializeSupportedExchanges()
{
    if (!m_supportedExchanges.isEmpty())
        return;
    m_supportedExchanges = {
        QStringLiteral("BINANCE"),
        QStringLiteral("COINBASE"),
        QStringLiteral("KRAKEN"),
        QStringLiteral("OKX"),
        QStringLiteral("BITGET"),
        QStringLiteral("BYBIT"),
    };
    std::sort(m_supportedExchanges.begin(), m_supportedExchanges.end());
    m_supportedExchanges.erase(
        std::unique(m_supportedExchanges.begin(), m_supportedExchanges.end()),
        m_supportedExchanges.end());
}

void Application::maybeAutoExportRiskHistory(const QDateTime& snapshotTimestamp)
{
    if (!m_riskHistoryAutoExportEnabled)
        return;

    if (m_riskHistoryExportLastDirectory.isEmpty()) {
        if (!m_riskHistoryAutoExportDirectoryWarned) {
            qCWarning(lcAppMetrics)
                << "Automatyczny eksport historii ryzyka pominięty – brak skonfigurowanego katalogu docelowego.";
            m_riskHistoryAutoExportDirectoryWarned = true;
        }
        return;
    }

    QString directoryPath;
    if (m_riskHistoryExportLastDirectory.isLocalFile() || m_riskHistoryExportLastDirectory.scheme().isEmpty())
        directoryPath = m_riskHistoryExportLastDirectory.toLocalFile();
    else
        directoryPath = m_riskHistoryExportLastDirectory.toString(QUrl::PreferLocalFile);

    directoryPath = directoryPath.trimmed();
    if (directoryPath.isEmpty()) {
        if (!m_riskHistoryAutoExportDirectoryWarned) {
            qCWarning(lcAppMetrics)
                << "Automatyczny eksport historii ryzyka pominięty – ścieżka katalogu docelowego jest pusta.";
            m_riskHistoryAutoExportDirectoryWarned = true;
        }
        return;
    }

    QDir directory(directoryPath);
    if (!directory.exists() && !directory.mkpath(QStringLiteral("."))) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka pominięty – nie udało się utworzyć katalogu"
            << directory.absolutePath();
        return;
    }

    const QDateTime nowUtc = QDateTime::currentDateTimeUtc();
    const int intervalSeconds = qMax(1, m_riskHistoryAutoExportIntervalMinutes) * 60;
    if (m_lastRiskHistoryAutoExportUtc.isValid()) {
        if (m_lastRiskHistoryAutoExportUtc.secsTo(nowUtc) < intervalSeconds)
            return;
    }

    const QDateTime exportTimestamp = snapshotTimestamp.isValid() ? snapshotTimestamp : nowUtc;
    const QString filePath = resolveAutoExportFilePath(directory, m_riskHistoryAutoExportBasename, exportTimestamp);
    if (filePath.isEmpty()) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka pominięty – nie udało się wyznaczyć docelowej nazwy pliku.";
        return;
    }

    int limit = -1;
    if (m_riskHistoryExportLimitEnabled)
        limit = m_riskHistoryExportLimitValue;

    if (!m_riskHistoryModel.exportToCsv(filePath, limit)) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka nie powiódł się do pliku" << filePath;
        return;
    }

    m_lastRiskHistoryAutoExportUtc = nowUtc;
    m_lastRiskHistoryAutoExportPath = QUrl::fromLocalFile(filePath);
    m_riskHistoryAutoExportDirectoryWarned = false;
    Q_EMIT riskHistoryLastAutoExportAtChanged();
    Q_EMIT riskHistoryLastAutoExportPathChanged();
    qCInfo(lcAppMetrics) << "Automatycznie wyeksportowano historię ryzyka do" << filePath;
}

QString Application::resolveAutoExportFilePath(const QDir& directory,
                                               const QString& basename,
                                               const QDateTime& timestamp) const
{
    const QString sanitizedBase = sanitizeAutoExportBasename(basename);
    const QString effectiveBase = sanitizedBase.isEmpty() ? QStringLiteral("risk-history") : sanitizedBase;
    const bool useLocalTime = m_riskHistoryAutoExportUseLocalTime;

    QDateTime normalizedTimestamp;
    if (timestamp.isValid())
        normalizedTimestamp = useLocalTime ? timestamp.toLocalTime() : timestamp.toUTC();
    else
        normalizedTimestamp = useLocalTime ? QDateTime::currentDateTime() : QDateTime::currentDateTimeUtc();

    const QString timePart = normalizedTimestamp.toString(QStringLiteral("yyyyMMdd_HHmmss"));

    QString baseName;
    if (useLocalTime) {
        const int offsetSeconds = normalizedTimestamp.offsetFromUtc();
        const int offsetMinutes = offsetSeconds / 60;
        const int offsetHours = offsetMinutes / 60;
        const int remainingMinutes = std::abs(offsetMinutes % 60);
        const QChar sign = offsetSeconds >= 0 ? QLatin1Char('+') : QLatin1Char('-');
        const QString zoneSuffix = QStringLiteral("%1%2%3")
                                      .arg(sign)
                                      .arg(std::abs(offsetHours), 2, 10, QLatin1Char('0'))
                                      .arg(remainingMinutes, 2, 10, QLatin1Char('0'));
        baseName = QStringLiteral("%1_%2_%3").arg(effectiveBase, timePart, zoneSuffix);
    } else {
        baseName = QStringLiteral("%1_%2").arg(effectiveBase, timePart);
    }
    QString candidate = directory.absoluteFilePath(baseName + QStringLiteral(".csv"));
    if (!QFileInfo::exists(candidate))
        return candidate;

    for (int attempt = 1; attempt <= 100; ++attempt) {
        const QString alternate = directory.absoluteFilePath(
            QStringLiteral("%1_%2.csv").arg(baseName).arg(attempt));
        if (!QFileInfo::exists(alternate))
            return alternate;
    }

    return {};
}

bool Application::setDecisionLogPathInternal(const QString& path, bool emitSignal)
{
    QString candidate = path.trimmed();
    if (candidate.isEmpty())
        return false;

    candidate = expandPath(candidate);
    if (candidate.isEmpty())
        return false;

    if (m_decisionLogPath == candidate) {
        if (emitSignal)
            Q_EMIT decisionLogPathChanged();
        return true;
    }

    m_decisionLogPath = candidate;
    m_decisionLogModel.setLogPath(m_decisionLogPath);
    if (emitSignal)
        Q_EMIT decisionLogPathChanged();
    return true;
}

bool Application::validateTransportConfiguration(const QString& endpoint,
                                                 const QString& datasetPath,
                                                 const TradingClient::TlsConfig& tradingTls,
                                                 const TelemetryTlsConfig& metricsTls,
                                                 const GrpcTlsConfig& healthTls,
                                                 const QString& metricsEndpoint,
                                                 const QString& healthEndpoint) const
{
    QStringList errors;

    auto requireExistingFile = [&](const QString& path, const QString& description) {
        const QString trimmed = path.trimmed();
        if (trimmed.isEmpty()) {
            errors.append(description);
            return;
        }
        const QFileInfo info(trimmed);
        if (!info.exists() || !info.isFile()) {
            errors.append(QStringLiteral("Plik '%1' wskazany dla %2 nie istnieje lub nie jest regularnym plikiem.")
                              .arg(trimmed, description));
        }
    };

    auto validateGrpcTls = [&](const TradingClient::TlsConfig& tls, const QString& contextLabel) {
        if (!tls.enabled) {
            if (tls.requireClientAuth) {
                errors.append(QStringLiteral("Włączono mTLS (%1), ale TLS jest wyłączony.").arg(contextLabel));
            }
            if (!tls.clientCertificatePath.trimmed().isEmpty() || !tls.clientKeyPath.trimmed().isEmpty()) {
                errors.append(QStringLiteral("Podano materiał klienta TLS (%1), lecz połączenie TLS jest wyłączone.")
                                  .arg(contextLabel));
            }
            return;
        }

        if (tls.rootCertificatePath.trimmed().isEmpty()) {
            errors.append(QStringLiteral("Brak ścieżki root CA dla %1 (pole --tls-root-cert lub grpc.tls.root_cert).").arg(contextLabel));
        } else {
            requireExistingFile(tls.rootCertificatePath, QStringLiteral("root CA (%1)").arg(contextLabel));
        }

        const bool certEmpty = tls.clientCertificatePath.trimmed().isEmpty();
        const bool keyEmpty = tls.clientKeyPath.trimmed().isEmpty();
        if (tls.requireClientAuth) {
            if (certEmpty || keyEmpty) {
                errors.append(QStringLiteral("Włączono mTLS dla %1, ale nie podano zarówno certyfikatu, jak i klucza klienta.")
                                  .arg(contextLabel));
            }
        }
        if (certEmpty != keyEmpty) {
            errors.append(QStringLiteral("Podano tylko część materiału klienta (certyfikat/klucz) dla %1.").arg(contextLabel));
        }
        if (!certEmpty) {
            requireExistingFile(tls.clientCertificatePath, QStringLiteral("certyfikat klienta (%1)").arg(contextLabel));
        }
        if (!keyEmpty) {
            requireExistingFile(tls.clientKeyPath, QStringLiteral("klucz klienta (%1)").arg(contextLabel));
        }
    };

    auto validateGrpcTlsHealth = [&](const GrpcTlsConfig& tls, const QString& contextLabel) {
        if (!tls.enabled) {
            if (tls.requireClientAuth) {
                errors.append(QStringLiteral("Włączono mTLS (%1), ale TLS jest wyłączony.").arg(contextLabel));
            }
            if (!tls.clientCertificatePath.trimmed().isEmpty() || !tls.clientKeyPath.trimmed().isEmpty()) {
                errors.append(QStringLiteral("Podano materiał klienta TLS (%1), lecz połączenie TLS jest wyłączone.")
                                  .arg(contextLabel));
            }
            return;
        }

        if (tls.rootCertificatePath.trimmed().isEmpty()) {
            errors.append(QStringLiteral("Brak ścieżki root CA dla %1 (parametr --health-tls-root-cert / grpc.tls.root_cert).").arg(contextLabel));
        } else {
            requireExistingFile(tls.rootCertificatePath, QStringLiteral("root CA (%1)").arg(contextLabel));
        }

        const bool certEmpty = tls.clientCertificatePath.trimmed().isEmpty();
        const bool keyEmpty = tls.clientKeyPath.trimmed().isEmpty();
        if (tls.requireClientAuth) {
            if (certEmpty || keyEmpty) {
                errors.append(QStringLiteral("Włączono mTLS dla %1, ale nie dostarczono pełnej pary certyfikat/klucz.").arg(contextLabel));
            }
        }
        if (certEmpty != keyEmpty) {
            errors.append(QStringLiteral("Podano tylko certyfikat lub tylko klucz klienta dla %1.").arg(contextLabel));
        }
        if (!certEmpty) {
            requireExistingFile(tls.clientCertificatePath, QStringLiteral("certyfikat klienta (%1)").arg(contextLabel));
        }
        if (!keyEmpty) {
            requireExistingFile(tls.clientKeyPath, QStringLiteral("klucz klienta (%1)").arg(contextLabel));
        }
    };

    auto validateMetricsTls = [&](const TelemetryTlsConfig& tls) {
        if (!tls.enabled) {
            if (!tls.clientCertificatePath.trimmed().isEmpty() || !tls.clientKeyPath.trimmed().isEmpty()
                || !tls.rootCertificatePath.trimmed().isEmpty() || !tls.pinnedServerSha256.trimmed().isEmpty()) {
                errors.append(QStringLiteral("Podano konfigurację TLS telemetrii, ale TLS jest wyłączony (użyj --metrics-use-tls).");
            }
            return;
        }

        if (tls.rootCertificatePath.trimmed().isEmpty()) {
            errors.append(QStringLiteral("Brak ścieżki root CA dla MetricsService (--metrics-root-cert / telemetry.root_cert)."));
        } else {
            requireExistingFile(tls.rootCertificatePath, QStringLiteral("root CA (MetricsService)"));
        }

        const bool certEmpty = tls.clientCertificatePath.trimmed().isEmpty();
        const bool keyEmpty = tls.clientKeyPath.trimmed().isEmpty();
        if (certEmpty != keyEmpty) {
            errors.append(QStringLiteral("Podano tylko certyfikat lub tylko klucz klienta dla MetricsService."));
        }
        if (!certEmpty) {
            requireExistingFile(tls.clientCertificatePath, QStringLiteral("certyfikat klienta (MetricsService)"));
        }
        if (!keyEmpty) {
            requireExistingFile(tls.clientKeyPath, QStringLiteral("klucz klienta (MetricsService)"));
        }
    };

    if (m_transportMode == TradingClient::TransportMode::Grpc) {
        const QString trimmedEndpoint = endpoint.trimmed();
        if (trimmedEndpoint.isEmpty()) {
            errors.append(QStringLiteral("Tryb gRPC wymaga poprawnego endpointu --endpoint host:port."));
        }

        validateGrpcTls(tradingTls, QStringLiteral("TradingService"));

        if (m_metricsEnabled) {
            if (metricsEndpoint.trimmed().isEmpty()) {
                errors.append(QStringLiteral("Telemetria jest włączona, ale nie podano --metrics-endpoint."));
            }
            validateMetricsTls(metricsTls);
        }

        const QString effectiveHealthEndpoint = healthEndpoint.trimmed().isEmpty() ? trimmedEndpoint : healthEndpoint.trimmed();
        if (effectiveHealthEndpoint.isEmpty()) {
            errors.append(QStringLiteral("Nie określono endpointu HealthService (--health-endpoint)."));
        }
        validateGrpcTlsHealth(healthTls, QStringLiteral("HealthService"));
    } else {
        if (datasetPath.trimmed().isEmpty()) {
            errors.append(QStringLiteral("Tryb in-process wymaga wskazania datasetu (--transport-dataset lub transport.dataset w konfiguracji)."));
        } else {
            const QFileInfo datasetInfo(datasetPath);
            if (!datasetInfo.exists() || !datasetInfo.isFile()) {
                errors.append(QStringLiteral("Dataset in-process '%1' nie istnieje.").arg(datasetPath));
            }
        }

        if (tradingTls.enabled) {
            errors.append(QStringLiteral("TLS dla TradingService nie jest wspierany w trybie in-process."));
        }
        if (healthTls.enabled) {
            errors.append(QStringLiteral("TLS dla HealthService nie jest wspierany w trybie in-process."));
        }
        if (metricsTls.enabled) {
            errors.append(QStringLiteral("TLS telemetrii nie jest wspierany w trybie in-process."));
        }

        if (m_metricsEnabled && !metricsEndpoint.trimmed().startsWith(QStringLiteral("in-process"), Qt::CaseInsensitive)) {
            errors.append(QStringLiteral("Tryb in-process wymaga ustawienia telemetrii na endpoint 'in-process'."));
        }

        if (!healthEndpoint.trimmed().isEmpty()
            && !healthEndpoint.trimmed().startsWith(QStringLiteral("in-process"), Qt::CaseInsensitive)) {
            errors.append(QStringLiteral("Tryb in-process wymaga ustawienia HealthService na endpoint 'in-process'."));
        }
    }

    if (!errors.isEmpty()) {
        for (const QString& error : std::as_const(errors)) {
            qCWarning(lcAppMetrics) << error;
        }
        return false;
    }

    return true;
}

void Application::configureLocalBotCoreService(const QCommandLineParser& parser, QString& endpoint)
{
    if (m_offlineMode) {
        if (m_localService && m_localServiceEnabled)
            m_localService->stop();
        m_localServiceEnabled = false;
        return;
    }

    if (parser.isSet(QStringLiteral("no-local-core"))) {
        if (m_localService && m_localServiceEnabled) {
            m_localService->stop();
        }
        m_localServiceEnabled = false;
        return;
    }

    bool requested = true;
    if (const auto envFlag = envBool("BOT_CORE_UI_LOCAL_CORE")) {
        requested = envFlag.value();
    }
    if (parser.isSet(QStringLiteral("local-core"))) {
        requested = true;
    }

    if (!requested)
        return;

    if (!m_localService)
        m_localService = std::make_unique<BotCoreLocalService>(this);

    if (m_repoRoot.isEmpty())
        m_repoRoot = locateRepoRoot();

    if (!m_repoRoot.isEmpty())
        m_localService->setRepoRoot(m_repoRoot);

    QString pythonExecutable = parser.value(QStringLiteral("local-core-python")).trimmed();
    if (pythonExecutable.isEmpty()) {
        if (const auto envPython = envValue("BOT_CORE_UI_LOCAL_CORE_PYTHON"))
            pythonExecutable = envPython->trimmed();
    }
    if (!pythonExecutable.isEmpty())
        m_localService->setPythonExecutable(expandPath(pythonExecutable));

    QString datasetPath = parser.value(QStringLiteral("local-core-dataset")).trimmed();
    if (datasetPath.isEmpty()) {
        if (const auto envDataset = envValue("BOT_CORE_UI_LOCAL_CORE_DATASET"))
            datasetPath = envDataset->trimmed();
    }
    if (!datasetPath.isEmpty())
        m_localService->setDatasetPath(expandPath(datasetPath));

    QString host = parser.value(QStringLiteral("local-core-host")).trimmed();
    if (host.isEmpty()) {
        if (const auto envHost = envValue("BOT_CORE_UI_LOCAL_CORE_HOST"))
            host = envHost->trimmed();
    }
    if (host.isEmpty())
        host = QStringLiteral("127.0.0.1");
    m_localService->setHost(host);

    int port = 0;
    bool portOk = false;
    if (!parser.value(QStringLiteral("local-core-port")).trimmed().isEmpty()) {
        port = parser.value(QStringLiteral("local-core-port")).toInt(&portOk);
        if (!portOk) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowa wartość --local-core-port" << parser.value("local-core-port");
            port = 0;
        }
    }
    if (const auto envPort = envValue("BOT_CORE_UI_LOCAL_CORE_PORT")) {
        bool envOk = false;
        const int candidate = envPort->toInt(&envOk);
        if (envOk)
            port = candidate;
    }
    m_localService->setPort(port);

    bool repeat = parser.isSet(QStringLiteral("local-core-stream-repeat"));
    if (const auto envRepeat = envBool("BOT_CORE_UI_LOCAL_CORE_STREAM_REPEAT"))
        repeat = envRepeat.value();
    m_localService->setStreamRepeat(repeat);

    double interval = parser.value(QStringLiteral("local-core-stream-interval")).toDouble();
    if (const auto envInterval = envValue("BOT_CORE_UI_LOCAL_CORE_STREAM_INTERVAL")) {
        bool ok = false;
        const double candidate = envInterval->toDouble(&ok);
        if (ok)
            interval = candidate;
    }
    m_localService->setStreamInterval(interval);

    if (!m_localService->start()) {
        qCWarning(lcAppMetrics)
            << "Nie udało się uruchomić lokalnego serwisu bot_core:" << m_localService->lastError();
        m_localServiceEnabled = false;
        return;
    }

    endpoint = m_localService->endpoint();
    m_localServiceEnabled = true;
    qCInfo(lcAppMetrics) << "Uruchomiono lokalny stub bot_core pod adresem" << endpoint;
}

void Application::start() {
    m_ohlcvModel.setMaximumSamples(m_maxSamples);
    m_regimeTimelineModel.setMaximumSnapshots(m_regimeTimelineMaximumSnapshots);
    m_riskModel.clear();
    m_riskHistoryModel.clear();
    m_alertsModel.reset();

    ensureFrameMonitor();

    // Jeśli QML już wczytany — podepnij okno
    if (!m_engine.rootObjects().isEmpty()) {
        attachWindow(m_engine.rootObjects().constFirst());
    }

    ensureTelemetry();
    if (m_telemetry) {
        m_telemetry->setWindowCount(m_windowCount);
    }

    if (m_healthController) {
        m_healthController->refresh();
    }

    if (m_offlineMode) {
        ensureOfflineBridge();
        if (m_offlineBridge) {
            const QUrl endpointUrl = QUrl::fromUserInput(m_offlineEndpoint);
            m_offlineBridge->setEndpoint(endpointUrl);
            m_offlineBridge->setInstrument(m_instrument);
            m_offlineBridge->setHistoryLimit(m_historyLimit);
            m_offlineBridge->setDatasetPath(m_inProcessDatasetPath);
            m_offlineBridge->setStrategyConfig(m_offlineStrategyConfig);
            m_offlineBridge->setAutoRunEnabled(m_offlineAutoRun);
            m_connectionStatus = m_offlineStatus;
            Q_EMIT connectionStatusChanged();
            m_offlineBridge->start();
        }
    } else {
        m_client.start();
    }
    if (m_marketplaceController)
        m_marketplaceController->refreshPresets();
    m_started = true;
    applyRiskRefreshTimerState();
}

void Application::stop() {
    if (m_offlineMode) {
        if (m_offlineBridge)
            m_offlineBridge->stop();
    } else {
        m_client.stop();
    }
    m_alertsModel.reset();
    m_riskHistoryModel.clear();
    if (m_localService && m_localServiceEnabled) {
        m_localService->stop();
        m_localServiceEnabled = false;
    }
    m_started = false;
    applyRiskRefreshTimerState();
}

void Application::handleHistory(const QList<OhlcvPoint>& candles) {
    m_ohlcvModel.resetWithHistory(candles);
    m_signalModel.resetWithSignals(QVector<SignalEventEntry>());
    m_regimeTimelineModel.resetWithSnapshots(QVector<MarketRegimeSnapshotEntry>());
}

void Application::handleCandle(const OhlcvPoint& candle) {
    m_ohlcvModel.applyIncrement(candle);
}

void Application::handleRiskState(const RiskSnapshotData& snapshot) {
    m_riskModel.updateFromSnapshot(snapshot);
    m_riskHistoryModel.recordSnapshot(snapshot);
    m_alertsModel.updateFromRiskSnapshot(snapshot);
    if (snapshot.generatedAt.isValid())
        m_lastRiskUpdateUtc = snapshot.generatedAt.toUTC();
    else
        m_lastRiskUpdateUtc = QDateTime::currentDateTimeUtc();
    if (!m_lastRiskRefreshRequestUtc.isValid())
        m_lastRiskRefreshRequestUtc = m_lastRiskUpdateUtc;
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::ensureOfflineBridge()
{
    if (m_offlineBridge)
        return;
    m_offlineBridge = std::make_unique<OfflineRuntimeBridge>(this);
    connect(m_offlineBridge.get(), &OfflineRuntimeBridge::historyReceived, this, &Application::handleHistory);
    connect(m_offlineBridge.get(), &OfflineRuntimeBridge::riskStateReceived, this, &Application::handleRiskState);
    connect(m_offlineBridge.get(), &OfflineRuntimeBridge::performanceGuardUpdated, this,
            [this](const PerformanceGuard& guard) {
                m_guard = guard;
                Q_EMIT performanceGuardChanged();
                if (m_frameMonitor)
                    m_frameMonitor->setPerformanceGuard(m_guard);
                if (m_performanceTelemetry)
                    m_performanceTelemetry->setPerformanceGuard(m_guard);
                reportOverlayTelemetry();
            });
    connect(m_offlineBridge.get(), &OfflineRuntimeBridge::connectionStateChanged, this,
            &Application::handleOfflineStatusChanged);
    connect(m_offlineBridge.get(), &OfflineRuntimeBridge::automationStateChanged, this,
            &Application::handleOfflineAutomationChanged);
}

void Application::handleOfflineStatusChanged(const QString& status)
{
    if (m_offlineStatus == status)
        return;
    m_offlineStatus = status;
    Q_EMIT offlineDaemonStatusChanged();
    if (m_offlineMode) {
        if (m_connectionStatus != status) {
            m_connectionStatus = status;
            Q_EMIT connectionStatusChanged();
            updateSupportBundleMetadata();
        }
    }
}

void Application::handleOfflineAutomationChanged(bool running)
{
    if (m_offlineAutomationRunning == running)
        return;
    m_offlineAutomationRunning = running;
    Q_EMIT offlineAutomationRunningChanged(running);
}

void Application::handleActivationErrorChanged()
{
    if (!m_activationController)
        return;

    const QString error = m_activationController->lastError();

    if (!error.isEmpty() || !m_lastSecurityError.isEmpty())
        processSecurityArtifactsUpdate();
    if (error.isEmpty()) {
        clearSecurityAlert(QStringLiteral("security:license-refresh"));
        return;
    }

    raiseSecurityAlert(QStringLiteral("security:license-refresh"),
                       AlertsModel::Critical,
                       tr("Błąd aktualizacji licencji"),
                       error);
}

void Application::handleActivationFingerprintChanged()
{
    processSecurityArtifactsUpdate();
    if (m_updateManager && m_licenseController)
        m_updateManager->setFingerprintOverride(m_licenseController->fingerprint().value(QStringLiteral("hash")).toString());
}

void Application::handleActivationLicensesChanged()
{
    processSecurityArtifactsUpdate();
}

void Application::handleActivationOemLicenseChanged()
{
    processSecurityArtifactsUpdate();
}

void Application::startOfflineAutomation()
{
    if (!m_offlineMode)
        return;
    ensureOfflineBridge();
    if (m_offlineBridge)
        m_offlineBridge->startAutomation();
}

void Application::stopOfflineAutomation()
{
    if (!m_offlineMode)
        return;
    if (m_offlineBridge)
        m_offlineBridge->stopAutomation();
}

bool Application::setDecisionLogPath(const QUrl& url)
{
    QString candidate;
    if (url.isLocalFile())
        candidate = url.toLocalFile();
    else
        candidate = url.toString(QUrl::PreferLocalFile);
    if (candidate.trimmed().isEmpty())
        return false;
    return setDecisionLogPathInternal(candidate, true);
}

bool Application::reloadDecisionLog()
{
    return m_decisionLogModel.reload();
}

QVariantMap Application::licenseRefreshSchedule() const
{
    QVariantMap schedule;
    schedule.insert(QStringLiteral("intervalSeconds"), m_licenseRefreshIntervalSeconds);
    schedule.insert(QStringLiteral("active"), m_licenseRefreshTimer.isActive());
    schedule.insert(QStringLiteral("lastRequestAt"),
                    m_lastLicenseRefreshRequestUtc.isValid()
                        ? m_lastLicenseRefreshRequestUtc.toString(Qt::ISODateWithMs)
                        : QString());
    schedule.insert(QStringLiteral("lastCompletedAt"),
                    m_lastLicenseRefreshUtc.isValid()
                        ? m_lastLicenseRefreshUtc.toString(Qt::ISODateWithMs)
                        : QString());
    schedule.insert(QStringLiteral("nextRefreshDueAt"),
                    m_nextLicenseRefreshUtc.isValid()
                        ? m_nextLicenseRefreshUtc.toString(Qt::ISODateWithMs)
                        : QString());
    double remainingSeconds = -1.0;
    if (m_nextLicenseRefreshUtc.isValid()) {
        const qint64 remainingMs = QDateTime::currentDateTimeUtc().msecsTo(m_nextLicenseRefreshUtc);
        remainingSeconds = remainingMs > 0 ? static_cast<double>(remainingMs) / 1000.0 : 0.0;
    }
    schedule.insert(QStringLiteral("nextRefreshInSeconds"), remainingSeconds);
    return schedule;
}

void Application::handleRiskHistorySnapshotRecorded(const QDateTime& timestamp)
{
    if (m_loadingUiSettings)
        return;

    maybeAutoExportRiskHistory(timestamp);
}

void Application::exposeToQml() {
    m_engine.rootContext()->setContextProperty(QStringLiteral("appController"), this);
    m_engine.rootContext()->setContextProperty(QStringLiteral("ohlcvModel"), &m_ohlcvModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("indicatorSeriesModel"), &m_indicatorModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("signalListModel"), &m_signalModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("marketRegimeTimelineModel"), &m_regimeTimelineModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &m_riskModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskHistoryModel"), &m_riskHistoryModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("alertsModel"), &m_alertsModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("alertsFilterModel"), &m_filteredAlertsModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), m_licenseController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("activationController"), m_activationController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("securityController"), m_securityController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("reportController"), m_reportController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("strategyController"), m_strategyController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("workbenchController"), m_workbenchController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("supportController"), m_supportController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("healthController"), m_healthController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("decisionLogModel"), &m_decisionLogModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("moduleManager"), m_moduleManager.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("moduleViewsModel"), m_moduleViewsModel.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("marketplaceController"), m_marketplaceController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("portfolioController"), m_portfolioController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("configurationWizard"), m_configurationWizard.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("updateManager"), m_updateManager.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("resultsDashboard"), m_resultsDashboard.get());
}

QObject* Application::activationController() const
{
    return m_activationController.get();
}

QObject* Application::reportController() const
{
    return m_reportController.get();
}

QObject* Application::strategyController() const
{
    return m_strategyController.get();
}

QObject* Application::workbenchController() const
{
    return m_workbenchController.get();
}

QObject* Application::supportController() const
{
    return m_supportController.get();
}

QObject* Application::healthController() const
{
    return m_healthController.get();
}

QObject* Application::decisionLogModel() const
{
    return const_cast<DecisionLogModel*>(&m_decisionLogModel);
}

QObject* Application::moduleManager() const
{
    return m_moduleManager.get();
}

QObject* Application::moduleViewsModel() const
{
    return m_moduleViewsModel.get();
}

QObject* Application::marketplaceController() const
{
    return m_marketplaceController.get();
}

QObject* Application::portfolioController() const
{
    return m_portfolioController.get();
}

QObject* Application::configurationWizard() const
{
    return m_configurationWizard.get();
}

QObject* Application::updateManager() const
{
    return m_updateManager.get();
}

QObject* Application::resultsDashboard() const
{
    return m_resultsDashboard.get();
}

void Application::setModuleManagerForTesting(std::unique_ptr<UiModuleManager> manager)
{
    if (manager)
        manager->setParent(this);

    m_moduleManager = std::move(manager);
    if (m_moduleViewsModel)
        m_moduleViewsModel->setModuleManager(m_moduleManager.get());

    if (m_started)
        m_engine.rootContext()->setContextProperty(QStringLiteral("moduleManager"), m_moduleManager.get());
    if (m_started)
        m_engine.rootContext()->setContextProperty(QStringLiteral("moduleViewsModel"), m_moduleViewsModel.get());
}

void Application::ensureFrameMonitor() {
    if (m_frameMonitor)
        return;

    m_frameMonitor = std::make_unique<FrameRateMonitor>(this);

    connect(m_frameMonitor.get(), &FrameRateMonitor::reduceMotionSuggested, this,
            [this](bool enabled) {
                if (m_reduceMotionActive == enabled) {
                    reportReduceMotionTelemetry(enabled);
                    return;
                }
                m_reduceMotionActive = enabled;
                m_pendingReduceMotionState = enabled;  // zapamiętaj do telemetrii
                Q_EMIT reduceMotionActiveChanged();
                reportReduceMotionTelemetry(enabled);
            });

    connect(m_frameMonitor.get(), &FrameRateMonitor::frameSampled, this, [this](double fps) {
        m_latestFpsSample = fps;
        if (m_pendingReduceMotionState.has_value()) {
            const bool pending = m_pendingReduceMotionState.value();
            m_pendingReduceMotionState.reset();
            reportReduceMotionTelemetry(pending);
        }
    });

    connect(m_frameMonitor.get(), &FrameRateMonitor::jankBudgetBreached, this,
            [this](double frameMs, double thresholdMs) { reportJankTelemetry(frameMs, thresholdMs); });

    m_frameMonitor->setPerformanceGuard(m_guard);
    if (m_performanceTelemetry) {
        m_performanceTelemetry->setFrameRateMonitor(m_frameMonitor.get());
    }
}

void Application::attachWindow(QObject* object) {
    auto* window = qobject_cast<QQuickWindow*>(object);
    if (!window && object) {
        window = object->findChild<QQuickWindow*>();
    }
    if (!window)
        return;

    ensureFrameMonitor();
    m_frameMonitor->setWindow(window);
    applyPreferredScreen(window);
    updateScreenInfo(window->screen());
    connect(window, &QQuickWindow::screenChanged, this,
            [this](QScreen* screen) { updateScreenInfo(screen); },
            Qt::UniqueConnection);
}

void Application::setTelemetryReporter(std::unique_ptr<TelemetryReporter> reporter) {
    if (auto* uiReporter = dynamic_cast<UiTelemetryReporter*>(m_telemetry.get())) {
        disconnect(uiReporter, &UiTelemetryReporter::pendingRetryCountChanged,
                   this, &Application::handleTelemetryPendingRetryCountChanged);
    }
    m_telemetry = std::move(reporter);
    updateTelemetryPendingRetryCount(0);
    ensureTelemetry();
    if (m_performanceTelemetry) {
        m_performanceTelemetry->setTelemetryReporter(m_telemetry.get());
    }
}

void Application::setMetricsClientOverrideForTesting(std::shared_ptr<MetricsClientInterface> client)
{
    m_metricsClientOverride = std::move(client);
    ensureTelemetry();
}

void Application::setInProcessDatasetPathForTesting(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = expandPath(normalized);

    if (m_inProcessDatasetPath == normalized)
        return;

    m_inProcessDatasetPath = normalized;
    m_client.setInProcessDatasetPath(m_inProcessDatasetPath);
    if (m_inProcessHealthClient)
        m_inProcessHealthClient->setDatasetPath(m_inProcessDatasetPath);
    if (m_offlineBridge)
        m_offlineBridge->setDatasetPath(m_inProcessDatasetPath);
}

std::shared_ptr<MetricsClientInterface> Application::activeMetricsClientForTesting() const
{
    return m_activeMetricsClient.lock();
}

void Application::notifyOverlayUsage(int activeCount, int allowedCount, bool reduceMotionActive) {
    OverlayState state;
    state.active = qMax(0, activeCount);
    state.allowed = qMax(0, allowedCount);
    state.reduceMotion = reduceMotionActive;
    m_lastOverlayState = state;
    reportOverlayTelemetry();
}

void Application::notifyWindowCount(int totalWindowCount) {
    m_windowCount = qMax(1, totalWindowCount);
    ensureTelemetry();
    if (m_telemetry) {
        m_telemetry->setWindowCount(m_windowCount);
    }
}

QVariantMap Application::instrumentConfigSnapshot() const {
    QVariantMap map;
    map.insert(QStringLiteral("exchange"), m_instrument.exchange);
    map.insert(QStringLiteral("symbol"), m_instrument.symbol);
    map.insert(QStringLiteral("venueSymbol"), m_instrument.venueSymbol);
    map.insert(QStringLiteral("quoteCurrency"), m_instrument.quoteCurrency);
    map.insert(QStringLiteral("baseCurrency"), m_instrument.baseCurrency);
    map.insert(QStringLiteral("granularity"), m_instrument.granularityIso8601);
    return map;
}

QVariantMap Application::performanceGuardSnapshot() const {
    QVariantMap map;
    map.insert(QStringLiteral("fpsTarget"), m_guard.fpsTarget);
    map.insert(QStringLiteral("reduceMotionAfter"), m_guard.reduceMotionAfterSeconds);
    map.insert(QStringLiteral("jankThresholdMs"), m_guard.jankThresholdMs);
    map.insert(QStringLiteral("maxOverlayCount"), m_guard.maxOverlayCount);
    map.insert(QStringLiteral("disableSecondaryWhenBelow"), m_guard.disableSecondaryWhenFpsBelow);
    return map;
}

QVariantMap Application::riskRefreshSnapshot() const
{
    QVariantMap map;
    map.insert(QStringLiteral("enabled"), m_riskRefreshEnabled);
    map.insert(QStringLiteral("intervalSeconds"), static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
    map.insert(QStringLiteral("active"), m_riskRefreshEnabled && m_riskRefreshTimer.isActive());
    map.insert(QStringLiteral("lastRequestAt"),
               m_lastRiskRefreshRequestUtc.isValid()
                   ? m_lastRiskRefreshRequestUtc.toString(Qt::ISODateWithMs)
                   : QString());
    map.insert(QStringLiteral("lastUpdateAt"),
               m_lastRiskUpdateUtc.isValid() ? m_lastRiskUpdateUtc.toString(Qt::ISODateWithMs) : QString());
    map.insert(QStringLiteral("nextRefreshDueAt"),
               m_nextRiskRefreshUtc.isValid() ? m_nextRiskRefreshUtc.toString(Qt::ISODateWithMs) : QString());
    double remainingSeconds = -1.0;
    if (m_nextRiskRefreshUtc.isValid()) {
        const qint64 remainingMs = QDateTime::currentDateTimeUtc().msecsTo(m_nextRiskRefreshUtc);
        remainingSeconds = remainingMs > 0 ? static_cast<double>(remainingMs) / 1000.0 : 0.0;
    }
    map.insert(QStringLiteral("nextRefreshInSeconds"), remainingSeconds);
    return map;
}

bool Application::updateInstrument(const QString& exchange,
                                   const QString& symbol,
                                   const QString& venueSymbol,
                                   const QString& quoteCurrency,
                                   const QString& baseCurrency,
                                   const QString& granularityIso8601) {
    TradingClient::InstrumentConfig config;
    config.exchange = exchange.trimmed().toUpper();
    config.symbol = symbol.trimmed();
    config.venueSymbol = venueSymbol.trimmed();
    config.quoteCurrency = quoteCurrency.trimmed();
    config.baseCurrency = baseCurrency.trimmed();
    config.granularityIso8601 = granularityIso8601.trimmed();

    if (config.exchange.isEmpty() || config.symbol.isEmpty() || config.venueSymbol.isEmpty()
        || config.quoteCurrency.isEmpty() || config.baseCurrency.isEmpty()
        || config.granularityIso8601.isEmpty()) {
        qCWarning(lcAppMetrics) << "Odmowa aktualizacji instrumentu – wymagane pola są puste";
        return false;
    }

    const auto cached = m_tradableInstrumentCache.constFind(config.exchange);
    if (cached != m_tradableInstrumentCache.constEnd() && !cached->isEmpty()) {
        const auto match = std::find_if(cached->cbegin(), cached->cend(), [&](const TradingClient::TradableInstrument& entry) {
            return entry.config.symbol.compare(config.symbol, Qt::CaseInsensitive) == 0
                && entry.config.venueSymbol.compare(config.venueSymbol, Qt::CaseInsensitive) == 0;
        });
        if (match == cached->cend()) {
            qCWarning(lcAppMetrics) << "Odmowa aktualizacji instrumentu – symbol" << config.symbol
                                    << "nie występuje w liście giełdy" << config.exchange;
            return false;
        }
        config.symbol = match->config.symbol;
        config.venueSymbol = match->config.venueSymbol;
        config.quoteCurrency = match->config.quoteCurrency;
        config.baseCurrency = match->config.baseCurrency;
    }

    const bool changed = config.exchange != m_instrument.exchange
        || config.symbol != m_instrument.symbol || config.venueSymbol != m_instrument.venueSymbol
        || config.quoteCurrency != m_instrument.quoteCurrency
        || config.baseCurrency != m_instrument.baseCurrency
        || config.granularityIso8601 != m_instrument.granularityIso8601;

    const bool wasStreaming = m_client.isStreaming();
    if (wasStreaming && !m_offlineMode)
        m_client.stop();

    m_client.setInstrument(config);
    m_instrument = config;
    if (m_offlineBridge)
        m_offlineBridge->setInstrument(m_instrument);
    updateSupportBundleMetadata();
    Q_EMIT instrumentChanged();

    if (wasStreaming && !m_offlineMode)
        m_client.start();

    if (changed && !m_loadingUiSettings)
        scheduleUiSettingsPersist();

    return true;
}

bool Application::updatePerformanceGuard(int fpsTarget,
                                         double reduceMotionAfter,
                                         double jankThresholdMs,
                                         int maxOverlayCount,
                                         int disableSecondaryWhenBelow) {
    PerformanceGuard guard;
    guard.fpsTarget = qMax(1, fpsTarget);
    guard.reduceMotionAfterSeconds = qMax(0.0, reduceMotionAfter);
    guard.jankThresholdMs = qMax(0.0, jankThresholdMs);
    guard.maxOverlayCount = qMax(0, maxOverlayCount);
    guard.disableSecondaryWhenFpsBelow = qMax(0, disableSecondaryWhenBelow);

    const bool changed = guard.fpsTarget != m_guard.fpsTarget
        || !qFuzzyCompare(guard.reduceMotionAfterSeconds + 1.0, m_guard.reduceMotionAfterSeconds + 1.0)
        || !qFuzzyCompare(guard.jankThresholdMs + 1.0, m_guard.jankThresholdMs + 1.0)
        || guard.maxOverlayCount != m_guard.maxOverlayCount
        || guard.disableSecondaryWhenFpsBelow != m_guard.disableSecondaryWhenFpsBelow;

    m_client.setPerformanceGuard(guard);
    m_guard = guard;
    Q_EMIT performanceGuardChanged();

    if (m_frameMonitor)
        m_frameMonitor->setPerformanceGuard(m_guard);

    if (changed && !m_loadingUiSettings)
        scheduleUiSettingsPersist();

    return true;
}

bool Application::updateRiskRefresh(bool enabled, double intervalSeconds)
{
    const bool previousEnabled = m_riskRefreshEnabled;
    const int previousInterval = m_riskRefreshIntervalMs;

    double effectiveInterval = intervalSeconds;
    if (enabled) {
        if (!std::isfinite(intervalSeconds) || intervalSeconds <= 0.0) {
            qCWarning(lcAppMetrics)
                << "Odmowa aktualizacji harmonogramu ryzyka – oczekiwano dodatniego interwału (s), otrzymano"
                << intervalSeconds;
            return false;
        }
    } else if (!std::isfinite(intervalSeconds) || intervalSeconds <= 0.0) {
        effectiveInterval = static_cast<double>(m_riskRefreshIntervalMs) / 1000.0;
    }

    configureRiskRefresh(enabled, effectiveInterval);
    applyRiskRefreshTimerState();

    if (!m_loadingUiSettings
        && (previousEnabled != m_riskRefreshEnabled || previousInterval != m_riskRefreshIntervalMs)) {
        scheduleUiSettingsPersist();
    }

    return true;
}

QVariantList Application::listTradableInstruments(const QString& exchange)
{
    QVariantList result;
    const QString normalized = exchange.trimmed().toUpper();
    if (normalized.isEmpty()) {
        return result;
    }

    const auto instruments = m_client.listTradableInstruments(normalized);
    if (instruments.isEmpty()) {
        m_tradableInstrumentCache.remove(normalized);
        return result;
    }

    m_tradableInstrumentCache.insert(normalized, instruments);
    result.reserve(instruments.size());
    for (const auto& entry : instruments) {
        QVariantMap map;
        map.insert(QStringLiteral("exchange"), entry.config.exchange);
        map.insert(QStringLiteral("symbol"), entry.config.symbol);
        map.insert(QStringLiteral("venueSymbol"), entry.config.venueSymbol);
        map.insert(QStringLiteral("quoteCurrency"), entry.config.quoteCurrency);
        map.insert(QStringLiteral("baseCurrency"), entry.config.baseCurrency);
        const QString label = entry.config.venueSymbol.isEmpty()
            ? entry.config.symbol
            : QStringLiteral("%1 (%2)").arg(entry.config.symbol, entry.config.venueSymbol);
        map.insert(QStringLiteral("label"), label);
        map.insert(QStringLiteral("priceStep"), entry.priceStep);
        map.insert(QStringLiteral("amountStep"), entry.amountStep);
        map.insert(QStringLiteral("minNotional"), entry.minNotional);
        map.insert(QStringLiteral("minAmount"), entry.minAmount);
        map.insert(QStringLiteral("maxAmount"), entry.maxAmount);
        map.insert(QStringLiteral("minPrice"), entry.minPrice);
        map.insert(QStringLiteral("maxPrice"), entry.maxPrice);
        result.append(map);
    }
    return result;
}

bool Application::triggerRiskRefreshNow()
{
    const QDateTime requestTime = QDateTime::currentDateTimeUtc();
    m_lastRiskRefreshRequestUtc = requestTime;
    if (m_offlineMode) {
        if (m_offlineBridge)
            m_offlineBridge->refreshRiskNow();
    } else {
        m_client.refreshRiskState();
    }

    if (m_riskRefreshEnabled && m_started) {
        m_riskRefreshTimer.start(m_riskRefreshIntervalMs);
        m_nextRiskRefreshUtc = requestTime.addMSecs(m_riskRefreshIntervalMs);
    } else if (!m_riskRefreshEnabled) {
        m_nextRiskRefreshUtc = {};
    }

    Q_EMIT riskRefreshScheduleChanged();
    return true;
}

bool Application::updateRiskHistoryLimit(int maximumEntries)
{
    if (maximumEntries < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa aktualizacji limitu historii ryzyka – oczekiwano dodatniej liczby próbek, otrzymano"
            << maximumEntries;
        return false;
    }

    m_riskHistoryModel.setMaximumEntries(maximumEntries);

    return true;
}

void Application::clearRiskHistory()
{
    m_riskHistoryModel.clear();
}

bool Application::exportRiskHistoryToCsv(const QUrl& destination, int limit)
{
    if (limit == 0 || limit < -1) {
        qCWarning(lcAppMetrics)
            << "Nieprawidłowy limit eksportu historii ryzyka:" << limit;
        return false;
    }

    if (limit > 0)
        setRiskHistoryExportLimitValue(limit);

    setRiskHistoryExportLimitEnabled(limit > 0);

    if (!destination.isValid()) {
        qCWarning(lcAppMetrics)
            << "Nieprawidłowy adres docelowy eksportu historii ryzyka:" << destination;
        return false;
    }

    QString path;
    if (destination.isLocalFile() || destination.scheme().isEmpty()) {
        path = destination.isLocalFile() ? destination.toLocalFile() : destination.toString(QUrl::PreferLocalFile);
    } else {
        path = destination.toLocalFile();
    }

    if (path.trimmed().isEmpty()) {
        qCWarning(lcAppMetrics) << "Brak ścieżki docelowej do eksportu historii ryzyka";
        return false;
    }

    QFileInfo info(path);
    if (info.fileName().isEmpty()) {
        qCWarning(lcAppMetrics) << "Ścieżka eksportu historii ryzyka nie zawiera nazwy pliku:" << path;
        return false;
    }

    QDir directory = info.dir();
    if (!directory.exists() && !directory.mkpath(QStringLiteral("."))) {
        qCWarning(lcAppMetrics)
            << "Nie udało się utworzyć katalogu docelowego eksportu historii ryzyka:" << directory.absolutePath();
        return false;
    }

    const bool ok = m_riskHistoryModel.exportToCsv(info.absoluteFilePath(), limit);
    if (!ok) {
        qCWarning(lcAppMetrics) << "Eksport historii ryzyka nie powiódł się do pliku" << info.absoluteFilePath();
        return false;
    }

    setRiskHistoryExportLastDirectory(QUrl::fromLocalFile(info.dir().absolutePath()));

    return true;
}

bool Application::setRiskHistoryExportLimitEnabled(bool enabled)
{
    if (m_riskHistoryExportLimitEnabled == enabled)
        return true;

    m_riskHistoryExportLimitEnabled = enabled;
    Q_EMIT riskHistoryExportLimitEnabledChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryExportLimitValue(int limit)
{
    if (limit < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa ustawienia limitu eksportu historii ryzyka na wartość mniejszą niż 1:" << limit;
        return false;
    }

    if (m_riskHistoryExportLimitValue == limit)
        return true;

    m_riskHistoryExportLimitValue = limit;
    Q_EMIT riskHistoryExportLimitValueChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryExportLastDirectory(const QUrl& directory)
{
    if (!directory.isValid() && !directory.isEmpty())
        return false;

    QString path;
    if (directory.isLocalFile() || directory.scheme().isEmpty()) {
        path = directory.isLocalFile() ? directory.toLocalFile() : directory.toString(QUrl::PreferLocalFile);
    } else {
        return false;
    }

    path = path.trimmed();
    if (path.isEmpty())
        return false;

    const QString normalizedPath = QDir(path).absolutePath();
    const QUrl normalizedUrl = QUrl::fromLocalFile(normalizedPath);

    if (m_riskHistoryExportLastDirectory == normalizedUrl)
        return true;

    m_riskHistoryExportLastDirectory = normalizedUrl;
    m_riskHistoryAutoExportDirectoryWarned = false;
    Q_EMIT riskHistoryExportLastDirectoryChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportEnabled(bool enabled)
{
    if (m_riskHistoryAutoExportEnabled == enabled)
        return true;

    m_riskHistoryAutoExportEnabled = enabled;
    Q_EMIT riskHistoryAutoExportEnabledChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportIntervalMinutes(int minutes)
{
    if (minutes < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa ustawienia interwału auto-eksportu historii ryzyka na wartość mniejszą niż 1 minuta:" << minutes;
        return false;
    }

    if (m_riskHistoryAutoExportIntervalMinutes == minutes)
        return true;

    m_riskHistoryAutoExportIntervalMinutes = minutes;
    Q_EMIT riskHistoryAutoExportIntervalMinutesChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportBasename(const QString& basename)
{
    const QString sanitized = sanitizeAutoExportBasename(basename);
    if (m_riskHistoryAutoExportBasename == sanitized)
        return true;

    m_riskHistoryAutoExportBasename = sanitized;
    Q_EMIT riskHistoryAutoExportBasenameChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportUseLocalTime(bool useLocalTime)
{
    if (m_riskHistoryAutoExportUseLocalTime == useLocalTime)
        return true;

    m_riskHistoryAutoExportUseLocalTime = useLocalTime;
    Q_EMIT riskHistoryAutoExportUseLocalTimeChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRegimeTimelineMaximumSnapshots(int maximumSnapshots)
{
    if (maximumSnapshots < 0) {
        qCWarning(lcAppMetrics)
            << "Odmowa ustawienia limitu próbek reżimu rynku – oczekiwano wartości >= 0, otrzymano"
            << maximumSnapshots;
        return false;
    }

    if (m_regimeTimelineMaximumSnapshots == maximumSnapshots)
        return true;

    m_regimeTimelineModel.setMaximumSnapshots(maximumSnapshots);
    return true;
}

bool Application::setUiTheme(const QString& theme)
{
    QString normalized = theme.trimmed();
    if (normalized.isEmpty())
        return false;
    normalized = normalized.toLower();
    if (normalized != QStringLiteral("dark") && normalized != QStringLiteral("light")
        && normalized != QStringLiteral("midnight")) {
        qCWarning(lcAppMetrics) << "Nieznany motyw UI" << theme;
        normalized = QStringLiteral("dark");
    }

    if (normalized == m_uiTheme)
        return true;

    m_uiTheme = normalized;
    applyUiThemePalette();
    Q_EMIT uiThemeChanged();
    if (!m_loadingUiSettings)
        scheduleUiSettingsPersist();
    return true;
}

bool Application::setUiLayoutMode(const QString& mode)
{
    QString normalized = mode.trimmed();
    if (normalized.isEmpty())
        return false;
    normalized = normalized.toLower();
    if (normalized != QStringLiteral("classic") && normalized != QStringLiteral("compact")
        && normalized != QStringLiteral("advanced")) {
        qCWarning(lcAppMetrics) << "Nieznany układ UI" << mode;
        normalized = QStringLiteral("classic");
    }

    if (normalized == m_uiLayoutMode)
        return true;

    m_uiLayoutMode = normalized;
    Q_EMIT uiLayoutModeChanged();
    if (!m_loadingUiSettings)
        scheduleUiSettingsPersist();
    return true;
}

void Application::setAlertToastsEnabled(bool enabled)
{
    if (m_alertToastsEnabled == enabled)
        return;
    m_alertToastsEnabled = enabled;
    Q_EMIT alertToastsEnabledChanged();
    if (!m_loadingUiSettings)
        scheduleUiSettingsPersist();
}

QVariantList Application::marketplaceListPresets()
{
    QVariantList items;
    const QVector<TradingClient::MarketplacePresetSummary> presets = m_client.listMarketplacePresets();
    items.reserve(presets.size());
    for (const auto& preset : presets) {
        items.append(buildMarketplacePresetVariant(preset));
    }
    return items;
}

QVariantMap Application::marketplaceImportPreset(const QUrl& sourceUrl)
{
    if (!sourceUrl.isValid()) {
        return buildMarketplaceErrorResult(tr("Nieprawidłowa lokalizacja pliku presetu."));
    }

    const QString path = sourceUrl.isLocalFile() ? sourceUrl.toLocalFile() : sourceUrl.toString();
    if (path.trimmed().isEmpty()) {
        return buildMarketplaceErrorResult(tr("Ścieżka pliku presetu jest wymagana."));
    }

    QFile file(expandPath(path));
    if (!file.exists()) {
        return buildMarketplaceErrorResult(tr("Plik %1 nie istnieje.").arg(path));
    }
    if (!file.open(QIODevice::ReadOnly)) {
        return buildMarketplaceErrorResult(tr("Nie udało się otworzyć pliku %1: %2").arg(path, file.errorString()));
    }

    const QByteArray payload = file.readAll();
    file.close();
    const QString filename = QFileInfo(path).fileName();
    const auto summary = m_client.importMarketplacePreset(payload, filename);

    if (summary.presetId.isEmpty()) {
        const QString error = summary.issues.isEmpty()
                                   ? tr("Backend marketplace odrzucił preset.")
                                   : summary.issues.join(QStringLiteral("\n"));
        QVariantMap result = buildMarketplaceErrorResult(error);
        if (!summary.issues.isEmpty()) {
            result.insert(QStringLiteral("issues"), summary.issues);
        }
        return result;
    }

    QVariantMap result;
    result.insert(QStringLiteral("success"), true);
    result.insert(QStringLiteral("preset"), buildMarketplacePresetVariant(summary));
    result.insert(QStringLiteral("sourcePath"), QFileInfo(file.fileName()).absoluteFilePath());
    if (!summary.issues.isEmpty()) {
        result.insert(QStringLiteral("issues"), summary.issues);
    }
    return result;
}

QVariantMap Application::marketplaceExportPreset(const QString& presetId,
                                                 const QString& format,
                                                 const QUrl& destinationUrl)
{
    const QString trimmedId = presetId.trimmed();
    if (trimmedId.isEmpty()) {
        return buildMarketplaceErrorResult(tr("Identyfikator presetu jest wymagany."));
    }

    TradingClient::MarketplacePresetSummary summary;
    QString exportedFilename;
    const QByteArray payload = m_client.exportMarketplacePreset(trimmedId, format, &summary, &exportedFilename);
    if (payload.isEmpty()) {
        const QString error = summary.issues.isEmpty()
                                   ? tr("Serwer nie zwrócił danych presetu.")
                                   : summary.issues.join(QStringLiteral("\n"));
        QVariantMap result = buildMarketplaceErrorResult(error);
        if (!summary.issues.isEmpty()) {
            result.insert(QStringLiteral("issues"), summary.issues);
        }
        return result;
    }

    QString writtenPath;
    const QString suggested = !exportedFilename.trimmed().isEmpty() ? exportedFilename
                              : !summary.sourcePath.trimmed().isEmpty() ? QFileInfo(summary.sourcePath).fileName()
                                                                         : QStringLiteral("marketplace_preset.%1")
                                                                               .arg(format.trimmed().isEmpty()
                                                                                        ? QStringLiteral("yaml")
                                                                                        : format.trimmed());
    if (!writeMarketplacePayloadToFile(payload, suggested, destinationUrl, &writtenPath)) {
        return buildMarketplaceErrorResult(tr("Nie udało się zapisać pliku presetu."));
    }

    QVariantMap result;
    result.insert(QStringLiteral("success"), true);
    result.insert(QStringLiteral("preset"), buildMarketplacePresetVariant(summary));
    result.insert(QStringLiteral("path"), writtenPath);
    if (!summary.issues.isEmpty()) {
        result.insert(QStringLiteral("issues"), summary.issues);
    }
    return result;
}

QVariantMap Application::marketplaceRemovePreset(const QString& presetId)
{
    const QString trimmedId = presetId.trimmed();
    if (trimmedId.isEmpty()) {
        return buildMarketplaceErrorResult(tr("Identyfikator presetu jest wymagany."));
    }

    const bool removed = m_client.removeMarketplacePreset(trimmedId);
    QVariantMap result;
    result.insert(QStringLiteral("success"), removed);
    result.insert(QStringLiteral("presetId"), trimmedId);
    if (!removed) {
        result.insert(QStringLiteral("error"),
                      tr("Nie udało się usunąć presetu %1. Sprawdź logi backendu.").arg(trimmedId));
    }
    return result;
}

QVariantMap Application::marketplaceActivatePreset(const QString& presetId)
{
    const QString trimmedId = presetId.trimmed();
    if (trimmedId.isEmpty()) {
        return buildMarketplaceErrorResult(tr("Identyfikator presetu jest wymagany."));
    }

    const auto summary = m_client.activateMarketplacePreset(trimmedId);
    if (summary.presetId.isEmpty()) {
        const QString error = summary.issues.isEmpty()
                                   ? tr("Aktywacja presetu nie powiodła się.")
                                   : summary.issues.join(QStringLiteral("\n"));
        QVariantMap result = buildMarketplaceErrorResult(error);
        if (!summary.issues.isEmpty()) {
            result.insert(QStringLiteral("issues"), summary.issues);
        }
        return result;
    }

    QVariantMap result;
    result.insert(QStringLiteral("success"), true);
    result.insert(QStringLiteral("preset"), buildMarketplacePresetVariant(summary));
    if (!summary.issues.isEmpty()) {
        result.insert(QStringLiteral("issues"), summary.issues);
    }
    return result;
}

QStringList Application::supportedExchanges() const
{
    return m_supportedExchanges;
}

QVariantMap Application::personalizationSnapshot() const
{
    return buildPersonalizationPayload();
}

void Application::saveUiSettingsImmediatelyForTesting()
{
    if (m_uiSettingsSaveTimer.isActive())
        m_uiSettingsSaveTimer.stop();
    persistUiSettings();
}

void Application::applyTradingTlsEnvironmentOverrides(const QCommandLineParser& parser)
{
    const bool cliTlsEnabled = parser.isSet("grpc-use-mtls");
    if (!cliTlsEnabled) {
        if (const auto tlsEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_GRPC_USE_MTLS")); tlsEnv.has_value())
            m_tradingTlsConfig.enabled = tlsEnv.value();
    }

    auto applyPathFromEnv = [&](const QByteArray& key, const QString& cliValue, QString TradingClient::TlsConfig::*field) {
        if (!cliValue.trimmed().isEmpty())
            return;
        if (const auto value = envValue(key)) {
            const QString trimmed = value->trimmed();
            if (trimmed.compare(QStringLiteral("NONE"), Qt::CaseInsensitive) == 0
                || trimmed.compare(QStringLiteral("NULL"), Qt::CaseInsensitive) == 0) {
                m_tradingTlsConfig.*field = QString();
            } else {
                m_tradingTlsConfig.*field = expandPath(trimmed);
            }
        }
    };

    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_ROOT_CERT"), parser.value("grpc-root-cert"),
                     &TradingClient::TlsConfig::rootCertificatePath);
    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_CLIENT_CERT"), parser.value("grpc-client-cert"),
                     &TradingClient::TlsConfig::clientCertificatePath);
    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_CLIENT_KEY"), parser.value("grpc-client-key"),
                     &TradingClient::TlsConfig::clientKeyPath);

    if (parser.value("tls-pinned-sha256").trimmed().isEmpty()) {
        if (const auto pinnedEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_PINNED_SHA256")); pinnedEnv.has_value()) {
            m_tradingTlsConfig.pinnedServerFingerprint = pinnedEnv->trimmed();
        }
    }

    if (parser.value("grpc-target-name").trimmed().isEmpty()) {
        if (const auto targetEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_TARGET_NAME")))
            m_tradingTlsConfig.targetNameOverride = targetEnv->trimmed();
    }
}

void Application::applyTradingAuthEnvironmentOverrides(const QCommandLineParser& parser,
                                                       bool cliTokenProvided,
                                                       bool cliTokenFileProvided,
                                                       bool cliRoleProvided,
                                                       bool cliScopesProvided,
                                                       QString& tradingToken,
                                                       QString& tradingTokenFile)
{
    Q_UNUSED(parser);

    if (!cliTokenProvided && !cliTokenFileProvided) {
        const auto envToken = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_AUTH_TOKEN"));
        const auto envTokenFile = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_AUTH_TOKEN_FILE"));
        const bool envTokenNonEmpty = envToken.has_value() && !envToken->trimmed().isEmpty();
        const bool envTokenFileNonEmpty = envTokenFile.has_value() && !envTokenFile->trimmed().isEmpty();

        if (envTokenNonEmpty && envTokenFileNonEmpty) {
            qCWarning(lcAppMetrics)
                << "Zmiennie BOT_CORE_UI_GRPC_AUTH_TOKEN oraz BOT_CORE_UI_GRPC_AUTH_TOKEN_FILE są ustawione równocześnie."
                << "Użyję wartości tokenu przekazanej w zmiennej BOT_CORE_UI_GRPC_AUTH_TOKEN.";
        }

        bool applied = false;
        if (envToken.has_value()) {
            const QString trimmed = envToken->trimmed();
            if (!trimmed.isEmpty()) {
                tradingToken = trimmed;
                tradingTokenFile.clear();
                m_tradingAuthToken = trimmed;
                applied = true;
            }
        }

        if (!applied && envTokenFileNonEmpty) {
            const QString expandedPath = expandPath(envTokenFile->trimmed());
            const QString tokenFromFile = readTokenFile(expandedPath, QStringLiteral("TradingService"));
            if (!tokenFromFile.isEmpty()) {
                tradingToken = tokenFromFile;
                tradingTokenFile = expandedPath;
                m_tradingAuthToken = tokenFromFile;
                applied = true;
            }
        }

        if (!applied && ((envToken.has_value() && envToken->trimmed().isEmpty())
                         || (envTokenFile.has_value() && envTokenFile->trimmed().isEmpty()))) {
            tradingToken.clear();
            tradingTokenFile.clear();
            m_tradingAuthToken.clear();
        }
    }

    if (!cliRoleProvided) {
        if (const auto roleEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_RBAC_ROLE")); roleEnv.has_value()) {
            m_tradingRbacRole = roleEnv->trimmed();
        }
    }

    if (!cliScopesProvided) {
        if (const auto scopesEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_RBAC_SCOPES")); scopesEnv.has_value()) {
            const QString trimmed = scopesEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_tradingRbacScopes.clear();
            } else {
                m_tradingRbacScopes = splitScopesList(trimmed);
            }
        }
    }
}

void Application::applyHealthEnvironmentOverrides(const QCommandLineParser& parser,
                                                  bool cliEndpointProvided,
                                                  bool cliTokenProvided,
                                                  bool cliTokenFileProvided,
                                                  bool cliRoleProvided,
                                                  bool cliScopesProvided,
                                                  bool cliIntervalProvided,
                                                  bool cliTlsEnableProvided,
                                                  bool cliTlsDisableProvided,
                                                  bool cliTlsRequireClientAuthProvided,
                                                  bool cliTlsRootProvided,
                                                  bool cliTlsClientCertProvided,
                                                  bool cliTlsClientKeyProvided,
                                                  bool cliTlsServerNameProvided,
                                                  bool cliTlsTargetNameProvided,
                                                  bool cliTlsPinnedProvided,
                                                  QString& healthToken,
                                                  QString& healthTokenFile)
{
    if (!cliTlsEnableProvided && !cliTlsDisableProvided) {
        if (const auto tlsEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_HEALTH_USE_TLS")); tlsEnv.has_value()) {
            m_healthTlsConfig.enabled = tlsEnv.value();
        }
    }

    if (!cliTlsRequireClientAuthProvided) {
        if (const auto requireEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_REQUIRE_CLIENT_AUTH"));
            requireEnv.has_value()) {
            m_healthTlsConfig.requireClientAuth = requireEnv.value();
        }
    }

    auto applyHealthTlsPath = [&](const QByteArray& key, bool cliProvided, QString GrpcTlsConfig::*field) {
        if (cliProvided)
            return;
        if (const auto value = envValue(key)) {
            const QString trimmed = value->trimmed();
            if (trimmed.compare(QStringLiteral("NONE"), Qt::CaseInsensitive) == 0
                || trimmed.compare(QStringLiteral("NULL"), Qt::CaseInsensitive) == 0) {
                m_healthTlsConfig.*field = QString();
            } else {
                m_healthTlsConfig.*field = expandPath(trimmed);
            }
        }
    };

    applyHealthTlsPath(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_ROOT_CERT"),
                       cliTlsRootProvided,
                       &GrpcTlsConfig::rootCertificatePath);
    applyHealthTlsPath(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_CLIENT_CERT"),
                       cliTlsClientCertProvided,
                       &GrpcTlsConfig::clientCertificatePath);
    applyHealthTlsPath(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_CLIENT_KEY"),
                       cliTlsClientKeyProvided,
                       &GrpcTlsConfig::clientKeyPath);

    if (!cliTlsServerNameProvided) {
        if (const auto serverEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_SERVER_NAME")); serverEnv.has_value()) {
            m_healthTlsConfig.serverNameOverride = serverEnv->trimmed();
        }
    }

    if (!cliTlsTargetNameProvided) {
        if (const auto targetEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_TARGET_NAME")); targetEnv.has_value()) {
            m_healthTlsConfig.targetNameOverride = targetEnv->trimmed();
        }
    }

    if (!cliTlsPinnedProvided) {
        if (const auto pinnedEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_TLS_PINNED_SHA256")); pinnedEnv.has_value()) {
            m_healthTlsConfig.pinnedServerFingerprint = pinnedEnv->trimmed();
        }
    }

    if (!cliEndpointProvided) {
        if (const auto endpointEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_ENDPOINT")); endpointEnv.has_value()) {
            m_healthEndpoint = endpointEnv->trimmed();
        }
    }

    if (!cliTokenProvided && !cliTokenFileProvided) {
        const auto envToken = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_AUTH_TOKEN"));
        const auto envTokenFile = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_AUTH_TOKEN_FILE"));
        const bool envTokenNonEmpty = envToken.has_value() && !envToken->trimmed().isEmpty();
        const bool envTokenFileNonEmpty = envTokenFile.has_value() && !envTokenFile->trimmed().isEmpty();

        if (envTokenNonEmpty && envTokenFileNonEmpty) {
            qCWarning(lcAppMetrics)
                << "Zmiennie BOT_CORE_UI_HEALTH_AUTH_TOKEN oraz BOT_CORE_UI_HEALTH_AUTH_TOKEN_FILE są ustawione jednocześnie."
                << "Użyję tokenu z BOT_CORE_UI_HEALTH_AUTH_TOKEN.";
        }

        bool applied = false;
        if (envToken.has_value()) {
            const QString trimmed = envToken->trimmed();
            if (!trimmed.isEmpty()) {
                healthToken = trimmed;
                healthTokenFile.clear();
                applied = true;
            }
        }

        if (!applied && envTokenFileNonEmpty) {
            const QString expandedPath = expandPath(envTokenFile->trimmed());
            const QString tokenFromFile = readTokenFile(expandedPath, QStringLiteral("HealthService"));
            if (!tokenFromFile.isEmpty()) {
                healthToken = tokenFromFile;
                healthTokenFile = expandedPath;
                applied = true;
            }
        }

        if (!applied && ((envToken.has_value() && envToken->trimmed().isEmpty())
                         || (envTokenFile.has_value() && envTokenFile->trimmed().isEmpty()))) {
            healthToken.clear();
            healthTokenFile.clear();
        }
    }

    if (!cliRoleProvided) {
        if (const auto roleEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_RBAC_ROLE")); roleEnv.has_value()) {
            m_healthRbacRole = roleEnv->trimmed();
        }
    }

    if (!cliScopesProvided) {
        if (const auto scopesEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_RBAC_SCOPES")); scopesEnv.has_value()) {
            const QString trimmed = scopesEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_healthRbacScopes.clear();
            } else {
                m_healthRbacScopes = splitScopesList(trimmed);
            }
        }
    }

    if (!cliIntervalProvided) {
        if (const auto intervalEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_HEALTH_REFRESH_SECONDS")); intervalEnv.has_value()) {
            bool ok = false;
            const double candidate = intervalEnv->toDouble(&ok);
            if (ok && candidate > 0.0) {
                m_healthRefreshIntervalSeconds = static_cast<int>(candidate);
            } else {
                qCWarning(lcAppMetrics)
                    << "Nieprawidłowa wartość BOT_CORE_UI_HEALTH_REFRESH_SECONDS:" << *intervalEnv
                    << "– oczekiwano liczby dodatniej.";
            }
        }
    }

    if (!parser.isSet("health-disable-auto-refresh")) {
        if (const auto autoEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_HEALTH_AUTO_REFRESH")); autoEnv.has_value()) {
            m_healthAutoRefreshEnabled = autoEnv.value();
        }
    }
}

void Application::ingestFpsSampleForTesting(double fps) {
    m_latestFpsSample = fps;
    if (m_pendingReduceMotionState.has_value()) {
        const bool pending = m_pendingReduceMotionState.value();
        m_pendingReduceMotionState.reset();
        reportReduceMotionTelemetry(pending);
    }
}

void Application::setReduceMotionStateForTesting(bool active) {
    m_reduceMotionActive = active;
    reportReduceMotionTelemetry(active);
}

void Application::simulateFrameIntervalForTesting(double seconds) {
    ensureFrameMonitor();
    if (!m_frameMonitor)
        return;
    m_frameMonitor->simulateFrameIntervalForTest(seconds);
}

void Application::startRiskRefreshTimerForTesting()
{
    m_started = true;
    applyRiskRefreshTimerState();
}

void Application::setLastRiskHistoryAutoExportForTesting(const QDateTime& timestamp)
{
    if (timestamp.isValid())
        m_lastRiskHistoryAutoExportUtc = timestamp.toUTC();
    else
        m_lastRiskHistoryAutoExportUtc = {};
    Q_EMIT riskHistoryLastAutoExportAtChanged();
}

void Application::setRiskHistoryAutoExportLastPathForTesting(const QUrl& url)
{
    m_lastRiskHistoryAutoExportPath = url;
    Q_EMIT riskHistoryLastAutoExportPathChanged();
}

void Application::setTradableInstrumentsForTesting(const QString& exchange,
                                                   const QVector<TradingClient::TradableInstrument>& items)
{
    m_tradableInstrumentCache.insert(exchange.trimmed().toUpper(), items);
}

void Application::applyMetricsEnvironmentOverrides(const QCommandLineParser& parser,
                                                    bool cliTokenProvided,
                                                    bool cliTokenFileProvided,
                                                    QString& metricsToken,
                                                    QString& metricsTokenFile) {
    if (const auto endpointEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_ENDPOINT"));
        endpointEnv.has_value()) {
        applyMetricsEndpoint(endpointEnv->trimmed());
    }

    if (const auto tagEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_TAG")); tagEnv.has_value()) {
        m_metricsTag = tagEnv->trimmed();
    }

    const bool disableFlagsProvided = parser.isSet("disable-metrics") || parser.isSet("no-metrics");
    if (!disableFlagsProvided) {
        if (const auto enabledEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_ENABLED")); enabledEnv.has_value()) {
            m_metricsEnabled = enabledEnv.value();
        }
        if (const auto disabledEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_DISABLED")); disabledEnv.has_value()) {
            m_metricsEnabled = !disabledEnv.value();
        }
    }

    bool envTlsExplicit = false;
    if (!parser.isSet("metrics-use-tls")) {
        if (const auto tlsEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_USE_TLS")); tlsEnv.has_value()) {
            m_tlsConfig.enabled = tlsEnv.value();
            envTlsExplicit = true;
        }
    }

    bool tlsMaterialProvided = false;
    auto overrideString = [&](QString TelemetryTlsConfig::*field, const QByteArray& key) {
        if (const auto value = envValue(key); value.has_value()) {
            const QString trimmed = value->trimmed();
            if (trimmed.isEmpty()) {
                m_tlsConfig.*field = QString();
            } else {
                m_tlsConfig.*field = expandPath(trimmed);
                tlsMaterialProvided = true;
            }
        }
    };

    overrideString(&TelemetryTlsConfig::rootCertificatePath, QByteArrayLiteral("BOT_CORE_UI_METRICS_ROOT_CERT"));
    overrideString(&TelemetryTlsConfig::clientCertificatePath, QByteArrayLiteral("BOT_CORE_UI_METRICS_CLIENT_CERT"));
    overrideString(&TelemetryTlsConfig::clientKeyPath, QByteArrayLiteral("BOT_CORE_UI_METRICS_CLIENT_KEY"));
    overrideString(&TelemetryTlsConfig::serverNameOverride, QByteArrayLiteral("BOT_CORE_UI_METRICS_SERVER_NAME"));

    if (const auto shaEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_SERVER_SHA256")); shaEnv.has_value()) {
        m_tlsConfig.pinnedServerSha256 = shaEnv->trimmed();
        if (!m_tlsConfig.pinnedServerSha256.isEmpty())
            tlsMaterialProvided = true;
    }

    if (tlsMaterialProvided && !m_tlsConfig.enabled && !envTlsExplicit && !parser.isSet("metrics-use-tls")) {
        qCDebug(lcAppMetrics) << "Wymuszam TLS dla telemetrii na podstawie zmiennych środowiskowych.";
        m_tlsConfig.enabled = true;
    }

    if (!cliTokenProvided && !cliTokenFileProvided) {
        const auto envToken = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_AUTH_TOKEN"));
        const auto envTokenFile = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE"));

        const bool envTokenNonEmpty = envToken.has_value() && !envToken->trimmed().isEmpty();
        const bool envTokenFileNonEmpty = envTokenFile.has_value() && !envTokenFile->trimmed().isEmpty();

        if (envTokenNonEmpty && envTokenFileNonEmpty) {
            qCWarning(lcAppMetrics)
                << "Zmiennie BOT_CORE_UI_METRICS_AUTH_TOKEN oraz BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE są ustawione równocześnie."
                << "Użyję wartości tokenu przekazanej w zmiennej BOT_CORE_UI_METRICS_AUTH_TOKEN.";
        }

        bool applied = false;
        if (envToken.has_value()) {
            const QString trimmed = envToken->trimmed();
            if (!trimmed.isEmpty()) {
                metricsToken = trimmed;
                metricsTokenFile.clear();
                applied = true;
            }
        }

        if (!applied && envTokenFileNonEmpty) {
            const QString expandedPath = expandPath(envTokenFile->trimmed());
            const QString tokenFromFile = readTokenFile(expandedPath);
            if (!tokenFromFile.isEmpty()) {
                metricsToken = tokenFromFile;
                metricsTokenFile = expandedPath;
                applied = true;
            }
        }

        if (!applied && ((envToken.has_value() && envToken->trimmed().isEmpty()) ||
                         (envTokenFile.has_value() && envTokenFile->trimmed().isEmpty()))) {
            metricsToken.clear();
            metricsTokenFile.clear();
        }
    }

    if (m_metricsRbacRole.trimmed().isEmpty()) {
        if (const auto roleEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_RBAC_ROLE")); roleEnv.has_value()) {
            m_metricsRbacRole = roleEnv->trimmed();
        }
    }
}

void Application::applyMetricsEndpoint(const QString& endpoint) {
    QString sanitized = endpoint.trimmed();
    if (sanitized.isEmpty())
        sanitized = QStringLiteral("in-process");
    if (m_metricsEndpoint == sanitized)
        return;
    m_metricsEndpoint = sanitized;
    Q_EMIT metricsEndpointChanged();
}

void Application::configureTokenWatcher(QFileSystemWatcher& watcher,
                                       QString& trackedFile,
                                       QStringList& trackedDirs,
                                       const QString& filePath,
                                       const char* label)
{
    if (!trackedFile.isEmpty())
        watcher.removePath(trackedFile);
    for (const QString& dir : std::as_const(trackedDirs))
        watcher.removePath(dir);

    trackedFile.clear();
    trackedDirs.clear();

    if (filePath.trimmed().isEmpty())
        return;

    QFileInfo info(filePath);
    trackedFile = info.absoluteFilePath();
    if (QFile::exists(trackedFile)) {
        if (!watcher.addPath(trackedFile)) {
            qCWarning(lcAppMetrics) << "Nie można obserwować pliku" << trackedFile << "dla" << label;
        }
    }

    const QStringList directories = watchableDirectories(info.absolutePath());
    for (const QString& directory : directories) {
        if (directory.isEmpty())
            continue;
        if (trackedDirs.contains(directory))
            continue;
        if (!watcher.addPath(directory)) {
            qCWarning(lcAppMetrics) << "Nie można obserwować katalogu" << directory << "dla" << label;
            continue;
        }
        trackedDirs.append(directory);
    }
}

bool Application::applyRegimeThresholdPath(const QString& path, bool warnIfMissing)
{
    const QString trimmed = path.trimmed();
    const QString expanded = expandPath(trimmed);

    QString normalized;
    if (!expanded.trimmed().isEmpty()) {
        QFileInfo info(expanded);
        normalized = QDir::cleanPath(info.absoluteFilePath());
    }

    if (normalized.isEmpty()) {
        configureTokenWatcher(m_regimeThresholdWatcher,
                              m_regimeThresholdWatcherFile,
                              m_regimeThresholdWatcherDirs,
                              QString(),
                              "MarketRegime thresholds");
        m_regimeThresholdPath.clear();
        m_client.setRegimeThresholdsPath(QString());
        return true;
    }

    const bool existed = QFile::exists(normalized);
    const bool pathChanged = normalized != m_regimeThresholdPath;

    configureTokenWatcher(m_regimeThresholdWatcher,
                          m_regimeThresholdWatcherFile,
                          m_regimeThresholdWatcherDirs,
                          normalized,
                          "MarketRegime thresholds");

    if (!pathChanged) {
        if (!existed && warnIfMissing) {
            qCWarning(lcAppMetrics)
                << "Plik progów reżimu rynku nie istnieje:" << normalized;
        }
        m_client.reloadRegimeThresholds();
        return existed;
    }

    m_regimeThresholdPath = normalized;
    if (!existed && warnIfMissing) {
        qCWarning(lcAppMetrics)
            << "Plik progów reżimu rynku nie istnieje:" << normalized;
    }

    m_client.setRegimeThresholdsPath(normalized);
    return existed;
}

void Application::configureTlsWatcher(QFileSystemWatcher& watcher,
                                      QStringList& trackedFiles,
                                      QStringList& trackedDirs,
                                      const QStringList& filePaths,
                                      const char* label)
{
    for (const QString& path : std::as_const(trackedFiles))
        watcher.removePath(path);
    for (const QString& path : std::as_const(trackedDirs))
        watcher.removePath(path);

    trackedFiles.clear();
    trackedDirs.clear();

    for (const QString& rawPath : filePaths) {
        const QString trimmed = rawPath.trimmed();
        if (trimmed.isEmpty())
            continue;

        const QString expanded = expandPath(trimmed);
        QFileInfo info(expanded);
        const QString absoluteFile = info.absoluteFilePath();
        if (QFile::exists(absoluteFile)) {
            if (!trackedFiles.contains(absoluteFile)) {
                if (!watcher.addPath(absoluteFile)) {
                    qCWarning(lcAppMetrics) << "Nie można obserwować pliku TLS" << absoluteFile << "dla" << label;
                }
                trackedFiles.append(absoluteFile);
            }
        }

        const QStringList directories = watchableDirectories(info.absolutePath());
        for (const QString& directory : directories) {
            if (directory.isEmpty())
                continue;
            if (trackedDirs.contains(directory))
                continue;
            if (!watcher.addPath(directory)) {
                qCWarning(lcAppMetrics) << "Nie można obserwować katalogu TLS" << directory << "dla" << label;
                continue;
            }
            trackedDirs.append(directory);
        }
    }
}

void Application::configureTradingTlsWatchers()
{
    const QStringList files{
        m_tradingTlsConfig.rootCertificatePath,
        m_tradingTlsConfig.clientCertificatePath,
        m_tradingTlsConfig.clientKeyPath
    };
    configureTlsWatcher(m_tradingTlsWatcher,
                        m_tradingTlsWatcherFiles,
                        m_tradingTlsWatcherDirs,
                        files,
                        "TradingService TLS");
}

void Application::configureMetricsTlsWatchers()
{
    const QStringList files{
        m_tlsConfig.rootCertificatePath,
        m_tlsConfig.clientCertificatePath,
        m_tlsConfig.clientKeyPath
    };
    configureTlsWatcher(m_metricsTlsWatcher,
                        m_metricsTlsWatcherFiles,
                        m_metricsTlsWatcherDirs,
                        files,
                        "MetricsService TLS");
}

void Application::applyMarketplaceEnvironmentOverrides(const QCommandLineParser& parser)
{
    if (!m_marketplaceController)
        return;

    auto applyPathEnv = [&](const QByteArray& envKey, const char* cliOption, auto setter) {
        if (parser.isSet(cliOption))
            return;
        const auto value = envValue(envKey);
        if (!value.has_value())
            return;
        const QString trimmed = value->trimmed();
        if (trimmed.isEmpty()) {
            setter(QString());
            return;
        }
        setter(expandPath(trimmed));
    };

    applyPathEnv(kMarketplaceBridgeEnv,
                 "marketplace-bridge",
                 [&](const QString& path) { m_marketplaceController->setBridgeScriptPath(path); });
    applyPathEnv(kMarketplacePresetsDirEnv,
                 "marketplace-presets-dir",
                 [&](const QString& path) { m_marketplaceController->setPresetsDirectory(path); });
    applyPathEnv(kMarketplaceLicensesPathEnv,
                 "marketplace-licenses-path",
                 [&](const QString& path) { m_marketplaceController->setLicensesPath(path); });

    if (!parser.isSet("marketplace-signing-key")) {
        const auto value = envValue(kMarketplaceSigningKeysEnv);
        if (value.has_value()) {
            const QStringList keys = parseMarketplaceList(*value);
            m_marketplaceController->setSigningKeys(keys);
        }
    }

    if (!parser.isSet("marketplace-signing-key-file")) {
        const auto value = envValue(kMarketplaceSigningKeyFilesEnv);
        if (value.has_value()) {
            QStringList files;
            for (const QString& entry : parseMarketplaceList(*value)) {
                if (entry.isEmpty())
                    continue;
                files.append(expandPath(entry));
            }
            m_marketplaceController->setSigningKeyFiles(files);
        }
    }

    if (!parser.isSet("marketplace-fingerprint")) {
        if (const auto value = envValue(kMarketplaceFingerprintEnv); value.has_value())
            m_marketplaceController->setFingerprintOverride(value->trimmed());
    }
}

void Application::configureHealthTlsWatchers()
{
    const QStringList files{
        m_healthTlsConfig.rootCertificatePath,
        m_healthTlsConfig.clientCertificatePath,
        m_healthTlsConfig.clientKeyPath
    };
    configureTlsWatcher(m_healthTlsWatcher,
                        m_healthTlsWatcherFiles,
                        m_healthTlsWatcherDirs,
                        files,
                        "HealthService TLS");
}

void Application::setTradingAuthTokenFile(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = expandPath(normalized);
    if (m_tradingAuthTokenFile == normalized)
        return;
    m_tradingAuthTokenFile = normalized;
    configureTokenWatcher(m_tradingTokenWatcher,
                          m_tradingTokenWatcherFile,
                          m_tradingTokenWatcherDirs,
                          m_tradingAuthTokenFile,
                          "TradingService");
    if (!m_tradingAuthTokenFile.isEmpty()) {
        reloadTradingTokenFromFile();
    }
}

void Application::setMetricsAuthTokenFile(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = expandPath(normalized);
    if (m_metricsAuthTokenFile == normalized)
        return;
    m_metricsAuthTokenFile = normalized;
    configureTokenWatcher(m_metricsTokenWatcher,
                          m_metricsTokenWatcherFile,
                          m_metricsTokenWatcherDirs,
                          m_metricsAuthTokenFile,
                          "MetricsService");
    if (!m_metricsAuthTokenFile.isEmpty()) {
        reloadMetricsTokenFromFile();
    } else {
        ensureTelemetry();
    }
}

void Application::setHealthAuthTokenFile(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = expandPath(normalized);
    if (m_healthAuthTokenFile == normalized)
        return;
    m_healthAuthTokenFile = normalized;
    configureTokenWatcher(m_healthTokenWatcher,
                          m_healthTokenWatcherFile,
                          m_healthTokenWatcherDirs,
                          m_healthAuthTokenFile,
                          "HealthService");
    if (!m_healthAuthTokenFile.isEmpty()) {
        reloadHealthTokenFromFile();
    } else {
        applyHealthAuthTokenToController();
    }
}

void Application::reloadTradingTokenFromFile()
{
    if (m_tradingAuthTokenFile.trimmed().isEmpty())
        return;

    const QString token = readTokenFile(m_tradingAuthTokenFile, QStringLiteral("TradingService"));
    if (token == m_tradingAuthToken)
        return;

    m_tradingAuthToken = token.trimmed();
    m_client.setAuthToken(m_tradingAuthToken);
}

void Application::reloadMetricsTokenFromFile()
{
    if (m_metricsAuthTokenFile.trimmed().isEmpty())
        return;

    const QString token = readTokenFile(m_metricsAuthTokenFile);
    if (token == m_metricsAuthToken)
        return;

    m_metricsAuthToken = token.trimmed();
    ensureTelemetry();
}

void Application::reloadHealthTokenFromFile()
{
    if (m_healthAuthTokenFile.trimmed().isEmpty())
        return;

    const QString token = readTokenFile(m_healthAuthTokenFile, QStringLiteral("HealthService"));
    if (token == m_healthAuthToken)
        return;

    m_healthAuthToken = token.trimmed();
    applyHealthAuthTokenToController();
}

void Application::updateUiModuleWatchTargets(const QStringList& directories, const QStringList& pluginFiles)
{
    const QStringList currentDirs = m_uiModuleWatcher.directories();
    if (!currentDirs.isEmpty())
        m_uiModuleWatcher.removePaths(currentDirs);
    const QStringList currentFiles = m_uiModuleWatcher.files();
    if (!currentFiles.isEmpty())
        m_uiModuleWatcher.removePaths(currentFiles);

    m_watchedUiModuleDirectories.clear();
    m_watchedUiModuleFiles.clear();

    auto appendUnique = [](QStringList& list, const QString& value) {
        if (value.isEmpty())
            return;
        if (!list.contains(value))
            list.append(value);
    };

    auto appendWatchableDirectories = [&](const QString& basePath) {
        const QStringList candidates = watchableDirectories(basePath);
        for (const QString& candidate : candidates)
            appendUnique(directoryTargets, candidate);
    };

    QStringList directoryTargets;
    QStringList fileTargets;

    for (const QString& entry : directories) {
        const QString trimmed = entry.trimmed();
        if (trimmed.isEmpty())
            continue;
        QFileInfo info(trimmed);
        if (info.exists() && info.isFile()) {
            appendUnique(fileTargets, info.absoluteFilePath());
            appendWatchableDirectories(info.absolutePath());
            continue;
        }

        appendWatchableDirectories(info.absoluteFilePath());
        if (!info.exists())
            appendWatchableDirectories(info.absolutePath());
    }

    for (const QString& plugin : pluginFiles) {
        const QString trimmed = plugin.trimmed();
        if (trimmed.isEmpty())
            continue;
        QFileInfo info(trimmed);
        if (!info.exists()) {
            appendWatchableDirectories(info.absolutePath());
            continue;
        }
        appendUnique(fileTargets, info.absoluteFilePath());
        appendWatchableDirectories(info.absolutePath());
    }

    if (!directoryTargets.isEmpty()) {
        const QStringList failures = m_uiModuleWatcher.addPaths(directoryTargets);
        QStringList filteredDirs = directoryTargets;
        for (const QString& failed : failures) {
            filteredDirs.removeAll(failed);
            qCWarning(lcAppMetrics) << "Nie można obserwować katalogu modułów UI" << failed;
        }
        directoryTargets = filteredDirs;
    }

    if (!fileTargets.isEmpty()) {
        const QStringList failures = m_uiModuleWatcher.addPaths(fileTargets);
        QStringList filteredFiles = fileTargets;
        for (const QString& failed : failures) {
            filteredFiles.removeAll(failed);
            qCWarning(lcAppMetrics) << "Nie można obserwować pliku modułu UI" << failed;
        }
        fileTargets = filteredFiles;
    }

    m_watchedUiModuleDirectories = directoryTargets;
    m_watchedUiModuleFiles = fileTargets;
}

void Application::applyHealthAuthTokenToController()
{
    if (!m_healthController)
        return;

    QString effective = m_healthAuthToken.trimmed();
    if (effective.isEmpty())
        effective = m_tradingAuthToken;
    m_healthController->setAuthToken(effective.trimmed());
}

void Application::applyTradingTlsConfig()
{
    ++m_tradingTlsReloadGeneration;
    m_client.setTlsConfig(m_tradingTlsConfig);
}

void Application::applyMetricsTlsConfig()
{
    ++m_metricsTlsReloadGeneration;
    ensureTelemetry();
}

void Application::applyHealthTlsConfig()
{
    ++m_healthTlsReloadGeneration;
    if (m_healthController)
        m_healthController->setTlsConfig(m_healthTlsConfig);
}

void Application::handleTradingTlsPathChanged(const QString&)
{
    configureTradingTlsWatchers();
    applyTradingTlsConfig();
}

void Application::handleMetricsTlsPathChanged(const QString&)
{
    configureMetricsTlsWatchers();
    applyMetricsTlsConfig();
}

void Application::handleHealthTlsPathChanged(const QString&)
{
    configureHealthTlsWatchers();
    applyHealthTlsConfig();
}

void Application::handleRegimeThresholdPathChanged(const QString&)
{
    if (m_regimeThresholdPath.isEmpty())
        return;

    applyRegimeThresholdPath(m_regimeThresholdPath, false);
}

QVariantMap Application::buildMarketplacePresetVariant(const TradingClient::MarketplacePresetSummary& preset) const
{
    QVariantMap map;
    map.insert(QStringLiteral("presetId"), preset.presetId);
    map.insert(QStringLiteral("name"), preset.name);
    map.insert(QStringLiteral("version"), preset.version);
    map.insert(QStringLiteral("profile"), preset.profile);
    map.insert(QStringLiteral("tags"), preset.tags);
    map.insert(QStringLiteral("signatureVerified"), preset.signatureVerified);
    map.insert(QStringLiteral("issues"), preset.issues);
    map.insert(QStringLiteral("sourcePath"), preset.sourcePath);
    return map;
}

QVariantMap Application::buildMarketplaceErrorResult(const QString& message) const
{
    QVariantMap result;
    result.insert(QStringLiteral("success"), false);
    result.insert(QStringLiteral("error"), message);
    return result;
}

bool Application::writeMarketplacePayloadToFile(const QByteArray& payload,
                                                const QString& suggestedName,
                                                const QUrl& destinationUrl,
                                                QString* writtenPath) const
{
    if (writtenPath) {
        writtenPath->clear();
    }
    if (payload.isEmpty()) {
        return false;
    }

    QString targetPath;
    if (destinationUrl.isValid() && !destinationUrl.isEmpty()) {
        if (!destinationUrl.isLocalFile() && !destinationUrl.scheme().isEmpty()
            && destinationUrl.scheme() != QStringLiteral("file")) {
            return false;
        }
        targetPath = destinationUrl.isLocalFile() ? destinationUrl.toLocalFile() : destinationUrl.path();
    }

    targetPath = expandPath(targetPath.trimmed());
    if (targetPath.isEmpty()) {
        const QString fallbackName = suggestedName.trimmed().isEmpty() ? QStringLiteral("marketplace_preset.yaml")
                                                                       : suggestedName.trimmed();
        targetPath = QDir::current().absoluteFilePath(fallbackName);
    } else if (QFileInfo(targetPath).isDir()) {
        const QString fallbackName = suggestedName.trimmed().isEmpty() ? QStringLiteral("marketplace_preset.yaml")
                                                                       : suggestedName.trimmed();
        targetPath = QDir(targetPath).absoluteFilePath(fallbackName);
    }

    if (targetPath.trimmed().isEmpty()) {
        return false;
    }

    QDir targetDir = QFileInfo(targetPath).absoluteDir();
    if (!targetDir.exists() && !targetDir.mkpath(QStringLiteral("."))) {
        return false;
    }

    QSaveFile outFile(targetPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        return false;
    }
    if (outFile.write(payload) != payload.size()) {
        return false;
    }
    if (!outFile.commit()) {
        return false;
    }

    if (writtenPath) {
        *writtenPath = QFileInfo(targetPath).absoluteFilePath();
    }
    return true;
}

void Application::handleTradingTokenPathChanged(const QString&)
{
    configureTokenWatcher(m_tradingTokenWatcher,
                          m_tradingTokenWatcherFile,
                          m_tradingTokenWatcherDirs,
                          m_tradingAuthTokenFile,
                          "TradingService");
    reloadTradingTokenFromFile();
}

void Application::handleMetricsTokenPathChanged(const QString&)
{
    configureTokenWatcher(m_metricsTokenWatcher,
                          m_metricsTokenWatcherFile,
                          m_metricsTokenWatcherDirs,
                          m_metricsAuthTokenFile,
                          "MetricsService");
    reloadMetricsTokenFromFile();
}

void Application::handleHealthTokenPathChanged(const QString&)
{
    configureTokenWatcher(m_healthTokenWatcher,
                          m_healthTokenWatcherFile,
                          m_healthTokenWatcherDirs,
                          m_healthAuthTokenFile,
                          "HealthService");
    reloadHealthTokenFromFile();
}

void Application::applyScreenEnvironmentOverrides(const QCommandLineParser& parser)
{
    if (!parser.isSet("screen-name")) {
        if (const auto nameEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_SCREEN_NAME"));
            nameEnv.has_value()) {
            const QString trimmed = nameEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_preferredScreenName.clear();
            } else {
                m_preferredScreenName = trimmed;
            }
        }
    }

    if (!parser.isSet("screen-index")) {
        if (const auto indexEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_SCREEN_INDEX"));
            indexEnv.has_value()) {
            const QString trimmed = indexEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_preferredScreenIndex = -1;
            } else {
                bool ok = false;
                const int value = trimmed.toInt(&ok);
                if (ok && value >= 0) {
                    m_preferredScreenIndex = value;
                } else {
                    qCWarning(lcAppMetrics)
                        << "Nieprawidłowa wartość" << trimmed
                        << "w zmiennej BOT_CORE_UI_SCREEN_INDEX – oczekiwano liczby całkowitej >= 0.";
                }
            }
        }
    }

    if (!parser.isSet("primary-screen")) {
        if (const auto primaryEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_SCREEN_PRIMARY"));
            primaryEnv.has_value()) {
            m_forcePrimaryScreen = primaryEnv.value();
        }
    }
}

void Application::applyPreferredScreen(QQuickWindow* window)
{
    if (!window)
    {
        updateScreenInfo(nullptr);
        return;
    }

    QScreen* target = resolvePreferredScreen();
    if (target && window->screen() != target) {
        window->setScreen(target);
        const QRect geometry = target->geometry();
        const QPoint desired = geometry.center() - QPoint(window->width() / 2, window->height() / 2);
        window->setPosition(desired);
    }

    QScreen* effectiveScreen = window->screen();
    if (!effectiveScreen)
        effectiveScreen = target;
    updateScreenInfo(effectiveScreen);
}

QScreen* Application::resolvePreferredScreen() const
{
    const QList<QScreen*> screens = QGuiApplication::screens();
    if (screens.isEmpty())
        return nullptr;

    if (m_forcePrimaryScreen) {
        if (auto* primary = QGuiApplication::primaryScreen())
            return primary;
    }

    if (m_preferredScreenIndex >= 0) {
        if (m_preferredScreenIndex < screens.size()) {
            return screens.at(m_preferredScreenIndex);
        }
        if (m_preferredScreenConfigured && !m_screenWarningLogged) {
            qCWarning(lcAppMetrics)
                << "Żądany indeks ekranu" << m_preferredScreenIndex
                << "wykracza poza dostępne ekrany (liczba ekranów:" << screens.size() << ").";
            m_screenWarningLogged = true;
        }
    }

    if (!m_preferredScreenName.trimmed().isEmpty()) {
        const QString normalized = m_preferredScreenName.trimmed().toLower();
        for (QScreen* screen : screens) {
            if (!screen)
                continue;
            const QString screenName = screen->name();
            if (screenName.compare(normalized, Qt::CaseInsensitive) == 0
                || screenName.toLower().contains(normalized)) {
                return screen;
            }
        }
        if (m_preferredScreenConfigured && !m_screenWarningLogged) {
            qCWarning(lcAppMetrics)
                << "Nie znaleziono ekranu o nazwie zawierającej" << m_preferredScreenName << '.';
            m_screenWarningLogged = true;
        }
    }

    if (m_forcePrimaryScreen) {
        if (auto* primary = QGuiApplication::primaryScreen())
            return primary;
    }

    return nullptr;
}

void Application::applyPreferredScreenForTesting(QQuickWindow* window)
{
    applyPreferredScreen(window);
}

QScreen* Application::pickPreferredScreenForTesting() const
{
    return resolvePreferredScreen();
}

void Application::updateScreenInfo(QScreen* screen)
{
    if (!screen) {
        m_screenInfo.reset();
        if (m_telemetry) {
            m_telemetry->clearScreenInfo();
        }
        return;
    }

    TelemetryReporter::ScreenInfo info;
    info.name = screen->name();
    info.manufacturer = screen->manufacturer();
    info.model = screen->model();
    info.serialNumber = screen->serialNumber();
    info.geometry = screen->geometry();
    info.availableGeometry = screen->availableGeometry();
    info.refreshRateHz = screen->refreshRate();
    info.devicePixelRatio = screen->devicePixelRatio();
    info.logicalDpiX = screen->logicalDotsPerInchX();
    info.logicalDpiY = screen->logicalDotsPerInchY();
    const QList<QScreen*> screens = QGuiApplication::screens();
    info.index = screens.indexOf(screen);

    m_screenInfo = info;
    if (m_telemetry) {
        m_telemetry->setScreenInfo(info);
    }
}

void Application::ensureTelemetry() {
    const bool shouldEnable = m_metricsEnabled && !m_metricsEndpoint.isEmpty();
    if (!m_telemetry) {
        if (!shouldEnable) {
            updateTelemetryPendingRetryCount(0);
            if (m_performanceTelemetry) {
                m_performanceTelemetry->setTelemetryReporter(nullptr);
            }
            return;
        }
        auto reporter = std::make_unique<UiTelemetryReporter>(this);
        if (m_transportMode == TradingClient::TransportMode::InProcess) {
            if (!m_inProcessMetricsClient)
                m_inProcessMetricsClient = std::make_shared<InProcessMetricsClient>();
            reporter->setMetricsClientForTesting(m_inProcessMetricsClient);
        }
        m_telemetry = std::move(reporter);
    }
    if (!m_telemetry)
        return;

    if (m_performanceTelemetry) {
        m_performanceTelemetry->setTelemetryReporter(m_telemetry.get());
        if (m_frameMonitor) {
            m_performanceTelemetry->setFrameRateMonitor(m_frameMonitor.get());
        }
    }

    if (auto* uiReporter = dynamic_cast<UiTelemetryReporter*>(m_telemetry.get())) {
        if (m_transportMode == TradingClient::TransportMode::InProcess) {
            if (!m_inProcessMetricsClient)
                m_inProcessMetricsClient = std::make_shared<InProcessMetricsClient>();
            if (!m_usingInProcessMetricsClient) {
                uiReporter->setMetricsClientForTesting(m_inProcessMetricsClient);
                m_usingInProcessMetricsClient = true;
            }
        } else if (m_usingInProcessMetricsClient) {
            if (!m_grpcMetricsClient)
                m_grpcMetricsClient = std::make_shared<MetricsClient>();
            uiReporter->setMetricsClientForTesting(m_grpcMetricsClient);
            m_usingInProcessMetricsClient = false;
        }
        connect(uiReporter, &UiTelemetryReporter::pendingRetryCountChanged,
                this, &Application::handleTelemetryPendingRetryCountChanged,
                Qt::UniqueConnection);
        updateTelemetryPendingRetryCount(uiReporter->pendingRetryCount());
    } else {
        updateTelemetryPendingRetryCount(0);
    }

    m_telemetry->setWindowCount(m_windowCount);
    m_telemetry->setNotesTag(m_metricsTag);
    m_telemetry->setEndpoint(m_metricsEndpoint);
    m_telemetry->setTlsConfig(m_tlsConfig);
    m_telemetry->setAuthToken(m_metricsAuthToken);
    m_telemetry->setRbacRole(m_metricsRbacRole);
    if (m_screenInfo.has_value()) {
        m_telemetry->setScreenInfo(m_screenInfo.value());
    } else {
        m_telemetry->clearScreenInfo();
    }
    m_telemetry->setEnabled(shouldEnable);
}

void Application::reportOverlayTelemetry() {
    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled() || !m_lastOverlayState)
        return;

    const OverlayState current = *m_lastOverlayState;
    if (m_lastOverlayTelemetryReported && *m_lastOverlayTelemetryReported == current)
        return;

    m_telemetry->reportOverlayBudget(m_guard, current.active, current.allowed, current.reduceMotion);
    m_lastOverlayTelemetryReported = current;
}

void Application::reportJankTelemetry(double frameMs, double thresholdMs) {
    if (frameMs <= 0.0)
        return;

    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled())
        return;

    if (!m_jankTelemetryTimerValid) {
        m_lastJankTelemetry.start();
        m_jankTelemetryTimerValid = true;
    } else if (m_lastJankTelemetry.isValid() && m_lastJankTelemetry.elapsed() < m_jankTelemetryCooldownMs) {
        return;
    } else if (m_lastJankTelemetry.isValid()) {
        m_lastJankTelemetry.restart();
    } else {
        m_lastJankTelemetry.start();
    }

    OverlayState overlay = m_lastOverlayState.value_or(OverlayState{});
    if (overlay.allowed == 0) {
        overlay.allowed = m_guard.maxOverlayCount > 0 ? m_guard.maxOverlayCount : 0;
    }

    m_telemetry->reportJankEvent(
        m_guard,
        frameMs,
        thresholdMs,
        m_reduceMotionActive,
        overlay.active,
        overlay.allowed
    );
}

void Application::reportReduceMotionTelemetry(bool enabled) {
    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled()) {
        m_pendingReduceMotionState.reset();
        return;
    }

    if (m_latestFpsSample <= 0.0) {
        m_pendingReduceMotionState = enabled;
        return;
    }

    if (m_lastReduceMotionReported && m_lastReduceMotionReported.value() == enabled)
        return;

    OverlayState overlay = m_lastOverlayState.value_or(OverlayState{});
    if (overlay.allowed == 0) {
        overlay.allowed = m_guard.maxOverlayCount > 0 ? m_guard.maxOverlayCount : 0;
    }

    m_telemetry->reportReduceMotion(m_guard, enabled, m_latestFpsSample, overlay.active, overlay.allowed);
    m_lastReduceMotionReported = enabled;
    m_pendingReduceMotionState.reset();
}

void Application::handleTelemetryPendingRetryCountChanged(int pending) {
    updateTelemetryPendingRetryCount(pending);
}

void Application::updateTelemetryPendingRetryCount(int pending) {
    if (m_pendingRetryCount == pending)
        return;
    m_pendingRetryCount = pending;
    Q_EMIT telemetryPendingRetryCountChanged(pending);
}
