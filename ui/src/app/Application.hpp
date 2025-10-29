#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>
#include <QDateTime>
#include <QElapsedTimer>
#include <QTimer>
#include <QUrl>
#include <QJsonObject>
#include <QDir>
#include <QFileSystemWatcher>
#include <QHash>
#include <QStringList>

#include <memory>
#include <optional>

#include "grpc/GrpcTlsConfig.hpp"
#include "grpc/TradingClient.hpp"
#include "models/AlertsModel.hpp"
#include "models/AlertsFilterProxyModel.hpp"
#include "models/DecisionLogFilterProxyModel.hpp"
#include "models/DecisionLogModel.hpp"
#include "models/IndicatorSeriesModel.hpp"
#include "models/MarketRegimeTimelineModel.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskStateModel.hpp"
#include "models/RiskHistoryModel.hpp"
#include "models/SignalListModel.hpp"
#include "utils/PerformanceGuard.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"
#include "telemetry/PerformanceTelemetryController.hpp"

class QQuickWindow;
class QScreen;
class ActivationController;            // forward decl (app/ActivationController.hpp)
class LicenseActivationController;     // forward decl (license/LicenseActivationController.hpp)
class SecurityAdminController;         // forward decl (security/SecurityAdminController.hpp)
class ReportCenterController;          // forward decl (reporting/ReportCenterController.hpp)
class BotCoreLocalService;             // forward decl (grpc/BotCoreLocalService.hpp)
class StrategyConfigController;        // forward decl (app/StrategyConfigController.hpp)
class StrategyWorkbenchController;     // forward decl (app/StrategyWorkbenchController.hpp)
class SupportBundleController;         // forward decl (support/SupportBundleController.hpp)
class HealthStatusController;          // forward decl (health/HealthStatusController.hpp)
class OfflineRuntimeBridge;            // forward decl (runtime/OfflineRuntimeBridge.hpp)
class UiModuleManager;                 // forward decl (app/UiModuleManager.hpp)
class UiModuleViewsModel;              // forward decl (app/UiModuleViewsModel.hpp)
class MetricsClientInterface;          // forward decl (grpc/MetricsClient.hpp)
class HealthClientInterface;           // forward decl (grpc/HealthClient.hpp)
class MarketplaceController;           // forward decl (app/MarketplaceController.hpp)
class PortfolioManagerController;      // forward decl (app/PortfolioManagerController.hpp)

class Application : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString          connectionStatus     READ connectionStatus    NOTIFY connectionStatusChanged)
    Q_PROPERTY(PerformanceGuard performanceGuard     READ performanceGuard    NOTIFY performanceGuardChanged)
    Q_PROPERTY(bool             reduceMotionActive   READ reduceMotionActive  NOTIFY reduceMotionActiveChanged)
    Q_PROPERTY(QString          instrumentLabel      READ instrumentLabel     NOTIFY instrumentChanged)
    Q_PROPERTY(QObject*         riskModel            READ riskModel           CONSTANT)
    Q_PROPERTY(QObject*         riskHistoryModel     READ riskHistoryModel    CONSTANT)
    Q_PROPERTY(QObject*         alertsModel         READ alertsModel         CONSTANT)
    Q_PROPERTY(QObject*         alertsFilterModel   READ alertsFilterModel   CONSTANT)
    Q_PROPERTY(QObject*         indicatorSeriesModel READ indicatorSeriesModel CONSTANT)
    Q_PROPERTY(QObject*         signalListModel READ signalListModel CONSTANT)
    Q_PROPERTY(QObject*         marketRegimeTimelineModel READ marketRegimeTimelineModel CONSTANT)
    Q_PROPERTY(int              regimeTimelineMaximumSnapshots READ regimeTimelineMaximumSnapshots WRITE setRegimeTimelineMaximumSnapshots NOTIFY regimeTimelineMaximumSnapshotsChanged)
    Q_PROPERTY(QObject*         activationController READ activationController CONSTANT)
    Q_PROPERTY(QObject*         reportController READ reportController CONSTANT)
    Q_PROPERTY(QObject*         strategyController READ strategyController CONSTANT)
    Q_PROPERTY(QObject*         workbenchController READ workbenchController CONSTANT)
    Q_PROPERTY(QObject*         supportController READ supportController CONSTANT)
    Q_PROPERTY(QObject*         healthController READ healthController CONSTANT)
    Q_PROPERTY(QObject*         decisionLogModel READ decisionLogModel CONSTANT)
    Q_PROPERTY(QObject*         moduleManager READ moduleManager CONSTANT)
    Q_PROPERTY(QObject*         moduleViewsModel READ moduleViewsModel CONSTANT)
    Q_PROPERTY(QObject*         marketplaceController READ marketplaceController CONSTANT)
    Q_PROPERTY(QObject*         portfolioController READ portfolioController CONSTANT)
    Q_PROPERTY(QString          decisionLogPath READ decisionLogPath NOTIFY decisionLogPathChanged)
    Q_PROPERTY(int              telemetryPendingRetryCount READ telemetryPendingRetryCount NOTIFY telemetryPendingRetryCountChanged)
    Q_PROPERTY(QVariantMap      riskRefreshSchedule READ riskRefreshSchedule NOTIFY riskRefreshScheduleChanged)
    Q_PROPERTY(QVariantMap      licenseRefreshSchedule READ licenseRefreshSchedule NOTIFY licenseRefreshScheduleChanged)
    Q_PROPERTY(QVariantMap      securityCache READ securityCache NOTIFY securityCacheChanged)
    Q_PROPERTY(bool             riskHistoryExportLimitEnabled READ riskHistoryExportLimitEnabled WRITE setRiskHistoryExportLimitEnabled NOTIFY riskHistoryExportLimitEnabledChanged)
    Q_PROPERTY(int              riskHistoryExportLimitValue READ riskHistoryExportLimitValue WRITE setRiskHistoryExportLimitValue NOTIFY riskHistoryExportLimitValueChanged)
    Q_PROPERTY(QUrl             riskHistoryExportLastDirectory READ riskHistoryExportLastDirectory WRITE setRiskHistoryExportLastDirectory NOTIFY riskHistoryExportLastDirectoryChanged)
    Q_PROPERTY(bool             riskHistoryAutoExportEnabled READ riskHistoryAutoExportEnabled WRITE setRiskHistoryAutoExportEnabled NOTIFY riskHistoryAutoExportEnabledChanged)
    Q_PROPERTY(int              riskHistoryAutoExportIntervalMinutes READ riskHistoryAutoExportIntervalMinutes WRITE setRiskHistoryAutoExportIntervalMinutes NOTIFY riskHistoryAutoExportIntervalMinutesChanged)
    Q_PROPERTY(QString          riskHistoryAutoExportBasename READ riskHistoryAutoExportBasename WRITE setRiskHistoryAutoExportBasename NOTIFY riskHistoryAutoExportBasenameChanged)
    Q_PROPERTY(bool             riskHistoryAutoExportUseLocalTime READ riskHistoryAutoExportUseLocalTime WRITE setRiskHistoryAutoExportUseLocalTime NOTIFY riskHistoryAutoExportUseLocalTimeChanged)
    Q_PROPERTY(QDateTime        riskHistoryLastAutoExportAt READ riskHistoryLastAutoExportAt NOTIFY riskHistoryLastAutoExportAtChanged)
    Q_PROPERTY(QUrl             riskHistoryLastAutoExportPath READ riskHistoryLastAutoExportPath NOTIFY riskHistoryLastAutoExportPathChanged)
    Q_PROPERTY(bool             offlineMode          READ offlineMode         CONSTANT)
    Q_PROPERTY(QString          offlineDaemonStatus  READ offlineDaemonStatus NOTIFY offlineDaemonStatusChanged)
    Q_PROPERTY(bool             offlineAutomationRunning READ offlineAutomationRunning NOTIFY offlineAutomationRunningChanged)
    Q_PROPERTY(QString          offlineStrategyPath  READ offlineStrategyPath NOTIFY offlineStrategyPathChanged)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);
    ~Application() override;

    // CLI
    void configureParser(QCommandLineParser& parser) const;
    bool applyParser(const QCommandLineParser& parser);

    // Umożliwia wstrzyknięcie mocka w testach
    void setTelemetryReporter(std::unique_ptr<TelemetryReporter> reporter);
    void setMetricsClientOverrideForTesting(std::shared_ptr<MetricsClientInterface> client);
    std::shared_ptr<MetricsClientInterface> activeMetricsClientForTesting() const;
    bool usingInProcessMetricsClientForTesting() const { return m_usingInProcessMetricsClient; }
    void setInProcessDatasetPathForTesting(const QString& path);
    void applyPreferredScreenForTesting(QQuickWindow* window);
    QScreen* pickPreferredScreenForTesting() const;

    // Getters
    QString          connectionStatus() const { return m_connectionStatus; }
    PerformanceGuard performanceGuard() const { return m_guard; }
    QString          instrumentLabel() const;
    bool             reduceMotionActive() const { return m_reduceMotionActive; }
    QObject*         riskModel() const { return const_cast<RiskStateModel*>(&m_riskModel); }
    QObject*         activationController() const;
    QObject*         reportController() const;
    QObject*         strategyController() const;
    QObject*         workbenchController() const;
    QObject*         supportController() const;
    QObject*         healthController() const;
    QObject*         decisionLogModel() const;
    QObject*         moduleManager() const;
    QObject*         moduleViewsModel() const;
    QObject*         marketplaceController() const;
    QObject*         portfolioController() const;
    QObject*         alertsModel() const { return const_cast<AlertsModel*>(&m_alertsModel); }
    QObject*         alertsFilterModel() const { return const_cast<AlertsFilterProxyModel*>(&m_filteredAlertsModel); }
    QObject*         riskHistoryModel() const { return const_cast<RiskHistoryModel*>(&m_riskHistoryModel); }
    int              telemetryPendingRetryCount() const { return m_pendingRetryCount; }
    bool             riskHistoryExportLimitEnabled() const { return m_riskHistoryExportLimitEnabled; }
    int              riskHistoryExportLimitValue() const { return m_riskHistoryExportLimitValue; }
    QUrl             riskHistoryExportLastDirectory() const { return m_riskHistoryExportLastDirectory; }
    bool             riskHistoryAutoExportEnabled() const { return m_riskHistoryAutoExportEnabled; }
    int              riskHistoryAutoExportIntervalMinutes() const { return m_riskHistoryAutoExportIntervalMinutes; }
    QString          riskHistoryAutoExportBasename() const { return m_riskHistoryAutoExportBasename; }
    bool             riskHistoryAutoExportUseLocalTime() const { return m_riskHistoryAutoExportUseLocalTime; }
    QDateTime        riskHistoryLastAutoExportAt() const { return m_lastRiskHistoryAutoExportUtc; }
    QUrl             riskHistoryLastAutoExportPath() const { return m_lastRiskHistoryAutoExportPath; }
    bool             offlineMode() const { return m_offlineMode; }
    QString          offlineDaemonStatus() const { return m_offlineStatus; }
    bool             offlineAutomationRunning() const { return m_offlineAutomationRunning; }
    QString          offlineStrategyPath() const { return m_offlineStrategyPath; }
    QString          decisionLogPath() const { return m_decisionLogPath; }

public slots:
    void start();
    void stop();

    // Z QML (np. ChartView/StatusFooter/MainWindow)
    Q_INVOKABLE void notifyOverlayUsage(int activeCount, int allowedCount, bool reduceMotionActive);
    Q_INVOKABLE void notifyWindowCount(int totalWindowCount);
    Q_INVOKABLE QVariantMap instrumentConfigSnapshot() const;
    Q_INVOKABLE QVariantMap performanceGuardSnapshot() const;
    Q_INVOKABLE QVariantMap riskRefreshSnapshot() const;
    QVariantMap riskRefreshSchedule() const { return riskRefreshSnapshot(); }
    QVariantMap licenseRefreshSchedule() const;
    QVariantMap securityCache() const { return m_securityCache; }
    Q_INVOKABLE bool updateInstrument(const QString& exchange,
                                      const QString& symbol,
                                      const QString& venueSymbol,
                                      const QString& quoteCurrency,
                                      const QString& baseCurrency,
                                      const QString& granularityIso8601);
    Q_INVOKABLE bool updatePerformanceGuard(int fpsTarget,
                                            double reduceMotionAfter,
                                            double jankThresholdMs,
                                            int maxOverlayCount,
                                            int disableSecondaryWhenBelow);
    Q_INVOKABLE bool updateRiskRefresh(bool enabled, double intervalSeconds);
    Q_INVOKABLE QVariantList listTradableInstruments(const QString& exchange);
    Q_INVOKABLE bool triggerRiskRefreshNow();
    Q_INVOKABLE bool updateRiskHistoryLimit(int maximumEntries);
    Q_INVOKABLE void clearRiskHistory();
    Q_INVOKABLE bool exportRiskHistoryToCsv(const QUrl& destination, int limit = -1);
    Q_INVOKABLE bool setRiskHistoryExportLimitEnabled(bool enabled);
    Q_INVOKABLE bool setRiskHistoryExportLimitValue(int limit);
    Q_INVOKABLE bool setRiskHistoryExportLastDirectory(const QUrl& directory);
    Q_INVOKABLE bool setRiskHistoryAutoExportEnabled(bool enabled);
    Q_INVOKABLE bool setRiskHistoryAutoExportIntervalMinutes(int minutes);
    Q_INVOKABLE bool setRiskHistoryAutoExportBasename(const QString& basename);
    Q_INVOKABLE bool setRiskHistoryAutoExportUseLocalTime(bool useLocalTime);
    Q_INVOKABLE bool setRegimeTimelineMaximumSnapshots(int maximumSnapshots);
    Q_INVOKABLE void startOfflineAutomation();
    Q_INVOKABLE void stopOfflineAutomation();
    Q_INVOKABLE bool setDecisionLogPath(const QUrl& url);
    Q_INVOKABLE bool reloadDecisionLog();
    Q_INVOKABLE bool reloadUiModules();

    // Test helpers (persistent UI state)
    void saveUiSettingsImmediatelyForTesting();
    QString uiSettingsPathForTesting() const { return m_uiSettingsPath; }
    bool uiSettingsPersistenceEnabledForTesting() const { return m_uiSettingsPersistenceEnabled; }
    void setLastRiskHistoryAutoExportForTesting(const QDateTime& timestamp);
    void setRiskHistoryAutoExportLastPathForTesting(const QUrl& url);
    QString decisionLogPathForTesting() const { return m_decisionLogPath; }
    DecisionLogModel* decisionLogModelForTesting() { return &m_decisionLogModel; }
    void setTradableInstrumentsForTesting(const QString& exchange,
                                          const QVector<TradingClient::TradableInstrument>& items);
    void setModuleManagerForTesting(std::unique_ptr<UiModuleManager> manager);
    UiModuleManager* moduleManagerForTesting() const { return m_moduleManager.get(); }
    UiModuleViewsModel* moduleViewsModelForTesting() const { return m_moduleViewsModel.get(); }
    QStringList uiModuleDirectoriesForTesting() const { return m_uiModuleDirectories; }
    MarketplaceController* marketplaceControllerForTesting() const { return m_marketplaceController.get(); }
    PortfolioManagerController* portfolioControllerForTesting() const { return m_portfolioController.get(); }

    // Test helpers
    void ingestFpsSampleForTesting(double fps);
    void setReduceMotionStateForTesting(bool active);
    void simulateFrameIntervalForTesting(double seconds);

signals:
    void connectionStatusChanged();
    void performanceGuardChanged();
    void instrumentChanged();
    void reduceMotionActiveChanged();
    void telemetryPendingRetryCountChanged(int pending);
    void riskRefreshScheduleChanged();
    void fingerprintRefreshScheduleChanged();
    void riskHistoryExportLimitEnabledChanged();
    void riskHistoryExportLimitValueChanged();
    void riskHistoryExportLastDirectoryChanged();
    void riskHistoryAutoExportEnabledChanged();
    void riskHistoryAutoExportIntervalMinutesChanged();
    void riskHistoryAutoExportBasenameChanged();
    void riskHistoryAutoExportUseLocalTimeChanged();
    void uiModuleDirectoriesChanged(const QStringList& directories);
    void uiModulesReloaded(bool success, const QVariantMap& report);
    void riskHistoryLastAutoExportAtChanged();
    void riskHistoryLastAutoExportPathChanged();
    void regimeTimelineMaximumSnapshotsChanged();
    void offlineDaemonStatusChanged();
    void offlineAutomationRunningChanged(bool running);
    void offlineStrategyPathChanged();
    void decisionLogPathChanged();
    void licenseRefreshScheduleChanged();
    void securityCacheChanged();

private slots:
    void handleHistory(const QList<OhlcvPoint>& candles);
    void handleCandle(const OhlcvPoint& candle);
    void handleRiskState(const RiskSnapshotData& snapshot);
    void handleTelemetryPendingRetryCountChanged(int pending);
    void handleRiskHistorySnapshotRecorded(const QDateTime& timestamp);
    void handleTradingTokenPathChanged(const QString& path);
    void handleMetricsTokenPathChanged(const QString& path);
    void handleHealthTokenPathChanged(const QString& path);
    void handleOfflineStatusChanged(const QString& status);
    void handleOfflineAutomationChanged(bool running);
    void handleActivationErrorChanged();
    void handleActivationFingerprintChanged();
    void handleActivationLicensesChanged();
    void handleActivationOemLicenseChanged();

private:
    // Rejestracja obiektów w kontekście QML
    void exposeToQml();

    // FPS/Reduce-motion
    void ensureFrameMonitor();
    void attachWindow(QObject* object);

    // Telemetria UI
    void ensureTelemetry();
    void ensureOfflineBridge();
    void reportOverlayTelemetry();
    void reportReduceMotionTelemetry(bool enabled);
    void reportJankTelemetry(double frameMs, double thresholdMs);
    void applyMetricsEnvironmentOverrides(const QCommandLineParser& parser,
                                          bool cliTokenProvided,
                                          bool cliTokenFileProvided,
                                          QString& metricsToken,
                                          QString& metricsTokenFile);
    void applyTradingTlsEnvironmentOverrides(const QCommandLineParser& parser);
    void applyTradingAuthEnvironmentOverrides(const QCommandLineParser& parser,
                                              bool cliTokenProvided,
                                              bool cliTokenFileProvided,
                                              bool cliRoleProvided,
                                              bool cliScopesProvided,
                                              QString& tradingToken,
                                              QString& tradingTokenFile);
    void applyHealthEnvironmentOverrides(const QCommandLineParser& parser,
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
                                         QString& healthTokenFile);
                                              bool cliScopesProvided);
    void applyScreenEnvironmentOverrides(const QCommandLineParser& parser);
    void applyPreferredScreen(QQuickWindow* window);
    QScreen* resolvePreferredScreen() const;
    void updateScreenInfo(QScreen* screen);
    void updateTelemetryPendingRetryCount(int pending);
    bool validateTransportConfiguration(const QString& endpoint,
                                        const QString& datasetPath,
                                        const TradingClient::TlsConfig& tradingTls,
                                        const TelemetryTlsConfig& metricsTls,
                                        const GrpcTlsConfig& healthTls,
                                        const QString& metricsEndpoint,
                                        const QString& healthEndpoint) const;
    void configureLocalBotCoreService(const QCommandLineParser& parser, QString& endpoint);
    QString locateRepoRoot() const;
    void configureRiskRefresh(bool enabled, double intervalSeconds);
    void applyRiskRefreshTimerState();
    void initializeUiSettingsStorage();
    void ensureUiSettingsTimerConfigured();
    void applyUiSettingsCliOverrides(const QCommandLineParser& parser);
    void applyRiskHistoryCliOverrides(const QCommandLineParser& parser);
    void configureStrategyBridge(const QCommandLineParser& parser);
    void configureSupportBundle(const QCommandLineParser& parser);
    void configureRegimeThresholds(const QCommandLineParser& parser);
    void configureDecisionLog(const QCommandLineParser& parser);
    void configureUiModules(const QCommandLineParser& parser);
    void setUiSettingsPersistenceEnabled(bool enabled);
    void setUiSettingsPath(const QString& path, bool reload = true);
    void loadUiSettings();
    void scheduleUiSettingsPersist();
    void persistUiSettings();
    QJsonObject buildUiSettingsPayload() const;
    void maybeAutoExportRiskHistory(const QDateTime& snapshotTimestamp);
    QString resolveAutoExportFilePath(const QDir& directory, const QString& basename, const QDateTime& timestamp) const;
    bool setDecisionLogPathInternal(const QString& path, bool emitSignal);
    void setTradingAuthTokenFile(const QString& path);
    void setMetricsAuthTokenFile(const QString& path);
    void setHealthAuthTokenFile(const QString& path);
    void reloadTradingTokenFromFile();
    void reloadMetricsTokenFromFile();
    void reloadHealthTokenFromFile();
    void updateUiModuleWatchTargets(const QStringList& directories, const QStringList& pluginFiles);
    void configureTokenWatcher(QFileSystemWatcher& watcher,
                               QString& trackedFile,
                               QStringList& trackedDirs,
                               const QString& filePath,
                               const char* label);
    bool applyRegimeThresholdPath(const QString& path, bool warnIfMissing);
    void configureTlsWatcher(QFileSystemWatcher& watcher,
                              QStringList& trackedFiles,
                              QStringList& trackedDirs,
                              const QStringList& filePaths,
                              const char* label);
    void configureTradingTlsWatchers();
    void configureMetricsTlsWatchers();
    void applyMarketplaceEnvironmentOverrides(const QCommandLineParser& parser);
    void configureHealthTlsWatchers();
    void applyHealthAuthTokenToController();
    void applyTradingTlsConfig();
    void applyMetricsTlsConfig();
    void applyHealthTlsConfig();
    void handleTradingTlsPathChanged(const QString& path);
    void handleMetricsTlsPathChanged(const QString& path);
    void handleHealthTlsPathChanged(const QString& path);
    void handleRegimeThresholdPathChanged(const QString& path);

    // --- Stan i komponenty ---
    QQmlApplicationEngine& m_engine;
    OhlcvListModel         m_ohlcvModel;
    IndicatorSeriesModel   m_indicatorModel;
    SignalListModel        m_signalModel;
    MarketRegimeTimelineModel m_regimeTimelineModel;
    RiskStateModel         m_riskModel;
    RiskHistoryModel       m_riskHistoryModel;
    DecisionLogModel       m_decisionLogModel;
    DecisionLogFilterProxyModel m_decisionLogFilter;
    TradingClient          m_client;
    std::unique_ptr<OfflineRuntimeBridge> m_offlineBridge;
    AlertsModel            m_alertsModel;
    AlertsFilterProxyModel m_filteredAlertsModel;
    std::unique_ptr<BotCoreLocalService> m_localService;

    QString                m_connectionStatus = QStringLiteral("idle");
    PerformanceGuard       m_guard{};
    int                    m_maxSamples = 10240;
    int                    m_historyLimit = 500;
    int                    m_regimeTimelineMaximumSnapshots = 720;
    TradingClient::TlsConfig m_tradingTlsConfig{};
    QString                m_tradingAuthToken;
    QString                m_tradingAuthTokenFile;
    QString                m_tradingRbacRole;
    QStringList            m_tradingRbacScopes;
    TradingClient::TransportMode m_transportMode = TradingClient::TransportMode::Grpc;
    QString                m_inProcessDatasetPath;
    int                    m_inProcessCandleIntervalMs = 150;
    QHash<QString, QVector<TradingClient::TradableInstrument>> m_tradableInstrumentCache;

    TradingClient::InstrumentConfig m_instrument{
        QStringLiteral("BINANCE"),
        QStringLiteral("BTC/USDT"),
        QStringLiteral("BTCUSDT"),
        QStringLiteral("USDT"),
        QStringLiteral("BTC"),
        QStringLiteral("PT1M")
    };

    std::unique_ptr<FrameRateMonitor>         m_frameMonitor;
    bool                                      m_reduceMotionActive = false;
    bool                                      m_offlineMode = false;
    QString                                   m_offlineEndpoint;
    QVariantMap                               m_offlineStrategyConfig;
    bool                                      m_offlineAutoRun = false;
    QString                                   m_offlineStatus;
    bool                                      m_offlineAutomationRunning = false;
    QString                                   m_offlineStrategyPath;
    QString                                   m_decisionLogPath;

    // Oba kontrolery – aktywacja (app) i licencje OEM (license)
    std::unique_ptr<ActivationController>     m_activationController;
    std::unique_ptr<LicenseActivationController> m_licenseController;
    std::unique_ptr<SecurityAdminController>   m_securityController;
    std::unique_ptr<ReportCenterController>    m_reportController;
    std::unique_ptr<StrategyConfigController>  m_strategyController;
    std::unique_ptr<StrategyWorkbenchController> m_workbenchController;
    std::unique_ptr<SupportBundleController>   m_supportController;
    std::unique_ptr<HealthStatusController>    m_healthController;
    std::unique_ptr<MarketplaceController>     m_marketplaceController;
    std::unique_ptr<PortfolioManagerController> m_portfolioController;
    std::unique_ptr<UiModuleManager>           m_moduleManager;
    std::unique_ptr<UiModuleViewsModel>        m_moduleViewsModel;

    // --- Telemetry state ---
    std::unique_ptr<PerformanceTelemetryController> m_performanceTelemetry;
    std::unique_ptr<TelemetryReporter> m_telemetry;
    QString                            m_metricsEndpoint;
    QString                            m_metricsTag;
    bool                               m_metricsEnabled = false;
    QString                            m_metricsAuthToken;
    QString                            m_metricsAuthTokenFile;
    QString                            m_metricsRbacRole;
    double                             m_latestFpsSample = 0.0;
    int                                m_windowCount = 1;
    TelemetryTlsConfig                 m_tlsConfig;
    GrpcTlsConfig                      m_healthTlsConfig{};
    QString                            m_healthEndpoint;
    QString                            m_healthAuthToken;
    QString                            m_healthAuthTokenFile;
    QString                            m_healthRbacRole;
    QStringList                        m_healthRbacScopes;
    std::shared_ptr<MetricsClientInterface> m_inProcessMetricsClient;
    std::shared_ptr<MetricsClientInterface> m_grpcMetricsClient;
    std::shared_ptr<MetricsClientInterface> m_metricsClientOverride;
    std::shared_ptr<HealthClientInterface>  m_inProcessHealthClient;
    std::shared_ptr<HealthClientInterface>  m_grpcHealthClient;
    bool                                    m_usingInProcessMetricsClient = false;
    std::weak_ptr<MetricsClientInterface>   m_activeMetricsClient;
    bool                                    m_usingInProcessHealthClient = false;
    int                                m_healthRefreshIntervalSeconds = 60;
    bool                               m_healthAutoRefreshEnabled = true;
    QString                            m_preferredScreenName;
    int                                m_preferredScreenIndex = -1;
    bool                               m_forcePrimaryScreen = false;
    bool                               m_preferredScreenConfigured = false;
    mutable bool                       m_screenWarningLogged = false;
    bool                               m_localServiceEnabled = false;
    QString                            m_repoRoot;
    QTimer                             m_riskRefreshTimer;
    int                                m_riskRefreshIntervalMs = 5000;
    bool                               m_riskRefreshEnabled = true;
    bool                               m_started = false;
    QDateTime                          m_lastRiskRefreshRequestUtc;
    QDateTime                          m_lastRiskUpdateUtc;
    QDateTime                          m_nextRiskRefreshUtc;
    QString                            m_uiSettingsPath;
    QTimer                             m_uiSettingsSaveTimer;
    bool                               m_loadingUiSettings = false;
    bool                               m_uiSettingsPersistenceEnabled = true;
    bool                               m_uiSettingsTimerConfigured = false;
    bool                               m_riskHistoryExportLimitEnabled = false;
    int                                m_riskHistoryExportLimitValue = 50;
    QUrl                               m_riskHistoryExportLastDirectory;
    bool                               m_riskHistoryAutoExportEnabled = false;
    int                                m_riskHistoryAutoExportIntervalMinutes = 15;
    QString                            m_riskHistoryAutoExportBasename = QStringLiteral("risk-history");
    bool                               m_riskHistoryAutoExportUseLocalTime = false;
    QDateTime                          m_lastRiskHistoryAutoExportUtc;
    QUrl                               m_lastRiskHistoryAutoExportPath;
    bool                               m_riskHistoryAutoExportDirectoryWarned = false;
    QTimer                             m_licenseRefreshTimer;
    int                                m_licenseRefreshIntervalSeconds = 600;
    QDateTime                          m_lastLicenseRefreshRequestUtc;
    QDateTime                          m_lastLicenseRefreshUtc;
    QDateTime                          m_nextLicenseRefreshUtc;
    QString                            m_licenseCachePath;
    QVariantMap                        m_securityCache;
    bool                               m_loadingSecurityCache = false;
    QString                            m_lastSecurityError;
    bool                               m_licenseRefreshTimerConfigured = false;

    struct OverlayState {
        int  active = 0;
        int  allowed = 0;
        bool reduceMotion = false;
        bool operator==(const OverlayState& other) const {
            return active == other.active && allowed == other.allowed && reduceMotion == other.reduceMotion;
        }
    };

    std::optional<OverlayState>                        m_lastOverlayState;
    std::optional<OverlayState>                        m_lastOverlayTelemetryReported;
    std::optional<bool>                                m_lastReduceMotionReported;
    std::optional<bool>                                m_pendingReduceMotionState;
    std::optional<TelemetryReporter::ScreenInfo>       m_screenInfo;
    QElapsedTimer                                      m_lastJankTelemetry;
    bool                                               m_jankTelemetryTimerValid = false;
    int                                                m_jankTelemetryCooldownMs = 400;
    int                                                m_pendingRetryCount = 0;
    QFileSystemWatcher                                 m_tradingTokenWatcher;
    QFileSystemWatcher                                 m_metricsTokenWatcher;
    QFileSystemWatcher                                 m_healthTokenWatcher;
    QFileSystemWatcher                                 m_tradingTlsWatcher;
    QFileSystemWatcher                                 m_metricsTlsWatcher;
    QFileSystemWatcher                                 m_healthTlsWatcher;
    QFileSystemWatcher                                 m_regimeThresholdWatcher;
    QString                                            m_tradingTokenWatcherFile;
    QStringList                                        m_tradingTokenWatcherDirs;
    QString                                            m_metricsTokenWatcherFile;
    QStringList                                        m_metricsTokenWatcherDirs;
    QString                                            m_healthTokenWatcherFile;
    QStringList                                        m_healthTokenWatcherDirs;
    QStringList                                        m_tradingTlsWatcherFiles;
    QStringList                                        m_tradingTlsWatcherDirs;
    QStringList                                        m_metricsTlsWatcherFiles;
    QStringList                                        m_metricsTlsWatcherDirs;
    QStringList                                        m_healthTlsWatcherFiles;
    QStringList                                        m_healthTlsWatcherDirs;
    QStringList                                        m_uiModuleDirectories;
    quint64                                            m_tradingTlsReloadGeneration = 0;
    quint64                                            m_metricsTlsReloadGeneration = 0;
    quint64                                            m_healthTlsReloadGeneration = 0;

public: // test helpers
    int  riskRefreshIntervalMsForTesting() const { return m_riskRefreshIntervalMs; }
    bool riskRefreshEnabledForTesting() const { return m_riskRefreshEnabled; }
    bool isRiskRefreshTimerActiveForTesting() const { return m_riskRefreshTimer.isActive(); }
    QDateTime lastRiskRefreshRequestUtcForTesting() const { return m_lastRiskRefreshRequestUtc; }
    QDateTime nextRiskRefreshDueUtcForTesting() const { return m_nextRiskRefreshUtc; }
    QDateTime lastRiskUpdateUtcForTesting() const { return m_lastRiskUpdateUtc; }
    void startRiskRefreshTimerForTesting();
    TradingClient* tradingClientForTesting() { return &m_client; }
    quint64 tradingTlsReloadGenerationForTesting() const { return m_tradingTlsReloadGeneration; }
    quint64 metricsTlsReloadGenerationForTesting() const { return m_metricsTlsReloadGeneration; }
    quint64 healthTlsReloadGenerationForTesting() const { return m_healthTlsReloadGeneration; }
    QVariantMap securityCacheForTesting() const { return m_securityCache; }
};
