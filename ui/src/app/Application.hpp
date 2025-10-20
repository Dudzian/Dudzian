#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>
#include <QDateTime>
#include <QElapsedTimer>
#include <QTimer>
#include <QJsonObject>

#include <memory>
#include <optional>

#include "grpc/TradingClient.hpp"
#include "models/AlertsModel.hpp"
#include "models/AlertsFilterProxyModel.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskStateModel.hpp"
#include "models/RiskHistoryModel.hpp"
#include "utils/PerformanceGuard.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"

class QQuickWindow;
class QScreen;
class ActivationController;            // forward decl (app/ActivationController.hpp)
class LicenseActivationController;     // forward decl (license/LicenseActivationController.hpp)
class SecurityAdminController;         // forward decl (security/SecurityAdminController.hpp)
class ReportCenterController;          // forward decl (reporting/ReportCenterController.hpp)
class BotCoreLocalService;             // forward decl (grpc/BotCoreLocalService.hpp)

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
    Q_PROPERTY(QObject*         activationController READ activationController CONSTANT)
    Q_PROPERTY(QObject*         reportController READ reportController CONSTANT)
    Q_PROPERTY(int              telemetryPendingRetryCount READ telemetryPendingRetryCount NOTIFY telemetryPendingRetryCountChanged)
    Q_PROPERTY(QVariantMap      riskRefreshSchedule READ riskRefreshSchedule NOTIFY riskRefreshScheduleChanged)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);
    ~Application() override;

    // CLI
    void configureParser(QCommandLineParser& parser) const;
    bool applyParser(const QCommandLineParser& parser);

    // Umożliwia wstrzyknięcie mocka w testach
    void setTelemetryReporter(std::unique_ptr<TelemetryReporter> reporter);
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
    QObject*         alertsModel() const { return const_cast<AlertsModel*>(&m_alertsModel); }
    QObject*         alertsFilterModel() const { return const_cast<AlertsFilterProxyModel*>(&m_filteredAlertsModel); }
    QObject*         riskHistoryModel() const { return const_cast<RiskHistoryModel*>(&m_riskHistoryModel); }
    int              telemetryPendingRetryCount() const { return m_pendingRetryCount; }

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
    Q_INVOKABLE bool triggerRiskRefreshNow();
    Q_INVOKABLE bool updateRiskHistoryLimit(int maximumEntries);
    Q_INVOKABLE void clearRiskHistory();

    // Test helpers (persistent UI state)
    void saveUiSettingsImmediatelyForTesting();
    QString uiSettingsPathForTesting() const { return m_uiSettingsPath; }
    bool uiSettingsPersistenceEnabledForTesting() const { return m_uiSettingsPersistenceEnabled; }

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

private slots:
    void handleHistory(const QList<OhlcvPoint>& candles);
    void handleCandle(const OhlcvPoint& candle);
    void handleRiskState(const RiskSnapshotData& snapshot);
    void handleTelemetryPendingRetryCountChanged(int pending);

private:
    // Rejestracja obiektów w kontekście QML
    void exposeToQml();

    // FPS/Reduce-motion
    void ensureFrameMonitor();
    void attachWindow(QObject* object);

    // Telemetria UI
    void ensureTelemetry();
    void reportOverlayTelemetry();
    void reportReduceMotionTelemetry(bool enabled);
    void reportJankTelemetry(double frameMs, double thresholdMs);
    void applyMetricsEnvironmentOverrides(const QCommandLineParser& parser,
                                          bool cliTokenProvided,
                                          bool cliTokenFileProvided);
    void applyTradingTlsEnvironmentOverrides(const QCommandLineParser& parser);
    void applyScreenEnvironmentOverrides(const QCommandLineParser& parser);
    void applyPreferredScreen(QQuickWindow* window);
    QScreen* resolvePreferredScreen() const;
    void updateScreenInfo(QScreen* screen);
    void updateTelemetryPendingRetryCount(int pending);
    void configureLocalBotCoreService(const QCommandLineParser& parser, QString& endpoint);
    QString locateRepoRoot() const;
    void configureRiskRefresh(bool enabled, double intervalSeconds);
    void applyRiskRefreshTimerState();
    void initializeUiSettingsStorage();
    void ensureUiSettingsTimerConfigured();
    void applyUiSettingsCliOverrides(const QCommandLineParser& parser);
    void setUiSettingsPersistenceEnabled(bool enabled);
    void setUiSettingsPath(const QString& path, bool reload = true);
    void loadUiSettings();
    void scheduleUiSettingsPersist();
    void persistUiSettings();
    QJsonObject buildUiSettingsPayload() const;

    // --- Stan i komponenty ---
    QQmlApplicationEngine& m_engine;
    OhlcvListModel         m_ohlcvModel;
    RiskStateModel         m_riskModel;
    RiskHistoryModel       m_riskHistoryModel;
    TradingClient          m_client;
    AlertsModel            m_alertsModel;
    AlertsFilterProxyModel m_filteredAlertsModel;
    std::unique_ptr<BotCoreLocalService> m_localService;

    QString                m_connectionStatus = QStringLiteral("idle");
    PerformanceGuard       m_guard{};
    int                    m_maxSamples = 10240;
    TradingClient::TlsConfig m_tradingTlsConfig{};

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

    // Oba kontrolery – aktywacja (app) i licencje OEM (license)
    std::unique_ptr<ActivationController>     m_activationController;
    std::unique_ptr<LicenseActivationController> m_licenseController;
    std::unique_ptr<SecurityAdminController>   m_securityController;
    std::unique_ptr<ReportCenterController>    m_reportController;

    // --- Telemetry state ---
    std::unique_ptr<TelemetryReporter> m_telemetry;
    QString                            m_metricsEndpoint;
    QString                            m_metricsTag;
    bool                               m_metricsEnabled = false;
    QString                            m_metricsAuthToken;
    QString                            m_metricsRbacRole;
    double                             m_latestFpsSample = 0.0;
    int                                m_windowCount = 1;
    TelemetryTlsConfig                 m_tlsConfig;
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

public: // test helpers
    int  riskRefreshIntervalMsForTesting() const { return m_riskRefreshIntervalMs; }
    bool riskRefreshEnabledForTesting() const { return m_riskRefreshEnabled; }
    bool isRiskRefreshTimerActiveForTesting() const { return m_riskRefreshTimer.isActive(); }
    QDateTime lastRiskRefreshRequestUtcForTesting() const { return m_lastRiskRefreshRequestUtc; }
    QDateTime nextRiskRefreshDueUtcForTesting() const { return m_nextRiskRefreshUtc; }
    QDateTime lastRiskUpdateUtcForTesting() const { return m_lastRiskUpdateUtc; }
    void startRiskRefreshTimerForTesting();
};
