#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>
#include <QElapsedTimer>

#include <memory>
#include <optional>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskStateModel.hpp"
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

class Application : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString          connectionStatus     READ connectionStatus    NOTIFY connectionStatusChanged)
    Q_PROPERTY(PerformanceGuard performanceGuard     READ performanceGuard    NOTIFY performanceGuardChanged)
    Q_PROPERTY(bool             reduceMotionActive   READ reduceMotionActive  NOTIFY reduceMotionActiveChanged)
    Q_PROPERTY(QString          instrumentLabel      READ instrumentLabel     NOTIFY instrumentChanged)
    Q_PROPERTY(QObject*         riskModel            READ riskModel           CONSTANT)
    Q_PROPERTY(QObject*         activationController READ activationController CONSTANT)
    Q_PROPERTY(QObject*         reportController READ reportController CONSTANT)
    Q_PROPERTY(int              telemetryPendingRetryCount READ telemetryPendingRetryCount NOTIFY telemetryPendingRetryCountChanged)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);

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
    int              telemetryPendingRetryCount() const { return m_pendingRetryCount; }

public slots:
    void start();
    void stop();

    // Z QML (np. ChartView/StatusFooter/MainWindow)
    Q_INVOKABLE void notifyOverlayUsage(int activeCount, int allowedCount, bool reduceMotionActive);
    Q_INVOKABLE void notifyWindowCount(int totalWindowCount);
    Q_INVOKABLE QVariantMap instrumentConfigSnapshot() const;
    Q_INVOKABLE QVariantMap performanceGuardSnapshot() const;
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

    // --- Stan i komponenty ---
    QQmlApplicationEngine& m_engine;
    OhlcvListModel         m_ohlcvModel;
    RiskStateModel         m_riskModel;
    TradingClient          m_client;

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
};
