#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>

#include <memory>
#include <optional>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskStateModel.hpp"
#include "utils/PerformanceGuard.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"

class Application : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString          connectionStatus    READ connectionStatus   NOTIFY connectionStatusChanged)
    Q_PROPERTY(PerformanceGuard performanceGuard    READ performanceGuard   NOTIFY performanceGuardChanged)
    Q_PROPERTY(bool             reduceMotionActive  READ reduceMotionActive NOTIFY reduceMotionActiveChanged)
    Q_PROPERTY(QString          instrumentLabel     READ instrumentLabel    NOTIFY instrumentChanged)
    Q_PROPERTY(QObject*         riskModel           READ riskModel          CONSTANT)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);

    // CLI
    void configureParser(QCommandLineParser& parser) const;
    bool applyParser(const QCommandLineParser& parser);

    // Umożliwia wstrzyknięcie mocka w testach
    void setTelemetryReporter(std::unique_ptr<TelemetryReporter> reporter);

    // Getters
    QString          connectionStatus() const { return m_connectionStatus; }
    PerformanceGuard performanceGuard() const { return m_guard; }
    QString          instrumentLabel() const;
    bool             reduceMotionActive() const { return m_reduceMotionActive; }
    QObject*         riskModel() const { return const_cast<RiskStateModel*>(&m_riskModel); }

public slots:
    void start();
    void stop();

    // Z QML (np. ChartView/StatusFooter/MainWindow)
    Q_INVOKABLE void notifyOverlayUsage(int activeCount, int allowedCount, bool reduceMotionActive);
    Q_INVOKABLE void notifyWindowCount(int totalWindowCount);

    // Test helpers
    void ingestFpsSampleForTesting(double fps);
    void setReduceMotionStateForTesting(bool active);

signals:
    void connectionStatusChanged();
    void performanceGuardChanged();
    void instrumentChanged();
    void reduceMotionActiveChanged();

private slots:
    void handleHistory(const QList<OhlcvPoint>& candles);
    void handleCandle(const OhlcvPoint& candle);
    void handleRiskState(const RiskSnapshotData& snapshot);

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

    // --- Stan i komponenty ---
    QQmlApplicationEngine& m_engine;
    OhlcvListModel         m_ohlcvModel;
    RiskStateModel         m_riskModel;
    TradingClient          m_client;

    QString          m_connectionStatus = QStringLiteral("idle");
    PerformanceGuard m_guard{};
    int              m_maxSamples = 10240;

    TradingClient::InstrumentConfig m_instrument{
        QStringLiteral("BINANCE"),
        QStringLiteral("BTC/USDT"),
        QStringLiteral("BTCUSDT"),
        QStringLiteral("USDT"),
        QStringLiteral("BTC"),
        QStringLiteral("PT1M")
    };

    std::unique_ptr<FrameRateMonitor> m_frameMonitor;
    bool                              m_reduceMotionActive = false;

    // --- Telemetry state ---
    std::unique_ptr<TelemetryReporter> m_telemetry;
    QString                            m_metricsEndpoint;
    QString                            m_metricsTag;
    bool                               m_metricsEnabled = false;
    double                             m_latestFpsSample = 0.0;
    int                                m_windowCount = 1;
    TelemetryTlsConfig                 m_tlsConfig;

    struct OverlayState {
        int  active = 0;
        int  allowed = 0;
        bool reduceMotion = false;
        bool operator==(const OverlayState& other) const {
            return active == other.active && allowed == other.allowed && reduceMotion == other.reduceMotion;
        }
    };

    std::optional<OverlayState> m_lastOverlayState;
    std::optional<OverlayState> m_lastOverlayTelemetryReported;
    std::optional<bool>         m_lastReduceMotionReported;
    std::optional<bool>         m_pendingReduceMotionState;
};
