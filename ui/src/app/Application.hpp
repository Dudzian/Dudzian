#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>

#include <memory>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "utils/PerformanceGuard.hpp"
#include "utils/FrameRateMonitor.hpp"

class Application : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString connectionStatus READ connectionStatus NOTIFY connectionStatusChanged)
    Q_PROPERTY(PerformanceGuard performanceGuard READ performanceGuard NOTIFY performanceGuardChanged)
    Q_PROPERTY(bool reduceMotionActive READ reduceMotionActive NOTIFY reduceMotionActiveChanged)
    Q_PROPERTY(QString instrumentLabel READ instrumentLabel NOTIFY instrumentChanged)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);

    void configureParser(QCommandLineParser& parser) const;
    bool applyParser(const QCommandLineParser& parser);

    QString connectionStatus() const { return m_connectionStatus; }
    PerformanceGuard performanceGuard() const { return m_guard; }
    QString instrumentLabel() const;
    bool reduceMotionActive() const { return m_reduceMotionActive; }

public slots:
    void start();
    void stop();

signals:
    void connectionStatusChanged();
    void performanceGuardChanged();
    void instrumentChanged();
    void reduceMotionActiveChanged();

private slots:
    void handleHistory(const QList<OhlcvPoint>& candles);
    void handleCandle(const OhlcvPoint& candle);

private:
    void exposeToQml();
    void ensureFrameMonitor();
    void attachWindow(QObject* object);

    QQmlApplicationEngine& m_engine;
    OhlcvListModel m_ohlcvModel;
    TradingClient m_client;
    QString m_connectionStatus = QStringLiteral("idle");
    PerformanceGuard m_guard;
    int m_maxSamples = 10240;
    TradingClient::InstrumentConfig m_instrument;
    std::unique_ptr<FrameRateMonitor> m_frameMonitor;
    bool m_reduceMotionActive = false;
};
