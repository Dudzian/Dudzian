#pragma once

#include <QObject>
#include <QQmlApplicationEngine>
#include <QPointer>
#include <QCommandLineParser>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "utils/PerformanceGuard.hpp"

class Application : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString connectionStatus READ connectionStatus NOTIFY connectionStatusChanged)
    Q_PROPERTY(PerformanceGuard performanceGuard READ performanceGuard NOTIFY performanceGuardChanged)

public:
    explicit Application(QQmlApplicationEngine& engine, QObject* parent = nullptr);

    void configureParser(QCommandLineParser& parser) const;
    bool applyParser(const QCommandLineParser& parser);

    QString connectionStatus() const { return m_connectionStatus; }
    PerformanceGuard performanceGuard() const { return m_guard; }

public slots:
    void start();
    void stop();

signals:
    void connectionStatusChanged();
    void performanceGuardChanged();

private slots:
    void handleHistory(const QList<OhlcvPoint>& candles);
    void handleCandle(const OhlcvPoint& candle);

private:
    void exposeToQml();

    QQmlApplicationEngine& m_engine;
    OhlcvListModel m_ohlcvModel;
    TradingClient m_client;
    QString m_connectionStatus = QStringLiteral("idle");
    PerformanceGuard m_guard;
    int m_maxSamples = 10240;
};
