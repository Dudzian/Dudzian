#pragma once

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QTimer>
#include <QUrl>
#include <QUrlQuery>
#include <QVariantMap>
#include <functional>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskTypes.hpp"
#include "utils/PerformanceGuard.hpp"

class QNetworkReply;

class OfflineRuntimeBridge : public QObject {
    Q_OBJECT

public:
    explicit OfflineRuntimeBridge(QObject* parent = nullptr);
    ~OfflineRuntimeBridge() override;

    void setEndpoint(const QUrl& endpoint);
    void setInstrument(const TradingClient::InstrumentConfig& config);
    void setHistoryLimit(int limit);
    void setAutoRunEnabled(bool enabled);
    void setStrategyConfig(const QVariantMap& config);

public slots:
    void start();
    void stop();
    void refreshRiskNow();
    void startAutomation();
    void stopAutomation();

signals:
    void connectionStateChanged(const QString& state);
    void historyReceived(const QList<OhlcvPoint>& candles);
    void riskStateReceived(const RiskSnapshotData& snapshot);
    void performanceGuardUpdated(const PerformanceGuard& guard);
    void automationStateChanged(bool running);

private slots:
    void handlePollTick();

private:
    void fetchStatus();
    void fetchHistory();
    void fetchRisk();
    void fetchPerformanceGuard();
    void pushStrategyConfig();
    void applyConnectionState(const QString& state);
    void handleAutomationPayload(const QJsonObject& object);

    QList<OhlcvPoint> parseCandles(const QJsonArray& array) const;
    RiskSnapshotData parseRisk(const QJsonObject& object) const;
    PerformanceGuard parseGuard(const QJsonObject& object) const;

    void postJson(const QString& path, const QJsonObject& payload, std::function<void(const QJsonObject&)> onSuccess);
    void getJson(const QString& path, const QUrlQuery& query, std::function<void(const QJsonObject&)> onSuccess);
    void requestJson(const QNetworkRequest& request, std::function<void(const QJsonDocument&)> callback);
    QUrl buildUrl(const QString& path, const QUrlQuery& query = {}) const;

    QNetworkAccessManager m_network;
    QTimer                m_pollTimer;
    QUrl                  m_endpoint;
    TradingClient::InstrumentConfig m_instrument{};
    QVariantMap           m_strategyConfig;
    int                   m_historyLimit = 500;
    bool                  m_running = false;
    bool                  m_autoRunEnabled = false;
    bool                  m_automationRunning = false;
    QString               m_connectionState;
};
