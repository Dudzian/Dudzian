#pragma once

#include <QObject>
#include <QUrl>
#include <QVariantMap>
#include <memory>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskTypes.hpp"
#include "utils/PerformanceGuard.hpp"

class OfflineRuntimeService;

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
    void setDatasetPath(const QString& path);
    void setStreamSnapshotPath(const QString& path);
    void setStreamingEnabled(bool enabled);

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

private:
    void ensureService();
    void configureService();
    void applyConnectionState(const QString& state);

    std::unique_ptr<OfflineRuntimeService> m_service;
    QUrl                                   m_endpoint;
    TradingClient::InstrumentConfig        m_instrument{};
    QVariantMap                            m_strategyConfig;
    QString                                m_datasetPath;
    QString                                m_streamSnapshotPath;
    int                                    m_historyLimit = 500;
    bool                                   m_running = false;
    bool                                   m_autoRunEnabled = false;
    bool                                   m_streamingEnabled = false;
    QString                                m_connectionState;
};
