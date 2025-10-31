#pragma once

#include <QObject>
#include <QDateTime>
#include <QList>
#include <QString>
#include <QVariantMap>

#include "grpc/TradingClient.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskTypes.hpp"
#include "utils/PerformanceGuard.hpp"

class OfflineRuntimeService : public QObject {
    Q_OBJECT

public:
    explicit OfflineRuntimeService(QObject* parent = nullptr);

    void setInstrument(const TradingClient::InstrumentConfig& config);
    void setHistoryLimit(int limit);
    void setAutoRunEnabled(bool enabled);
    void setStrategyConfig(const QVariantMap& config);
    void setDatasetPath(const QString& path);
    void setStreamingEnabled(bool enabled);
    void setStreamSnapshotPath(const QString& path);

public slots:
    void start();
    void stop();
    void refreshRisk();
    void startAutomation();
    void stopAutomation();

signals:
    void connectionStateChanged(const QString& state);
    void historyReady(const QList<OhlcvPoint>& candles);
    void riskReady(const RiskSnapshotData& snapshot);
    void guardReady(const PerformanceGuard& guard);
    void automationStateChanged(bool running);

private:
    void ensureDatasetLoaded();
    bool tryLoadStreamSnapshot(QList<OhlcvPoint>& out) const;
    QList<OhlcvPoint> limitedHistory() const;
    RiskSnapshotData buildRiskSnapshot() const;
    PerformanceGuard buildPerformanceGuard() const;
    void applyAutomationPreference();
    void buildSyntheticDataset(QList<OhlcvPoint>& out) const;

    TradingClient::InstrumentConfig m_instrument{};
    QVariantMap                    m_strategyConfig;
    QString                        m_datasetPath;
    QString                        m_loadedDatasetPath;
    QString                        m_streamSnapshotPath;
    QString                        m_loadedStreamSnapshotPath;
    QList<OhlcvPoint>              m_history;
    int                            m_historyLimit = 500;
    bool                           m_running = false;
    bool                           m_autoRunEnabled = false;
    bool                           m_automationRunning = false;
    bool                           m_datasetDirty = true;
    bool                           m_streamingEnabled = false;
};
