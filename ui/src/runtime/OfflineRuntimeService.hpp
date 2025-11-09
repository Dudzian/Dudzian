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
    QVariantMap alertPreferences() const { return m_alertPreferences; }
    void setAlertPreferences(const QVariantMap& preferences);
    QVariantMap buildAutoModeSnapshot() const;
    Q_INVOKABLE QVariantMap previewPreset(const QVariantMap& selector);

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
    void alertPreferencesChanged(const QVariantMap& preferences);

private:
    void ensureDatasetLoaded();
    QList<OhlcvPoint> limitedHistory() const;
    RiskSnapshotData buildRiskSnapshot() const;
    PerformanceGuard buildPerformanceGuard() const;
    void applyAutomationPreference();
    void buildSyntheticDataset(QList<OhlcvPoint>& out) const;
    QVariantList buildEquityCurveSeries() const;
    QVariantList buildRiskHeatmapCells(const RiskSnapshotData& risk) const;
    QVariantMap buildAutomationMetrics(const RiskSnapshotData& risk) const;
    QVariantList buildAutomationRecommendations(const QVariantMap& metrics, const RiskSnapshotData& risk) const;
    QVariantList buildRiskAlerts(const RiskSnapshotData& risk) const;
    QVariantMap buildPerformanceSummary() const;
    QVariantMap buildRecentPerformanceSummary(int hours = 24) const;
    QVariantMap buildPerformanceSummaryFor(const QList<OhlcvPoint>& candles) const;
    QVariantMap loadDecisionSnapshotFromBackend() const;
    QString resolvePresetPath(const QVariantMap& selector) const;
    static QString presetsDirectory();
    static QString normalisePresetKey(const QString& value);
    QVariantList buildRiskDiff(const QVariantMap& presetRisk,
                               const QVariantMap& championRisk) const;

    TradingClient::InstrumentConfig m_instrument{};
    QVariantMap                    m_strategyConfig;
    QString                        m_datasetPath;
    QString                        m_loadedDatasetPath;
    QList<OhlcvPoint>              m_history;
    int                            m_historyLimit = 500;
    bool                           m_running = false;
    bool                           m_autoRunEnabled = false;
    bool                           m_automationRunning = false;
    bool                           m_datasetDirty = true;
    QVariantMap                    m_alertPreferences;
};
