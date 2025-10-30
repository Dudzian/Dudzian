#pragma once

#include <QObject>
#include <QPointer>
#include <QVariantList>

class RiskHistoryModel;
class RiskStateModel;
class DecisionLogModel;

class ResultsDashboardModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(double cumulativeReturn READ cumulativeReturn NOTIFY summaryChanged)
    Q_PROPERTY(double maxDrawdown READ maxDrawdown NOTIFY summaryChanged)
    Q_PROPERTY(double annualizedVolatility READ annualizedVolatility NOTIFY summaryChanged)
    Q_PROPERTY(double sharpeRatio READ sharpeRatio NOTIFY summaryChanged)
    Q_PROPERTY(double winRate READ winRate NOTIFY summaryChanged)
    Q_PROPERTY(double averageExposureUtilization READ averageExposureUtilization NOTIFY summaryChanged)
    Q_PROPERTY(int sampleCount READ sampleCount NOTIFY summaryChanged)
    Q_PROPERTY(QVariantList equityTimeline READ equityTimeline NOTIFY timelineChanged)

public:
    explicit ResultsDashboardModel(QObject* parent = nullptr);

    double cumulativeReturn() const { return m_cumulativeReturn; }
    double maxDrawdown() const { return m_maxDrawdown; }
    double annualizedVolatility() const { return m_annualizedVolatility; }
    double sharpeRatio() const { return m_sharpeRatio; }
    double winRate() const { return m_winRate; }
    double averageExposureUtilization() const { return m_averageExposure; }
    int sampleCount() const { return m_sampleCount; }
    QVariantList equityTimeline() const { return m_equityTimeline; }

    void setRiskHistoryModel(RiskHistoryModel* history);
    void setRiskStateModel(RiskStateModel* riskModel);
    void setDecisionLogModel(DecisionLogModel* decisionModel);

    Q_INVOKABLE QVariantMap summarySnapshot() const;
    Q_INVOKABLE QVariantList exposureHighlights() const;

signals:
    void summaryChanged();
    void timelineChanged();

private:
    void recompute();
    double computeSharpe(const QVector<double>& returns) const;
    QVariantList buildTimeline() const;

    QPointer<RiskHistoryModel> m_history;
    QPointer<RiskStateModel> m_riskModel;
    QPointer<DecisionLogModel> m_decisionModel;

    double m_cumulativeReturn = 0.0;
    double m_maxDrawdown = 0.0;
    double m_annualizedVolatility = 0.0;
    double m_sharpeRatio = 0.0;
    double m_winRate = 0.0;
    double m_averageExposure = 0.0;
    int m_sampleCount = 0;
    QVariantList m_equityTimeline;
};
