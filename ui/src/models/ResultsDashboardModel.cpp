#include "ResultsDashboardModel.hpp"

#include <QAbstractItemModel>
#include <QDateTime>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>
#include <QtMath>

#include "models/DecisionLogModel.hpp"
#include "models/RiskHistoryModel.hpp"
#include "models/RiskStateModel.hpp"

namespace {
double safeDivision(double numerator, double denominator)
{
    if (qFuzzyIsNull(denominator))
        return 0.0;
    return numerator / denominator;
}
}

ResultsDashboardModel::ResultsDashboardModel(QObject* parent)
    : QObject(parent)
{
}

void ResultsDashboardModel::setRiskHistoryModel(RiskHistoryModel* history)
{
    if (m_history == history)
        return;

    if (m_history)
        disconnect(m_history, nullptr, this, nullptr);

    m_history = history;
    if (m_history) {
        connect(m_history,
                &RiskHistoryModel::historyChanged,
                this,
                &ResultsDashboardModel::recompute);
        connect(m_history,
                &RiskHistoryModel::snapshotRecorded,
                this,
                &ResultsDashboardModel::recompute);
    }
    recompute();
}

void ResultsDashboardModel::setRiskStateModel(RiskStateModel* riskModel)
{
    if (m_riskModel == riskModel)
        return;
    if (m_riskModel)
        disconnect(m_riskModel, nullptr, this, nullptr);
    m_riskModel = riskModel;
    if (m_riskModel) {
        connect(m_riskModel,
                &RiskStateModel::riskStateChanged,
                this,
                &ResultsDashboardModel::recompute);
    }
    recompute();
}

void ResultsDashboardModel::setDecisionLogModel(DecisionLogModel* decisionModel)
{
    if (m_decisionModel == decisionModel)
        return;
    if (m_decisionModel)
        disconnect(m_decisionModel, nullptr, this, nullptr);
    m_decisionModel = decisionModel;
    if (m_decisionModel) {
        connect(m_decisionModel,
                &DecisionLogModel::dataChanged,
                this,
                &ResultsDashboardModel::recompute);
        connect(m_decisionModel,
                &DecisionLogModel::modelReset,
                this,
                &ResultsDashboardModel::recompute);
        connect(m_decisionModel,
                &DecisionLogModel::rowsInserted,
                this,
                &ResultsDashboardModel::recompute);
    }
    recompute();
}

QVariantMap ResultsDashboardModel::summarySnapshot() const
{
    QVariantMap map;
    map.insert(QStringLiteral("cumulativeReturn"), m_cumulativeReturn);
    map.insert(QStringLiteral("maxDrawdown"), m_maxDrawdown);
    map.insert(QStringLiteral("annualizedVolatility"), m_annualizedVolatility);
    map.insert(QStringLiteral("sharpeRatio"), m_sharpeRatio);
    map.insert(QStringLiteral("winRate"), m_winRate);
    map.insert(QStringLiteral("averageExposureUtilization"), m_averageExposure);
    map.insert(QStringLiteral("sampleCount"), m_sampleCount);
    return map;
}

QVariantList ResultsDashboardModel::exposureHighlights() const
{
    QVariantList highlights;
    if (!m_riskModel)
        return highlights;

    for (int row = 0; row < m_riskModel->rowCount(); ++row) {
        const QModelIndex idx = m_riskModel->index(row, 0);
        QVariantMap item;
        item.insert(QStringLiteral("code"), m_riskModel->data(idx, RiskStateModel::CodeRole));
        item.insert(QStringLiteral("current"), m_riskModel->data(idx, RiskStateModel::CurrentValueRole));
        item.insert(QStringLiteral("threshold"), m_riskModel->data(idx, RiskStateModel::ThresholdValueRole));
        item.insert(QStringLiteral("breached"), m_riskModel->data(idx, RiskStateModel::BreachRole));
        highlights.append(item);
    }
    return highlights;
}

void ResultsDashboardModel::recompute()
{
    QVector<double> equitySeries;
    QVector<double> stepReturns;
    QVector<double> exposureSeries;

    if (m_history) {
        const int rows = m_history->rowCount();
        equitySeries.reserve(rows);
        exposureSeries.reserve(rows);
        for (int row = 0; row < rows; ++row) {
            const QModelIndex idx = m_history->index(row, 0);
            const double value = m_history->data(idx, RiskHistoryModel::PortfolioValueRole).toDouble();
            const double exposure = m_history->data(idx, RiskHistoryModel::MaxExposureUtilizationRole).toDouble();
            if (value > 0.0)
                equitySeries.append(value);
            else
                equitySeries.append(0.0);
            exposureSeries.append(exposure);
        }
        for (int i = 1; i < equitySeries.size(); ++i) {
            const double previous = equitySeries.at(i - 1);
            const double current = equitySeries.at(i);
            if (previous > 0.0)
                stepReturns.append((current - previous) / previous);
        }
        m_sampleCount = equitySeries.size();
    } else {
        m_sampleCount = 0;
    }

    if (!equitySeries.isEmpty()) {
        const double firstValue = equitySeries.first();
        const double lastValue = equitySeries.last();
        m_cumulativeReturn = firstValue > 0.0 ? (lastValue - firstValue) / firstValue : 0.0;

        double peak = equitySeries.first();
        double maxDrawdownValue = 0.0;
        for (double value : equitySeries) {
            peak = qMax(peak, value);
            if (peak > 0.0)
                maxDrawdownValue = qMax(maxDrawdownValue, (peak - value) / peak);
        }
        m_maxDrawdown = maxDrawdownValue;

        double exposureSum = 0.0;
        for (double exposure : exposureSeries)
            exposureSum += exposure;
        m_averageExposure = exposureSeries.isEmpty() ? 0.0 : exposureSum / exposureSeries.size();
    } else {
        m_cumulativeReturn = 0.0;
        m_maxDrawdown = 0.0;
        m_averageExposure = 0.0;
    }

    if (!stepReturns.isEmpty()) {
        double mean = 0.0;
        for (double r : stepReturns)
            mean += r;
        mean /= stepReturns.size();

        double variance = 0.0;
        for (double r : stepReturns)
            variance += qPow(r - mean, 2);
        variance = stepReturns.size() > 1 ? variance / (stepReturns.size() - 1) : 0.0;

        const double dailyVolatility = qSqrt(variance);
        m_annualizedVolatility = dailyVolatility * qSqrt(252.0);
        m_sharpeRatio = computeSharpe(stepReturns);

        int wins = 0;
        for (double r : stepReturns) {
            if (r > 0.0)
                ++wins;
        }
        m_winRate = safeDivision(static_cast<double>(wins), static_cast<double>(stepReturns.size()));
    } else {
        m_annualizedVolatility = 0.0;
        m_sharpeRatio = 0.0;
        m_winRate = 0.0;
    }

    const QVariantList timeline = buildTimeline();
    if (timeline != m_equityTimeline) {
        m_equityTimeline = timeline;
        Q_EMIT timelineChanged();
    }

    Q_EMIT summaryChanged();
}

double ResultsDashboardModel::computeSharpe(const QVector<double>& returns) const
{
    if (returns.isEmpty())
        return 0.0;
    double mean = 0.0;
    for (double r : returns)
        mean += r;
    mean /= returns.size();

    double variance = 0.0;
    for (double r : returns)
        variance += qPow(r - mean, 2);
    variance = returns.size() > 1 ? variance / (returns.size() - 1) : 0.0;
    const double stddev = qSqrt(variance);
    if (qFuzzyIsNull(stddev))
        return 0.0;
    return (mean * 252.0) / stddev;
}

QVariantList ResultsDashboardModel::buildTimeline() const
{
    QVariantList timeline;
    if (!m_history)
        return timeline;

    const int rows = m_history->rowCount();
    timeline.reserve(rows);
    for (int row = 0; row < rows; ++row) {
        const QModelIndex idx = m_history->index(row, 0);
        QVariantMap entry;
        entry.insert(QStringLiteral("timestamp"),
                     m_history->data(idx, RiskHistoryModel::TimestampRole));
        entry.insert(QStringLiteral("portfolio"),
                     m_history->data(idx, RiskHistoryModel::PortfolioValueRole));
        entry.insert(QStringLiteral("drawdown"),
                     m_history->data(idx, RiskHistoryModel::DrawdownRole));
        entry.insert(QStringLiteral("breach"),
                     m_history->data(idx, RiskHistoryModel::HasBreachRole));
        timeline.append(entry);
    }
    return timeline;
}
