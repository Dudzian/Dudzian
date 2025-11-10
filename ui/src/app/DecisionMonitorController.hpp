#pragma once

#include <QDateTime>
#include <QModelIndex>
#include <QPointer>
#include <QObject>
#include <QVariant>

class DecisionLogModel;

class DecisionMonitorController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList recentDecisions READ recentDecisions NOTIFY recentDecisionsChanged)
    Q_PROPERTY(QVariantMap outcomeSummary READ outcomeSummary NOTIFY outcomeSummaryChanged)
    Q_PROPERTY(QVariantList scheduleSummary READ scheduleSummary NOTIFY scheduleSummaryChanged)
    Q_PROPERTY(QVariantList flaggedDecisions READ flaggedDecisions NOTIFY flaggedDecisionsChanged)
    Q_PROPERTY(QDateTime lastUpdated READ lastUpdated NOTIFY lastUpdatedChanged)
    Q_PROPERTY(int recentLimit READ recentLimit WRITE setRecentLimit NOTIFY recentLimitChanged)

public:
    explicit DecisionMonitorController(QObject* parent = nullptr);

    QVariantList recentDecisions() const { return m_recentDecisions; }
    QVariantMap outcomeSummary() const { return m_outcomeSummary; }
    QVariantList scheduleSummary() const { return m_scheduleSummary; }
    QVariantList flaggedDecisions() const { return m_flaggedDecisions; }
    QDateTime lastUpdated() const { return m_lastUpdated; }
    int recentLimit() const { return m_recentLimit; }

    Q_INVOKABLE void refresh();
    void setDecisionModel(DecisionLogModel* model);
    void setRecentLimit(int limit);

signals:
    void recentDecisionsChanged();
    void outcomeSummaryChanged();
    void scheduleSummaryChanged();
    void flaggedDecisionsChanged();
    void lastUpdatedChanged();
    void recentLimitChanged();

private:
    void rebuildSummaries();
    void connectToModel();
    void disconnectFromModel();

    QPointer<DecisionLogModel> m_decisionModel;
    QVariantList m_recentDecisions;
    QVariantMap m_outcomeSummary;
    QVariantList m_scheduleSummary;
    QVariantList m_flaggedDecisions;
    QDateTime m_lastUpdated;
    int m_recentLimit = 50;
};

