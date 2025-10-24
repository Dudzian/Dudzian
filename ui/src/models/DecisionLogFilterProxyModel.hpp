#pragma once

#include <QDateTime>
#include <QSortFilterProxyModel>
#include <QUrl>

#include "models/DecisionLogModel.hpp"

class DecisionLogFilterProxyModel : public QSortFilterProxyModel {
    Q_OBJECT
    Q_PROPERTY(QString searchText READ searchText WRITE setSearchText NOTIFY filterChanged)
    Q_PROPERTY(int approvalFilter READ approvalFilter WRITE setApprovalFilter NOTIFY filterChanged)
    Q_PROPERTY(QString strategyFilter READ strategyFilter WRITE setStrategyFilter NOTIFY filterChanged)
    Q_PROPERTY(QString regimeFilter READ regimeFilter WRITE setRegimeFilter NOTIFY filterChanged)
    Q_PROPERTY(QString environmentFilter READ environmentFilter WRITE setEnvironmentFilter NOTIFY filterChanged)
    Q_PROPERTY(QString portfolioFilter READ portfolioFilter WRITE setPortfolioFilter NOTIFY filterChanged)
    Q_PROPERTY(QString riskProfileFilter READ riskProfileFilter WRITE setRiskProfileFilter NOTIFY filterChanged)
    Q_PROPERTY(QString scheduleFilter READ scheduleFilter WRITE setScheduleFilter NOTIFY filterChanged)
    Q_PROPERTY(QString sideFilter READ sideFilter WRITE setSideFilter NOTIFY filterChanged)
    Q_PROPERTY(QString decisionStateFilter READ decisionStateFilter WRITE setDecisionStateFilter NOTIFY filterChanged)
    Q_PROPERTY(QString decisionModeFilter READ decisionModeFilter WRITE setDecisionModeFilter NOTIFY filterChanged)
    Q_PROPERTY(QString decisionReasonFilter READ decisionReasonFilter WRITE setDecisionReasonFilter NOTIFY filterChanged)
    Q_PROPERTY(QString eventFilter READ eventFilter WRITE setEventFilter NOTIFY filterChanged)
    Q_PROPERTY(QString symbolFilter READ symbolFilter WRITE setSymbolFilter NOTIFY filterChanged)
    Q_PROPERTY(QDateTime startTimeFilter READ startTimeFilter WRITE setStartTimeFilter NOTIFY filterChanged)
    Q_PROPERTY(QDateTime endTimeFilter READ endTimeFilter WRITE setEndTimeFilter NOTIFY filterChanged)

public:
    enum ApprovalMode {
        All = 0,
        ApprovedOnly,
        PendingOnly,
    };
    Q_ENUM(ApprovalMode)

    explicit DecisionLogFilterProxyModel(QObject* parent = nullptr);

    QString searchText() const { return m_searchText; }
    void setSearchText(const QString& text);

    int approvalFilter() const { return m_approvalFilter; }
    void setApprovalFilter(int mode);

    QString strategyFilter() const { return m_strategyFilter; }
    void setStrategyFilter(const QString& strategy);

    QString regimeFilter() const { return m_regimeFilter; }
    void setRegimeFilter(const QString& regime);

    QString environmentFilter() const { return m_environmentFilter; }
    void setEnvironmentFilter(const QString& environment);

    QString portfolioFilter() const { return m_portfolioFilter; }
    void setPortfolioFilter(const QString& portfolio);

    QString riskProfileFilter() const { return m_riskProfileFilter; }
    void setRiskProfileFilter(const QString& riskProfile);

    QString scheduleFilter() const { return m_scheduleFilter; }
    void setScheduleFilter(const QString& schedule);

    QString sideFilter() const { return m_sideFilter; }
    void setSideFilter(const QString& side);

    QString decisionStateFilter() const { return m_decisionStateFilter; }
    void setDecisionStateFilter(const QString& state);

    QString decisionModeFilter() const { return m_decisionModeFilter; }
    void setDecisionModeFilter(const QString& mode);

    QString decisionReasonFilter() const { return m_decisionReasonFilter; }
    void setDecisionReasonFilter(const QString& reason);

    QString eventFilter() const { return m_eventFilter; }
    void setEventFilter(const QString& event);

    QString symbolFilter() const { return m_symbolFilter; }
    void setSymbolFilter(const QString& symbol);

    QDateTime startTimeFilter() const { return m_startTimeFilter; }
    void setStartTimeFilter(const QDateTime& value);

    QDateTime endTimeFilter() const { return m_endTimeFilter; }
    void setEndTimeFilter(const QDateTime& value);

    Q_INVOKABLE void clearStartTimeFilter();
    Q_INVOKABLE void clearEndTimeFilter();

    Q_INVOKABLE bool exportFilteredToCsv(const QUrl& destination) const;

signals:
    void filterChanged();

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const override;

private:
    QString m_searchText;
    int     m_approvalFilter = All;
    QString m_strategyFilter;
    QString m_regimeFilter;
    QString m_environmentFilter;
    QString m_portfolioFilter;
    QString m_riskProfileFilter;
    QString m_scheduleFilter;
    QString m_sideFilter;
    QString m_decisionStateFilter;
    QString m_decisionModeFilter;
    QString m_decisionReasonFilter;
    QString m_eventFilter;
    QString m_symbolFilter;
    QDateTime m_startTimeFilter;
    QDateTime m_endTimeFilter;
};

