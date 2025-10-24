#pragma once

#include <QSortFilterProxyModel>
#include <QUrl>

#include "models/DecisionLogModel.hpp"

class DecisionLogFilterProxyModel : public QSortFilterProxyModel {
    Q_OBJECT
    Q_PROPERTY(QString searchText READ searchText WRITE setSearchText NOTIFY filterChanged)
    Q_PROPERTY(int approvalFilter READ approvalFilter WRITE setApprovalFilter NOTIFY filterChanged)
    Q_PROPERTY(QString strategyFilter READ strategyFilter WRITE setStrategyFilter NOTIFY filterChanged)
    Q_PROPERTY(QString regimeFilter READ regimeFilter WRITE setRegimeFilter NOTIFY filterChanged)

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
};

