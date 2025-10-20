#pragma once

#include <QSortFilterProxyModel>
#include <QString>

#include "AlertsModel.hpp"

class AlertsFilterProxyModel : public QSortFilterProxyModel {
    Q_OBJECT
    Q_PROPERTY(bool hideAcknowledged READ hideAcknowledged WRITE setHideAcknowledged NOTIFY filterChanged)
    Q_PROPERTY(SeverityFilter severityFilter READ severityFilter WRITE setSeverityFilter NOTIFY filterChanged)
    Q_PROPERTY(QString searchText READ searchText WRITE setSearchText NOTIFY filterChanged)
    Q_PROPERTY(SortMode sortMode READ sortMode WRITE setSortMode NOTIFY filterChanged)

public:
    enum SeverityFilter {
        AllSeverities = 0,
        WarningsAndCritical = 1,
        CriticalOnly = 2,
        WarningOnly = 3,
    };
    Q_ENUM(SeverityFilter)

    enum SortMode {
        NewestFirst = 0,
        OldestFirst = 1,
        SeverityDescending = 2,
        SeverityAscending = 3,
        TitleAscending = 4,
    };
    Q_ENUM(SortMode)

    explicit AlertsFilterProxyModel(QObject* parent = nullptr);

    bool hideAcknowledged() const { return m_hideAcknowledged; }
    void setHideAcknowledged(bool hide);

    SeverityFilter severityFilter() const { return m_severityFilter; }
    void setSeverityFilter(SeverityFilter filter);

    QString searchText() const { return m_searchText; }
    void setSearchText(const QString& text);

    SortMode sortMode() const { return m_sortMode; }
    void setSortMode(SortMode mode);

signals:
    void filterChanged();

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const override;
    bool lessThan(const QModelIndex& sourceLeft, const QModelIndex& sourceRight) const override;

private:
    bool matchesSeverity(int severity) const;

    bool           m_hideAcknowledged = false;
    SeverityFilter m_severityFilter = AllSeverities;
    QString        m_searchText;
    QString        m_searchNeedle;
    SortMode       m_sortMode = NewestFirst;
};
