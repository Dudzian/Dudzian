#include "DecisionLogFilterProxyModel.hpp"

#include <QDateTime>
#include <QFile>
#include <QTextStream>

namespace {
QString normalize(const QString& text)
{
    QString normalized = text;
    normalized.replace('\n', QLatin1Char(' '));
    normalized.replace('\r', QLatin1Char(' '));
    return normalized.trimmed();
}
}

DecisionLogFilterProxyModel::DecisionLogFilterProxyModel(QObject* parent)
    : QSortFilterProxyModel(parent)
{
    setDynamicSortFilter(true);
}

void DecisionLogFilterProxyModel::setSearchText(const QString& text)
{
    if (m_searchText == text) {
        return;
    }
    m_searchText = text;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setApprovalFilter(int mode)
{
    if (m_approvalFilter == mode) {
        return;
    }
    m_approvalFilter = mode;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setStrategyFilter(const QString& strategy)
{
    if (m_strategyFilter == strategy) {
        return;
    }
    m_strategyFilter = strategy;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setRegimeFilter(const QString& regime)
{
    if (m_regimeFilter == regime) {
        return;
    }
    m_regimeFilter = regime;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setEnvironmentFilter(const QString& environment)
{
    if (m_environmentFilter == environment) {
        return;
    }
    m_environmentFilter = environment;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setPortfolioFilter(const QString& portfolio)
{
    if (m_portfolioFilter == portfolio) {
        return;
    }
    m_portfolioFilter = portfolio;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setRiskProfileFilter(const QString& riskProfile)
{
    if (m_riskProfileFilter == riskProfile) {
        return;
    }
    m_riskProfileFilter = riskProfile;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setScheduleFilter(const QString& schedule)
{
    if (m_scheduleFilter == schedule) {
        return;
    }
    m_scheduleFilter = schedule;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setSideFilter(const QString& side)
{
    if (m_sideFilter == side) {
        return;
    }
    m_sideFilter = side;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setDecisionStateFilter(const QString& state)
{
    if (m_decisionStateFilter == state) {
        return;
    }
    m_decisionStateFilter = state;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setDecisionModeFilter(const QString& mode)
{
    if (m_decisionModeFilter == mode) {
        return;
    }
    m_decisionModeFilter = mode;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setDecisionReasonFilter(const QString& reason)
{
    if (m_decisionReasonFilter == reason) {
        return;
    }
    m_decisionReasonFilter = reason;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setEventFilter(const QString& event)
{
    if (m_eventFilter == event) {
        return;
    }
    m_eventFilter = event;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setSymbolFilter(const QString& symbol)
{
    if (m_symbolFilter == symbol) {
        return;
    }
    m_symbolFilter = symbol;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setStartTimeFilter(const QDateTime& value)
{
    QDateTime normalized = value.isValid() ? value.toUTC() : QDateTime();
    bool changed = (m_startTimeFilter != normalized);
    if (normalized.isValid() && m_endTimeFilter.isValid() && normalized > m_endTimeFilter) {
        m_endTimeFilter = normalized;
        changed = true;
    }
    if (!changed) {
        return;
    }
    m_startTimeFilter = normalized;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setEndTimeFilter(const QDateTime& value)
{
    QDateTime normalized = value.isValid() ? value.toUTC() : QDateTime();
    bool changed = (m_endTimeFilter != normalized);
    if (normalized.isValid() && m_startTimeFilter.isValid() && normalized < m_startTimeFilter) {
        m_startTimeFilter = normalized;
        changed = true;
    }
    if (!changed) {
        return;
    }
    m_endTimeFilter = normalized;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::clearStartTimeFilter()
{
    setStartTimeFilter(QDateTime());
}

void DecisionLogFilterProxyModel::clearEndTimeFilter()
{
    setEndTimeFilter(QDateTime());
}

bool DecisionLogFilterProxyModel::exportFilteredToCsv(const QUrl& destination) const
{
    if (!destination.isLocalFile()) {
        return false;
    }
    QFile file(destination.toLocalFile());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream stream(&file);
    stream << "timestamp,event,environment,portfolio,risk_profile,schedule,strategy,symbol,side,quantity,price,approved,state,reason,mode,regime" << '\n';

    for (int row = 0; row < rowCount(); ++row) {
        const QModelIndex idx = index(row, 0);
        const QString timestamp = data(idx, DecisionLogModel::TimestampDisplayRole).toString();
        const QString event = normalize(data(idx, DecisionLogModel::EventRole).toString());
        const QString environment = normalize(data(idx, DecisionLogModel::EnvironmentRole).toString());
        const QString portfolio = normalize(data(idx, DecisionLogModel::PortfolioRole).toString());
        const QString risk = normalize(data(idx, DecisionLogModel::RiskProfileRole).toString());
        const QString schedule = normalize(data(idx, DecisionLogModel::ScheduleRole).toString());
        const QString strategy = normalize(data(idx, DecisionLogModel::StrategyRole).toString());
        const QString symbol = normalize(data(idx, DecisionLogModel::SymbolRole).toString());
        const QString side = normalize(data(idx, DecisionLogModel::SideRole).toString());
        const QString quantity = normalize(data(idx, DecisionLogModel::QuantityRole).toString());
        const QString price = normalize(data(idx, DecisionLogModel::PriceRole).toString());
        const QString approved = data(idx, DecisionLogModel::ApprovedRole).toBool() ? QStringLiteral("true") : QStringLiteral("false");
        const QString state = normalize(data(idx, DecisionLogModel::DecisionStateRole).toString());
        const QString reason = normalize(data(idx, DecisionLogModel::DecisionReasonRole).toString());
        const QString mode = normalize(data(idx, DecisionLogModel::DecisionModeRole).toString());
        const QString regime = normalize(data(idx, DecisionLogModel::TelemetryNamespaceRole).toString());

        const QStringList values = {
            timestamp,
            event,
            environment,
            portfolio,
            risk,
            schedule,
            strategy,
            symbol,
            side,
            quantity,
            price,
            approved,
            state,
            reason,
            mode,
            regime
        };

        QStringList escaped;
        escaped.reserve(values.size());
        for (const QString& value : values) {
            QString field = value;
            field.replace('"', QStringLiteral(""""));
            escaped.append('"' + field + '"');
        }
        stream << escaped.join(',') << '\n';
    }
    return true;
}

bool DecisionLogFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const
{
    const QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    if (idx.row() < 0) {
        return false;
    }

    const bool hasTimeConstraint = m_startTimeFilter.isValid() || m_endTimeFilter.isValid();
    QDateTime timestampUtc;
    if (hasTimeConstraint) {
        timestampUtc = sourceModel()->data(idx, DecisionLogModel::TimestampRole).toDateTime().toUTC();
        if (!timestampUtc.isValid()) {
            return false;
        }
        if (m_startTimeFilter.isValid() && timestampUtc < m_startTimeFilter) {
            return false;
        }
        if (m_endTimeFilter.isValid() && timestampUtc > m_endTimeFilter) {
            return false;
        }
    }

    if (!m_searchText.isEmpty()) {
        const QString haystack = (sourceModel()->data(idx, DecisionLogModel::EventRole).toString() + ' '
                                  + sourceModel()->data(idx, DecisionLogModel::StrategyRole).toString() + ' '
                                  + sourceModel()->data(idx, DecisionLogModel::DecisionReasonRole).toString() + ' '
                                  + sourceModel()->data(idx, DecisionLogModel::SymbolRole).toString()).toLower();
        if (!haystack.contains(m_searchText.toLower())) {
            return false;
        }
    }

    if (m_approvalFilter == ApprovedOnly && !sourceModel()->data(idx, DecisionLogModel::ApprovedRole).toBool()) {
        return false;
    }
    if (m_approvalFilter == PendingOnly && sourceModel()->data(idx, DecisionLogModel::ApprovedRole).toBool()) {
        return false;
    }

    if (!m_strategyFilter.isEmpty()) {
        if (sourceModel()->data(idx, DecisionLogModel::StrategyRole).toString().compare(m_strategyFilter, Qt::CaseInsensitive) != 0) {
            return false;
        }
    }

    if (!m_regimeFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::TelemetryNamespaceRole).toString().contains(m_regimeFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_environmentFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::EnvironmentRole).toString().contains(m_environmentFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_portfolioFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::PortfolioRole).toString().contains(m_portfolioFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_riskProfileFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::RiskProfileRole).toString().contains(m_riskProfileFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_scheduleFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::ScheduleRole).toString().contains(m_scheduleFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_sideFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::SideRole).toString().contains(m_sideFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_decisionStateFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::DecisionStateRole).toString().contains(m_decisionStateFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_decisionModeFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::DecisionModeRole).toString().contains(m_decisionModeFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_decisionReasonFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::DecisionReasonRole).toString().contains(m_decisionReasonFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_eventFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::EventRole).toString().contains(m_eventFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_symbolFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::SymbolRole).toString().contains(m_symbolFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    return true;
}

