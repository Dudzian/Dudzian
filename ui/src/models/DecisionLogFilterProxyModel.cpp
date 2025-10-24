#include "DecisionLogFilterProxyModel.hpp"

#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QMetaType>
#include <QTextStream>
#include <QVariant>

#include <cmath>

namespace {
QString normalize(const QString& text)
{
    QString normalized = text;
    normalized.replace('\n', QLatin1Char(' '));
    normalized.replace('\r', QLatin1Char(' '));
    return normalized.trimmed();
}

std::optional<double> toOptionalDouble(const QVariant& value)
{
    if (!value.isValid()) {
        return std::nullopt;
    }

    if (value.metaType().id() == QMetaType::QString) {
        const QString text = value.toString().trimmed();
        if (text.isEmpty()) {
            return std::nullopt;
        }
        bool ok = false;
        const double numeric = text.toDouble(&ok);
        if (!ok || std::isnan(numeric)) {
            return std::nullopt;
        }
        return numeric;
    }

    bool ok = false;
    const double numeric = value.toDouble(&ok);
    if (!ok || std::isnan(numeric)) {
        return std::nullopt;
    }
    return numeric;
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

void DecisionLogFilterProxyModel::setTelemetryNamespaceFilter(const QString& telemetryNamespace)
{
    if (m_telemetryNamespaceFilter == telemetryNamespace) {
        return;
    }
    m_telemetryNamespaceFilter = telemetryNamespace;
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

void DecisionLogFilterProxyModel::setQuantityFilter(const QString& quantity)
{
    if (m_quantityFilter == quantity) {
        return;
    }
    m_quantityFilter = quantity;
    invalidateFilter();
    Q_EMIT filterChanged();
}

void DecisionLogFilterProxyModel::setPriceFilter(const QString& price)
{
    if (m_priceFilter == price) {
        return;
    }
    m_priceFilter = price;
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

void DecisionLogFilterProxyModel::setDetailsFilter(const QString& details)
{
    if (m_detailsFilter == details) {
        return;
    }
    m_detailsFilter = details;
    invalidateFilter();
    Q_EMIT filterChanged();
}

QVariant DecisionLogFilterProxyModel::minQuantityFilter() const
{
    if (!m_minQuantityFilter.has_value()) {
        return {};
    }
    return QVariant::fromValue(*m_minQuantityFilter);
}

void DecisionLogFilterProxyModel::setMinQuantityFilter(const QVariant& value)
{
    const std::optional<double> next = toOptionalDouble(value);
    bool changed = (m_minQuantityFilter != next);
    bool maxChanged = false;
    if (next.has_value() && m_maxQuantityFilter.has_value() && *next > *m_maxQuantityFilter) {
        m_maxQuantityFilter = next;
        maxChanged = true;
    }
    if (!changed && !maxChanged) {
        return;
    }
    m_minQuantityFilter = next;
    invalidateFilter();
    Q_EMIT filterChanged();
}

QVariant DecisionLogFilterProxyModel::maxQuantityFilter() const
{
    if (!m_maxQuantityFilter.has_value()) {
        return {};
    }
    return QVariant::fromValue(*m_maxQuantityFilter);
}

void DecisionLogFilterProxyModel::setMaxQuantityFilter(const QVariant& value)
{
    const std::optional<double> next = toOptionalDouble(value);
    bool changed = (m_maxQuantityFilter != next);
    bool minChanged = false;
    if (next.has_value() && m_minQuantityFilter.has_value() && *next < *m_minQuantityFilter) {
        m_minQuantityFilter = next;
        minChanged = true;
    }
    if (!changed && !minChanged) {
        return;
    }
    m_maxQuantityFilter = next;
    invalidateFilter();
    Q_EMIT filterChanged();
}

QVariant DecisionLogFilterProxyModel::minPriceFilter() const
{
    if (!m_minPriceFilter.has_value()) {
        return {};
    }
    return QVariant::fromValue(*m_minPriceFilter);
}

void DecisionLogFilterProxyModel::setMinPriceFilter(const QVariant& value)
{
    const std::optional<double> next = toOptionalDouble(value);
    bool changed = (m_minPriceFilter != next);
    bool maxChanged = false;
    if (next.has_value() && m_maxPriceFilter.has_value() && *next > *m_maxPriceFilter) {
        m_maxPriceFilter = next;
        maxChanged = true;
    }
    if (!changed && !maxChanged) {
        return;
    }
    m_minPriceFilter = next;
    invalidateFilter();
    Q_EMIT filterChanged();
}

QVariant DecisionLogFilterProxyModel::maxPriceFilter() const
{
    if (!m_maxPriceFilter.has_value()) {
        return {};
    }
    return QVariant::fromValue(*m_maxPriceFilter);
}

void DecisionLogFilterProxyModel::setMaxPriceFilter(const QVariant& value)
{
    const std::optional<double> next = toOptionalDouble(value);
    bool changed = (m_maxPriceFilter != next);
    bool minChanged = false;
    if (next.has_value() && m_minPriceFilter.has_value() && *next < *m_minPriceFilter) {
        m_minPriceFilter = next;
        minChanged = true;
    }
    if (!changed && !minChanged) {
        return;
    }
    m_maxPriceFilter = next;
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

void DecisionLogFilterProxyModel::clearAllFilters()
{
    bool changed = false;

    const auto resetString = [&](QString& value) {
        if (!value.isEmpty()) {
            value.clear();
            changed = true;
        }
    };

    const auto resetOptional = [&](std::optional<double>& value) {
        if (value.has_value()) {
            value.reset();
            changed = true;
        }
    };

    resetString(m_searchText);
    resetString(m_strategyFilter);
    resetString(m_regimeFilter);
    resetString(m_telemetryNamespaceFilter);
    resetString(m_environmentFilter);
    resetString(m_portfolioFilter);
    resetString(m_riskProfileFilter);
    resetString(m_scheduleFilter);
    resetString(m_sideFilter);
    resetString(m_decisionStateFilter);
    resetString(m_decisionModeFilter);
    resetString(m_decisionReasonFilter);
    resetString(m_eventFilter);
    resetString(m_quantityFilter);
    resetString(m_priceFilter);
    resetString(m_symbolFilter);
    resetString(m_detailsFilter);

    if (m_approvalFilter != All) {
        m_approvalFilter = All;
        changed = true;
    }

    resetOptional(m_minQuantityFilter);
    resetOptional(m_maxQuantityFilter);
    resetOptional(m_minPriceFilter);
    resetOptional(m_maxPriceFilter);

    if (m_startTimeFilter.isValid()) {
        m_startTimeFilter = QDateTime();
        changed = true;
    }

    if (m_endTimeFilter.isValid()) {
        m_endTimeFilter = QDateTime();
        changed = true;
    }

    if (!changed) {
        return;
    }

    invalidateFilter();
    Q_EMIT filterChanged();
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

    if (!m_telemetryNamespaceFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::TelemetryNamespaceRole).toString().contains(m_telemetryNamespaceFilter, Qt::CaseInsensitive)) {
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

    if (!m_quantityFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::QuantityRole).toString().contains(m_quantityFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (m_minQuantityFilter.has_value() || m_maxQuantityFilter.has_value()) {
        bool ok = false;
        double quantityValue = sourceModel()->data(idx, DecisionLogModel::QuantityRole).toDouble(&ok);
        if (!ok) {
            const QString quantityText = sourceModel()->data(idx, DecisionLogModel::QuantityRole).toString();
            quantityValue = quantityText.toDouble(&ok);
        }
        if (!ok) {
            return false;
        }
        if (m_minQuantityFilter.has_value() && quantityValue < *m_minQuantityFilter) {
            return false;
        }
        if (m_maxQuantityFilter.has_value() && quantityValue > *m_maxQuantityFilter) {
            return false;
        }
    }

    if (!m_priceFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::PriceRole).toString().contains(m_priceFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (m_minPriceFilter.has_value() || m_maxPriceFilter.has_value()) {
        bool ok = false;
        double priceValue = sourceModel()->data(idx, DecisionLogModel::PriceRole).toDouble(&ok);
        if (!ok) {
            const QString priceText = sourceModel()->data(idx, DecisionLogModel::PriceRole).toString();
            priceValue = priceText.toDouble(&ok);
        }
        if (!ok) {
            return false;
        }
        if (m_minPriceFilter.has_value() && priceValue < *m_minPriceFilter) {
            return false;
        }
        if (m_maxPriceFilter.has_value() && priceValue > *m_maxPriceFilter) {
            return false;
        }
    }

    if (!m_symbolFilter.isEmpty()) {
        if (!sourceModel()->data(idx, DecisionLogModel::SymbolRole).toString().contains(m_symbolFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    if (!m_detailsFilter.isEmpty()) {
        const QVariant detailsVariant = sourceModel()->data(idx, DecisionLogModel::DetailsRole);
        QString detailsText;
        if (detailsVariant.canConvert<QVariantMap>() || detailsVariant.canConvert<QVariantList>()) {
            detailsText = QString::fromUtf8(QJsonDocument::fromVariant(detailsVariant).toJson(QJsonDocument::Compact));
        } else {
            detailsText = detailsVariant.toString();
        }
        if (!detailsText.contains(m_detailsFilter, Qt::CaseInsensitive)) {
            return false;
        }
    }

    return true;
}

