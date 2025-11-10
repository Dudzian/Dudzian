#include "DecisionMonitorController.hpp"

#include "models/DecisionLogModel.hpp"

#include <QAbstractItemModel>
#include <QHash>
#include <QList>
#include <QPersistentModelIndex>
#include <QStringList>
#include <QVariantMap>
#include <QVector>

#include <algorithm>

namespace {
constexpr int kFlaggedLimit = 20;
}

DecisionMonitorController::DecisionMonitorController(QObject* parent)
    : QObject(parent)
{
}

void DecisionMonitorController::setDecisionModel(DecisionLogModel* model)
{
    if (m_decisionModel == model)
        return;

    disconnectFromModel();
    m_decisionModel = model;
    connectToModel();
    refresh();
}

void DecisionMonitorController::setRecentLimit(int limit)
{
    const int sanitized = limit <= 0 ? 1 : limit;
    if (m_recentLimit == sanitized)
        return;
    m_recentLimit = sanitized;
    Q_EMIT recentLimitChanged();
    rebuildSummaries();
}

void DecisionMonitorController::refresh()
{
    rebuildSummaries();
}

void DecisionMonitorController::disconnectFromModel()
{
    if (!m_decisionModel)
        return;
    disconnect(m_decisionModel, nullptr, this, nullptr);
}

void DecisionMonitorController::connectToModel()
{
    if (!m_decisionModel)
        return;

    connect(m_decisionModel, &DecisionLogModel::countChanged, this, &DecisionMonitorController::rebuildSummaries);
    connect(m_decisionModel, &QAbstractItemModel::modelReset, this, &DecisionMonitorController::rebuildSummaries);
    connect(m_decisionModel,
            &QAbstractItemModel::rowsInserted,
            this,
            [this](const QModelIndex&, int, int) { rebuildSummaries(); });
    connect(m_decisionModel,
            &QAbstractItemModel::rowsRemoved,
            this,
            [this](const QModelIndex&, int, int) { rebuildSummaries(); });
    connect(m_decisionModel,
            &QAbstractItemModel::dataChanged,
            this,
            [this](const QModelIndex&, const QModelIndex&, const QVector<int>&) { rebuildSummaries(); });
    connect(m_decisionModel,
            &QAbstractItemModel::layoutChanged,
            this,
            [this](const QList<QPersistentModelIndex>&, QAbstractItemModel::LayoutChangeHint) {
                rebuildSummaries();
            });
}

void DecisionMonitorController::rebuildSummaries()
{
    QVariantList recent;
    QVariantList flagged;
    QVariantMap outcome;
    QVariantList schedules;
    QDateTime updated;

    outcome.insert(QStringLiteral("total"), 0);
    outcome.insert(QStringLiteral("executed"), 0);
    outcome.insert(QStringLiteral("rejected"), 0);
    outcome.insert(QStringLiteral("pending"), 0);
    outcome.insert(QStringLiteral("approved"), 0);
    outcome.insert(QStringLiteral("declined"), 0);
    outcome.insert(QStringLiteral("manual"), 0);
    outcome.insert(QStringLiteral("automated"), 0);

    if (!m_decisionModel) {
        if (m_recentDecisions != recent) {
            m_recentDecisions = recent;
            Q_EMIT recentDecisionsChanged();
        }
        if (m_flaggedDecisions != flagged) {
            m_flaggedDecisions = flagged;
            Q_EMIT flaggedDecisionsChanged();
        }
        if (m_outcomeSummary != outcome) {
            m_outcomeSummary = outcome;
            Q_EMIT outcomeSummaryChanged();
        }
        if (m_scheduleSummary != schedules) {
            m_scheduleSummary = schedules;
            Q_EMIT scheduleSummaryChanged();
        }
        if (m_lastUpdated.isValid()) {
            m_lastUpdated = {};
            Q_EMIT lastUpdatedChanged();
        }
        return;
    }

    const int total = m_decisionModel->rowCount();
    outcome[QStringLiteral("total")] = total;

    const int limit = qMax(1, m_recentLimit);
    const QHash<int, QByteArray> roleNames = m_decisionModel->roleNames();

    struct ScheduleStats {
        int total = 0;
        int executed = 0;
        int rejected = 0;
        QVariantMap lastDecision;
    };

    QHash<QString, ScheduleStats> scheduleStats;

    int appended = 0;
    int flaggedCount = 0;

    for (int row = total - 1; row >= 0; --row) {
        const QModelIndex index = m_decisionModel->index(row, 0);
        if (!index.isValid())
            continue;

        QVariantMap entry;
        for (auto it = roleNames.constBegin(); it != roleNames.constEnd(); ++it)
            entry.insert(QString::fromUtf8(it.value()), m_decisionModel->data(index, it.key()));
        entry.insert(QStringLiteral("row"), row);

        const QString state = entry.value(QStringLiteral("decisionState")).toString().toLower();
        const QString mode = entry.value(QStringLiteral("decisionMode")).toString().toLower();
        const bool approved = entry.value(QStringLiteral("approved")).toBool();
        const QString reason = entry.value(QStringLiteral("decisionReason")).toString();

        if (appended < limit) {
            recent.append(entry);
            ++appended;
        }

        if (!approved || (!state.isEmpty() && state != QStringLiteral("executed")) || !reason.trimmed().isEmpty()) {
            if (flaggedCount < kFlaggedLimit) {
                flagged.append(entry);
                ++flaggedCount;
            }
        }

        outcome[QStringLiteral("approved")] = outcome.value(QStringLiteral("approved")).toInt() + (approved ? 1 : 0);
        outcome[QStringLiteral("declined")] = outcome.value(QStringLiteral("declined")).toInt() + (approved ? 0 : 1);

        if (state == QStringLiteral("executed"))
            outcome[QStringLiteral("executed")] = outcome.value(QStringLiteral("executed")).toInt() + 1;
        else if (!state.isEmpty())
            outcome[QStringLiteral("rejected")] = outcome.value(QStringLiteral("rejected")).toInt() + 1;
        else
            outcome[QStringLiteral("pending")] = outcome.value(QStringLiteral("pending")).toInt() + 1;

        if (mode == QStringLiteral("manual"))
            outcome[QStringLiteral("manual")] = outcome.value(QStringLiteral("manual")).toInt() + 1;
        else if (!mode.isEmpty())
            outcome[QStringLiteral("automated")] = outcome.value(QStringLiteral("automated")).toInt() + 1;

        const QString schedule = entry.value(QStringLiteral("schedule")).toString();
        ScheduleStats stats = scheduleStats.value(schedule);
        ++stats.total;
        if (state == QStringLiteral("executed"))
            ++stats.executed;
        else if (!state.isEmpty())
            ++stats.rejected;
        if (!stats.lastDecision.contains(QStringLiteral("timestamp"))
            || entry.value(QStringLiteral("timestamp")) > stats.lastDecision.value(QStringLiteral("timestamp"))) {
            stats.lastDecision = entry;
        }
        scheduleStats.insert(schedule, stats);

        if (!updated.isValid()) {
            const QVariant timestampValue = entry.value(QStringLiteral("timestamp"));
            if (timestampValue.canConvert<QDateTime>())
                updated = timestampValue.toDateTime();
            else
                updated = QDateTime::currentDateTimeUtc();
        }
    }

    if (!scheduleStats.isEmpty()) {
        QStringList keys = scheduleStats.keys();
        std::sort(keys.begin(), keys.end(), [](const QString& a, const QString& b) {
            return a.toLower() < b.toLower();
        });
        for (const QString& key : keys) {
            const ScheduleStats stats = scheduleStats.value(key);
            QVariantMap map;
            map.insert(QStringLiteral("schedule"), key);
            map.insert(QStringLiteral("total"), stats.total);
            map.insert(QStringLiteral("executed"), stats.executed);
            map.insert(QStringLiteral("rejected"), stats.rejected);
            map.insert(QStringLiteral("lastDecision"), stats.lastDecision);
            schedules.append(map);
        }
    }

    if (m_recentDecisions != recent) {
        m_recentDecisions = recent;
        Q_EMIT recentDecisionsChanged();
    }
    if (m_flaggedDecisions != flagged) {
        m_flaggedDecisions = flagged;
        Q_EMIT flaggedDecisionsChanged();
    }
    if (m_outcomeSummary != outcome) {
        m_outcomeSummary = outcome;
        Q_EMIT outcomeSummaryChanged();
    }
    if (m_scheduleSummary != schedules) {
        m_scheduleSummary = schedules;
        Q_EMIT scheduleSummaryChanged();
    }

    const QDateTime normalized = updated.isValid() ? updated.toUTC() : QDateTime::currentDateTimeUtc();
    if (m_lastUpdated != normalized) {
        m_lastUpdated = normalized;
        Q_EMIT lastUpdatedChanged();
    }
