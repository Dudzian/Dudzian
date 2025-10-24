#include <QtTest>
#include <QDateTime>
#include <QFile>
#include <QTemporaryDir>
#include <QUrl>

#include "models/DecisionLogFilterProxyModel.hpp"

namespace {
class FakeDecisionModel : public QAbstractListModel {
    Q_OBJECT
public:
    struct Entry {
        QDateTime timestamp;
        QString environment;
        QString portfolio;
        QString riskProfile;
        QString schedule;
        QString strategy;
        QString symbol;
        QString side;
        QString decisionState;
        QString decisionReason;
        bool approved = false;
        QString decisionMode;
        QString event = QStringLiteral("order");
    };

    explicit FakeDecisionModel(QObject* parent = nullptr)
        : QAbstractListModel(parent) {}

    int rowCount(const QModelIndex& parent = QModelIndex()) const override {
        if (parent.isValid())
            return 0;
        return m_entries.size();
    }

    QVariant data(const QModelIndex& index, int role) const override {
        if (!index.isValid() || index.row() < 0 || index.row() >= m_entries.size())
            return {};
        const Entry& entry = m_entries.at(index.row());
        switch (role) {
        case DecisionLogModel::TimestampRole:
            return entry.timestamp;
        case DecisionLogModel::TimestampDisplayRole:
            return entry.timestamp.isValid() ? entry.timestamp.toString(Qt::ISODate) : QString();
        case DecisionLogModel::EnvironmentRole:
            return entry.environment;
        case DecisionLogModel::PortfolioRole:
            return entry.portfolio;
        case DecisionLogModel::RiskProfileRole:
            return entry.riskProfile;
        case DecisionLogModel::ScheduleRole:
            return entry.schedule;
        case DecisionLogModel::StrategyRole:
            return entry.strategy;
        case DecisionLogModel::SymbolRole:
            return entry.symbol;
        case DecisionLogModel::SideRole:
            return entry.side;
        case DecisionLogModel::DecisionStateRole:
            return entry.decisionState;
        case DecisionLogModel::DecisionReasonRole:
            return entry.decisionReason;
        case DecisionLogModel::ApprovedRole:
            return entry.approved;
        case DecisionLogModel::DecisionModeRole:
            return entry.decisionMode;
        case DecisionLogModel::EventRole:
            return entry.event;
        case DecisionLogModel::TelemetryNamespaceRole:
            return QStringLiteral("trend");
        default:
            return {};
        }
    }

    QHash<int, QByteArray> roleNames() const override {
        return {
            {DecisionLogModel::TimestampRole, "timestamp"},
            {DecisionLogModel::TimestampDisplayRole, "timestampDisplay"},
            {DecisionLogModel::EnvironmentRole, "environment"},
            {DecisionLogModel::PortfolioRole, "portfolio"},
            {DecisionLogModel::RiskProfileRole, "riskProfile"},
            {DecisionLogModel::ScheduleRole, "schedule"},
            {DecisionLogModel::StrategyRole, "strategy"},
            {DecisionLogModel::SymbolRole, "symbol"},
            {DecisionLogModel::SideRole, "side"},
            {DecisionLogModel::DecisionStateRole, "decisionState"},
            {DecisionLogModel::DecisionReasonRole, "decisionReason"},
            {DecisionLogModel::ApprovedRole, "approved"},
            {DecisionLogModel::DecisionModeRole, "decisionMode"},
            {DecisionLogModel::EventRole, "event"},
            {DecisionLogModel::TelemetryNamespaceRole, "regime"},
        };
    }

    void setEntries(const QVector<Entry>& entries) {
        beginResetModel();
        m_entries = entries;
        endResetModel();
    }

private:
    QVector<Entry> m_entries;
};
} // namespace

class DecisionLogFilterProxyModelTest : public QObject {
    Q_OBJECT
private slots:
    void filtersBySearchText();
    void exportsFilteredRows();
    void filtersByTimeRange();
    void filtersBySymbol();
    void filtersByRiskProfileAndSchedule();
    void filtersByEnvironmentAndPortfolio();
    void filtersBySideStateAndMode();
    void filtersByEventAndReason();
};

void DecisionLogFilterProxyModelTest::filtersBySearchText() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "weekly", "mean_reversion", "BTC/USDT", "SELL", "pending", "Rebalance", false, "manual"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 2);

    proxy.setSearchText("Breakout");
    QCOMPARE(proxy.rowCount(), 1);
    QModelIndex idx = proxy.index(0, 0);
    QCOMPARE(proxy.data(idx, DecisionLogModel::DecisionReasonRole).toString(), QStringLiteral("Breakout"));

    proxy.setSearchText("mean");
    QCOMPARE(proxy.rowCount(), 1);
    idx = proxy.index(0, 0);
    QCOMPARE(proxy.data(idx, DecisionLogModel::StrategyRole).toString(), QStringLiteral("mean_reversion"));

    proxy.setSearchText("");
    proxy.setApprovalFilter(DecisionLogFilterProxyModel::ApprovedOnly);
    QCOMPARE(proxy.rowCount(), 1);
    QVERIFY(proxy.data(proxy.index(0, 0), DecisionLogModel::ApprovedRole).toBool());
}

void DecisionLogFilterProxyModelTest::exportsFilteredRows() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "weekly", "mean_reversion", "BTC/USDT", "SELL", "pending", "Rebalance", false, "manual"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);
    proxy.setApprovalFilter(DecisionLogFilterProxyModel::ApprovedOnly);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString filePath = dir.filePath("export.csv");
    const QUrl url = QUrl::fromLocalFile(filePath);

    QVERIFY(proxy.exportFilteredToCsv(url));

    QFile file(filePath);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString contents = QString::fromUtf8(file.readAll());
    QVERIFY(contents.contains(QStringLiteral("Breakout")));
    QVERIFY(!contents.contains(QStringLiteral("Rebalance")));
}

void DecisionLogFilterProxyModelTest::filtersByTimeRange() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "mean_reversion", "BTC/USDT", "SELL", "pending", "Rebalance", false, "manual"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "gamma", "balanced", "session", "trend", "ETH/USDT", "BUY", "approved", "Continuation", true, "auto"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setStartTimeFilter(QDateTime::fromString("2024-01-01T10:04:00Z", Qt::ISODate));
    QCOMPARE(proxy.rowCount(), 2);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setEndTimeFilter(QDateTime::fromString("2024-01-01T10:06:00Z", Qt::ISODate));
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::StrategyRole).toString(), QStringLiteral("mean_reversion"));

    proxy.setEndTimeFilter(QDateTime::fromString("2024-01-01T09:55:00Z", Qt::ISODate));
    QCOMPARE(proxy.rowCount(), 0);

    proxy.setEndTimeFilter(QDateTime::fromString("2024-01-01T10:08:00Z", Qt::ISODate));
    proxy.setStartTimeFilter(QDateTime::fromString("2024-01-01T10:09:00Z", Qt::ISODate));
    QVERIFY(proxy.endTimeFilter().isValid());
    QCOMPARE(proxy.endTimeFilter(), proxy.startTimeFilter());

    proxy.clearEndTimeFilter();
    proxy.clearStartTimeFilter();
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersBySymbol() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "trend", "ETH/USDT", "SELL", "approved", "Continuation", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "gamma", "conservative", "weekly", "trend", "SOL/USDT", "SELL", "approved", "Rebalance", true, "auto"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setSymbolFilter("ETH");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setSymbolFilter("usdt");
    QCOMPARE(proxy.rowCount(), 3);

    proxy.setSymbolFilter("SOL/USDT");
    QCOMPARE(proxy.rowCount(), 1);

    proxy.setSymbolFilter("");
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersByRiskProfileAndSchedule() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-02T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-02T11:00:00Z", Qt::ISODate), "prod", "alpha", "aggressive", "session", "trend", "ETH/USDT", "SELL", "approved", "Rebalance", true, "auto"},
        {QDateTime::fromString("2024-01-02T12:00:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "mean_reversion", "SOL/USDT", "BUY", "pending", "Rotation", false, "manual"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setRiskProfileFilter("agg");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setRiskProfileFilter("balanced");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setRiskProfileFilter("");
    proxy.setScheduleFilter("weekly");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("SOL/USDT"));

    proxy.setScheduleFilter("sess");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setScheduleFilter("");
    proxy.setRiskProfileFilter("conservative");
    proxy.setScheduleFilter("weekly");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("SOL/USDT"));
}

void DecisionLogFilterProxyModelTest::filtersByEnvironmentAndPortfolio() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "staging", "alpha", "aggressive", "session", "trend", "ETH/USDT", "SELL", "approved", "Continuation", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "momentum", "SOL/USDT", "SELL", "approved", "Rebalance", true, "auto"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setEnvironmentFilter("prod");
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setPortfolioFilter("alpha");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setEnvironmentFilter("staging");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setEnvironmentFilter("prod");
    proxy.setPortfolioFilter("beta");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("SOL/USDT"));

    proxy.setEnvironmentFilter("prod");
    proxy.setPortfolioFilter("");
    QCOMPARE(proxy.rowCount(), 2);
}

void DecisionLogFilterProxyModelTest::filtersBySideStateAndMode() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-03T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-03T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "pending", "Rebalance", false, "manual"},
        {QDateTime::fromString("2024-01-03T10:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "scalping", "SOL/USDT", "SELL", "rejected", "Risk limit", false, "auto"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setSideFilter("buy");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setSideFilter("sel");
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setDecisionStateFilter("pend");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionStateRole).toString(), QStringLiteral("pending"));

    proxy.setDecisionModeFilter("manual");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionModeRole).toString(), QStringLiteral("manual"));

    proxy.setDecisionStateFilter("approved");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionStateRole).toString(), QStringLiteral("approved"));

    proxy.setDecisionModeFilter("auto");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setDecisionModeFilter("");
    proxy.setDecisionStateFilter("");
    proxy.setSideFilter("");
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersByEventAndReason() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-04T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto", "order_fill"},
        {QDateTime::fromString("2024-01-04T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "pending", "Risk limit", false, "manual", "risk_alert"},
        {QDateTime::fromString("2024-01-04T10:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "scalping", "SOL/USDT", "SELL", "rejected", "Cooldown", false, "auto", "order_fill"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setDecisionReasonFilter("risk");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionReasonRole).toString(), QStringLiteral("Risk limit"));

    proxy.setDecisionReasonFilter("cool");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionReasonRole).toString(), QStringLiteral("Cooldown"));

    proxy.setDecisionReasonFilter("");
    proxy.setEventFilter("order");
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setEventFilter("risk");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DecisionModeRole).toString(), QStringLiteral("manual"));

    proxy.setEventFilter("");
    QCOMPARE(proxy.rowCount(), 3);
}

QTEST_MAIN(DecisionLogFilterProxyModelTest)
#include "DecisionLogFilterProxyModelTest.moc"
