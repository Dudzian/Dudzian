#include <QtTest>
#include <QDateTime>
#include <QFile>
#include <QTemporaryDir>
#include <QUrl>
#include <QVariant>

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
        QString quantity;
        QString price;
        QString decisionState;
        QString decisionReason;
        bool approved = false;
        QString decisionMode;
        QString event = QStringLiteral("order");
        QString telemetryNamespace = QStringLiteral("trend");
        QVariantMap details;
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
        case DecisionLogModel::QuantityRole:
            return entry.quantity;
        case DecisionLogModel::PriceRole:
            return entry.price;
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
            return entry.telemetryNamespace;
        case DecisionLogModel::DetailsRole:
            return entry.details;
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
            {DecisionLogModel::QuantityRole, "quantity"},
            {DecisionLogModel::PriceRole, "price"},
            {DecisionLogModel::DecisionStateRole, "decisionState"},
            {DecisionLogModel::DecisionReasonRole, "decisionReason"},
            {DecisionLogModel::ApprovedRole, "approved"},
            {DecisionLogModel::DecisionModeRole, "decisionMode"},
            {DecisionLogModel::EventRole, "event"},
            {DecisionLogModel::TelemetryNamespaceRole, "telemetryNamespace"},
            {DecisionLogModel::DetailsRole, "details"},
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
    void filtersByTelemetryNamespace();
    void filtersBySideStateAndMode();
    void filtersByEventAndReason();
    void filtersByQuantityAndPrice();
    void filtersByQuantityAndPriceRange();
    void filtersByDetails();
    void clearsAllFilters();
};

void DecisionLogFilterProxyModelTest::filtersBySearchText() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "weekly", "mean_reversion", "BTC/USDT", "SELL", "2.0", "45200", "pending", "Rebalance", false, "manual"}
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
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "weekly", "mean_reversion", "BTC/USDT", "SELL", "2.0", "45200", "pending", "Rebalance", false, "manual"}
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
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "mean_reversion", "BTC/USDT", "SELL", "2.1", "45150", "pending", "Rebalance", false, "manual"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "gamma", "balanced", "session", "trend", "ETH/USDT", "BUY", "3.0", "2400", "approved", "Continuation", true, "auto"}
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
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "trend", "ETH/USDT", "SELL", "2.0", "2350", "approved", "Continuation", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "gamma", "conservative", "weekly", "trend", "SOL/USDT", "SELL", "4.2", "100", "approved", "Rebalance", true, "auto"}
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
        {QDateTime::fromString("2024-01-02T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.2", "44800", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-02T11:00:00Z", Qt::ISODate), "prod", "alpha", "aggressive", "session", "trend", "ETH/USDT", "SELL", "2.5", "2380", "approved", "Rebalance", true, "auto"},
        {QDateTime::fromString("2024-01-02T12:00:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "mean_reversion", "SOL/USDT", "BUY", "5.0", "95", "pending", "Rotation", false, "manual"}
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
        {QDateTime::fromString("2024-01-01T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:05:00Z", Qt::ISODate), "staging", "alpha", "aggressive", "session", "trend", "ETH/USDT", "SELL", "2.0", "2365", "approved", "Continuation", true, "auto"},
        {QDateTime::fromString("2024-01-01T10:10:00Z", Qt::ISODate), "prod", "beta", "conservative", "weekly", "momentum", "SOL/USDT", "SELL", "4.0", "98", "approved", "Rebalance", true, "auto"}
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

void DecisionLogFilterProxyModelTest::filtersByTelemetryNamespace() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-04T09:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.1", "44950", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-04T09:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "trend", "ETH/USDT", "SELL", "2.4", "2375", "pending", "Rebalance", false, "manual"},
        {QDateTime::fromString("2024-01-04T09:10:00Z", Qt::ISODate), "prod", "gamma", "balanced", "weekly", "momentum", "SOL/USDT", "SELL", "3.2", "99", "approved", "Trailing", true, "auto"}
    };
    entries[0].telemetryNamespace = QStringLiteral("regime/trend");
    entries[1].telemetryNamespace = QStringLiteral("signals/alerts");
    entries[2].telemetryNamespace = QStringLiteral("regime/momentum");
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setTelemetryNamespaceFilter("regime/");
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setTelemetryNamespaceFilter("momentum");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("SOL/USDT"));

    proxy.setTelemetryNamespaceFilter("signals");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setTelemetryNamespaceFilter("");
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersBySideStateAndMode() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-03T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.7", "45120", "approved", "Breakout", true, "auto"},
        {QDateTime::fromString("2024-01-03T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "2.3", "2395", "pending", "Rebalance", false, "manual"},
        {QDateTime::fromString("2024-01-03T10:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "scalping", "SOL/USDT", "SELL", "3.8", "102", "rejected", "Risk limit", false, "auto"}
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
        {QDateTime::fromString("2024-01-04T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.4", "44950", "approved", "Breakout", true, "auto", "order_fill"},
        {QDateTime::fromString("2024-01-04T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "2.2", "2375", "pending", "Risk limit", false, "manual", "risk_alert"},
        {QDateTime::fromString("2024-01-04T10:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "scalping", "SOL/USDT", "SELL", "3.5", "99", "rejected", "Cooldown", false, "auto", "order_fill"}
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

void DecisionLogFilterProxyModelTest::filtersByQuantityAndPrice() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-05T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "0.8", "44850", "approved", "Breakout", true, "auto", "order_fill"},
        {QDateTime::fromString("2024-01-05T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "3.1", "2390", "pending", "Risk limit", false, "manual", "risk_alert"},
        {QDateTime::fromString("2024-01-05T10:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "mean_reversion", "SOL/USDT", "BUY", "6.0", "101", "approved", "Rotation", true, "auto", "order_fill"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setQuantityFilter("0.8");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setQuantityFilter("3.");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETH/USDT"));

    proxy.setQuantityFilter("");
    proxy.setPriceFilter("101");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("SOL/USDT"));

    proxy.setPriceFilter("44");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::SymbolRole).toString(), QStringLiteral("BTC/USDT"));

    proxy.setPriceFilter("");
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersByQuantityAndPriceRange() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-07T11:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.5", "45000", "approved", "Breakout", true, "auto", "order_fill"},
        {QDateTime::fromString("2024-01-07T11:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "2.5", "45500", "pending", "Risk limit", false, "manual", "risk_alert"},
        {QDateTime::fromString("2024-01-07T11:10:00Z", Qt::ISODate), "staging", "gamma", "conservative", "weekly", "mean_reversion", "SOL/USDT", "BUY", "3.5", "46000", "approved", "Rotation", true, "auto", "order_fill"}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 3);

    proxy.setMinQuantityFilter(2.0);
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setMaxQuantityFilter(3.0);
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::QuantityRole).toString(), QStringLiteral("2.5"));

    proxy.setMinQuantityFilter(QVariant());
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setMaxQuantityFilter(QVariant());
    QCOMPARE(proxy.rowCount(), 3);

    proxy.setMinPriceFilter(45500.0);
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setMaxPriceFilter(45600.0);
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::PriceRole).toString(), QStringLiteral("45500"));

    proxy.setMaxPriceFilter(45000.0);
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::PriceRole).toString(), QStringLiteral("45000"));

    proxy.setMinPriceFilter(QVariant());
    proxy.setMaxPriceFilter(QVariant());
    QCOMPARE(proxy.rowCount(), 3);
}

void DecisionLogFilterProxyModelTest::filtersByDetails() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-06T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.2", "44900", "approved", "Breakout", true, "auto", "order_fill", {{"algo", "trend"}, {"note", "primary"}}},
        {QDateTime::fromString("2024-01-06T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "2.4", "2380", "pending", "Risk limit", false, "manual", "risk_alert", {{"algo", "mean"}, {"note", "backup"}}}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 2);

    proxy.setDetailsFilter("primary");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DetailsRole).toMap().value(QStringLiteral("note")).toString(), QStringLiteral("primary"));

    proxy.setDetailsFilter("MEAN");
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.data(proxy.index(0, 0), DecisionLogModel::DetailsRole).toMap().value(QStringLiteral("algo")).toString(), QStringLiteral("mean"));

    proxy.setDetailsFilter("");
    QCOMPARE(proxy.rowCount(), 2);
}

void DecisionLogFilterProxyModelTest::clearsAllFilters() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {QDateTime::fromString("2024-01-06T10:00:00Z", Qt::ISODate), "prod", "alpha", "balanced", "daily", "trend", "BTC/USDT", "BUY", "1.2", "44900", "approved", "Breakout", true, "auto", "order_fill", {{"algo", "trend"}, {"note", "primary"}}},
        {QDateTime::fromString("2024-01-06T10:05:00Z", Qt::ISODate), "prod", "beta", "aggressive", "session", "momentum", "ETH/USDT", "SELL", "2.4", "2380", "pending", "Risk limit", false, "manual", "risk_alert", {{"algo", "mean"}, {"note", "backup"}}},
        {QDateTime::fromString("2024-01-06T10:10:00Z", Qt::ISODate), "uat", "gamma", "conservative", "session", "breakout", "ETH/USDT", "BUY", "3.5", "2400", "approved", "Continuation", true, "hybrid", "trade_signal", {{"tag", "momentum"}}}
    };
    source.setEntries(entries);

    DecisionLogFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    proxy.setSearchText("Breakout");
    proxy.setApprovalFilter(DecisionLogFilterProxyModel::ApprovedOnly);
    proxy.setStrategyFilter("trend");
    proxy.setRegimeFilter("trend");
    proxy.setTelemetryNamespaceFilter("stream");
    proxy.setEnvironmentFilter("prod");
    proxy.setPortfolioFilter("alpha");
    proxy.setRiskProfileFilter("balanced");
    proxy.setScheduleFilter("daily");
    proxy.setSideFilter("BUY");
    proxy.setDecisionStateFilter("approved");
    proxy.setDecisionModeFilter("auto");
    proxy.setDecisionReasonFilter("Breakout");
    proxy.setEventFilter("order_fill");
    proxy.setQuantityFilter("1.2");
    proxy.setPriceFilter("44900");
    proxy.setSymbolFilter("BTC/USDT");
    proxy.setDetailsFilter("primary");
    proxy.setMinQuantityFilter(1.0);
    proxy.setMaxQuantityFilter(2.0);
    proxy.setMinPriceFilter(44000.0);
    proxy.setMaxPriceFilter(50000.0);
    proxy.setStartTimeFilter(entries.first().timestamp);
    proxy.setEndTimeFilter(entries[1].timestamp);

    QVERIFY(proxy.rowCount() < entries.size());

    proxy.clearAllFilters();

    QCOMPARE(proxy.rowCount(), entries.size());
    QVERIFY(proxy.searchText().isEmpty());
    QCOMPARE(proxy.approvalFilter(), DecisionLogFilterProxyModel::All);
    QVERIFY(proxy.strategyFilter().isEmpty());
    QVERIFY(proxy.regimeFilter().isEmpty());
    QVERIFY(proxy.telemetryNamespaceFilter().isEmpty());
    QVERIFY(proxy.environmentFilter().isEmpty());
    QVERIFY(proxy.portfolioFilter().isEmpty());
    QVERIFY(proxy.riskProfileFilter().isEmpty());
    QVERIFY(proxy.scheduleFilter().isEmpty());
    QVERIFY(proxy.sideFilter().isEmpty());
    QVERIFY(proxy.decisionStateFilter().isEmpty());
    QVERIFY(proxy.decisionModeFilter().isEmpty());
    QVERIFY(proxy.decisionReasonFilter().isEmpty());
    QVERIFY(proxy.eventFilter().isEmpty());
    QVERIFY(proxy.quantityFilter().isEmpty());
    QVERIFY(proxy.priceFilter().isEmpty());
    QVERIFY(proxy.symbolFilter().isEmpty());
    QVERIFY(proxy.detailsFilter().isEmpty());
    QVERIFY(!proxy.minQuantityFilter().isValid());
    QVERIFY(!proxy.maxQuantityFilter().isValid());
    QVERIFY(!proxy.minPriceFilter().isValid());
    QVERIFY(!proxy.maxPriceFilter().isValid());
    QVERIFY(!proxy.startTimeFilter().isValid());
    QVERIFY(!proxy.endTimeFilter().isValid());
}

QTEST_MAIN(DecisionLogFilterProxyModelTest)
#include "DecisionLogFilterProxyModelTest.moc"
